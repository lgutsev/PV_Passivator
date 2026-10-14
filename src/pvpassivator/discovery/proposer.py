"""Representation proposers (provider-independent) and evidence-only feedback.

A proposer maps a feedback summary to a search-representation JSON (space.py). Implementations:

* ``ReplayProposer``    replays recorded proposals in order (offline, deterministic; used by tests and the pilot);
* ``CallableProposer``  wraps any ``fn(prompt: str) -> str | (str, usage_dict)``, so an Anthropic, OpenAI or local
                        model client plugs in without this package importing any provider SDK.

The feedback summary is built only from the store's search split and from representation-level statistics. It
never contains held-out results, oracle internals, or the acceptance thresholds in editable form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path

import numpy as np

from .objectives import ObjectiveSpec, nondominated
from .space import SCHEMA, PROTECTED_KEYS
from .store import EvaluationStore


@dataclass
class ProposalLog:
    round: int
    prompt_chars: int
    response_chars: int
    spec_id: str | None
    accepted: bool
    errors: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)        # tokens / cost reported by the provider, if any


class Proposer:
    name = "proposer"

    def propose(self, prompt: str) -> tuple[str, dict]:   # pragma: no cover - interface
        raise NotImplementedError


class ReplayProposer(Proposer):
    name = "replay"

    def __init__(self, paths: list[str | Path]):
        self.texts = [Path(p).read_text(encoding="utf-8") for p in paths]
        self.i = 0

    def propose(self, prompt):
        if self.i >= len(self.texts):
            return "", {"exhausted": True}
        t = self.texts[self.i]
        self.i += 1
        return t, {"replayed": True}


class CallableProposer(Proposer):
    name = "callable"

    def __init__(self, fn, name: str = "callable"):
        self.fn = fn
        self.name = name

    def propose(self, prompt):
        out = self.fn(prompt)
        return (out if isinstance(out, tuple) else (out, {}))


def extract_json(text: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else text.strip()
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def representation_stats(store: EvaluationStore, spec: ObjectiveSpec) -> dict[str, dict]:
    """Per-representation evidence from the search split only."""
    rows = [r for r in store.search_view() if spec.complete(r.values)]
    feas = [r for r in rows if spec.feasible(r.values)]
    front = set()
    if feas:
        Y = np.array([spec.as_min(r.values) for r in feas])
        front = {feas[i].key for i in np.flatnonzero(nondominated(Y))}
    out: dict[str, dict] = {}
    for r in rows:
        d = out.setdefault(r.source, {"evaluated": 0, "feasible": 0, "on_front": 0})
        d["evaluated"] += 1
        d["feasible"] += spec.feasible(r.values)
        d["on_front"] += r.key in front
    return out


def feedback_prompt(store: EvaluationStore, spec: ObjectiveSpec, history: list[dict], top_n: int = 8) -> str:
    """Evidence-grounded prompt. Objectives are shown read-only; protected fields are named as off-limits."""
    rows = [r for r in store.search_view() if spec.complete(r.values) and spec.feasible(r.values)]
    lines = [
        "You design SEARCH REPRESENTATIONS for organic passivating ions, not individual answers.",
        f"Return one JSON object with schema {SCHEMA!r}, kind 'combinatorial' (scaffolds with [*:k] attachment points "
        "and substituent lists using a single [*]) or 'library'.",
        f"You may NOT set any of: {sorted(PROTECTED_KEYS)}. Objectives, constraints and eligibility are fixed.",
        "Objectives (read-only): " + "; ".join(
            f"{c.name} ({c.role}{', ' + c.direction if c.role == 'objective' else ''})" for c in spec.criteria),
        f"Budget used so far: {store.spent:.2f} of {store.budget:.2f} DFT-equivalents.",
        "Evidence per previous representation (evaluated / feasible / on current Pareto front):",
    ]
    stats = representation_stats(store, spec)
    for h in history:
        s = stats.get(h["id"], {"evaluated": 0, "feasible": 0, "on_front": 0})
        lines.append(f"- {h['id']}: valid {h['n_valid']}/{h['n_enumerated']}, evaluated {s['evaluated']}, "
                     f"feasible {s['feasible']}, on front {s['on_front']}. Rationale was: {h.get('rationale', '')[:200]}")
    if rows:
        objs = spec.objectives
        rows = sorted(rows, key=lambda r: tuple(spec.as_min(r.values)))[:top_n]
        lines.append("Best feasible evaluated candidates (search split only):")
        for r in rows:
            lines.append(f"- {r.smiles} [{r.source}] " + ", ".join(f"{c.name}={r.values[c.name]:.3g}" for c in objs))
    lines.append("Propose ONE revised representation that explores chemistry the evidence suggests is under-sampled "
                 "or promising. Explain the design variables in 'rationale'.")
    return "\n".join(lines)
