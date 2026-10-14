"""Closed loop (arm C) and the matched-budget three-arm benchmark.

Arms share: eligibility rules, objective spec, oracle, evaluation budget, batch size, and starting knowledge (none).
  A  expert library, evaluated in the expert's priority order (conventional screening);
  B  conventional active learning: GPParEGO over a fixed, conventionally enumerated space;
  C  representation search: each round a proposer supplies a representation; the same GPParEGO searches the
     union of all accepted representations so far.
B and C use the same optimizer, so differences isolate the effect of the proposed representations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chem import Candidate, Eligibility
from .objectives import ObjectiveSpec
from .optimize import Optimizer
from .oracles import Oracle, NotComputed
from .proposer import Proposer, ProposalLog, extract_json, feedback_prompt
from .space import enumerate_spec, spec_hash
from .store import EvaluationStore, Record


@dataclass
class ArmResult:
    arm: str
    evaluated: list[Candidate]
    store: EvaluationStore
    pool_size: int
    proposals: list[ProposalLog] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    unused_budget: float = 0.0


def _evaluate(batch: list[Candidate], oracle: Oracle, store: EvaluationStore, arm: str, rnd: int) -> list[Candidate]:
    done = []
    for c in batch:
        if store.has(c.inchikey, oracle.id) or not store.charge_ok(oracle.cost):
            continue
        try:
            res = oracle.evaluate(c)
        except NotComputed:
            continue
        store.add(Record(c.inchikey, c.smiles, oracle.id, res.status, res.values, res.cost, "search",
                         c.source or arm, rnd, meta=res.meta))
        done.append(c)
    return done


def run_pool(arm: str, pool: list[Candidate], optimizer: Optimizer, oracle: Oracle, spec: ObjectiveSpec,
             budget: float, batch: int, seed: int, store_path: str | Path | None = None) -> ArmResult:
    """Arms A and B: one fixed pool, batches until the budget or the pool is exhausted."""
    rng = np.random.default_rng(seed)
    store = EvaluationStore(store_path, spec.hash, budget)
    byk = {c.inchikey: c for c in pool}
    evaluated = [byk[r.key] for r in store.search_view() if r.key in byk]   # resume
    rnd = 0
    while store.charge_ok(oracle.cost):
        seen = {c.inchikey for c in evaluated}
        remaining = [c for c in pool if c.inchikey not in seen]
        if not remaining:
            break
        observed = [(c, store.get(c.inchikey, oracle.id).values) for c in evaluated]
        k = min(batch, int((budget - store.spent) / oracle.cost + 1e-9) if oracle.cost else batch)
        idx = optimizer.suggest(remaining, observed, k, rng)
        got = _evaluate([remaining[i] for i in idx], oracle, store, arm, rnd)
        if not got:
            break
        evaluated += got
        rnd += 1
    return ArmResult(arm, evaluated, store, len(pool), unused_budget=budget - store.spent)


def run_representation_loop(proposer: Proposer, optimizer: Optimizer, oracle: Oracle, spec: ObjectiveSpec,
                            budget: float, batch: int, rounds: int, seed: int,
                            rules: Eligibility = Eligibility(), seed_specs: list[dict] | None = None,
                            store_path: str | Path | None = None, min_valid: int = 5) -> ArmResult:
    """Arm C. Each round: propose -> validate/enumerate -> optimise over the accepted union -> evaluate -> feedback.

    Proposals that fail validation are logged and cost no evaluations. ``seed_specs`` lets every arm start from
    the same knowledge if desired (default: none).
    """
    rng = np.random.default_rng(seed)
    store = EvaluationStore(store_path, spec.hash, budget)
    pool: dict[str, Candidate] = {}
    history: list[dict] = []
    logs: list[ProposalLog] = []
    evaluated: list[Candidate] = []
    per_round = budget / rounds

    def accept(d: dict):
        cands, rep = enumerate_spec(d, rules)
        if rep.ok and rep.n_valid >= min_valid:
            for c in cands:
                pool.setdefault(c.inchikey, c)
            history.append({"id": d["id"], "hash": spec_hash(d), "n_enumerated": rep.n_enumerated,
                            "n_valid": rep.n_valid, "rationale": d.get("rationale", ""),
                            "rejections": rep.rejections})
        return rep

    for d in seed_specs or []:
        accept(d)
    for rnd in range(rounds):
        prompt = feedback_prompt(store, spec, history)
        text, usage = proposer.propose(prompt)
        d = extract_json(text) if text else None
        if d is None:
            logs.append(ProposalLog(rnd, len(prompt), len(text or ""), None, False, ["no JSON object"], usage))
        else:
            rep = accept(d)
            ok = rep.ok and rep.n_valid >= min_valid
            errs = rep.errors or ([] if ok else [f"only {rep.n_valid} valid candidates"])
            logs.append(ProposalLog(rnd, len(prompt), len(text), d.get("id"), ok, errs, usage))
        target = min(budget, per_round * (rnd + 1))
        while store.spent + oracle.cost <= target + 1e-9:
            seen = {c.inchikey for c in evaluated}
            remaining = [c for c in pool.values() if c.inchikey not in seen]
            if not remaining:
                break
            observed = [(c, store.get(c.inchikey, oracle.id).values) for c in evaluated]
            k = max(1, min(batch, int((target - store.spent) / oracle.cost + 1e-9) if oracle.cost else batch))
            idx = optimizer.suggest(remaining, observed, k, rng)
            got = _evaluate([remaining[i] for i in idx], oracle, store, "C", rnd)
            if not got:
                break
            evaluated += got
    return ArmResult("C", evaluated, store, len(pool), logs, history, budget - store.spent)
