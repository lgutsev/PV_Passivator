"""Objective and constraint specification, frozen by hash, plus Pareto analysis.

Criteria are kept separate: each is an objective (min/max) or a constraint (bounds). There is no weighted
scalar objective in reporting. Optimizers may scalarise *internally* for acquisition only (ParEGO-style random
Chebyshev weights, see optimize.py); results are always judged by feasibility + Pareto dominance.

A spec is loaded once per campaign; ``ObjectiveSpec.hash`` is written into every store record and the loop
refuses to continue if it changes (``store.EvaluationStore`` checks it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Criterion:
    name: str
    role: str                      # "objective" | "constraint"
    direction: str = "min"         # objectives: "min" | "max"
    lower: float | None = None     # constraints: feasible if lower <= value <= upper
    upper: float | None = None
    units: str = ""
    tier: str = ""                 # where the value comes from (tier0 / tier1 / tier2_dft / synthetic)
    description: str = ""


@dataclass(frozen=True)
class ObjectiveSpec:
    id: str
    criteria: tuple[Criterion, ...]
    notes: str = ""
    raw: str = field(default="", compare=False, repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "ObjectiveSpec":
        text = Path(path).read_text(encoding="utf-8")
        d = json.loads(text)
        crit = tuple(Criterion(**c) for c in d["criteria"])
        spec = cls(d["id"], crit, d.get("notes", ""), json.dumps(d, sort_keys=True))
        spec.check()
        return spec

    def check(self) -> None:
        names = [c.name for c in self.criteria]
        if len(set(names)) != len(names):
            raise ValueError("duplicate criterion names")
        objs = self.objectives
        if len(objs) < 2:
            raise ValueError("at least two objectives are required (a single proxy is not accepted)")
        for c in self.criteria:
            if c.role == "objective" and c.direction not in ("min", "max"):
                raise ValueError(f"{c.name}: direction must be min or max")
            if c.role == "constraint" and c.lower is None and c.upper is None:
                raise ValueError(f"{c.name}: constraint needs a bound")
            if c.role not in ("objective", "constraint"):
                raise ValueError(f"{c.name}: role must be objective or constraint")

    @property
    def hash(self) -> str:
        return hashlib.sha256((self.raw or repr(self.criteria)).encode()).hexdigest()[:16]

    @property
    def objectives(self) -> list[Criterion]:
        return [c for c in self.criteria if c.role == "objective"]

    @property
    def constraints(self) -> list[Criterion]:
        return [c for c in self.criteria if c.role == "constraint"]

    def complete(self, values: dict) -> bool:
        return all(values.get(c.name) is not None for c in self.criteria)

    def feasible(self, values: dict) -> bool:
        for c in self.constraints:
            v = values.get(c.name)
            if v is None:
                return False
            if c.lower is not None and v < c.lower:
                return False
            if c.upper is not None and v > c.upper:
                return False
        return True

    def as_min(self, values: dict) -> np.ndarray:
        """Objective vector in 'smaller is better' orientation."""
        return np.array([values[c.name] if c.direction == "min" else -values[c.name] for c in self.objectives])


def nondominated(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows of Y (minimisation)."""
    n = len(Y)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(Y <= Y[i], 1) & np.any(Y < Y[i], 1)
        if dom.any():
            keep[i] = False
    return keep


def hypervolume(Y: np.ndarray, ref: np.ndarray, n_mc: int = 200_000, seed: int = 0) -> float:
    """Dominated hypervolume w.r.t. ``ref`` (minimisation). Exact for 2 objectives, Monte Carlo above."""
    Y = np.asarray(Y, float)
    if len(Y) == 0:
        return 0.0
    Y = Y[np.all(Y < ref, 1)]
    if len(Y) == 0:
        return 0.0
    Y = Y[nondominated(Y)]
    if Y.shape[1] == 2:
        Y = Y[np.argsort(Y[:, 0])]
        hv, prev = 0.0, ref[1]
        for x, y in Y:
            if y < prev:
                hv += (ref[0] - x) * (prev - y)
                prev = y
        return float(hv)
    lo = Y.min(0)
    rng = np.random.default_rng(seed)
    P = lo + rng.random((n_mc, Y.shape[1])) * (ref - lo)
    dom = np.zeros(n_mc, bool)
    for y in Y:
        dom |= np.all(P >= y, 1)
    return float(dom.mean() * np.prod(ref - lo))


def front_report(spec: ObjectiveSpec, rows: list[dict]) -> dict:
    """Feasible Pareto set of evaluated rows ({'key':..., 'values': {...}})."""
    ok = [r for r in rows if spec.complete(r["values"]) and spec.feasible(r["values"])]
    if not ok:
        return {"n_feasible": 0, "front": []}
    Y = np.array([spec.as_min(r["values"]) for r in ok])
    m = nondominated(Y)
    return {"n_feasible": len(ok), "front": [ok[i]["key"] for i in np.flatnonzero(m)]}
