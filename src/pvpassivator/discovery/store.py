"""Append-only evaluation store: provenance, status labels, dedup, budget accounting, held-out isolation.

Every record carries a ``status`` from STATUSES so hypotheses, surrogate predictions, completed calculations and
validated conclusions can never be confused. Records of the ``holdout`` split are written by the benchmark only;
``search_view`` (what optimizers and proposers see) excludes them, so held-out results cannot leak into model
selection or prompt revision.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
import time
from pathlib import Path

STATUSES = ("hypothesis", "prediction", "computed", "validated")


class BudgetExceeded(RuntimeError):
    pass


class SpecChanged(RuntimeError):
    pass


@dataclass
class Record:
    key: str                         # InChIKey (or salt key "cation|anion")
    smiles: str
    oracle: str                      # oracle id + version
    status: str
    values: dict
    cost: float                      # DFT-equivalent units charged
    split: str = "search"            # "search" | "holdout"
    source: str = ""                 # representation / arm that proposed it
    round: int = -1
    objective_hash: str = ""
    meta: dict = field(default_factory=dict)
    time: float = field(default_factory=time.time)


class EvaluationStore:
    """JSONL-backed store. Re-opening the same path resumes (checkpoint = the file itself)."""

    def __init__(self, path: str | Path | None, objective_hash: str, budget: float):
        self.path = Path(path) if path else None
        self.objective_hash = objective_hash
        self.budget = budget
        self.records: list[Record] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = Record(**json.loads(line))
                    if r.objective_hash and r.objective_hash != objective_hash:
                        raise SpecChanged(f"store {self.path} was written under objective spec {r.objective_hash}, "
                                          f"not {objective_hash}; start a new campaign instead of mixing")
                    self.records.append(r)

    @property
    def spent(self) -> float:
        return sum(r.cost for r in self.records if r.split == "search")

    def has(self, key: str, oracle: str) -> bool:
        return any(r.key == key and r.oracle == oracle for r in self.records)

    def get(self, key: str, oracle: str) -> Record | None:
        return next((r for r in self.records if r.key == key and r.oracle == oracle), None)

    def charge_ok(self, cost: float) -> bool:
        return self.spent + cost <= self.budget + 1e-9

    def add(self, rec: Record) -> Record:
        if rec.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if rec.split == "search" and not self.charge_ok(rec.cost):
            raise BudgetExceeded(f"budget {self.budget} would be exceeded ({self.spent} + {rec.cost})")
        if self.has(rec.key, rec.oracle):
            raise ValueError(f"duplicate evaluation {rec.key} with {rec.oracle}")
        rec.objective_hash = self.objective_hash
        self.records.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
        return rec

    def search_view(self) -> list[Record]:
        return [r for r in self.records if r.split == "search"]
