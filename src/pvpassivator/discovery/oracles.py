"""Evaluators ("oracles"). Each declares an id/version, a tier, a cost in DFT-equivalent units and the status of
what it returns. Oracles above ``APPROVAL_COST`` refuse to run unless the campaign was explicitly approved.

* ``SyntheticOracle``   engineering test only. A seeded random landscape over generic descriptors; it has no
                        physical meaning and its results must never be reported as passivation evidence.
* ``CachedDFTOracle``   returns completed calculations from a JSON results file (e.g. built from
                        ``pvpass labels`` output); raises ``NotComputed`` for anything else. It never launches jobs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .chem import Candidate

APPROVAL_COST = 1.0          # anything at or above one DFT-equivalent per call needs explicit approval


class NotComputed(KeyError):
    pass


class ApprovalRequired(PermissionError):
    pass


@dataclass
class OracleResult:
    values: dict
    status: str              # "prediction" | "computed"
    cost: float
    meta: dict


class Oracle:
    id = "oracle"
    tier = ""
    cost = 0.0
    status = "prediction"

    def __init__(self, approved: bool = False):
        if self.cost >= APPROVAL_COST and not approved:
            raise ApprovalRequired(f"{self.id} costs {self.cost} DFT-equivalents per call; needs PI approval")

    def evaluate(self, cand: Candidate) -> OracleResult:   # pragma: no cover - interface
        raise NotImplementedError


class SyntheticOracle(Oracle):
    """Smooth random landscape (random Fourier features of standardised generic descriptors) + noise.

    Returned criteria mirror the *shape* of the physical problem (two objectives, two constraints) under
    ``syn_`` names so they cannot be mistaken for physics. The hidden weights depend only on ``seed``; neither the
    representations nor the optimizers can see them.
    """
    tier = "synthetic"
    status = "prediction"
    FEATS = ("heavy_atoms", "tpsa", "logp", "hbd", "hba", "rotatable", "aromatic_rings", "fraction_csp3",
             "n_fluorine", "n_halogen", "n_heteroatoms")
    SCALE = {"heavy_atoms": (12, 5), "tpsa": (30, 20), "logp": (0, 2), "hbd": (2, 1.5), "hba": (1, 1.5),
             "rotatable": (3, 2), "aromatic_rings": (1, 1), "fraction_csp3": (0.4, 0.3), "n_fluorine": (1, 2),
             "n_halogen": (1, 2), "n_heteroatoms": (3, 2)}

    def __init__(self, seed: int = 0, noise: float = 0.05, cost: float = 0.01, n_features: int = 64):
        self.cost = cost
        super().__init__()
        self.id = f"synthetic/v1/seed{seed}"
        rng = np.random.default_rng(10_000 + seed)
        d = len(self.FEATS)
        self._W = rng.normal(0, 0.7, (4, n_features, d))
        self._b = rng.uniform(0, 2 * np.pi, (4, n_features))
        self._a = rng.normal(0, 1, (4, n_features)) / np.sqrt(n_features)
        self._noise = noise
        self._seed = seed

    def _x(self, c: Candidate) -> np.ndarray:
        return np.array([(c.descriptors[f] - self.SCALE[f][0]) / self.SCALE[f][1] for f in self.FEATS])

    def _f(self, k: int, x: np.ndarray) -> float:
        return float(self._a[k] @ np.cos(self._W[k] @ x + self._b[k]) * np.sqrt(2))

    def evaluate(self, cand: Candidate) -> OracleResult:
        x = self._x(cand)
        h = int(hashlib.sha256(f"{cand.inchikey}|{self._seed}".encode()).hexdigest()[:8], 16)  # reproducible
        eps = np.random.default_rng(h).normal(0, self._noise, 4)
        vals = {
            "syn_trap_suppression": self._f(0, x) + eps[0],      # objective, maximise
            "syn_binding": self._f(1, x) - 1.0 + eps[1],          # constraint window
            "syn_integrity": self._f(2, x) + eps[2],              # constraint lower bound
            "sa_score": cand.descriptors["sa_score"],             # real (RDKit SA score), objective, minimise
        }
        return OracleResult(vals, self.status, self.cost, {"synthetic": True})

    def truth(self, cand: Candidate) -> dict:
        """Noise-free values, for benchmark scoring only (never given to optimizers or proposers)."""
        x = self._x(cand)
        return {"syn_trap_suppression": self._f(0, x), "syn_binding": self._f(1, x) - 1.0,
                "syn_integrity": self._f(2, x), "sa_score": cand.descriptors["sa_score"]}


class CachedDFTOracle(Oracle):
    """Completed DFT results keyed by InChIKey: {"profile": ..., "results": {inchikey: {criterion: value}}}.

    Charges no budget (the calculations are already done) but records status "computed" and the DFT profile.
    """
    tier = "tier2_dft"
    status = "computed"
    cost = 0.0

    def __init__(self, path: str | Path):
        super().__init__()
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.profile = d["profile"]
        self.results = d["results"]
        self.id = f"cached_dft/{self.profile}"

    def evaluate(self, cand: Candidate) -> OracleResult:
        if cand.inchikey not in self.results:
            raise NotComputed(cand.inchikey)
        return OracleResult(dict(self.results[cand.inchikey]), self.status, 0.0, {"profile": self.profile})
