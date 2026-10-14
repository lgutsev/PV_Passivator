"""Optimizers select which enumerated candidates to evaluate next. They see only the search split of the store.

Interface: ``suggest(pool, observed, k, rng) -> list[int]`` where ``pool`` is the list of candidates still eligible
for this arm, ``observed`` the (candidate, values) pairs already evaluated (search split), and the return value
indexes into ``pool``.

``GPParEGO`` (the default numerical optimizer for arms B and C):
* surrogate: exact GP with a Tanimoto kernel on binary Morgan fingerprints (a standard choice for small molecular
  data sets; no deep model is fitted to tens of points), noise variance chosen by marginal likelihood on a grid;
* multi-objective acquisition: ParEGO, i.e. one random Chebyshev scalarisation per batch slot, so the batch
  spreads along the Pareto front without fixing preference weights (Knowles 2006);
* constraints: an independent GP per constraint gives P(feasible); acquisition = EI x P(feasible);
* batch diversity: a candidate whose Tanimoto similarity to one already chosen in this batch exceeds
  ``max_batch_similarity`` is skipped.

SAASBO (used by the reference P2O work) targets continuous, high-dimensional parameters with a sparsity prior; for
discrete molecular identities with fingerprints, a Tanimoto GP is the more established baseline. Molecule selection
and adsorption-geometry optimisation are kept separate: pose search happens inside the oracle (poses.py).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .chem import Candidate, fingerprints, tanimoto
from .objectives import ObjectiveSpec


class Optimizer:
    name = "optimizer"

    def suggest(self, pool: list[Candidate], observed: list[tuple[Candidate, dict]], k: int,
                rng: np.random.Generator) -> list[int]:   # pragma: no cover - interface
        raise NotImplementedError


class RandomOptimizer(Optimizer):
    name = "random"

    def suggest(self, pool, observed, k, rng):
        return list(rng.choice(len(pool), size=min(k, len(pool)), replace=False))


class FixedOrderOptimizer(Optimizer):
    """Conventional screening: evaluate in the order given (an expert's priority list)."""
    name = "fixed_order"

    def suggest(self, pool, observed, k, rng):
        return list(range(min(k, len(pool))))


class TanimotoGP:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.mu = float(y.mean())
        self.sd = float(y.std()) or 1.0
        z = (y - self.mu) / self.sd
        K = tanimoto(X, X)
        best = None
        for amp in (0.3, 1.0, 3.0, 10.0):                       # signal variance x noise variance, by evidence
            for noise in (1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3):
                try:
                    L = np.linalg.cholesky(amp * K + noise * np.eye(len(z)))
                except np.linalg.LinAlgError:
                    continue
                a = np.linalg.solve(L.T, np.linalg.solve(L, z))
                lml = -0.5 * z @ a - np.log(np.diag(L)).sum()
                if best is None or lml > best[0]:
                    best = (lml, L, a, amp)
        if best is None:
            raise np.linalg.LinAlgError("GP covariance not positive definite at any noise level")
        _, self.L, self.alpha, self.amp = best

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Ks = self.amp * tanimoto(Xs, self.X)
        m = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(self.amp - (v * v).sum(0), 1e-12)
        return m * self.sd + self.mu, np.sqrt(var) * self.sd


class GPParEGO(Optimizer):
    name = "gp_parego"

    def __init__(self, spec: ObjectiveSpec, n_init: int = 6, max_batch_similarity: float = 0.85):
        self.spec = spec
        self.n_init = n_init
        self.max_sim = max_batch_similarity
        self._fp: dict[str, np.ndarray] = {}

    def _fps(self, cands: list[Candidate]) -> np.ndarray:
        miss = [c.smiles for c in cands if c.smiles not in self._fp]
        if miss:
            for s, f in zip(miss, fingerprints(miss)):
                self._fp[s] = f
        return np.array([self._fp[c.smiles] for c in cands])

    def suggest(self, pool, observed, k, rng):
        obs = [(c, v) for c, v in observed if self.spec.complete(v)]
        if len(obs) < self.n_init:
            return RandomOptimizer().suggest(pool, observed, k, rng)
        Xo = self._fps([c for c, _ in obs])
        Xp = self._fps(pool)
        # objectives, oriented for minimisation, normalised on observed data
        Y = np.array([self.spec.as_min(v) for _, v in obs])
        lo, hi = Y.min(0), Y.max(0)
        Yn = (Y - lo) / np.where(hi > lo, hi - lo, 1.0)
        # constraint feasibility probabilities
        pfeas = np.ones(len(pool))
        for c in self.spec.constraints:
            yc = np.array([v[c.name] for _, v in obs])
            m, s = TanimotoGP(Xo, yc).predict(Xp)
            p = np.ones(len(pool))
            if c.lower is not None:
                p *= 1 - norm.cdf((c.lower - m) / s)
            if c.upper is not None:
                p *= norm.cdf((c.upper - m) / s)
            pfeas *= p
        chosen: list[int] = []
        sim_block = np.zeros(len(pool))
        for _ in range(min(k, len(pool))):
            w = rng.dirichlet(np.ones(Y.shape[1]))
            f = np.max(w * Yn, 1) + 0.05 * (w * Yn).sum(1)          # augmented Chebyshev (ParEGO)
            gp = TanimotoGP(Xo, f)
            m, s = gp.predict(Xp)
            best = f.min()
            z = (best - m) / s
            ei = (best - m) * norm.cdf(z) + s * norm.pdf(z)
            acq = ei * pfeas
            acq[chosen] = -np.inf
            acq[sim_block > self.max_sim] = -np.inf
            if not np.isfinite(acq).any():
                break
            i = int(np.argmax(acq))
            chosen.append(i)
            sim_block = np.maximum(sim_block, tanimoto(Xp[i:i + 1], Xp)[0])
        if len(chosen) < k:                                          # fill with random if diversity filter bit
            rest = [i for i in rng.permutation(len(pool)) if i not in chosen]
            chosen += rest[:k - len(chosen)]
        return chosen
