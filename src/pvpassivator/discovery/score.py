"""Device-referenced composite score: per-molecule physical descriptors -> interface trap density -> device PCE.

Why this form instead of a weighted sum: the device simulator supplies the weighting. Each descriptor enters
through a physical mechanism with its own units, and the only aggregation is the drift-diffusion model's response
to the resulting interface trap density (and, optionally, band offset). Criteria that the device model does
NOT contain (chemical robustness, synthetic practicality, surface integrity) stay separate: they are constraints
or extra Pareto axes in ``objectives.py``, never folded in.

Chain (all inputs per candidate, with uncertainties):

    coverage   theta = 1 / (1 + exp((dG_bind - mu) / kT))            Langmuir / Fermi-Dirac occupancy of the defect
                                                                     site; dG_bind is a *free* energy against a
                                                                     declared reference state (solution or salt),
                                                                     mu the effective processing chemical potential.
                                                                     Saturates: binding beyond ~ -0.3 eV past mu buys
                                                                     nothing, so "more negative E_ads" is not
                                                                     rewarded without limit.
    traps      N_it = N_res + N0 * (1 - theta * eta) + theta * N_new eta in [0, 1]: fraction of the defect's in-gap
                                                                     states removed when the site is occupied (from
                                                                     electronic structure, not from E_ads);
                                                                     N0: traps of the targeted defect type;
                                                                     N_res: traps from defects this passivant does not
                                                                     address (sets the best achievable device);
                                                                     N_new: deep states the passivant itself adds.
    device     PCE = f(log10 N_it [, d_chi])                         interpolated from a SCAPS-1D sweep table of the
                                                                     calibrated reference device.

The SCAPS table, N0, mu and kT are inputs, not constants of this module. They must come from the calibrated device
model and from experiment (see docs/DISCOVERY_P2O.md, "Composite score"); the test suite uses a labelled toy table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path

import numpy as np

KB_EV = 8.617333262e-5


@dataclass(frozen=True)
class DeviceResponse:
    """PCE (%) as a function of interface trap density (cm^-2), from a SCAPS-1D sweep.

    ``log10_nit`` must be strictly increasing. Outside the table the score is undefined (NaN), never extrapolated.
    """
    log10_nit: tuple[float, ...]
    pce: tuple[float, ...]
    source: str                         # provenance: SCAPS version, device stack, which interface was swept

    def __post_init__(self):
        x = np.asarray(self.log10_nit)
        if len(x) < 2 or np.any(np.diff(x) <= 0) or len(self.pce) != len(x):
            raise ValueError("log10_nit must be strictly increasing and match pce in length")

    @classmethod
    def from_csv(cls, path: str | Path, source: str = "") -> "DeviceResponse":
        """CSV with columns ``nit_cm2,pce_percent`` (as exported from a SCAPS batch sweep)."""
        rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()))
        rows.sort(key=lambda r: float(r["nit_cm2"]))
        return cls(tuple(np.log10([float(r["nit_cm2"]) for r in rows])), tuple(float(r["pce_percent"]) for r in rows),
                   source or str(path))

    def __call__(self, nit: np.ndarray) -> np.ndarray:
        lx = np.log10(np.maximum(np.asarray(nit, float), 1e-300))
        x = np.asarray(self.log10_nit)
        y = np.interp(lx, x, self.pce)
        return np.where((lx < x[0] - 1e-9) | (lx > x[-1] + 1e-9), np.nan, y)


@dataclass(frozen=True)
class ScoreModel:
    response: DeviceResponse
    n0_cm2: float                       # interface traps of the targeted defect type, unpassivated device
    n_res_cm2: float                    # residual interface traps from defects this passivant does not address
    mu_eV: float                        # effective processing chemical potential of the passivant
    temperature_K: float = 373.0        # annealing / deposition temperature at which coverage is set
    notes: str = ""

    def coverage(self, dg_bind: np.ndarray) -> np.ndarray:
        kt = KB_EV * self.temperature_K
        return 1.0 / (1.0 + np.exp(np.clip((np.asarray(dg_bind, float) - self.mu_eV) / kt, -200, 200)))

    def nit(self, dg_bind, eta, n_new=0.0) -> np.ndarray:
        th = self.coverage(dg_bind)
        eta = np.clip(np.asarray(eta, float), 0.0, 1.0)
        return self.n_res_cm2 + self.n0_cm2 * (1.0 - th * eta) + th * np.asarray(n_new, float)

    def pce(self, dg_bind, eta, n_new=0.0) -> np.ndarray:
        return self.response(self.nit(dg_bind, eta, n_new))


@dataclass
class Descriptor:
    """A value with a 1-sigma uncertainty and the tier it came from (mlip / dft / experiment / synthetic)."""
    mean: float
    sigma: float = 0.0
    tier: str = ""


@dataclass
class ScoreResult:
    pce_mean: float
    pce_p10: float
    pce_p90: float
    delta_pce_mean: float               # vs the unpassivated reference (theta = 0)
    nit_median_cm2: float
    coverage_median: float
    tiers: dict[str, str] = field(default_factory=dict)


def score(model: ScoreModel, dg_bind: Descriptor, eta: Descriptor, n_new: Descriptor | None = None,
          n_samples: int = 4000, seed: int = 0) -> ScoreResult:
    """Monte Carlo propagation of descriptor uncertainty into the device score.

    MLIP-derived descriptors carry their benchmarked error as ``sigma``; the optimizer can then use p10/p90
    rather than a point value, so an uncertain MLIP number cannot masquerade as a DFT-quality one.
    """
    rng = np.random.default_rng(seed)
    g = rng.normal(dg_bind.mean, dg_bind.sigma, n_samples)
    e = np.clip(rng.normal(eta.mean, eta.sigma, n_samples), 0, 1)
    nn = (np.maximum(rng.normal(n_new.mean, n_new.sigma, n_samples), 0) if n_new else np.zeros(n_samples))
    p = model.pce(g, e, nn)
    ref = float(model.response(np.array([model.n0_cm2 + model.n_res_cm2]))[0])
    th = model.coverage(g)
    tiers = {"dg_bind": dg_bind.tier, "eta": eta.tier}
    if n_new:
        tiers["n_new"] = n_new.tier
    if np.isnan(p).any():
        bad = float(np.isnan(p).mean())
        raise ValueError(f"{bad:.0%} of samples fall outside the SCAPS table; extend the sweep, do not extrapolate")
    return ScoreResult(float(p.mean()), float(np.percentile(p, 10)), float(np.percentile(p, 90)),
                       float(p.mean() - ref), float(np.median(model.nit(g, e, nn))), float(np.median(th)), tiers)
