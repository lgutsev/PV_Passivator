"""Benchmark MLIP energies against archived DFT runs (energies only; archives keep no forces).

Workflow:
1. ``discover_runs`` turns run folders into ``BenchRun`` records: final energy F from OSZICAR, the
   final structure (CONTCAR), composition, cell charge and the reasons a run is not a trusted label.
2. An external runner evaluates each MLIP on the CONTCAR as is (single point) and stores the energy
   per (run, model), plus a D3 term per run and profile (D3 is geometry-only, so it is shared).
3. The analysis functions here compare MLIP and DFT on quantities that do not depend on unknown
   or mismatched references:
   - ``same_composition_deltas``: energy differences between runs with identical composition;
   - ``offset_fit``: per-element offsets fitted to E_DFT - E_MLIP, with leave-one-out residuals.
     A constant per-element shift (different pseudopotentials, functional reference energies)
     is absorbed; what remains is the MLIP error on any balanced reaction among these runs;
   - ``matched_adsorption``: complex = slab + reference matched by composition, giving
     E_ads = E(complex) - E(slab) - E(ref) for DFT and MLIP alike.
Runs from different DFT profiles, or with different settings within a profile (k-point mesh in
particular: the same slab differs by ~0.9 eV between Gamma-only and 2x2x1), are never mixed in one
comparison; ``settings_groups`` splits them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from .ledger import formula_key
from .vasp_io import parse_lite
from .recovery import ZVAL_ASSUMED

SKIP_DIRS = ("X_", "BandRuns")


@dataclass
class BenchRun:
    run_id: str               # "<profile>/<set>/<folder>"
    profile: str
    path: str
    composition: dict[str, int]
    natoms: int
    energy_F: float | None
    status: str
    n_ionic: int
    cell_charge: float
    problems: list[str] = field(default_factory=list)       # reasons it is not a trusted label
    contcar: str | None = None
    last_dE: float | None = None
    sp_problems: list[str] = field(default_factory=list)    # reasons it is unusable even as a single point
    settings: str = ""        # normalized INCAR+KPOINTS fingerprint (NELECT excluded), JSON

    @property
    def accepted(self) -> bool:
        """Trusted label: converged relaxation, sound electronic state."""
        return not self.problems

    @property
    def single_point_ok(self) -> bool:
        """Usable for an energy-at-this-structure comparison: an unconverged relaxation still pairs a DFT
        energy with a structure, provided the last ionic step barely changed the energy."""
        return not self.sp_problems

    def evaluable(self) -> bool:
        """Worth an MLIP single point at all: a DFT energy, a CONTCAR, a neutral cell and a sound
        electronic state. The last-step tolerance is applied at analysis time."""
        return not [p for p in self.sp_problems if not p.startswith("last ionic step")] and             self.energy_F is not None and self.contcar is not None and self.n_ionic > 1 or             (self.status == "single_point" and not self.sp_problems)

    @property
    def formula(self) -> str:
        return formula_key(self.composition)


def _run_dirs(root: Path):
    """Run folders directly under ``root`` (or under its single nested copy, as OutPackLite zips
    extract to ``name/name``), skipping archive and band-structure subfolders."""
    root = Path(root)
    inner = root / root.name
    if inner.is_dir():
        root = inner
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith(SKIP_DIRS):
            continue
        if (d / "OSZICAR").exists() or (d / "POSCAR").exists():
            yield d


def discover_runs(root: str | Path, profile: str, set_name: str | None = None,
                  zval: dict[str, float] = ZVAL_ASSUMED, max_last_dE: float = 0.01) -> list[BenchRun]:
    """``max_last_dE`` (eV): largest last-step energy change for a run to count as a single point;
    the CONTCAR is one ionic step past the last OSZICAR energy."""
    out = []
    set_name = set_name or Path(root).name
    for d in _run_dirs(Path(root)):
        r = parse_lite(d)
        missing = sorted(set(r.composition) - set(zval))
        if missing:
            problems, q = [f"no assumed valence for {', '.join(missing)}"], 0.0
        else:
            problems, q = r.acceptance(zval), r.cell_charge(zval)
        cont = d / "CONTCAR"
        if not cont.exists() or cont.stat().st_size == 0:
            problems.append("no CONTCAR")
        elif _contcar_counts(cont) != (r.species, r.counts):
            problems.append("CONTCAR species/counts differ from POSCAR")
        sp = [p for p in problems if not p.startswith(("relaxation stopped at NSW", "only ", "charged cell (q"))]
        if r.status not in ("single_point",) and r.last_dE is not None and abs(r.last_dE) > max_last_dE:
            sp.append(f"last ionic step changed E by {r.last_dE:+.3g} eV")
        if q:
            sp.append("charged cell; needs a charge-aware model")
        sp = list(dict.fromkeys(sp))
        fp = r.fingerprint()
        fp.pop("NELECT", None)
        out.append(BenchRun(f"{profile}/{set_name}/{d.name}", profile, str(d), r.composition,
                            sum(r.counts), r.energy_F, r.status, r.n_ionic, q, problems,
                            str(cont) if cont.exists() else None, r.last_dE, sp,
                            json.dumps(fp, sort_keys=True)))
    return out


def _contcar_counts(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    return lines[5].split(), [int(x) for x in lines[6].split()]


def save_runs(runs: list[BenchRun], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(r) for r in runs], indent=1))


def load_runs(path: str | Path) -> list[BenchRun]:
    return [BenchRun(**d) for d in json.loads(Path(path).read_text())]


def settings_groups(runs: list[BenchRun]) -> dict[tuple[str, str], list[BenchRun]]:
    """Runs keyed by (profile, settings fingerprint)."""
    out: dict[tuple[str, str], list[BenchRun]] = {}
    for r in runs:
        out.setdefault((r.profile, r.settings), []).append(r)
    return out


# ----------------------------------------------------------------------------- analysis
def _stats(err: np.ndarray) -> dict:
    err = np.asarray(err, float)
    if len(err) == 0:
        return {"n": 0}
    return {"n": int(len(err)), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "max_abs": float(np.abs(err).max())}


def same_composition_deltas(runs: list[BenchRun], e_dft: dict[str, float], e_mlip: dict[str, float]) -> dict:
    """Within each composition group of >= 2 runs: E_i - E_min(DFT) for DFT and MLIP (same reference run).
    Returns rows and error statistics in eV."""
    groups: dict[str, list[BenchRun]] = {}
    for r in runs:
        if r.run_id in e_dft and r.run_id in e_mlip:
            groups.setdefault(r.formula, []).append(r)
    rows = []
    for f, g in groups.items():
        if len(g) < 2:
            continue
        ref = min(g, key=lambda r: e_dft[r.run_id])
        for r in g:
            if r is ref:
                continue
            d = e_dft[r.run_id] - e_dft[ref.run_id]
            m = e_mlip[r.run_id] - e_mlip[ref.run_id]
            rows.append({"formula": f, "run": r.run_id, "ref": ref.run_id, "dE_dft": d, "dE_mlip": m, "err": m - d})
    return {"rows": rows, "stats": _stats([x["err"] for x in rows])}


def offset_fit(runs: list[BenchRun], e_dft: dict[str, float], e_mlip: dict[str, float]) -> dict:
    """Fit E_DFT - E_MLIP = sum_el n_el mu_el by least squares; report in-sample and leave-one-out
    residuals (eV per structure). Elements present in only one run make that run's LOO undefined
    and are reported, not fitted around."""
    rs = [r for r in runs if r.run_id in e_dft and r.run_id in e_mlip]
    els = sorted({e for r in rs for e in r.composition})
    A = np.array([[r.composition.get(e, 0) for e in els] for r in rs], float)
    y = np.array([e_dft[r.run_id] - e_mlip[r.run_id] for r in rs])
    if len(rs) <= len(els):
        raise ValueError(f"{len(rs)} runs cannot fit {len(els)} element offsets")
    mu, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ mu
    H = A @ np.linalg.pinv(A.T @ A) @ A.T
    h = np.clip(np.diag(H), 0, 1)
    loo = np.where(h < 1 - 1e-8, res / np.maximum(1 - h, 1e-12), np.nan)
    rank = int(np.linalg.matrix_rank(A))
    rows = [{"run": r.run_id, "natoms": r.natoms, "residual": float(a), "loo": float(b), "leverage": float(c)}
            for r, a, b, c in zip(rs, res, loo, h)]
    fin = loo[np.isfinite(loo)]
    return {"elements": els, "mu": dict(zip(els, map(float, mu))), "rank": rank, "rows": rows,
            "in_sample": _stats(res), "loo": _stats(fin), "loo_undefined": int((~np.isfinite(loo)).sum())}


def matched_adsorption(complexes: list[BenchRun], slabs: list[BenchRun], refs: list[BenchRun],
                       e_dft: dict[str, float], e_mlip: dict[str, float], max_refs: int = 1) -> dict:
    """Every (complex, slab, ref) with composition(complex) = composition(slab) + composition(ref),
    all with energies in both sets. E_ads = E(complex) - E(slab) - E(ref) (eV, per reference unit)."""
    def has(r):
        return r.run_id in e_dft and r.run_id in e_mlip

    def diff(a, b):
        out = dict(a)
        for k, v in b.items():
            out[k] = out.get(k, 0) - v
        return {k: v for k, v in out.items() if v}
    by_formula: dict[str, list[BenchRun]] = {}
    for r in refs:
        if has(r):
            by_formula.setdefault(r.formula, []).append(r)
    rows = []
    for c in filter(has, complexes):
        for s in filter(has, slabs):
            if s.run_id == c.run_id:
                continue
            d = diff(c.composition, s.composition)
            if not d or any(v < 0 for v in d.values()):
                continue
            for ref in by_formula.get(formula_key(d), [])[:max_refs]:
                ed = e_dft[c.run_id] - e_dft[s.run_id] - e_dft[ref.run_id]
                em = e_mlip[c.run_id] - e_mlip[s.run_id] - e_mlip[ref.run_id]
                rows.append({"complex": c.run_id, "slab": s.run_id, "ref": ref.run_id,
                             "E_ads_dft": ed, "E_ads_mlip": em, "err": em - ed})
    return {"rows": rows, "stats": _stats([x["err"] for x in rows])}


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
