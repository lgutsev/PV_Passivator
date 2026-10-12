"""Reproduce archived adsorption energies from raw OSZICAR/POSCAR data.

Each case names the complex, slab and reference folders (relative to archive roots) and the
spreadsheet value it should reproduce. Compositions come from POSCARs; the reaction is
balance-checked before any energy is computed. Missing / unconverged runs are reported,
never filled in.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .ledger import Species, Reaction
from .vasp_io import parse_lite, LiteRun

# Standard PBE POTCAR valences used by the archived runs (Pb_d=14, I=7, N=5, C=4, H=1, O=6,
# F=7, Cl=7). Recovered from NELECT arithmetic in charged runs (e.g. 2682 -> 2681/2683);
# confirm against the POTCARs before trusting charged-state accounting.
ZVAL_ASSUMED = {"Pb": 14, "I": 7, "N": 5, "C": 4, "H": 1, "O": 6, "F": 7, "Cl": 7, "S": 6,
                "Na": 7, "K": 9, "Cs": 9, "Br": 7, "B": 3, "P": 5, "Si": 4}


def ensure_extracted(zip_path: Path, cache: Path) -> Path:
    """Extract an OutPackLite archive into a cache dir (never into the research tree)."""
    dest = cache / zip_path.stem
    if not dest.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    inner = dest / zip_path.stem
    return inner if inner.is_dir() else dest


@dataclass
class Case:
    case_id: str
    complex: str
    slab: str
    refs: list[tuple[float, str]]
    n_units: float
    sheet_value: float | None = None       # E_ads reported in spreadsheet (eV/unit)
    sheet_cell: str = ""
    surface: str = ""
    energy_kind: str = "adsorption_relaxed"
    notes: str = ""


@dataclass
class CaseResult:
    case: Case
    equation: str = ""
    balance_problems: list[str] = field(default_factory=list)
    run_status: dict = field(default_factory=dict)
    energies: dict = field(default_factory=dict)
    fingerprint_problems: list[str] = field(default_factory=list)
    e_ads: float | None = None
    delta_vs_sheet: float | None = None
    verdict: str = ""

    def row(self) -> dict:
        return {"case_id": self.case.case_id, "surface": self.case.surface,
                "equation": self.equation, "E_ads_eV_per_unit": self.e_ads,
                "sheet_eV": self.case.sheet_value, "delta_eV": self.delta_vs_sheet,
                "verdict": self.verdict, "run_status": self.run_status,
                "balance_problems": self.balance_problems,
                "fingerprint_problems": self.fingerprint_problems, "notes": self.case.notes}


def _species(name: str, run: LiteRun) -> Species:
    return Species(name, run.composition, cell_charge=int(round(run.cell_charge(ZVAL_ASSUMED))))


def evaluate(case: Case, resolve) -> CaseResult:
    """``resolve(path_key) -> folder Path``."""
    res = CaseResult(case)
    runs = {"complex": parse_lite(resolve(case.complex)), "slab": parse_lite(resolve(case.slab))}
    for i, (_, r) in enumerate(case.refs):
        runs[f"ref{i}:{r}"] = parse_lite(resolve(r))
    res.run_status = {k: f"{v.status} ({v.n_ionic} steps)" for k, v in runs.items()}
    sp = {k: _species(Path(v.path).name, v) for k, v in runs.items()}
    rxn = Reaction([(1, sp["slab"])] + [(nu, sp[f"ref{i}:{r}"]) for i, (nu, r) in enumerate(case.refs)],
                   [(1, sp["complex"])], energy_kind=case.energy_kind, n_units=case.n_units)
    res.equation = rxn.equation()
    res.balance_problems = rxn.check()
    # Hamiltonian consistency (k-points excluded: molecule refs legitimately use Gamma)
    fps = {k: {kk: vv for kk, vv in v.fingerprint().items() if kk != "KPOINTS"} for k, v in runs.items()}
    base = fps["complex"]
    for k, f in fps.items():
        for key in set(base) | set(f):
            if base.get(key) != f.get(key):
                res.fingerprint_problems.append(f"{key}: complex={base.get(key)} {k}={f.get(key)}")
    if runs["slab"].kpoints != runs["complex"].kpoints:
        res.fingerprint_problems.append(
            f"k-mesh differs between slab ({runs['slab'].kpoints}) and complex ({runs['complex'].kpoints})")
    missing = [k for k, v in runs.items() if v.energy_F is None]
    if missing:
        res.verdict = f"NOT REPRODUCIBLE: no energy in archive for {missing}"
        return res
    res.energies = {sp[k].name: v.energy_F for k, v in runs.items()}
    if res.balance_problems:
        res.verdict = "UNBALANCED"
        return res
    out = rxn.delta_e(res.energies)
    res.e_ads = out["delta_E_per_unit_eV"]
    if case.sheet_value is not None:
        res.delta_vs_sheet = res.e_ads - case.sheet_value
    caveats = []
    if any(v.status == "hit_nsw" for v in runs.values()):
        caveats.append("a run hit NSW (relaxation not converged)")
    if res.fingerprint_problems:
        caveats.append("settings mismatch")
    if res.delta_vs_sheet is None:
        head = "COMPUTED (no sheet value)"
    elif abs(res.delta_vs_sheet) <= 2e-3:
        head = "REPRODUCED"
    else:
        head = f"MISMATCH vs sheet ({res.delta_vs_sheet:+.3f} eV)"
    res.verdict = head + ("; caveats: " + "; ".join(caveats) if caveats else "")
    return res


def load_cases(path: str | Path) -> tuple[dict, list[Case]]:
    cfg = json.loads(Path(path).read_text())
    cases = [Case(**{**c, "refs": [tuple(r) for r in c["refs"]]}) for c in cfg["cases"]]
    return cfg, cases


def run_config(path: str | Path, cache: str | Path) -> list[CaseResult]:
    cfg, cases = load_cases(path)
    roots = {}
    for key, spec in cfg["archives"].items():
        p = Path(spec)
        roots[key] = ensure_extracted(p, Path(cache)) if p.suffix == ".zip" else p

    def resolve(k: str) -> Path:
        root, rel = k.split(":", 1)
        return roots[root] / rel

    return [evaluate(c, resolve) for c in cases]
