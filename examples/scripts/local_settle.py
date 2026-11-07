"""Local settle of rebuilt complexes: step A3 of docs/PACKAGE53_REBUILD_PLAN.md (a pre-relax tool, never a label).

    python examples/scripts/local_settle.py --rebuild-dir data/cache/rebuild_a2_dryrun --out data/cache/settle_a3 \
        [--cases PEAI,TFPAI] [--radius 8] [--fmax 0.1] [--steps 150] [--device cuda]

Run in an environment with mace-torch and torch-dftd (``maceh``). Reads each ``<case>/POSCAR`` and ``rebuild.json``
written by ``rebuild_on_slab.py`` (the archived complex is re-read from the report's ``source`` path).

Per complex:
1. Mobile atoms: everything within ``radius`` (default 8 A) of the salt, plus the salt and the relocated units
   (``moved_indices``). Organic molecules move whole or not at all, as in ``spectator.freeze_far_field``. All other
   atoms are fixed (FixAtoms) and checked afterwards to be bit-for-bit unchanged, so they stay the slab's.
2. FIRE with the foundation MACE-matpes-r2scan + D3(BJ, r2SCAN) [torch-dftd] until fmax over the mobile atoms
   <= ``--fmax`` (default 0.1 eV/A) or ``--steps`` (default 150). The goal is to remove transplant strain and short
   contacts, not to reach the model's minimum: near the r2SCAN minimum the foundation model's force error exceeds
   the residual DFT forces (rebuild plan, "Foundation MLIP test"), so a long relax walks away from DFT.
3. The rebuild's local checks again (``rebuild_on_slab.local_checks``) against the archived complex, plus:
   ``frozen_unchanged`` (max change of fixed atoms == 0) and ``max_move_A`` (largest displacement of a mobile atom;
   above 1.0 A the settle moved more than a transplant should need and the case is flagged).

``status``: pass (every check) / review (only the salt-H contact checks still fail, or the move exceeds 1 A) /
fail (a clash < 1 A, a site that left its local Pb frame, or a frozen atom moved). Output: ``<case>/POSCAR`` (no
selective-dynamics flags: A4's ``freeze_beyond_A`` writes them), ``<case>/settle.json``, ``summary.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import FIRE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import rebuild_on_slab as R  # noqa: E402

RADIUS_A = 8.0
FMAX = 0.1
STEPS = 150
MAX_MOVE_A = 1.0
MODEL = Path("D:/MLIP_Foundational_Models/mace/MACE-matpes-r2scan-omat-ft.model")
SOFT = ("salt_H_lattice_A", "salt_slab_A")


def mobile_mask(atoms: Atoms, centre, radius: float, always=()) -> np.ndarray:
    """Atoms within ``radius`` of any ``centre`` atom, plus ``always``; organic molecules whole (any atom in -> all)."""
    from ase.neighborlist import NeighborList, natural_cutoffs
    from scipy.sparse.csgraph import connected_components
    centre = np.asarray(centre, int)
    d = R._dmat(atoms.positions, atoms.positions[centre], atoms.cell).min(1)
    mob = d <= radius
    mob[np.asarray(always, int)] = True
    sym = np.array(atoms.get_chemical_symbols())
    org = np.flatnonzero(~np.isin(sym, ["Pb", "I", "Br", "Cl", "Sn", "Cs"]))
    sub = atoms[org]
    nl = NeighborList(natural_cutoffs(sub, mult=1.15), self_interaction=False, bothways=True)
    nl.update(sub)
    _, lab = connected_components(nl.get_connectivity_matrix(sparse=True))
    for c in np.unique(lab):
        idx = org[lab == c]
        if mob[idx].any():
            mob[idx] = True
    return mob


def settle(atoms: Atoms, mobile: np.ndarray, calc, fmax: float = FMAX, steps: int = STEPS,
           log: str | None = None) -> tuple[Atoms, dict]:
    """FIRE over the mobile atoms only; returns the settled copy and a report."""
    a = atoms.copy()
    a.set_constraint(FixAtoms(indices=np.flatnonzero(~mobile)))
    a.calc = calc
    start = a.positions.copy()
    t0 = time.time()
    e0 = float(a.get_potential_energy())
    f0 = float(np.linalg.norm(a.get_forces()[mobile], axis=1).max())
    opt = FIRE(a, logfile=log)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    e1 = float(a.get_potential_energy())
    f1 = float(np.linalg.norm(a.get_forces()[mobile], axis=1).max())
    move = np.linalg.norm(a.positions - start, axis=1)
    out = a.copy()
    out.set_constraint()
    out.calc = None
    return out, {
        "n_mobile": int(mobile.sum()), "n_fixed": int((~mobile).sum()), "steps": int(opt.nsteps),
        "converged": converged, "fmax_target": fmax, "fmax_start": round(f0, 3), "fmax_end": round(f1, 3),
        "energy_change_eV": round(e1 - e0, 4), "seconds": round(time.time() - t0, 1),
        "mean_move_A": round(float(move[mobile].mean()), 3), "max_move_A": round(float(move[mobile].max()), 3),
        "frozen_max_change_A": float(move[~mobile].max()) if (~mobile).any() else 0.0,
    }


def make_calc(model: Path, device: str, dtype: str = "float64"):
    import torch
    from ase.calculators.mixing import SumCalculator
    from mace.calculators import MACECalculator
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    mace = MACECalculator(model_paths=str(model), device=device, default_dtype=dtype)
    d3 = TorchDFTD3Calculator(damping="bj", xc="r2scan", device=device, dtype=torch.float32)
    return SumCalculator([mace, d3])


def evaluate(cx: Atoms, out: Atoms, rep: dict, srep: dict, h_tol: float = R.H_TOL_A) -> dict:
    checks = R.local_checks(cx, out, rep["salt_indices"], rep["moved_indices"], rep["relocated_anchors"], h_tol)
    checks["frozen_unchanged"] = {"value": srep["frozen_max_change_A"], "pass": srep["frozen_max_change_A"] == 0.0}
    checks["max_move_A"] = {"value": srep["max_move_A"], "limit": MAX_MOVE_A, "pass": srep["max_move_A"] <= MAX_MOVE_A}
    hard = all(c["pass"] for k, c in checks.items() if k not in SOFT + ("max_move_A",))
    ok = all(c["pass"] for c in checks.values())
    return {"checks": checks, "status": "pass" if ok else ("review" if hard else "fail")}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main(argv=None, calc=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebuild-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", default="", help="comma-separated; default: every case in the rebuild summary")
    ap.add_argument("--radius", type=float, default=RADIUS_A)
    ap.add_argument("--fmax", type=float, default=FMAX)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--include-failed", action="store_true", help="also settle cases whose rebuild status is fail")
    a = ap.parse_args(argv)

    summ = json.loads((a.rebuild_dir / "summary.json").read_text())
    names = [n.strip() for n in a.cases.split(",") if n.strip()] or list(summ["cases"])
    a.out.mkdir(parents=True, exist_ok=True)
    injected = calc is not None
    calc = calc or make_calc(a.model, a.device, a.dtype)
    result = {"rebuild_dir": str(a.rebuild_dir), "slab": summ["slab"], "radius_A": a.radius, "fmax": a.fmax,
              "steps": a.steps, "model": "injected (test)" if injected else str(a.model),
              "cases": {}}
    for n in names:
        rep = json.loads((a.rebuild_dir / n / "rebuild.json").read_text())
        if rep.get("status") == "fail" and not a.include_failed:
            result["cases"][n] = {"status": "skipped", "reason": "rebuild status fail"}
            print(f"{n:10s} skipped (rebuild failed)")
            continue
        src = a.rebuild_dir / n / "POSCAR"
        atoms = read(src, format="vasp")
        cx = read(rep["source"], format="vasp")
        mob = mobile_mask(atoms, rep["salt_indices"], a.radius, rep["moved_indices"])
        d = a.out / n
        d.mkdir(exist_ok=True)
        out, srep = settle(atoms, mob, calc, a.fmax, a.steps, log=str(d / "fire.log"))
        ev = evaluate(cx, out, rep, srep)
        write(d / "POSCAR", out, format="vasp", direct=True, sort=False)
        full = {"name": n, "source_poscar": str(src), "source_sha256_16": _sha(src), "radius_A": a.radius,
                "settle": srep, **ev, "poscar_sha256_16": _sha(d / "POSCAR")}
        (d / "settle.json").write_text(json.dumps(full, indent=1))
        failed = [k for k, c in ev["checks"].items() if not c["pass"]]
        result["cases"][n] = {"status": ev["status"], "failed": failed, "steps": srep["steps"],
                              "fmax_end": srep["fmax_end"], "max_move_A": srep["max_move_A"]}
        print(f"{n:10s} {ev['status']:7s} steps={srep['steps']} fmax {srep['fmax_start']}->{srep['fmax_end']} "
              f"dE={srep['energy_change_eV']} max_move={srep['max_move_A']} failed={failed} ({srep['seconds']} s)",
              flush=True)
    (a.out / "summary.json").write_text(json.dumps(result, indent=1))
    return 0 if all(c["status"] in ("pass", "review", "skipped") for c in result["cases"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
