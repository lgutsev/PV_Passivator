"""Read the exported D0 statics of package 105 (chirality) and decide the k-mesh before the relaxations.

    python examples/scripts/analyze_chirality_d0.py <exported package dir> [--json out.json]

Reads ``outputs/<job>/static/`` (vasprun.xml.gz, else OUTCAR) for every job that has DONE; missing jobs are reported,
never filled in. Checks, with thresholds fixed in examples/chirality/plan_matrices.json (D0a-D0c, D2):

* k-mesh (D0a): R_kdown and R_kref against R_kup, rac_kref against rac_kup. Max per-atom force change <= 5 meV/A
  and the R - rac energy per formula unit changing by <= 1 meV/f.u. between the reference and the dense meshes.
* ENCUT: R_kref against R_kref_e600 with the same criteria.
* Mirror (D2): Rmirror_kref is the inversion of R with identical atom order, so E must match and F_mirror = -F_R.
  Pass if |dE| <= 1 meV per cell and max |F_mirror + F_R| <= 5 meV/A. This checks settings, not physics.
* SOC (D0c): force change from LSORBIT per element; SOC enters the structural protocol if any Pb or I atom moves by
  more than 20 meV/A.
* Timing (D0b): LOOP+ seconds and total elapsed time per job.

Recommendation: the reference mesh if both k-mesh checks and the ENCUT check pass, otherwise the denser mesh (and the
relax inputs must be regenerated). Nothing here relaxes, submits or edits the package.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

F_TOL = 0.005        # eV/A
E_FU_TOL = 0.001     # eV per formula unit
MIRROR_E_TOL = 0.001 # eV per cell
SOC_F_TOL = 0.020    # eV/A on Pb / I
N_FU = {"R": 4, "rac": 2}


def load_job(d: Path):
    """Final frame (energy sigma->0, forces) of one static output dir, or None if it is not DONE."""
    from ase.io import read
    if not (d / "DONE").exists():
        return None
    for name, fmt in (("vasprun.xml.gz", "vasp-xml"), ("OUTCAR", "vasp-out")):
        if (d / name).exists():
            at = read(d / name, index=-1, format=fmt)
            break
    else:
        return None
    at.info["timing"] = timing(d / "OUTCAR")
    return at


def timing(outcar: Path) -> dict:
    if not outcar.exists():
        return {}
    txt = outcar.read_text(errors="replace")
    loops = [float(m) for m in re.findall(r"LOOP\+:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)", txt)]
    el = re.search(r"Elapsed time \(sec\):\s+([\d.]+)", txt)
    return {"loop_plus_s": loops, "elapsed_s": float(el.group(1)) if el else None}


def fdiff(a, b) -> float:
    return float(np.linalg.norm(a.get_forces() - b.get_forces(), axis=1).max())


def analyze(jobs: dict) -> dict:
    """``jobs``: name -> Atoms with energy/forces (None or absent = missing). Pure function."""
    have = {k: v for k, v in jobs.items() if v is not None}
    missing = sorted(k for k, v in jobs.items() if v is None)
    E = {k: v.get_potential_energy() for k, v in have.items()}
    out = {"missing": missing, "energies_eV": {k: round(e, 6) for k, e in E.items()}, "checks": {}}
    c = out["checks"]

    def de_fu(r, ra):   # E_R/fu - E_rac/fu
        return E[r] / N_FU["R"] - E[ra] / N_FU["rac"]

    if {"R_kref", "R_kup"} <= have.keys():
        c["R_kref_vs_kup_fmax_meV_A"] = round(1000 * fdiff(have["R_kref"], have["R_kup"]), 2)
    if {"R_kdown", "R_kup"} <= have.keys():
        c["R_kdown_vs_kup_fmax_meV_A"] = round(1000 * fdiff(have["R_kdown"], have["R_kup"]), 2)
    if {"rac_kref", "rac_kup"} <= have.keys():
        c["rac_kref_vs_kup_fmax_meV_A"] = round(1000 * fdiff(have["rac_kref"], have["rac_kup"]), 2)
    if {"R_kref", "R_kup", "rac_kref", "rac_kup"} <= have.keys():
        c["dE_R_rac_kref_meV_fu"] = round(1000 * de_fu("R_kref", "rac_kref"), 2)
        c["dE_R_rac_kup_meV_fu"] = round(1000 * de_fu("R_kup", "rac_kup"), 2)
        c["dE_R_rac_mesh_change_meV_fu"] = round(abs(c["dE_R_rac_kref_meV_fu"] - c["dE_R_rac_kup_meV_fu"]), 2)
    if {"R_kref", "R_kref_e600"} <= have.keys():
        c["R_encut_fmax_meV_A"] = round(1000 * fdiff(have["R_kref"], have["R_kref_e600"]), 2)
        c["R_encut_dE_meV_atom"] = round(1000 * (E["R_kref_e600"] - E["R_kref"]) / len(have["R_kref"]), 3)

    kmesh_ok = (c.get("R_kref_vs_kup_fmax_meV_A", np.inf) <= 1000 * F_TOL
                and c.get("rac_kref_vs_kup_fmax_meV_A", np.inf) <= 1000 * F_TOL
                and c.get("dE_R_rac_mesh_change_meV_fu", np.inf) <= 1000 * E_FU_TOL)
    encut_ok = c.get("R_encut_fmax_meV_A", np.inf) <= 1000 * F_TOL
    out["kmesh_pass"], out["encut_pass"] = bool(kmesh_ok), bool(encut_ok)

    if {"R_kref", "Rmirror_kref"} <= have.keys():
        r, m = have["R_kref"], have["Rmirror_kref"]
        if r.get_chemical_symbols() != m.get_chemical_symbols():
            raise ValueError("mirror and R atom order differ; the inversion check needs identical ordering")
        dE = E["Rmirror_kref"] - E["R_kref"]
        dF = float(np.linalg.norm(m.get_forces() + r.get_forces(), axis=1).max())
        out["mirror"] = {"dE_meV_cell": round(1000 * dE, 4), "max_F_mismatch_meV_A": round(1000 * dF, 3),
                         "pass": bool(abs(dE) <= MIRROR_E_TOL and dF <= F_TOL)}

    soc = {}
    for base in ("R_kref", "rac_kref"):
        if {base, base + "_soc"} <= have.keys():
            a, b = have[base], have[base + "_soc"]
            d = np.linalg.norm(a.get_forces() - b.get_forces(), axis=1)
            sym = np.array(a.get_chemical_symbols())
            per = {el: round(1000 * float(d[sym == el].max()), 2) for el in sorted(set(sym))}
            soc[base] = {"max_dF_meV_A_by_element": per,
                         "dE_soc_meV_atom": round(1000 * (E[base + "_soc"] - E[base]) / len(a), 3),
                         "soc_needed_for_structure": bool(max(per.get("Pb", 0), per.get("I", 0)) > 1000 * SOC_F_TOL)}
    out["soc"] = soc

    out["start_forces"] = {k: {"fmax_eV_A": round(float(np.linalg.norm(have[k].get_forces(), axis=1).max()), 3)}
                           for k in ("R_kref", "rac_kref") if k in have}
    out["timing"] = {k: v.info.get("timing", {}) for k, v in have.items()}
    if kmesh_ok and encut_ok:
        out["recommendation"] = "keep the staged meshes (R 3x3x1, rac 3x3x2) and ENCUT 520; relax inputs need no change"
    elif missing:
        out["recommendation"] = "incomplete: decide after the missing jobs are exported"
    else:
        out["recommendation"] = ("use the denser mesh (or ENCUT) that failed the check: regenerate the relax/static "
                                 "inputs before releasing them")
    return out


def analyze_diag(jobs: dict) -> dict:
    """Package 108 (D0e): which setting removes the iodide force error seen in package 105.

    ``jobs``: R_kref (520, Auto) and R_kref_e600 (600, Auto) from 105; R_e520_lrealF, R_e600_lrealF, R_e700_lrealF and
    R_e700 (700, Auto) from 108. All share one start and mesh. Max per-atom force differences (meV/A), per element.
      * LREAL is the cause if the three LREAL=.FALSE. runs agree within 5 meV/A -> relax at 520 with LREAL=.FALSE.
      * otherwise the lowest LREAL=.FALSE. cutoff within 5 meV/A of the next one up (520 or 600); if even 600 vs 700
        fails, no setting here is converged and a higher cutoff is needed (no recommendation is forced).
    """
    have = {k: v for k, v in jobs.items() if v is not None}
    need = ["R_kref", "R_kref_e600", "R_e520_lrealF", "R_e600_lrealF", "R_e700_lrealF", "R_e700"]
    out = {"missing": [k for k in need if k not in have], "pairs": {}}
    if out["missing"]:
        out["recommendation"] = "incomplete: decide after the missing jobs are exported"
        return out
    sym = np.array(have["R_kref"].get_chemical_symbols())

    def cmp(a, b):
        d = np.linalg.norm(have[a].get_forces() - have[b].get_forces(), axis=1)
        return {"max": round(1000 * float(d.max()), 2),
                **{el: round(1000 * float(d[sym == el].max()), 2) for el in sorted(set(sym))}}

    pairs = {"F520_vs_F600": ("R_e520_lrealF", "R_e600_lrealF"), "F600_vs_F700": ("R_e600_lrealF", "R_e700_lrealF"),
             "F520_vs_F700": ("R_e520_lrealF", "R_e700_lrealF"), "Auto520_vs_F520": ("R_kref", "R_e520_lrealF"),
             "Auto600_vs_F600": ("R_kref_e600", "R_e600_lrealF"), "Auto700_vs_F700": ("R_e700", "R_e700_lrealF"),
             "Auto520_vs_Auto600": ("R_kref", "R_kref_e600"), "Auto600_vs_Auto700": ("R_kref_e600", "R_e700")}
    for k, (a, b) in pairs.items():
        out["pairs"][k] = cmp(a, b)
    tol = 1000 * F_TOL
    p = out["pairs"]
    if p["F520_vs_F700"]["max"] <= tol and p["F520_vs_F600"]["max"] <= tol:
        out["cause"], out["recommendation"] = "LREAL", "relax/static at ENCUT 520 with LREAL=.FALSE."
    elif p["F600_vs_F700"]["max"] <= tol:
        out["cause"], out["recommendation"] = "basis", "relax/static at ENCUT 600 with LREAL=.FALSE."
    else:
        out["cause"] = "unresolved"
        out["recommendation"] = "not converged at 700 with LREAL=.FALSE.: needs a higher-cutoff check before any relax"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--diag", type=Path, help="exported package 108 (D0e); 'package' is then package 105")
    a = ap.parse_args(argv)
    if a.diag:
        jobs = {n: load_job(a.package / "outputs" / n / "static") for n in ("R_kref", "R_kref_e600")}
        jobs.update({n: load_job(a.diag / "outputs" / n / "static")
                     for n in ("R_e520_lrealF", "R_e600_lrealF", "R_e700_lrealF", "R_e700")})
        res = analyze_diag(jobs)
        txt = json.dumps(res, indent=1)
        if a.json:
            a.json.write_text(txt)
        print(txt)
        return 0
    names = ["R_kref", "R_kdown", "R_kup", "R_kref_e600", "Rmirror_kref", "rac_kref", "rac_kup",
             "R_kref_soc", "rac_kref_soc"]
    jobs = {n: load_job(a.package / "outputs" / n / "static") for n in names}
    res = analyze(jobs)
    txt = json.dumps(res, indent=1)
    if a.json:
        a.json.write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
