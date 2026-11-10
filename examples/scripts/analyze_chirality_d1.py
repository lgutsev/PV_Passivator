"""D1 relaxations of package 105 rev1: integrity, cells and handedness descriptors, relaxed vs start.

    python examples/scripts/analyze_chirality_d1.py <exported package 105 dir> [--json out.json]

Per relax job (outputs/<job>/relax/CONTCAR + OUTCAR): final max force, cell lengths/volume against the start, and a
fresh pvpassivator.chirality topology of the relaxed cell. Integrity checks: the same stereocentre labels and molecule
count as the start, every Pb six-coordinate with 2 axial / 4 bridging I, every N still NH3+. Descriptor means and
per-link values are reported for the start, the relaxed cell and (for R) the exact inversion of the relaxed cell, which
must give exactly the negated values. Nothing is submitted; statics are not needed for geometry.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pvpassivator import chirality as ch  # noqa: E402

JOBS = ("R_fixcell", "R_fullcell", "rac_fixcell", "rac_fullcell")


def last_fmax(outcar: Path) -> float | None:
    from ase.io import read as r
    try:
        a = r(outcar, index=-1, format="vasp-out")
    except Exception:
        return None
    return float(np.linalg.norm(a.get_forces(), axis=1).max())


def describe(atoms) -> dict:
    top = ch.build_topology(atoms)
    d = ch.frame_descriptors(top, atoms)
    s = top.summary()
    return {
        "stereocentres": s["stereocentres"], "n_molecules": s["n_molecules"], "n_layers": s["n_layers"],
        "mean": {k: round(d[f"mean_{k}"], 5) for k in ("ax", "eq", "org")},
        "links": {k: sorted(set(np.round(d[f"tau_{k}"], 4).tolist())) for k in ("ax", "eq", "org")},
        "Pb_I_Pb_deg": sorted(set(np.round(ch.mxm_angles(top, atoms), 1).tolist())),
        "r_HB_A": [round(float(x), 3) for x in sorted(set(np.round(ch.cation_hbond_length(top, atoms), 3)))],
        "headgroup_tilt_deg": sorted(set(np.round(ch.headgroup_tilt(top, atoms), 1).tolist())),
    }


def analyze(pkg: Path) -> dict:
    out = {}
    for job in JOBS:
        d = pkg / "outputs" / job / "relax"
        if not (d / "DONE").exists():
            out[job] = {"status": "missing"}
            continue
        start = read(pkg / "inputs" / job / "relax" / "POSCAR", format="vasp")
        fin = read(d / "CONTCAR", format="vasp")
        try:
            ds, df = describe(start), describe(fin)
            intact = (ds["stereocentres"] == df["stereocentres"] and ds["n_molecules"] == df["n_molecules"])
            err = None
        except ch.TopologyError as e:          # coordination change, broken NH3+, merged molecules...
            ds, df, intact, err = describe(start), None, False, str(e)
        rec = {"status": "intact" if intact else "BROKEN", "error": err,
               "fmax_last_eV_A": round(last_fmax(d / "OUTCAR") or float("nan"), 4),
               "cell_start": np.round(start.cell.cellpar(), 3).tolist(), "cell_final": np.round(fin.cell.cellpar(), 3).tolist(),
               "volume_change_pct": round(100 * (fin.get_volume() / start.get_volume() - 1), 2),
               "start": ds, "relaxed": df}
        if job.startswith("R_") and df:
            inv = describe(ch.invert(fin))
            rec["relaxed_inverted_mean"] = inv["mean"]
            rec["mirror_exact"] = all(abs(inv["mean"][k] + df["mean"][k]) < 1e-5 for k in inv["mean"])
        out[job] = rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = analyze(a.package)
    txt = json.dumps(res, indent=1)
    if a.json:
        a.json.write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
