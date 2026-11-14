"""Turn arbitrary saved geometries into an audit pool that prepare_static_audit.py can read.

    python examples/scripts/geoms_to_audit_pool.py --source-pool POOL --model MODEL --out NEW_POOL
           NAME=FILE[::GROUP] [NAME=FILE[::GROUP] ...]

Each FILE is an extxyz on the source pool's cell (for example a geometry saved by fmax_study.py).  The model's energy
and forces (MODEL + D3(BJ, r2SCAN), MACE in float64) are evaluated at exactly that geometry and stored with it, as
adsearch_v1_pool.py does.  Bare slabs may be listed like any other structure: prepare_static_audit.py groups by
composition, so they form their own group.  The source pool's slab and molecule are copied as references (the slab
fixes the dipole centre; prepare_static_audit.py can be told not to run them again).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+")
    ap.add_argument("--source-pool", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    from local_settle import make_calc
    src, out = Path(a.source_pool), Path(a.out)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    summ = json.loads((src / "pool.json").read_text())
    cell0 = read(src / summ["references"]["slab"]["file"]).cell.array
    calc = make_calc(Path(a.model), a.device, "float64")
    recs = []
    for item in a.items:
        name, rest = item.split("=", 1)
        f, group = rest.rsplit("::", 1) if "::" in rest else (rest, "")
        x = read(f)
        assert np.allclose(x.cell.array, cell0, atol=1e-6), f"{name}: not on the source pool's cell"
        x.calc = None
        x.set_constraint()
        c = x.copy()
        c.calc = calc
        e, frc = float(c.get_potential_energy()), c.get_forces()
        x.set_array("model_forces", frc)
        x.info = {"model_energy": e}
        fn = f"structures/{name}.extxyz"
        write(out / fn, x, format="extxyz")
        mob = x.arrays["mobile"] == 1 if "mobile" in x.arrays else np.ones(len(x), bool)
        recs.append({"name": name, "group": group or "geometry", "from": str(f), "n_atoms": len(x),
                     "formula": x.get_chemical_formula(), "relaxed": True, "model_energy_eV": e,
                     ("model_fmax_free_eV_A" if "mobile" in x.arrays else "model_fmax_all_eV_A"): float(np.linalg.norm(frc[mob], axis=1).max()), "file": fn})
        print(name, len(x), round(e, 4), flush=True)
    for k in ("slab", "molecule"):
        shutil.copy(src / summ["references"][k]["file"], out / summ["references"][k]["file"])
    prov = dict(summ["provenance"])
    prov["evaluator"] = {"model": str(a.model), "model_sha256": hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
                         "dispersion": "D3(BJ, r2SCAN) torch-dftd", "dtype": "float64 (MACE)"}
    prov["geometries"] = {"source_pool": str(src), "note": a.note}
    refs = json.loads(json.dumps(summ["references"]))
    for k in refs:
        refs[k]["model_energy_eV"] = None
    (out / "pool.json").write_text(json.dumps({"schema": summ.get("schema"), "provenance": prov, "structures": recs,
                                               "references": refs}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
