"""Turn re-relaxed structures (rerelax_pool.py) into an audit pool that prepare_static_audit.py can read.

    python examples/scripts/rerelax_to_audit_pool.py RERELAX_DIR --source-pool POOL --model MODEL --out NEW_POOL
           --names a,b,c [--prefix rr_]

For each named structure the model's energy and forces (MODEL + D3(BJ, r2SCAN), MACE in float64) are evaluated at
exactly the re-relaxed geometry and stored with it, as adsearch_v1_pool.py does, so the DFT single point can be
compared atom by atom.  The bare slab and the isolated molecule are copied from the source pool as references (their
geometry is unchanged; prepare_static_audit.py can be told not to run them again).
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
    ap.add_argument("rerelax")
    ap.add_argument("--source-pool", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--prefix", default="rr_")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from local_settle import make_calc
    rr, src, out = Path(a.rerelax), Path(a.source_pool), Path(a.out)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    summ = json.loads((src / "pool.json").read_text())
    runs = {r["name"]: r for r in json.loads((rr / "rerelax.json").read_text())["structures"]}
    calc = make_calc(Path(a.model), a.device, "float64")
    recs = []
    for name in a.names.split(","):
        r = runs[name]
        x = read(rr / "structures" / f"{name}.extxyz")
        x.calc = None
        c = x.copy()
        c.calc = calc
        e, f = float(c.get_potential_energy()), c.get_forces()
        x.set_array("model_forces", f)
        x.info = {"model_energy": e}
        new = a.prefix + name
        fn = f"structures/{new}.extxyz"
        write(out / fn, x, format="extxyz")
        recs.append({"name": new, "group": "rerelaxed", "from": name, "n_atoms": len(x), "formula": x.get_chemical_formula(),
                     "relaxed": True, "converged": r["converged"], "steps": r["steps"], "substrate_ok": True,
                     "final_state": "intact_adsorption" if r["molecules_intact"] else "changed",
                     "model_energy_eV": e, "model_fmax_all_eV_A": float(np.linalg.norm(f, axis=1).max()),
                     "N_Pb_min_A": r["N_Pb_min_end_A"], "max_move_molecule_A": r["max_move_molecule_A"], "file": fn})
        print(new, round(e, 4), flush=True)
    for k in ("slab", "molecule"):
        shutil.copy(src / summ["references"][k]["file"], out / summ["references"][k]["file"])
    prov = dict(summ["provenance"])
    prov["evaluator"] = {"model": str(a.model), "model_sha256": hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
                         "dispersion": "D3(BJ, r2SCAN) torch-dftd", "dtype": "float64 (MACE)"}
    prov["rerelaxed_from"] = {"pool": str(src), "rerelax": str(rr),
                              "note": "geometries relaxed by the evaluator model, starting from the source pool's minima"}
    refs = json.loads(json.dumps(summ["references"]))
    for k in refs:      # the stored reference energies belong to the source pool's model, not to this evaluator
        refs[k]["model_energy_eV"] = None
    (out / "pool.json").write_text(json.dumps({"schema": summ.get("schema"), "provenance": prov, "structures": recs,
                                               "references": refs}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
