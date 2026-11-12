"""Continue the relaxations of a v1 pool with another model, to see how far its minima are from the first model's.

    python examples/scripts/rerelax_pool.py POOL --model MODEL --out DIR [--fmax 0.05] [--steps 300] [--names a,b]

Every relaxed structure of POOL (adsearch_v1_pool.py) starts from the geometry the pool's model left it in and is
relaxed with MODEL + D3(BJ, r2SCAN), with the SAME free atoms (the stored ``mobile`` array) and a fixed cell.  The
relaxation runs in float32 for speed; the reported energy and forces are one float64 evaluation of the final
geometry.  Per structure: steps, energy at the start and at the end (both with MODEL, float64), largest displacement
of a molecule atom and of a slab atom, the closest N-Pb distance before and after, and whether the molecules are
still intact.  Resumable: finished structures are skipped.

This says how much the minima move and whether the order changes.  It is not a validation: only DFT at the new
geometries can test the new model where it chose to go.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.geometry import find_mic, get_distances
from ase.io import read, write
from ase.optimize import FIRE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def n_pb_min(a) -> float:
    s = np.array(a.get_chemical_symbols())
    n = (a.arrays["mol_id"] > 0) & (s == "N")
    return float(get_distances(a.positions[n], a.positions[s == "Pb"], cell=a.cell, pbc=a.pbc)[1].min())


def molecules_intact(a0, a1) -> bool:
    """Same intramolecular bonds (pairs within 1.7 A among each molecule's atoms) before and after."""
    for k in sorted(set(a0.arrays["mol_id"]) - {0}):
        i = np.where(a0.arrays["mol_id"] == k)[0]
        h = a0.numbers[i] == 1
        lim = np.where(h[:, None] | h[None, :], 1.25, 1.65)          # X-H and heavy-heavy bond lengths
        d0 = get_distances(a0.positions[i], cell=a0.cell, pbc=a0.pbc)[1]
        d1 = get_distances(a1.positions[i], cell=a1.cell, pbc=a1.pbc)[1]
        bonded = (d0 < lim) & (d0 > 0)
        if (d1[bonded] > 1.2 * lim[bonded]).any() or ((d1 < 0.85 * lim) & ~bonded & (d0 > 0)).any():
            return False                                             # a bond broke, or a new one formed
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--names", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from local_settle import make_calc
    pool, out = Path(a.pool), Path(a.out)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    recs = json.loads((pool / "pool.json").read_text())["structures"]
    names = [r["name"] for r in recs if r["relaxed"] and (not a.names or r["name"] in a.names.split(","))]
    fast, exact = make_calc(Path(a.model), a.device, "float32"), None
    results = []
    for name in names:
        fj = out / "structures" / f"{name}.json"
        if fj.exists():
            results.append(json.loads(fj.read_text()))
            continue
        a0 = read(pool / "structures" / f"{name}.extxyz")
        frozen = a0.arrays["mobile"] == 0
        x = a0.copy()
        for k in ("model_forces", "mobile"):
            del x.arrays[k]
        x.set_constraint(FixAtoms(mask=frozen))
        x.calc = fast
        t0 = time.perf_counter()
        f0 = x.get_forces(apply_constraint=False)
        opt = FIRE(x, logfile=None)
        conv = bool(opt.run(fmax=a.fmax, steps=a.steps))
        secs = time.perf_counter() - t0
        rec = {"name": name, "group": a0.get_chemical_formula(), "steps": int(opt.nsteps), "converged": conv,
               "fmax_start_free": float(np.linalg.norm(f0[~frozen], axis=1).max()), "seconds": round(secs, 1),
               "frozen_max_move_A": float(np.abs(x.positions[frozen] - a0.positions[frozen]).max()),
               "N_Pb_min_start_A": round(n_pb_min(a0), 3), "N_Pb_min_end_A": round(n_pb_min(x), 3),
               "molecules_intact": molecules_intact(a0, x)}
        d = find_mic(x.positions - a0.positions, a0.cell, a0.pbc)[1]
        mol = a0.arrays["mol_id"] > 0
        rec["max_move_molecule_A"] = float(d[mol].max())
        rec["max_move_slab_A"] = float(d[~mol].max())
        rec["_start"], rec["_end"] = a0.positions.tolist(), None
        x.set_constraint()
        write(out / "structures" / f"{name}.extxyz", x, format="extxyz")
        rec.pop("_start"), rec.pop("_end")
        fj.write_text(json.dumps(rec, indent=1) + "\n")
        results.append(rec)
        print(json.dumps(rec), flush=True)
    # float64 energies at start and end geometries with the same model (one calculator at a time on an 8 GB GPU)
    del fast
    import torch
    torch.cuda.empty_cache()
    exact = make_calc(Path(a.model), a.device, "float64")
    for rec in results:
        if "energy_end_eV" in rec:
            continue
        for tag, f in (("start", pool / "structures" / f"{rec['name']}.extxyz"), ("end", out / "structures" / f"{rec['name']}.extxyz")):
            c = read(f)
            c.calc = exact
            rec[f"energy_{tag}_eV"] = float(c.get_potential_energy())
            if tag == "end":
                fr = c.get_forces()
                rec["fmax_end_free_f64"] = float(np.linalg.norm(fr[read(pool / 'structures' / f"{rec['name']}.extxyz").arrays['mobile'] > 0], axis=1).max())
        rec["energy_gain_eV"] = rec["energy_end_eV"] - rec["energy_start_eV"]
        rec["molecules_intact"] = molecules_intact(read(pool / "structures" / f"{rec['name']}.extxyz"),
                                                   read(out / "structures" / f"{rec['name']}.extxyz"))
        (out / "structures" / f"{rec['name']}.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "rerelax.json").write_text(json.dumps({"schema": "pool_rerelax/1", "pool": str(pool), "model": str(a.model),
                                                  "fmax": a.fmax, "steps": a.steps, "structures": results}, indent=1) + "\n")
    for r in results:
        print(f"{r['name']:18s} steps {r['steps']:3d} conv {r['converged']} dE {r['energy_gain_eV']:+.3f} "
              f"mol {r['max_move_molecule_A']:.2f} slab {r['max_move_slab_A']:.2f} N-Pb {r['N_Pb_min_start_A']}->{r['N_Pb_min_end_A']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
