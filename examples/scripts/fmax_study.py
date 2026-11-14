"""How tight must a relaxation be before energy differences between arrangements stop changing?

    python examples/scripts/fmax_study.py POOL --model MODEL --out DIR --names a,b,c,d
           [--thresholds 0.1,0.05,0.03,0.02,0.01] [--optimizers FIRE,LBFGS] [--max-steps 250] [--threads 4]

Each named structure of POOL (adsearch_v1_pool.py: stored geometry and ``mobile`` array) is relaxed with
MODEL + D3(BJ, r2SCAN) at fixed cell and with the same free atoms, once per optimizer, through a ladder of force
thresholds in a single trajectory.  The geometry is saved each time a threshold is first met.  Afterwards every saved
geometry gets one float64 energy, so the numbers compared are free of float32 noise.

Reported: for each threshold, the energy of each structure relative to the tightest threshold reached (what looser
convergence leaves on the table), the spread of that quantity across structures (what it does to energy DIFFERENCES,
the thing a ranking uses), and the disagreement between optimizers at the same threshold.  A threshold is adequate for
a resolution R when both of the last two are well below R.

Resumable per (structure, optimizer).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--thresholds", default="0.1,0.05,0.03,0.02,0.01")
    ap.add_argument("--optimizers", default="FIRE,LBFGS")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    from ase.constraints import FixAtoms
    from ase.io import read, write
    from ase.optimize import FIRE, LBFGS
    from local_settle import make_calc
    OPT = {"FIRE": FIRE, "LBFGS": LBFGS}
    thr = [float(x) for x in a.thresholds.split(",")]
    pool, out = Path(a.pool), Path(a.out)
    (out / "geoms").mkdir(parents=True, exist_ok=True)
    names, opts = a.names.split(","), a.optimizers.split(",")
    fast = make_calc(Path(a.model), a.device, "float32")
    runs = {}
    for name in names:
        a0 = read(pool / "structures" / f"{name}.extxyz")
        frozen = a0.arrays["mobile"] == 0
        for on in opts:
            fj = out / f"{name}__{on}.json"
            if fj.exists():
                runs[(name, on)] = json.loads(fj.read_text())
                continue
            x = a0.copy()
            for k in ("model_forces", "mobile"):
                del x.arrays[k]
            x.set_constraint(FixAtoms(mask=frozen))
            x.calc = fast
            opt = OPT[on](x, logfile=None)
            rec = {"name": name, "optimizer": on, "n_free": int((~frozen).sum()), "levels": []}
            t0 = time.perf_counter()
            for f in thr:
                left = a.max_steps - opt.nsteps
                ok = bool(opt.run(fmax=f, steps=left)) if left > 0 else False
                fm = float(np.linalg.norm(x.get_forces()[~frozen], axis=1).max())
                if not (ok and fm <= f * 1.001):
                    rec["stopped"] = f"step cap {a.max_steps} before fmax {f} (fmax now {fm:.4f})"
                    break
                g = x.copy()
                g.set_constraint()
                write(out / "geoms" / f"{name}__{on}__{f}.extxyz", g, format="extxyz")
                rec["levels"].append({"fmax": f, "steps": int(opt.nsteps), "fmax_reached": fm,
                                      "energy_f32": float(x.get_potential_energy()),
                                      "max_move_from_start_A": float(np.linalg.norm(x.positions - a0.positions, axis=1).max())})
            rec["seconds"] = round(time.perf_counter() - t0, 1)
            fj.write_text(json.dumps(rec, indent=1) + "\n")
            runs[(name, on)] = rec
            print(name, on, [(lv["fmax"], lv["steps"]) for lv in rec["levels"]], rec.get("stopped", ""), rec["seconds"], flush=True)
    del fast
    torch.cuda.empty_cache()
    exact = make_calc(Path(a.model), a.device, "float64")
    for (name, on), rec in runs.items():
        changed = False
        for lv in rec["levels"]:
            if "energy" not in lv:
                c = read(out / "geoms" / f"{name}__{on}__{lv['fmax']}.extxyz")
                c.calc = exact
                lv["energy"] = float(c.get_potential_energy())
                changed = True
        if changed:
            (out / f"{name}__{on}.json").write_text(json.dumps(rec, indent=1) + "\n")
    # ---- summary
    E = {(n, o, lv["fmax"]): lv["energy"] for (n, o), r in runs.items() for lv in r["levels"]}
    P = {}
    summary = {"schema": "fmax_study/1", "model": str(a.model), "pool": str(pool), "thresholds": thr, "names": names,
               "optimizers": opts, "runs": [runs[k] for k in runs], "by_threshold": {}}
    common = [f for f in thr if all((n, o, f) in E for n in names for o in opts)]
    summary["thresholds_reached_by_all"] = common
    if common:
        tight = common[-1]
        ref = {n: min(E[(n, o, tight)] for o in opts) for n in names}        # best energy at the tightest common level
        for f in common:
            d = {o: np.array([E[(n, o, f)] - ref[n] for n in names]) for o in opts}
            allv = np.concatenate(list(d.values()))
            row = {"left_on_table_eV": {o: dict(zip(names, d[o].round(4).tolist())) for o in opts},
                   "left_on_table_max_eV": float(allv.max()),
                   # effect on energy differences between structures: spread of the residual across structures
                   "difference_error_max_eV": {o: float(d[o].max() - d[o].min()) for o in opts},
                   "steps": {o: {n: next(lv["steps"] for lv in runs[(n, o)]["levels"] if lv["fmax"] == f) for n in names} for o in opts}}
            if len(opts) >= 2:
                dis = np.array([abs(E[(n, opts[0], f)] - E[(n, opts[1], f)]) for n in names])
                row["optimizer_disagreement_eV"] = dict(zip(names, dis.round(4).tolist()))
                row["optimizer_disagreement_max_eV"] = float(dis.max())
                pa = [read(out / "geoms" / f"{n}__{opts[0]}__{f}.extxyz").positions for n in names]
                pb = [read(out / "geoms" / f"{n}__{opts[1]}__{f}.extxyz").positions for n in names]
                row["optimizer_max_dpos_A"] = float(max(np.linalg.norm(x - y, axis=1).max() for x, y in zip(pa, pb)))
            summary["by_threshold"][str(f)] = row
            print(f"fmax {f}: left on table max {row['left_on_table_max_eV']:.3f} eV; difference error "
                  f"{ {o: round(v, 3) for o, v in row['difference_error_max_eV'].items()} }; optimizer disagreement "
                  f"{row.get('optimizer_disagreement_max_eV', float('nan')):.3f} eV; steps "
                  f"{ {o: int(np.mean(list(v.values()))) for o, v in row['steps'].items()} }")
    (out / "fmax_study.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
