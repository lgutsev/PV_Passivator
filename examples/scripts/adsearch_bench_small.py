"""Laptop-sized pilot of the random-vs-evolutionary search benchmark (docs/acat_feasibility.md, section 7).

    python examples/scripts/adsearch_bench_small.py CONFIG --slab POSCAR --model MODEL --out DIR
           [--budget 16] [--n-init 8] [--seeds 0,1] [--screen-steps 80] [--screen-fmax 0.1]
           [--refine-top 6] [--refine-steps 400] [--refine-fmax 0.03] [--threads 4]

Two stages, because a relaxation that reaches the deep minima costs several hundred steps per candidate (section 8g):

  screen   every valid proposal of both arms is relaxed with LBFGS to a loose threshold with a small step cap.  This
           is the energy the search ranks by, so the evolutionary arm selects parents on SHALLOW energies.
  refine   the lowest unique screened structures over all runs are relaxed further with LBFGS (tight threshold,
           large cap) and given one float64 energy, to see whether the screening order survives.

Every relaxation is cached on disk under the hash of its starting geometry.  For one seed both arms start from the
same proposals, so those are relaxed once; the arm that reuses a result is still charged its calls.  Rerunning the
same command resumes: finished relaxations are read back, the generators replay the same proposals.

This is a pilot: with this budget and two seeds it supports no claim about which arm is better.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))


class Cached:
    """Disk cache around an evaluator, keyed by the starting geometry."""

    def __init__(self, inner, folder: Path):
        self.inner, self.folder, self.protocol = inner, folder, inner.protocol
        folder.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def __call__(self, atoms, genes):
        from ase.io import read, write
        from pvpassivator import adsearch as A
        key = A.structure_hash(atoms)
        fj, fx = self.folder / f"{key}.json", self.folder / f"{key}.extxyz"
        if fj.exists() and fx.exists():
            self.hits += 1
            m = json.loads(fj.read_text())
            return A.EvalResult(read(fx), m["energy"], m["n_calls"], m["converged"], {**m["info"], "cached": True})
        self.misses += 1
        r = self.inner(atoms, genes)
        write(fx, r.atoms, format="extxyz")
        fj.write_text(json.dumps({"energy": r.energy, "n_calls": r.n_calls, "converged": r.converged, "info": r.info}))
        return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"), ap.add_argument("--slab", required=True), ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=16), ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--screen-steps", type=int, default=80), ap.add_argument("--screen-fmax", type=float, default=0.1)
    ap.add_argument("--refine-top", type=int, default=6)
    ap.add_argument("--refine-steps", type=int, default=400), ap.add_argument("--refine-fmax", type=float, default=0.03)
    ap.add_argument("--threads", type=int, default=4), ap.add_argument("--device", default="cuda")
    ap.add_argument("--stop-file", default=None, help="if this file exists, stop cleanly before the next relaxation")
    a = ap.parse_args(argv)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    from ase.io import write
    import adsearch_pilot as P
    from local_settle import make_calc
    from pvpassivator import adsearch as A

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    cfg["slab"]["path"] = str(Path(a.slab).resolve())
    space, prov = P.load_space(cfg)
    b = cfg.get("benchmark", {})
    rx = b.get("relax", {})
    seeds = [int(x) for x in a.seeds.split(",")]
    calc = make_calc(Path(a.model), a.device, "float32")
    stop = Path(a.stop_file) if a.stop_file else None

    class Stop(Exception):
        pass

    def guard(ev):
        def run(atoms, genes):
            if stop is not None and stop.exists():
                raise Stop()
            return ev(atoms, genes)
        run.protocol = ev.protocol
        return run

    kw = dict(radius=rx.get("radius_A", 8.0), mobile=rx.get("mobile", "face"), space=space, optimizer="LBFGS")
    screen = Cached(A.AseRelaxEvaluator(calc, fmax=a.screen_fmax, steps=a.screen_steps, **kw), out / "cache_screen")
    refine = Cached(A.AseRelaxEvaluator(calc, fmax=a.refine_fmax, steps=a.refine_steps, **kw), out / "cache_refine")
    prov["evaluator"] = {"kind": "mace+d3", "model": str(a.model),
                         "model_sha256_16": hashlib.sha256(Path(a.model).read_bytes()).hexdigest()[:16],
                         "dtype": "float32 (MACE and D3) in relaxations; float64 MACE for the refined energies",
                         "d3": "BJ, xc=r2scan (torch-dftd)", "screen": screen.protocol, "refine": refine.protocol}
    t0 = time.time()
    logs = []
    try:
        for s in seeds:
            for arm in ("random", "evolutionary"):
                ev = guard(screen)
                log = A.run_search(space, ev, arm, a.budget, s, n_init=a.n_init, pop_size=b.get("pop_size", 10),
                                   dup_tol=b.get("dup_tol_A", 0.5))
                logs.append(log)
                print(arm, s, log["counts"], "calls", log["total_calls"], "ranked", log["ranked"], "best",
                      log["best_energy"], f"cache {screen.hits}/{screen.hits + screen.misses}", flush=True)
        A.write_manifest(out, space, logs, prov, write_structures="unique")
        # ---- refine the lowest unique screened structures over all runs
        cand = sorted(((r["energy"], f"{log['arm']}_s{log['seed']}", r) for log in logs for r in log["records"]
                       if r["ranked"]), key=lambda x: x[0])
        picked = []
        for e, tag, r in cand:
            twin = next((p for p in picked if space.same_structure(r["_atoms"], p["atoms"], b.get("dup_tol_A", 0.5))), None)
            if twin is not None:
                twin["found_by"].append(f"{tag}:a{r['attempt']}")
                continue
            if len(picked) < a.refine_top:
                picked.append({"atoms": r["_atoms"], "screen_energy": e, "found_by": [f"{tag}:a{r['attempt']}"],
                               "modes": "+".join(sorted(g["mode_id"] for g in r["genes"])), "operator": r["operator"]})
        exact = None
        rows = []
        for k, p in enumerate(picked):
            res = guard(refine)(p["atoms"], None)
            if exact is None:
                exact = make_calc(Path(a.model), a.device, "float64")
            c = res.atoms.copy()
            c.calc = exact
            e64 = float(c.get_potential_energy())
            write(out / f"refined_{k}.extxyz", res.atoms, format="extxyz")
            move = float(np.linalg.norm(res.atoms.positions - p["atoms"].positions, axis=1).max())
            rows.append({"rank_screen": k, "screen_energy": p["screen_energy"], "refined_energy_f32": res.energy,
                         "refined_energy_f64": e64, "converged": res.converged, "steps": res.info["steps"],
                         "fmax_end": res.info["fmax_end"], "max_move_A": round(move, 3), "modes": p["modes"],
                         "operator_first": p["operator"], "found_by": p["found_by"], "file": f"refined_{k}.extxyz"})
            print("refined", k, p["modes"], round(p["screen_energy"], 3), "->", round(e64, 3), "steps", res.info["steps"],
                  "conv", res.converged, flush=True)
    except Stop:
        print("stopped on request; rerun the same command to resume", flush=True)
        return 3
    summ = {"schema": "adsearch_bench_small/1", "provenance": prov, "budget_attempts": a.budget, "n_init": a.n_init,
            "seeds": seeds, "wall_seconds": round(time.time() - t0, 1),
            "claim_status": "pilot: 2 seeds and a small budget support no claim about which arm is better",
            "runs": [{k: v for k, v in log.items() if k != "records"} |
                     {"diversity": A.low_energy_diversity(space, log, 0.1),
                      "relax_seconds": round(sum(r.get("eval_info", {}).get("seconds", 0) for r in log["records"]), 1),
                      "steps": [r["eval_info"]["steps"] for r in log["records"] if r["evaluated"]]} for log in logs],
            "refined": rows}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=A._json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
