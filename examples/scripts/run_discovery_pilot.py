"""Matched-budget three-arm pilot on the SYNTHETIC oracle (engineering test of the discovery loop; no physics).

    python examples/scripts/run_discovery_pilot.py --out data/discovery/pilot_synthetic_v1 [--seeds 10] [--budget 30]

Arms (same eligibility, objective spec, oracle, budget, batch size and starting knowledge):
  A    expert library in priority order
  B    GPParEGO over the conventional grid
  B0   random over the conventional grid (calibrates whether the GP helps at all)
  C    GPParEGO over the union of recorded representation proposals (ReplayProposer; one per round)

Scoring uses the oracle's noise-free values, which no arm sees. For each oracle seed the reference set is the union
of all arms' pools; metrics are the hypervolume of the feasible evaluated points divided by the hypervolume of the
reference set's true feasible front (HV ratio), the fraction of evaluations spent on truly infeasible candidates,
and how many true-front members each arm evaluated.

What this can show: that the loop runs, respects budgets and dedup, and how much the *pool* matters relative to the
optimizer on a landscape nobody designed for. What it cannot show: anything about passivation, or that an LLM
improves discovery (the proposals were written once, without feedback, by the same session that wrote this code).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pvpassivator.discovery.chem import Eligibility  # noqa: E402
from pvpassivator.discovery.loop import run_pool, run_representation_loop  # noqa: E402
from pvpassivator.discovery.objectives import ObjectiveSpec, hypervolume, nondominated  # noqa: E402
from pvpassivator.discovery.optimize import GPParEGO, RandomOptimizer, FixedOrderOptimizer  # noqa: E402
from pvpassivator.discovery.oracles import SyntheticOracle  # noqa: E402
from pvpassivator.discovery.proposer import ReplayProposer  # noqa: E402
from pvpassivator.discovery.space import enumerate_spec, load_spec  # noqa: E402

EX = Path(__file__).resolve().parents[1] / "discovery"


def true_front_metrics(spec, oracle, evaluated, reference):
    def pts(cands):
        rows = [oracle.truth(c) for c in cands]
        ok = [(c, r) for c, r in zip(cands, rows) if spec.feasible(r)]
        return ok, (np.array([spec.as_min(r) for _, r in ok]) if ok else np.zeros((0, len(spec.objectives))))
    ref_ok, Yref = pts(reference)
    lo, hi = Yref.min(0), Yref.max(0)
    norm = lambda Y: (Y - lo) / np.where(hi > lo, hi - lo, 1.0)
    refpt = np.full(len(spec.objectives), 1.1)
    hv_ref = hypervolume(norm(Yref), refpt)
    ev_ok, Yev = pts(evaluated)
    hv = hypervolume(norm(Yev), refpt) if len(Yev) else 0.0
    front_keys = {ref_ok[i][0].inchikey for i in np.flatnonzero(nondominated(Yref))}
    n_inf = sum(not spec.feasible(oracle.truth(c)) for c in evaluated)
    return {"hv_ratio": hv / hv_ref if hv_ref else 0.0, "n_eval": len(evaluated),
            "infeasible_fraction": n_inf / len(evaluated) if evaluated else 0.0,
            "front_found": len(front_keys & {c.inchikey for c in evaluated}), "front_size": len(front_keys)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=30, help="evaluations per arm")
    ap.add_argument("--batch", type=int, default=3)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)

    spec = ObjectiveSpec.load(EX / "objectives_synthetic.json")
    rules = Eligibility()
    lib, _ = enumerate_spec(load_spec(EX / "expert_library.json"), rules)
    grid, _ = enumerate_spec(load_spec(EX / "conventional_space.json"), rules)
    rep_files = sorted((EX / "representations").glob("rep_*.json"))
    rep_pool = {}
    for f in rep_files:
        for c in enumerate_spec(load_spec(f), rules)[0]:
            rep_pool.setdefault(c.inchikey, c)
    reference = list({c.inchikey: c for c in lib + grid + list(rep_pool.values())}.values())

    rows = []
    t0 = time.time()
    for s in range(a.seeds):
        oracle = SyntheticOracle(seed=s)
        cost = oracle.cost
        budget = a.budget * cost
        arms = {
            "A": run_pool("A", lib, FixedOrderOptimizer(), oracle, spec, budget, a.batch, s),
            "B": run_pool("B", grid, GPParEGO(spec), oracle, spec, budget, a.batch, s),
            "B0": run_pool("B0", grid, RandomOptimizer(), oracle, spec, budget, a.batch, s),
            "C": run_representation_loop(ReplayProposer(rep_files), GPParEGO(spec), oracle, spec, budget, a.batch,
                                         rounds=len(rep_files), seed=s, rules=rules),
        }
        for name, res in arms.items():
            m = true_front_metrics(spec, oracle, res.evaluated, reference)
            m.update({"seed": s, "arm": name, "pool": res.pool_size, "unused_budget_evals":
                      round(res.unused_budget / cost), "proposals_rejected":
                      sum(not p.accepted for p in res.proposals)})
            rows.append(m)
        print(f"seed {s}: " + "  ".join(f"{r['arm']} hv={r['hv_ratio']:.2f}" for r in rows[-4:]), flush=True)

    summary = {}
    for arm in ("A", "B", "B0", "C"):
        r = [x for x in rows if x["arm"] == arm]
        hv = np.array([x["hv_ratio"] for x in r])
        summary[arm] = {"hv_ratio_mean": float(hv.mean()), "hv_ratio_sem": float(hv.std(ddof=1) / np.sqrt(len(hv))),
                        "infeasible_fraction_mean": float(np.mean([x["infeasible_fraction"] for x in r])),
                        "front_found_mean": float(np.mean([x["front_found"] for x in r])),
                        "front_size_mean": float(np.mean([x["front_size"] for x in r])),
                        "n_eval_mean": float(np.mean([x["n_eval"] for x in r])), "pool": r[0]["pool"]}
    cb = np.array([x["hv_ratio"] for x in rows if x["arm"] == "C"]) - \
        np.array([x["hv_ratio"] for x in rows if x["arm"] == "B"])
    summary["C_minus_B_paired"] = {"mean": float(cb.mean()), "sem": float(cb.std(ddof=1) / np.sqrt(len(cb))),
                                   "wins": int((cb > 0).sum()), "n": len(cb)}
    meta = {"warning": "SYNTHETIC ORACLE: engineering test only, no physical meaning",
            "objective_spec": spec.id, "objective_hash": spec.hash, "eligibility_hash": rules.hash(),
            "budget_evals": a.budget, "batch": a.batch, "seeds": a.seeds, "reference_pool": len(reference),
            "rep_files": [f.name for f in rep_files], "wall_s": round(time.time() - t0, 1)}
    (out / "pilot_rows.json").write_text(json.dumps(rows, indent=1))
    (out / "pilot_summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
