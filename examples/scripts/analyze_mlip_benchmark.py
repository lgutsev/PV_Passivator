"""Compare MLIP single points with archived DFT energies; writes a JSON report and prints a summary.

    python examples/scripts/analyze_mlip_benchmark.py --inventory data/benchmark/runs.json \
        --energies data/benchmark/energies.jsonl --out data/benchmark/report.json

MLIP total = model energy + the D3 term matching the run's DFT profile (none of the models includes
dispersion). Tiers: "strict" = runs whose last ionic step changed E by <= 10 meV (CONTCAR matches the
energy); "loose" = up to ``--loose`` eV (unconverged relaxations, CONTCAR one step past the energy).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np  # noqa: E402
from pvpassivator.benchmark import (load_runs, same_composition_deltas, offset_fit,  # noqa: E402
                                    matched_adsorption, spearman, settings_groups, _stats)

PROFILE_D3 = {"pbe_d3zero": "d3:zero-pbe", "r2scan_d3bj": "d3:bj-r2scan"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--energies", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loose", type=float, default=0.2)
    a = ap.parse_args(argv)

    runs = load_runs(a.inventory)
    E: dict[tuple[str, str], float] = {}
    for line in Path(a.energies).read_text().splitlines():
        d = json.loads(line)
        E[(d["run_id"], d["term"])] = d["energy_eV"]
    models = sorted({t for _, t in E if not t.startswith("d3:")})

    def tier_ok(r, tier):
        if not r.evaluable():
            return False
        if tier == "strict":
            return r.single_point_ok
        return r.last_dE is None or abs(r.last_dE) <= a.loose or r.status == "single_point"

    report = {"models": models, "tiers": {"strict": "last dE <= 10 meV", "loose": f"last dE <= {a.loose} eV"},
              "results": {}}
    for prof, d3 in PROFILE_D3.items():
        for model in models:
            for tier in ("strict", "loose"):
                pool = [r for r in runs if r.profile == prof and tier_ok(r, tier)
                        and (r.run_id, model) in E and (r.run_id, d3) in E]
                e_dft = {r.run_id: r.energy_F for r in pool}
                e_ml = {r.run_id: E[(r.run_id, model)] + E[(r.run_id, d3)] for r in pool}
                e_ml_nod3 = {r.run_id: E[(r.run_id, model)] for r in pool}
                res = {"n_runs": len(pool), "groups": []}
                sc_rows, loo, loo_nod3, ads_rows = [], [], [], []
                # every comparison stays inside one settings group (same k-mesh, cutoff, functional)
                for (_, settings), rs in settings_groups(pool).items():
                    g = {"settings": json.loads(settings), "n_runs": len(rs)}
                    sc = same_composition_deltas(rs, e_dft, e_ml)
                    sc_rows += sc["rows"]
                    try:
                        of = offset_fit(rs, e_dft, e_ml)
                        g["offset_fit"] = of
                        loo += [x for x in of["rows"] if np.isfinite(x["loo"])]
                        loo_nod3 += [x["loo"] for x in offset_fit(rs, e_dft, e_ml_nod3)["rows"] if np.isfinite(x["loo"])]
                    except ValueError as err:
                        g["offset_fit"] = {"error": str(err)}
                    # references are isolated molecules/salts in vacuum boxes: their energy barely depends on
                    # the k-mesh, so they may come from any group of this profile whose other settings match
                    no_k = {k: v for k, v in json.loads(settings).items() if k != "KPOINTS"}
                    refs = [r for r in pool if "/REAG/" in r.run_id and
                            {k: v for k, v in json.loads(r.settings).items() if k != "KPOINTS"} == no_k]
                    # adsorption: relaxed structures only (an unrelaxed NORELAX slab is not a minimum)
                    surf = [r for r in rs if "/REAG/" not in r.run_id and r.status != "single_point"]
                    if refs:
                        ads = matched_adsorption(surf, surf, refs, e_dft, e_ml)
                        for x in ads["rows"]:
                            x["labels"] = all(next(r for r in pool if r.run_id == x[k]).accepted
                                              for k in ("complex", "slab", "ref"))
                            x["ref_kmesh_differs"] = next(r for r in pool if r.run_id == x["ref"]).settings != settings
                        ads_rows += ads["rows"]
                    res["groups"].append(g)
                res["same_composition"] = {"rows": sc_rows, "stats": _stats([x["err"] for x in sc_rows])}
                res["offset_loo"] = {"rows": loo, "stats": _stats([x["loo"] for x in loo]),
                                     "stats_without_d3": _stats(loo_nod3)}
                if ads_rows:
                    res["adsorption"] = {"rows": ads_rows, "stats": _stats([x["err"] for x in ads_rows]),
                                         "spearman": spearman([x["E_ads_dft"] for x in ads_rows],
                                                              [x["E_ads_mlip"] for x in ads_rows]),
                                         "stats_labels_only": _stats([x["err"] for x in ads_rows if x["labels"]])}
                report["results"][f"{prof}|{model}|{tier}"] = res

    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"{'set | model | tier':48s} {'N':>4s} {'sameC MAE':>9s} {'n':>3s} {'LOO MAE':>8s} {'noD3':>6s} {'n':>3s} "
          f"{'Eads MAE':>8s} {'n':>3s} {'rho':>5s}")
    f = lambda x: f"{x:.3f}" if isinstance(x, float) else "-"
    for key, res in report["results"].items():
        sc, lo, ad = res["same_composition"]["stats"], res["offset_loo"], res.get("adsorption", {})
        print(f"{key:48s} {res['n_runs']:4d} {f(sc.get('mae')):>9s} {sc.get('n', 0):3d} {f(lo['stats'].get('mae')):>8s} "
              f"{f(lo['stats_without_d3'].get('mae')):>6s} {lo['stats'].get('n', 0):3d} "
              f"{f(ad.get('stats', {}).get('mae')):>8s} {ad.get('stats', {}).get('n', 0):3d} {f(ad.get('spearman')):>5s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
