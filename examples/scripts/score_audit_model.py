"""Score a MACE model on an audit-frame dataset (export_audit_frames.py), per split.

    python examples/scripts/score_audit_model.py DATASET --model MODEL [--device cuda] --out report.json
    python examples/scripts/score_audit_model.py DATASET --stored --out report.json      # the --align-to model's numbers

Compared at fixed geometry against REF_energy / REF_forces (DFT minus D3 when the dataset was exported that way, so
no dispersion is added on either side).  Reported per split:
  energy   absolute error against the aligned target, and errors of energy DIFFERENCES inside one composition group
           (the quantity a ranking uses); Spearman only as an indication when a split has few structures.
  forces   RMSE over all atoms, and over the "near" set (molecule atoms and substrate atoms within 4 A of one), plus
           the mean cosine for near atoms with |F_ref| > 0.05 eV/A.
Thresholds are the ones fixed for the audit (prepare_static_audit.THRESHOLDS).  Only the TEST split is evidence about a
fine-tuned model; train and valid are printed to show how much was fitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.geometry import get_distances
from ase.io import read

TH = {"relative_energy_mae_eV": 0.10, "relative_energy_max_eV": 0.20, "force_rmse_near_eV_A": 0.05, "force_cos_min": 0.9}


def near_mask(a) -> np.ndarray:
    mol = a.arrays["mol_id"] > 0
    if not mol.any() or mol.all():
        return mol.copy()
    d = get_distances(a.positions, a.positions[mol], cell=a.cell, pbc=a.pbc)[1].min(axis=1)
    return mol | (d <= 4.0)


def score(dataset: Path, model=None, device="cuda", stored=False) -> dict:
    calc = None
    if not stored:
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=str(model), device=device, default_dtype="float64")
    out = {"schema": "audit_model_score/1", "dataset": str(dataset), "model": "stored align_model_*" if stored else str(model),
           "thresholds": TH, "splits": {}}
    for split in ("train", "valid", "test"):
        f = dataset / f"{split}.extxyz"
        frames = read(f, index=":") if f.exists() and f.stat().st_size else []
        rows, nd, nm, ad, am = {}, [], [], [], []
        for a in frames:
            if stored:
                e, fm = a.info["align_model_energy"], a.arrays["align_model_forces"]
            else:
                c = a.copy()
                c.calc = calc
                e, fm = float(c.get_potential_energy()), c.get_forces()
            fr = a.arrays["REF_forces"]
            near = near_mask(a)
            rows[a.info["job"]] = {"group": a.info["group"], "family": a.info["family"], "n_atoms": len(a),
                                   "E_ref": a.info["REF_energy"], "E_model": e, "dE_abs_eV": e - a.info["REF_energy"],
                                   "force_rmse_all": float(np.sqrt(((fm - fr) ** 2).mean())),
                                   "force_rmse_near": float(np.sqrt(((fm - fr)[near] ** 2).mean())) if near.any() else None}
            nd.append(fr[near]), nm.append(fm[near]), ad.append(fr), am.append(fm)
        s = {"n": len(frames), "structures": rows, "groups": {}}
        if frames:
            D, M = np.concatenate(nd), np.concatenate(nm)
            big = np.linalg.norm(D, axis=1) > 0.05
            cos = (D[big] * M[big]).sum(1) / (np.linalg.norm(D[big], axis=1) * np.linalg.norm(M[big], axis=1) + 1e-12)
            A, B = np.concatenate(ad), np.concatenate(am)
            s["forces"] = {"rmse_all_eV_A": float(np.sqrt(((A - B) ** 2).mean())),
                           "rmse_near_eV_A": float(np.sqrt(((D - M) ** 2).mean())), "n_near": int(len(D)),
                           "mean_cos_near": float(cos.mean()) if big.any() else None}
            s["energy_abs_mae_eV"] = float(np.mean([abs(r["dE_abs_eV"]) for r in rows.values()]))
            for g in sorted({r["group"] for r in rows.values()}):
                names = [n for n, r in rows.items() if r["group"] == g]
                if len(names) < 2:
                    continue
                er, em = np.array([rows[n]["E_ref"] for n in names]), np.array([rows[n]["E_model"] for n in names])
                d = (er - er.mean()) - (em - em.mean())
                gg = {"names": names, "ref_span_eV": float(np.ptp(er)), "relative_mae_eV": float(np.abs(d).mean()),
                      "relative_max_eV": float(np.abs(d).max()),
                      "ref_rel_eV": (er - er.min()).round(3).tolist(), "model_rel_eV": (em - em[er.argmin()]).round(3).tolist()}
                if len(names) >= 4:
                    rk = lambda x: np.argsort(np.argsort(x))
                    gg["spearman"] = float(np.corrcoef(rk(er), rk(em))[0, 1])
                s["groups"][g] = gg
        out["splits"][split] = s
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--model")
    ap.add_argument("--stored", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not a.stored and not a.model:
        ap.error("give --model or --stored")
    rep = score(Path(a.dataset), a.model, a.device, a.stored)
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    for split, s in rep["splits"].items():
        if not s["n"]:
            continue
        print(split, "n", s["n"], "| E abs MAE", round(s["energy_abs_mae_eV"], 3), "| F rmse all", round(s["forces"]["rmse_all_eV_A"], 3),
              "near", round(s["forces"]["rmse_near_eV_A"], 3), "cos", s["forces"]["mean_cos_near"] and round(s["forces"]["mean_cos_near"], 2))
        for g, gg in s["groups"].items():
            print("   ", g, "n", len(gg["names"]), "span", round(gg["ref_span_eV"], 2), "rel MAE", round(gg["relative_mae_eV"], 3),
                  "max", round(gg["relative_max_eV"], 3), "spearman", gg.get("spearman") and round(gg["spearman"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
