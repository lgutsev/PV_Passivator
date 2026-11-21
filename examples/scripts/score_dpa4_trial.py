"""Score the returned DPA-4 fold predictions against the round-3 MACE cross-validation, on the same footing.

    python examples/scripts/score_dpa4_trial.py RETURNED_PACKAGE --mace-cv data/cache/adsearch/ft110/cv_v3 [--out report.json]

RETURNED_PACKAGE is the exported package (prepare_dpa4_trial.py): data/fold{k}/test/<formula>/ (references, mol_id,
names) and outputs/fold{k}/pred/test/<formula>.npz (energy, force, names).  The MACE numbers are read from
cv_v3/fold{k}/score.json, which score_audit_model.py wrote with the same definitions:

  relative energy     within one fold and one composition, (E_model - E_ref) minus its mean over that fold's test
                      frames of that composition; reported per fold and pooled over folds (MAE, worst).  A constant
                      per-composition offset (DPA-4's PBE reference, MACE's alignment) cancels.
  force, near         RMSE over molecule atoms and substrate atoms within 4 A of them (score_audit_model.near_mask),
                      pooled over atoms.
  force, all atoms    RMSE over every atom.

Frames are matched by name, so a fold missing in either model is reported, not silently dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

sys.path.insert(0, str(Path(__file__).parent))
from score_audit_model import near_mask  # noqa: E402


def dpa4_fold(pkg: Path, k: int) -> dict:
    rows = {}
    pred = pkg / "outputs" / f"fold{k}" / "pred" / "test"
    for s in sorted(p for p in (pkg / "data" / f"fold{k}" / "test").iterdir() if (p / "type.raw").exists()):
        f = pred / f"{s.name}.npz"
        if not f.exists():
            raise FileNotFoundError(f"fold {k}: no predictions for {s.name}")
        z = np.load(f)
        names = [ln.split("\t")[0] for ln in (s / "names.txt").read_text().splitlines()]
        if [str(x) for x in z["names"]] != names:
            raise ValueError(f"fold {k} {s.name}: frame order differs between data and predictions")
        tmap = (s / "type_map.raw").read_text().split()
        sym = [tmap[int(t)] for t in (s / "type.raw").read_text().split()]
        coord, box = np.load(s / "set.000" / "coord.npy"), np.load(s / "set.000" / "box.npy")
        e0, f0 = np.load(s / "set.000" / "energy.npy"), np.load(s / "set.000" / "force.npy")
        mol = np.load(s / "mol_id.npy")
        for i, n in enumerate(names):
            a = Atoms(sym, positions=coord[i].reshape(-1, 3), cell=box[i].reshape(3, 3), pbc=True)
            a.set_array("mol_id", mol[i])
            fr, fm = f0[i].reshape(-1, 3), z["force"][i]
            rows[n] = {"group": s.name, "E_ref": float(e0[i]), "E_model": float(z["energy"][i]),
                       "d_near": (fm - fr)[near_mask(a)], "d_all": fm - fr}
    return rows


def mace_fold(cv: Path, k: int) -> dict:
    s = json.loads((cv / f"fold{k}" / "score.json").read_text())["splits"]["test"]
    return {n: {"group": r["group"], "E_ref": r["E_ref"], "E_model": r["E_model"]} for n, r in s["structures"].items()}, s["forces"]


def relative(rows: dict) -> dict:
    out = {}
    for g in sorted({r["group"] for r in rows.values()}):
        names = [n for n, r in rows.items() if r["group"] == g]
        if len(names) < 2:
            continue
        d = np.array([rows[n]["E_model"] - rows[n]["E_ref"] for n in names])
        out[g] = dict(zip(names, (d - d.mean()).tolist()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package"), ap.add_argument("--mace-cv", required=True), ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    pkg, cv = Path(a.package), Path(a.mace_cv)
    folds = sorted(int(p.name[4:]) for p in (pkg / "data").iterdir() if p.name.startswith("fold"))
    rep = {"schema": "dpa4_vs_mace/1", "package": str(pkg), "mace_cv": str(cv), "folds": {}, "missing": []}
    pool = {"dpa4": {}, "mace": {}}
    near_sq = {"dpa4": [0.0, 0], "mace": [0.0, 0]}
    all_sq = {"dpa4": [0.0, 0], "mace": [0.0, 0]}
    for k in folds:
        try:
            d = dpa4_fold(pkg, k)
        except FileNotFoundError as ex:
            rep["missing"].append(str(ex))
            continue
        m, mf = mace_fold(cv, k)
        if set(d) != set(m):
            raise ValueError(f"fold {k}: test frames differ: only DPA-4 {sorted(set(d) - set(m))}, only MACE {sorted(set(m) - set(d))}")
        rd, rm = relative(d), relative(m)
        dn = np.concatenate([r["d_near"] for r in d.values()])
        da = np.concatenate([r["d_all"] for r in d.values()])
        f = {"dpa4": {"rmse_near_eV_A": float(np.sqrt((dn ** 2).sum(1).mean() / 3)) if len(dn) else None,
                      "rmse_all_eV_A": float(np.sqrt((da ** 2).mean())), "n_near": int(len(dn))},
             "mace": {"rmse_near_eV_A": mf["rmse_near_eV_A"], "rmse_all_eV_A": mf["rmse_all_eV_A"], "n_near": mf["n_near"]}}
        near_sq["dpa4"][0] += float((dn ** 2).sum()); near_sq["dpa4"][1] += dn.size
        all_sq["dpa4"][0] += float((da ** 2).sum()); all_sq["dpa4"][1] += da.size
        near_sq["mace"][0] += mf["rmse_near_eV_A"] ** 2 * mf["n_near"] * 3; near_sq["mace"][1] += mf["n_near"] * 3
        rec = {"test_frames": sorted(d), "forces": f, "relative": {}}
        for model, rr in (("dpa4", rd), ("mace", rm)):
            for g, vals in rr.items():
                v = np.abs(np.array(list(vals.values())))
                rec["relative"].setdefault(g, {})[model] = {"n": len(v), "mae_eV": float(v.mean()), "max_eV": float(v.max())}
                pool[model].setdefault(g, []).extend(v.tolist())
        rep["folds"][k] = rec
        print(f"fold {k}:", {g: {mm: (round(x['mae_eV'], 3), round(x['max_eV'], 3)) for mm, x in v.items()} for g, v in rec["relative"].items()},
              "F near", {mm: round(x["rmse_near_eV_A"], 4) if x["rmse_near_eV_A"] is not None else None for mm, x in f.items()})
    rep["pooled"] = {model: {"relative": {g: {"n": len(v), "mae_eV": float(np.mean(v)), "max_eV": float(np.max(v))} for g, v in pool[model].items()},
                             "force_rmse_near_eV_A": float(np.sqrt(near_sq[model][0] / near_sq[model][1])) if near_sq[model][1] else None}
                     for model in ("dpa4", "mace")}
    rep["pooled"]["dpa4"]["force_rmse_all_eV_A"] = float(np.sqrt(all_sq["dpa4"][0] / all_sq["dpa4"][1])) if all_sq["dpa4"][1] else None
    rep["note"] = ("relative energies are centred within each fold, so they are not the cross-fold 'out_of_fold' numbers of "
                   "cv_audit_finetune.py; both models are scored here the same way")
    print(json.dumps(rep["pooled"], indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
