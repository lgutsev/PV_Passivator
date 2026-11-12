"""Leave-arrangements-out cross-validation of the audit-frame fine-tune.

    python examples/scripts/cv_audit_finetune.py DATASET --config FINETUNE.yaml --folds FOLDS.json --out DIR [--execute]

DATASET is one export of export_audit_frames.py made with --subtract-d3 and --align-to (every frame carries
REF_energy_unaligned and the foundation model's energy).  FOLDS.json is a list of lists of FAMILY names; each inner
list is one fold's test set.  For fold k the validation set is the first two complex families of fold k+1 that are
not references, and everything else trains.  References (bare slab, isolated molecule) always train.  The
composition offset is refitted on each fold's own training frames, so no test energy enters its own target.

Without --execute only the fold folders (data + config) are written.  With it each fold is trained
(python -m mace.cli.run_train) and scored with score_audit_model.py; folds that already have a score are skipped.
The summary pools the per-fold TEST results: every arrangement is tested exactly once, by a model that never saw it
or anything descended from the same start.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).parent))
import score_audit_model as S  # noqa: E402


def load(dataset: Path):
    frames = []
    for s in ("train", "valid", "test"):
        f = dataset / f"{s}.extxyz"
        if f.exists() and f.stat().st_size:
            frames += read(f, index=":")
    return frames


def make_fold(frames, test_f, valid_f, d: Path, cfg_text: str, name: str):
    part = {"train": [], "valid": [], "test": []}
    for a in frames:
        a = a.copy()
        fam, ref = a.info["family"], a.info["config_type"] != "complex"
        split = "train" if ref else "test" if fam in test_f else "valid" if fam in valid_f else "train"
        a.info["split"] = split
        part[split].append(a)
    zs = sorted({int(z) for a in frames for z in a.numbers})
    A = np.array([[int((a.numbers == z).sum()) for z in zs] for a in part["train"]], float)
    y = np.array([a.info["REF_energy_unaligned"] - a.info["align_model_energy"] for a in part["train"]])
    shift, *_ = np.linalg.lstsq(A, y, rcond=None)
    for ats in part.values():
        for a in ats:
            off = float(sum(s * (a.numbers == z).sum() for s, z in zip(shift, zs)))
            a.info["align_offset"] = off
            a.info["REF_energy"] = a.info["REF_energy_unaligned"] - off
    (d / "data").mkdir(parents=True, exist_ok=True)
    for s, ats in part.items():
        write(d / "data" / f"{s}.extxyz", ats, format="extxyz")
    lines = []
    for line in cfg_text.splitlines():
        key = line.split(":")[0].strip()
        if key == "name":
            line = f"name: {name}"
        elif key in ("train_file", "valid_file", "test_file"):
            line = f"{key}: data/{key.split('_')[0]}.extxyz"
        lines.append(line)
    (d / "finetune.yaml").write_text("\n".join(lines) + "\n")
    return {s: sorted(a.info["job"] for a in ats) for s, ats in part.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    frames = load(Path(a.dataset))
    folds = json.loads(Path(a.folds).read_text())
    fams = {f.info["family"] for f in frames if f.info["config_type"] == "complex"}
    flat = [x for f in folds for x in f]
    if set(flat) != fams or len(flat) != len(set(flat)):
        raise SystemExit(f"folds must cover every complex family exactly once; missing {sorted(fams - set(flat))}, "
                         f"unknown {sorted(set(flat) - fams)}")
    out = Path(a.out)
    cfg = Path(a.config).read_text()
    summary = {"schema": "audit_cv/1", "dataset": str(a.dataset), "folds": []}
    for k, test_f in enumerate(folds):
        valid_f = folds[(k + 1) % len(folds)][:2]
        d = out / f"fold{k}"
        name = f"cv_fold{k}"
        splits = make_fold(frames, set(test_f), set(valid_f), d, cfg, name)
        rec = {"fold": k, "test_families": test_f, "valid_families": valid_f, "splits": splits}
        score_f = d / "score.json"
        if a.execute and not score_f.exists():
            with open(d / "train.log", "w") as log:
                r = subprocess.run([sys.executable, "-m", "mace.cli.run_train", "--config", "finetune.yaml"], cwd=d,
                                   stdout=log, stderr=subprocess.STDOUT)
            if r.returncode:
                rec["failed"] = f"training exit {r.returncode}"
                summary["folds"].append(rec)
                continue
            rep = S.score(d / "data", d / "run" / f"{name}.model", a.device)
            score_f.write_text(json.dumps(rep, indent=1) + "\n")
            for ck in (d / "run" / "checkpoints").glob("*.pt"):
                ck.unlink()                                   # tens of MB each; the .model file is kept
        if score_f.exists():
            rec["test"] = json.loads(score_f.read_text())["splits"]["test"]
        summary["folds"].append(rec)
        print("fold", k, "test", splits["test"], "->",
              {g: (round(v["relative_mae_eV"], 3), round(v["relative_max_eV"], 3)) for g, v in rec.get("test", {}).get("groups", {}).items()},
              rec.get("test", {}).get("forces"), flush=True)
    done = [f for f in summary["folds"] if "test" in f]
    if done:
        rows = {n: {**r, "fold": f["fold"]} for f in done for n, r in f["test"]["structures"].items()}
        near = np.array([r["force_rmse_near"] for r in rows.values() if r["force_rmse_near"] is not None])
        natn = None
        summary["pooled_test"] = {
            "n_structures": len(rows),
            "energy_abs_mae_eV": float(np.mean([abs(r["dE_abs_eV"]) for r in rows.values()])),
            "energy_abs_max_eV": float(np.max([abs(r["dE_abs_eV"]) for r in rows.values()])),
            "force_rmse_near_per_structure_mean": float(near.mean()), "force_rmse_near_per_structure_max": float(near.max()),
            "per_fold_relative": [{"fold": f["fold"], **{g: {"n": len(v["names"]), "mae": v["relative_mae_eV"], "max": v["relative_max_eV"]}
                                                         for g, v in f["test"]["groups"].items()}} for f in done],
            "per_fold_forces": [{"fold": f["fold"], **f["test"]["forces"]} for f in done],
            "structures": rows,
        }
        # out-of-fold energies on the unaligned scale: E_model + that fold's composition offset, against DFT - D3
        by = {}
        for f in done:
            off = {x.info["job"]: (x.info["align_offset"], x.info["REF_energy_unaligned"], x.info["group"])
                   for x in read(out / f"fold{f['fold']}" / "data" / "test.extxyz", index=":")}
            for n, r in f["test"]["structures"].items():
                by.setdefault(off[n][2], []).append((n, off[n][1], r["E_model"] + off[n][0]))
        oof = {}
        for g, lst in by.items():
            if len(lst) < 3:
                continue
            names = [x[0] for x in lst]
            er, em = np.array([x[1] for x in lst]), np.array([x[2] for x in lst])
            dlt = (er - er.mean()) - (em - em.mean())
            rk = lambda x: np.argsort(np.argsort(x))
            oof[g] = {"names": names, "n": len(names), "ref_span_eV": float(np.ptp(er)), "relative_mae_eV": float(np.abs(dlt).mean()),
                      "relative_max_eV": float(np.abs(dlt).max()), "spearman": float(np.corrcoef(rk(er), rk(em))[0, 1]),
                      "ref_rel_eV": (er - er.min()).round(3).tolist(), "model_rel_eV": (em - em[er.argmin()]).round(3).tolist(),
                      "note": "each structure predicted by the fold model that never saw it; folds differ, so this mixes models"}
        summary["out_of_fold"] = oof
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
