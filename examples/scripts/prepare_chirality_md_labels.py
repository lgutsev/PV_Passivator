"""Label tranche B (D8 training pool, D9 held-out test): r2SCAN single points on pre-relaxer MD frames.

    python examples/scripts/prepare_chirality_md_labels.py --frames data/cache/chirality_premd_ft1 \
        --pkg105 <exported package 105 dir> --config data/chirality_d0/config_labels.json --out <LONI_staging>/<package>

Frames come from run_chirality_premd.py (fine-tuned ft1 + D3, Langevin NVT on the 105 rev1 fixed-cell minima). They
are configurations only: no MLIP energy or force is used. Every frame is re-checked here (pvpassivator.chirality
topology with the Pb-I cutoff the run used, unchanged stereocentres, cell identical to the 105 minimum of its system)
and a frame that fails is refused. The profile is identical to tranche A and package 105 rev1 (520 eV,
LREAL=.FALSE., ISYM=0, NSW=0); meshes R 3x3x1, rac 3x3x2 from the same k-spacing.

Names are ``<run>_f<frame>`` with the run name carrying system, temperature, block (d8/d9) and seed, so the training
pool and the held-out set never share a trajectory; the manifest records block, split, T, seed and MD time per point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from pvpassivator import chirality as ch  # noqa: E402

STATIC = {"ENCUT": 520, "IBRION": -1, "NSW": 0, "LREAL": ".FALSE."}


def collect(frames_dir: Path, base: dict) -> tuple[dict, dict, list]:
    """name -> Atoms and name -> metadata for every frame of every run; refused frames listed with the reason."""
    structs, meta, refused = {}, {}, []
    ref = {k: ch.build_topology(v).summary() for k, v in base.items()}
    for f in sorted(frames_dir.glob("*.extxyz")):
        done = frames_dir / (f.stem + ".DONE")
        rec = json.loads(done.read_text()) if done.exists() else {}
        mx = float(rec.get("mx_cut_A", 3.8))
        for i, at in enumerate(read(f, ":")):
            name = f"{f.stem}_f{i:02d}"
            key = at.info["system"]
            m = {"block": "D8" if at.info["split"] == "train_pool" else "D9", "system": key, "split": at.info["split"],
                 "T_K": int(at.info["T_K"]), "seed": int(at.info["seed"]), "time_fs": float(at.info["time_fs"]),
                 "run": at.info["run"], "frame": i, "run_status": rec.get("status", "unknown"), "mx_cut_A": mx,
                 "generator": at.info.get("generator", "")}
            why = None
            if not np.allclose(at.cell.array, base[key].cell.array, atol=1e-6):
                why = "cell differs from the 105 fixed-cell minimum"
            elif at.get_chemical_symbols() != base[key].get_chemical_symbols():
                why = "atom order differs from the 105 minimum"
            else:
                try:
                    s = ch.build_topology(at, mx_cut=mx).summary()
                    if s["stereocentres"] != ref[key]["stereocentres"] or s["n_molecules"] != ref[key]["n_molecules"]:
                        why = f"stereocentres/molecules changed: {s['stereocentres']}"
                except ch.TopologyError as e:
                    why = f"topology: {e}"
            if why:
                refused.append({"frame": name, "reason": why})
                continue
            structs[name], meta[name] = at, m
    return structs, meta, refused


def main(argv=None) -> int:
    from prepare_bulk_refs import kmesh
    from prepare_chirality_d0 import stage
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--pkg105", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    cfg["_sha16"] = hashlib.sha256(a.config.read_bytes()).hexdigest()[:16]
    cfg["title"] = ("MBA2PbI4 chirality label tranche B (D8 training pool, D9 held-out) on ft1 pre-relaxer MD frames "
                    "(r2SCAN+D3(BJ), 520, LREAL=.FALSE., ISYM=0)")
    base, prov = {}, {"base": {}, "profile": STATIC, "frames_dir": str(a.frames)}
    for key in ("R", "rac"):
        p = a.pkg105 / "outputs" / f"{key}_fixcell" / "relax" / "CONTCAR"
        if not (p.parent / "DONE").exists():
            raise SystemExit(f"{p.parent} is not DONE")
        base[key] = read(p, format="vasp")
        prov["base"][key] = {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    structs, meta, refused = collect(a.frames, base)
    if refused:
        raise SystemExit(f"{len(refused)} frames refused, e.g. {refused[:3]}")
    k = {key: kmesh(base[key], cfg["kspacing"]) for key in base}
    jobs = [("labels", n, "static", "static", n, dict(STATIC), k[meta[n]["system"]], "") for n in structs]
    prov.update({"points": meta, "frame_files_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                                         for f in sorted(a.frames.glob("*.extxyz"))}})
    stage(cfg, None, a.out, a.overwrite, structures=structs, jobs=jobs, provenance=prov)
    by = {}
    for n in structs:
        kk = f"{meta[n]['block']}_{meta[n]['system']}"
        by[kk] = by.get(kk, 0) + 1
    print("points:", len(structs), by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
