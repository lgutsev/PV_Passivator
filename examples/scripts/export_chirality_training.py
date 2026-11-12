"""Export the chirality DFT labels (packages 105 rev1 + 113, later D8/D9) as a MACE fine-tuning dataset.

    D:/MLIP_Work_Folder/envs/maceh/Scripts/python.exe examples/scripts/export_chirality_training.py \
        --relax-pkg <exported 105> --label-pkg <exported 113> [--label-pkg ...] --out data/training/chirality_v1

Reuses the training-pilot readers in ``pvpassivator.training_data`` / ``vasp_io`` (run acceptance, per-frame SCF
checks on the XML, Hamiltonian fingerprints, exact-geometry hashes) and writes the same ``pvpass_training/1`` layout
(train/valid/test.extxyz + dataset.json), so ``prepare_mace_finetune.py`` and ``run_mace_pilot.py`` work unchanged.

Differences from the reaction-bundle exporter, all deliberate:
* Sources are single points and relaxations of two bulk crystals, not reactions; splits come from
  ``examples/chirality/finetune_splits.json`` (whole scans / seeds / trajectories, fixed before labels returned).
* R (3x3x1) and rac (3x3x2) use one k-spacing (0.25 1/A, converged in D0), so k-meshes may differ; every other
  Hamiltonian key must match the reference profile (520 eV, LREAL=.FALSE., r2SCAN+D3(BJ)). Mismatching runs, e.g. the
  free-cell relaxations at ENCUT 676 and the LREAL=Auto D0 checks, are rejected with a reason, not pooled.
* Package 105: fixed-cell relax trajectories (every ``--relax-stride``-th frame plus the last) and the four 520 eV
  statics (fixed- and free-cell minima); all go to train. A static supersedes its own relax endpoint.
* Target: by default D3(BJ, r2SCAN) is subtracted (torch-dftd, default cutoffs; reproduces the VASP Edisp to 0.1 meV
  per cell on the 105 minima), so the model learns the r2SCAN part like its foundation model and D3 is added back at
  inference. ``--keep-d3`` trains on the full DFT energy instead. The DFT values are kept as DFT_energy/DFT_forces.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")   # shared laptop: keep local jobs to a few cores

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pvpassivator.training_data import geometry_key, sha256, xml_frames  # noqa: E402
from pvpassivator.vasp_io import compatible, parse_run  # noqa: E402

SPLITS = ROOT / "examples" / "chirality" / "finetune_splits.json"
RELAX_JOBS = {"R_fixcell": "R", "rac_fixcell": "rac"}
STATIC_JOBS = {"R_fixcell": "R", "rac_fixcell": "rac", "R_fullcell": "R", "rac_fullcell": "rac"}


def assign_split(meta: dict, rules: list[dict]) -> str:
    """First rule whose ``match`` keys all equal the point's metadata (``abs_strain`` derived) wins; default train."""
    m = dict(meta)
    if "inplane_strain" in m:
        m["abs_strain"] = round(abs(m["inplane_strain"]), 4)
    for r in rules:
        if all(k in m and m[k] == v for k, v in r["match"].items()):
            return r["split"]
    return "train"


def d3_terms(at, device="cpu"):
    """(energy, forces) of D3(BJ) with r2SCAN parameters, float64, torch-dftd default cutoffs (as VASP IVDW=12)."""
    import torch
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    a = at.copy()
    a.calc = TorchDFTD3Calculator(damping="bj", xc="r2scan", device=device, dtype=torch.float64)
    return float(a.get_potential_energy()), a.get_forces().copy()


def _sources(relax_pkgs, label_pkgs):
    """(package root, job, step, metadata) for every candidate run."""
    out = []
    for pkg in relax_pkgs:
        for job, system in RELAX_JOBS.items():
            out.append((Path(pkg), job, "relax", {"block": "D1", "system": system, "job": job}))
        for job, system in STATIC_JOBS.items():
            out.append((Path(pkg), job, "static", {"block": "D1", "system": system, "job": job}))
    for pkg in label_pkgs:
        man = json.loads((Path(pkg) / "manifest.json").read_text())
        points = man.get("settle", {}).get("points", {})
        for j in man["jobs"]:
            meta = dict(points.get(j["job"], {}))
            meta.setdefault("block", "unknown")
            meta.setdefault("system", j.get("start", ""))
            meta["job"] = j["job"]
            out.append((Path(pkg), j["job"], j["step"], meta))
    return out


def export(relax_pkgs, label_pkgs, out, splits_file=SPLITS, relax_stride=4, subtract_d3=True, device="cpu",
           reference=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be new or empty")
    rules = json.loads(Path(splits_file).read_text())["rules"]
    accepted, rejected = [], []
    for root, job, step, meta in _sources(relax_pkgs, label_pkgs):
        folder = root / "outputs" / job / step
        key = f"{sha256(root / 'manifest.json')[:16]}/{job}"
        if not (folder / "DONE").exists():
            rejected.append({"job": key, "step": step, "reason": "missing DONE"})
            continue
        run = parse_run(folder)
        bad = run.acceptance(require_relax=(step == "relax"))
        if run.cell_charge is None or abs(run.cell_charge) > 1e-6:
            bad.append("neutral electron count unverified")
        xml = folder / "vasprun.xml.gz"
        if not xml.exists():
            xml = folder / "vasprun.xml"
        if not xml.exists():
            bad.append("XML trajectory missing")
        if bad:
            rejected.append({"job": key, "step": step, "reason": "; ".join(bad)})
            continue
        accepted.append((root, job, step, meta, key, folder, run, xml))
    if not accepted:
        raise ValueError("no accepted runs")
    # reference profile = first accepted label single point, else the first accepted run
    ref = reference or next((a[6] for a in accepted if a[2] == "static"), accepted[0][6])
    frames, runs = [], []
    for root, job, step, meta, key, folder, run, xml in accepted:
        mism = compatible([ref, run], allow_kpoint_difference=True)
        if mism:
            rejected.append({"job": key, "step": step, "reason": "Hamiltonian differs from reference: " + "; ".join(mism)})
            continue
        try:
            symbols = read(root / "inputs" / job / step / "POSCAR", format="vasp").get_chemical_symbols()
            got, skipped = xml_frames(xml, symbols, int(float(run.incar.get("NELM", 60))),
                                      float(run.incar.get("EDIFF", 1e-4)), run.nelect, run.incar)
        except (ValueError, ET.ParseError, OSError) as exc:
            rejected.append({"job": key, "step": step, "reason": str(exc)})
            continue
        rejected += [dict(job=key, step=step, **r) for r in skipped]
        if step == "relax" and got:
            got = got[::relax_stride] + ([got[-1]] if (len(got) - 1) % relax_stride else [])
        split = "train" if meta["block"] == "D1" else assign_split(meta, rules)
        src = sha256(xml)
        for i, at in got:
            at.info.update({k: v for k, v in meta.items() if isinstance(v, (int, float, str))})
            at.info.update(job_id=key, step=step, frame_index=i, source_sha256=src, split=split,
                           config_type=f"{meta['block']}_{meta['system']}", energy_kind="vasp_free_energy")
            frames.append(at)
        runs.append(run)
    partition = {s: [] for s in ("train", "valid", "test")}
    seen, duplicates, frames_kept = {}, [], []
    for at in frames:
        g = geometry_key(at)
        if g in seen:
            prev = seen[g]
            if prev.info["split"] != at.info["split"]:
                # A held-out point identical to a training geometry (the 0-degree point of a scan is the DFT minimum)
                # is no held-out data: keep the train copy, drop the other, record the label agreement.
                if "train" not in (prev.info["split"], at.info["split"]):
                    raise ValueError(f"duplicate geometry crosses valid/test: {prev.info['job_id']} / {at.info['job_id']}")
                keep, drop = (prev, at) if prev.info["split"] == "train" else (at, prev)
                duplicates.append({"kept": f"{keep.info['job_id']}/{keep.info['step']}",
                                   "dropped": f"{drop.info['job_id']}/{drop.info['step']}",
                                   "reason": f"identical to a train geometry; removed from {drop.info['split']}",
                                   "dE_meV": round(1000 * (drop.info["REF_energy"] - keep.info["REF_energy"]), 4),
                                   "max_dF_meV_A": round(1000 * float(abs(drop.arrays["REF_forces"] - keep.arrays["REF_forces"]).max()), 4)})
                if keep is at:
                    frames_kept[frames_kept.index(prev)] = at
                    seen[g] = at
                continue
            if prev.info["job_id"] == at.info["job_id"] and prev.info["step"] == "relax" and at.info["step"] == "static":
                # the final static is the authoritative recalculation of the relax endpoint (as in training_data)
                duplicates.append({"kept": at.info["job_id"] + "/static", "dropped": prev.info["job_id"] + "/relax",
                                   "frame": prev.info["frame_index"]})
                frames_kept[frames_kept.index(prev)] = at
                seen[g] = at
                continue
            duplicates.append({"kept": seen[g].info["job_id"], "dropped": at.info["job_id"], "frame": at.info["frame_index"]})
            continue
        seen[g] = at
        frames_kept.append(at)
    for at in frames_kept:
        e, f = at.info.pop("REF_energy"), at.arrays.pop("REF_forces")
        at.info["DFT_energy"] = e
        at.set_array("DFT_forces", f)
        if subtract_d3:
            ed, fd = d3_terms(at, device)
            at.info["D3_energy"] = ed
            e, f = e - ed, f - fd
        at.info["REF_energy"] = e
        at.set_array("REF_forces", f)
        partition[at.info["split"]].append(at)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, ats in partition.items():
        p = out / f"{split}.extxyz"
        if ats:
            write(p, ats, format="extxyz")
        else:
            p.write_text("")
        blocks = {}
        for a in ats:
            blocks[a.info["config_type"]] = blocks.get(a.info["config_type"], 0) + 1
        files[split] = {"file": p.name, "sha256": sha256(p), "frames": len(ats), "by_config_type": blocks}
    target = ("r2SCAN only: VASP free energy and forces minus D3(BJ, r2SCAN); add D3(BJ, r2SCAN) at inference"
              if subtract_d3 else "full VASP free energy including D3(BJ); do not add D3 at inference")
    report = {"schema": "pvpass_training/1", "energy_kind": "vasp_free_energy", "force_units": "eV/Angstrom",
              "target": target, "d3_subtracted": subtract_d3, "files": files,
              "reference_run": ref.path, "reference_fingerprint": ref.fingerprint(),
              "splits_file": str(splits_file), "splits_sha256": sha256(splits_file), "relax_stride": relax_stride,
              "rejected": rejected, "duplicates": duplicates,
              "ready_for_pilot": bool(partition["train"] and partition["valid"]),
              "limitations": ["No stress labels (fixed-cell data only)",
                              "Two crystals of one composition: per-element reference energies are not identifiable "
                              "from these data; use the foundation model's E0s",
                              "Rigid rotor scans and random Pb/I displacements, not relaxed barriers or symmetry-adapted modes",
                              "Exact-geometry hashes only; near duplicates within a trajectory are thinned by the stride"]}
    (out / "dataset.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relax-pkg", type=Path, action="append", default=[])
    ap.add_argument("--label-pkg", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=SPLITS)
    ap.add_argument("--relax-stride", type=int, default=4)
    ap.add_argument("--keep-d3", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    rep = export(a.relax_pkg, a.label_pkg, a.out, a.splits, a.relax_stride, not a.keep_d3, a.device)
    print(json.dumps({"files": rep["files"], "rejected": len(rep["rejected"]), "duplicates": len(rep["duplicates"]),
                      "ready_for_pilot": rep["ready_for_pilot"]}, indent=1))
    for r in rep["rejected"]:
        print("rejected:", r["job"], r["step"], r["reason"][:160])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
