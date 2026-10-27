"""Export relaxation frames from a label-set package as a PRE-RELAXER training set (not labels).

    python examples/scripts/export_prerelax_frames.py PACKAGE --out DIR --test-jobs complex_TFPAI complex_THCl \
        [--valid-tail 0.15] [--subtract-d3 bj r2scan --device cuda]

Unlike pvpassivator.training_data.export_dataset (label grade: DONE + complete XML only), this reads every *complete*
ionic step from OUTCAR / OUTCAR.segN of slab-type relaxes that may still be running. A frame is kept only if VASP printed
"aborting loop because EDIFF is reached" for it and its force block and free energy are complete. The result is marked
label_grade = false and must never be merged into a label dataset: its only use is to train a tool that brings
geometries closer to the r2SCAN minimum before DFT finishes them.

Splits: every frame of a --test-jobs job is test; the last --valid-tail fraction (at least 5 frames) of each other job's
trajectory is valid; the rest is train. Exact duplicate geometries (restart boundaries) are dropped.

--subtract-d3: VASP's IVDW=12 D3(BJ) is part of the DFT energy. The foundation model (MatPES r2SCAN) was trained without
dispersion, so the fitted target is DFT minus D3 (REF_energy, REF_forces); add the same D3 back at inference. torch-dftd
parameters must equal the INCAR's VDW_S8/A1/A2 (checked). Without the flag REF_* is the full DFT target.
Self-contained (numpy + ase; torch-dftd only with --subtract-d3) so the same file can run on the cluster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

SLAB_ROLES = ("bare_slab", "complex")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def incar(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].split("!")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip().upper()] = v.strip()
    return out


def outcar_frames(outcar: Path, natoms: int):
    """Complete ionic steps of one OUTCAR: list of dicts (positions, forces, free_energy, scf_ok).

    A step is complete when its POSITION/TOTAL-FORCE block (natoms rows) and the following 'free  energy   TOTEN'
    line were both written; a step cut off by a walltime kill or a live save is dropped. scf_ok is True only if
    'aborting loop because EDIFF is reached' appeared within that step.
    """
    frames, pos, frc, block, scf_ok, pending = [], [], [], False, False, None
    with open(outcar, errors="replace") as fh:
        for line in fh:
            if "aborting loop because EDIFF is reached" in line:
                scf_ok = True
            elif "aborting loop EDIFF was not reached" in line:
                scf_ok = False
            elif "POSITION" in line and "TOTAL-FORCE" in line:
                block, pos, frc = True, [], []
                next(fh)
            elif block:
                if line.startswith(" ---"):
                    block = False
                    pending = (np.array(pos), np.array(frc), scf_ok) if len(pos) == natoms else None
                else:
                    v = line.split()
                    pos.append([float(x) for x in v[:3]])
                    frc.append([float(x) for x in v[3:6]])
            elif "free  energy   TOTEN" in line and pending is not None:
                p, f, ok = pending
                frames.append({"positions": p, "forces": f, "free_energy": float(line.split("=")[1].split()[0]),
                               "scf_ok": ok})
                pending, scf_ok = None, False
    return frames


def geometry_key(at: Atoms) -> str:
    rows = np.column_stack((at.numbers, np.round(at.get_scaled_positions(wrap=True), 7)))
    rows = rows[np.lexsort(tuple(rows[:, i] for i in reversed(range(4))))]
    return hashlib.sha256(np.round(at.cell.array, 7).tobytes() + rows.tobytes()).hexdigest()


def export(package, out, test_jobs, valid_tail=0.15, d3=None, fmax_sane=20.0):
    root, out = Path(package).resolve(), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be new or empty")
    manifest = json.loads((root / "manifest.json").read_text())
    common = np.array(manifest["common_cell_A"])
    jobs = sorted(j for j, rec in manifest["jobs"].items() if rec.get("role") in SLAB_ROLES)
    unknown = set(test_jobs) - set(jobs)
    if unknown:
        raise ValueError(f"unknown test jobs: {sorted(unknown)}")
    split_of, rejected, per_job, seen, trajs = {}, [], {}, {}, {}
    partition = {"train": [], "valid": [], "test": []}
    hamiltonian = None
    for job in sorted(jobs, key=lambda j: (j not in test_jobs, j)):     # test jobs first: they keep shared geometries
        d = root / "outputs" / job / "relax"
        segs = sorted(d.glob("OUTCAR.seg*"), key=lambda p: int(p.name.removeprefix("OUTCAR.seg"))) + [d / "OUTCAR"]
        segs = [s for s in segs if s.exists()]
        if not segs:
            rejected.append({"job": job, "reason": "no OUTCAR exported"})
            continue
        ref = read(root / "inputs" / job / "relax" / "POSCAR", format="vasp")
        inc = incar(root / "inputs" / job / "relax" / "INCAR")
        h = {k: inc.get(k) for k in ("METAGGA", "IVDW", "VDW_S8", "VDW_A1", "VDW_A2", "ENCUT", "PREC", "ISMEAR",
                                     "SIGMA", "LASPH", "ISPIN", "LDIPOL", "IDIPOL")}
        if hamiltonian is None:
            hamiltonian = h
        elif h != hamiltonian:
            raise ValueError(f"{job}: Hamiltonian differs from the other jobs: {h} vs {hamiltonian}")
        if not np.allclose(ref.cell.array, common, atol=1e-5):
            raise ValueError(f"{job}: cell differs from the manifest common cell")
        traj = []
        for seg in segs:
            for k, fr in enumerate(outcar_frames(seg, len(ref))):
                tag = {"job": job, "source": seg.name, "frame": k}
                if not fr["scf_ok"]:
                    rejected.append({**tag, "reason": "SCF did not reach EDIFF"})
                    continue
                fn = np.linalg.norm(fr["forces"], axis=1)
                if not (np.isfinite(fr["forces"]).all() and np.isfinite(fr["free_energy"])) or fn.max() > fmax_sane:
                    rejected.append({**tag, "reason": f"nonfinite or |F| > {fmax_sane} eV/A"})
                    continue
                at = Atoms(ref.get_chemical_symbols(), positions=fr["positions"], cell=ref.cell, pbc=True)
                at.info.update(DFT_energy=fr["free_energy"], job=job, source=seg.name, source_sha256=None, frame=k,
                               energy_kind="vasp_free_energy", label_grade=False, config_type=manifest["jobs"][job]["role"])
                at.set_array("DFT_forces", fr["forces"])
                key = geometry_key(at)
                if key in seen:       # restart boundary, or a geometry another job already has
                    rejected.append({**tag, "reason": f"duplicate geometry of {seen[key]}"})
                    continue
                seen[key] = f"{job}/{seg.name}/{k}"
                at.info["source_sha256"] = sha256(seg)[:16]
                traj.append(at)
        trajs[job] = traj
    for job, traj in trajs.items():
        n = len(traj)
        n_valid = 0 if job in test_jobs else min(n, max(5, int(round(valid_tail * n))))
        for i, at in enumerate(traj):
            split = "test" if job in test_jobs else ("valid" if i >= n - n_valid else "train")
            at.info["split"] = split
            partition[split].append(at)
        split_of[job] = "test" if job in test_jobs else "train+valid(tail)"
        per_job[job] = {"segments": [s.name for s in segs], "frames_kept": sum(1 for s in partition.values() for a in s
                                                                               if a.info["job"] == job)}
    if d3 is not None:
        damping, xc, device = d3
        _subtract_d3(partition, damping, xc, device, hamiltonian)
    else:
        for ats in partition.values():
            for at in ats:
                at.info["REF_energy"] = at.info["DFT_energy"]
                at.set_array("REF_forces", at.arrays["DFT_forces"].copy())
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, ats in partition.items():
        path = out / f"{split}.extxyz"
        write(path, ats, format="extxyz") if ats else path.write_text("")
        files[split] = {"file": path.name, "sha256": sha256(path), "frames": len(ats),
                        "atoms": int(sum(len(a) for a in ats))}
    report = {"schema": "pvpass_prerelax/1", "label_grade": False,
              "purpose": "train a pre-relaxer only; never merge into label datasets",
              "package": str(root), "manifest_sha256": sha256(root / "manifest.json"), "hamiltonian": hamiltonian,
              "target": ("DFT free energy minus D3(%s,%s) [torch-dftd]" % d3[:2]) if d3 else "DFT free energy (incl. D3)",
              "add_back_at_inference": f"D3({d3[0]},{d3[1]})" if d3 else None,
              "test_jobs": list(test_jobs), "valid_tail": valid_tail, "splits_by_job": split_of, "jobs": per_job,
              "files": files, "rejected": rejected}
    (out / "dataset.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def _subtract_d3(partition, damping, xc, device, hamiltonian):
    import torch
    from torch_dftd.dftd3_xc_params import get_dftd3_default_params
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    p = get_dftd3_default_params(damping, xc)
    want = {"rs6": float(hamiltonian["VDW_A1"]), "s18": float(hamiltonian["VDW_S8"]), "rs18": float(hamiltonian["VDW_A2"])}
    if any(abs(p[k] - v) > 1e-4 for k, v in want.items()):
        raise ValueError(f"torch-dftd {damping}/{xc} parameters {p} differ from the INCAR {want}")
    calc = TorchDFTD3Calculator(damping=damping, xc=xc, device=device, dtype=torch.float64)
    for ats in partition.values():
        for at in ats:
            a = at.copy()
            a.calc = calc
            e, f = a.get_potential_energy(), a.get_forces()
            at.info["D3_energy"] = float(e)
            at.info["REF_energy"] = at.info["DFT_energy"] - float(e)
            at.set_array("REF_forces", at.arrays["DFT_forces"] - f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-jobs", nargs="*", default=[])
    ap.add_argument("--valid-tail", type=float, default=0.15)
    ap.add_argument("--subtract-d3", nargs=2, metavar=("DAMPING", "XC"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    d3 = (*a.subtract_d3, a.device) if a.subtract_d3 else None
    r = export(a.package, a.out, a.test_jobs, a.valid_tail, d3)
    print(json.dumps({"files": r["files"], "rejected": len(r["rejected"]), "target": r["target"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
