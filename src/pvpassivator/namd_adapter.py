"""Adapter replacing the hard-coded part of NAMD_Template/FolderPrep.py.

Original behaviour (recovered): read XDATCAR_FINAL, take the last NSCF=2000 frames via
negative indexing, write POSCARs into zero-padded folders 0001..NSCF.
This adapter keeps that output layout but makes every choice explicit and validated.
The original script is not modified.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np


class FrameSelectionError(ValueError):
    pass


@dataclass
class NamdExportConfig:
    trajectory: str
    out_dir: str
    n_frames: int = 2000              # NSCF in FolderPrep.py
    stride: int = 1
    start: int | None = None          # explicit first frame (0-based). None = take the last window
    equilibration_cutoff: int = 0     # frames before this are never used
    timestep_fs: float | None = None  # MD POTIM
    save_interval: int = 1            # NBLOCK (steps between saved frames)
    temperature_K: float | None = None
    thermostat: str | None = None
    calculator_profile: str | None = None  # e.g. "VASP PBE 400eV AIMD" or MLIP checkpoint hash
    source_kind: str = "DFT"          # DFT | MLIP
    nelect: float | None = None
    overwrite: bool = False
    max_step_displacement: float = 1.0   # A, continuity check between consecutive saved frames
    meta: dict = field(default_factory=dict)


def select_window(n_avail: int, cfg: NamdExportConfig) -> list[int]:
    usable = n_avail - cfg.equilibration_cutoff
    need = (cfg.n_frames - 1) * cfg.stride + 1
    if cfg.start is None:
        first = n_avail - need
    else:
        first = cfg.start
    if first < cfg.equilibration_cutoff or first + need > n_avail or usable <= 0:
        raise FrameSelectionError(
            f"Requested {cfg.n_frames} frames (stride {cfg.stride}, span {need}) but trajectory has "
            f"{n_avail} frames with equilibration cutoff {cfg.equilibration_cutoff} "
            f"({max(usable,0)} usable). Reduce n_frames/stride or extend the MD; frames are never repeated.")
    return list(range(first, first + need, cfg.stride))


def _check_frames(frames, idx, cfg) -> list[str]:
    ref = frames[0]
    syms = ref.get_chemical_symbols()
    issues = []
    for k, i in enumerate(idx):
        a = frames[i]
        if a.get_chemical_symbols() != syms:
            raise FrameSelectionError(f"frame {i}: atom order/species differ from frame 0")
        if not np.allclose(a.cell.array, ref.cell.array, atol=1e-6):
            issues.append(f"frame {i}: cell differs from frame 0 (variable-cell trajectory)")
    # continuity with minimum image between consecutive *original* frames in window
    for i0, i1 in zip(idx[:-1], idx[1:]):
        a0, a1 = frames[i0], frames[i1]
        d = a1.get_scaled_positions(wrap=False) - a0.get_scaled_positions(wrap=False)
        d -= np.round(d)
        dmax = float(np.linalg.norm(d @ a0.cell.array, axis=1).max())
        if dmax > cfg.max_step_displacement * cfg.stride:
            issues.append(f"frames {i0}->{i1}: max displacement {dmax:.2f} A (discontinuity/duplicate?)")
        if dmax < 1e-8:
            raise FrameSelectionError(f"frames {i0} and {i1} are identical (duplicated frame)")
    return issues


def export_frames(cfg: NamdExportConfig, frames=None) -> dict:
    from ase.io import read, write
    if frames is None:
        frames = read(cfg.trajectory, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    idx = select_window(len(frames), cfg)
    issues = _check_frames(frames, idx, cfg)
    out = Path(cfg.out_dir)
    if out.exists() and any(out.iterdir()) and not cfg.overwrite:
        raise FileExistsError(f"{out} is not empty; set overwrite=True explicitly")
    out.mkdir(parents=True, exist_ok=True)
    width = len(str(cfg.n_frames))
    # One atom map for all frames: ASE's vasp writer groups by species only if sort=True.
    # We sort once and apply the same permutation to every frame.
    syms = frames[idx[0]].get_chemical_symbols()
    species_order = list(dict.fromkeys(syms))
    perm = sorted(range(len(syms)), key=lambda i: (species_order.index(syms[i]), i))
    records = []
    dt = (cfg.timestep_fs or 0) * cfg.save_interval
    for k, i in enumerate(idx):
        d = out / f"{k + 1:0{width}d}"
        d.mkdir(exist_ok=True)
        a = frames[i][perm]
        write(d / "POSCAR", a, format="vasp", direct=True, sort=False)
        records.append({"folder": d.name, "source_frame": i,
                        "time_fs": i * dt if cfg.timestep_fs else None})
    atom_map = {"export_index_to_source_index": perm, "species_order": species_order}
    manifest = {
        "schema_version": "namd_export/1",
        "config": asdict(cfg),
        "n_source_frames": len(frames),
        "frames": records,
        "atom_map": atom_map,
        "cell": frames[idx[0]].cell.array.tolist(),
        "issues": issues,
        "contains_velocities": False,
        "status": {"geometry_ready": True, "nuclear_trajectory_ready": not issues,
                   "electronic_frame_ready": False, "nac_ready": False},
        "trajectory_sha256": _sha(cfg.trajectory),
    }
    (out / "namd_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _sha(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_nac_alignment(n_frames: int, eig_rows: int, nac_rows: int) -> list[str]:
    """NACs couple frames t and t+1: N frames -> N-1 NAC rows. Report, don't guess."""
    p = []
    if eig_rows != n_frames:
        p.append(f"EIGTXT rows {eig_rows} != frames {n_frames}")
    if nac_rows not in (n_frames - 1, n_frames):
        p.append(f"NATXT rows {nac_rows} incompatible with {n_frames} frames")
    elif nac_rows == n_frames:
        p.append("NATXT has N rows for N frames: confirm which frame pair each NAC row belongs to")
    return p
