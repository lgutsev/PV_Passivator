"""Per-frame slab-normal dipole labels from finished LDIPOL/IDIPOL relaxations (no new DFT).

    python examples/scripts/extract_slab_dipoles.py <package dir> --out dipoles.extxyz [--json summary.json]

For each <package>/outputs/<job>/relax the OUTCAR segments (OUTCAR.seg1, OUTCAR.seg2, ..., OUTCAR) are read in order.
Per ionic step the label is the last `dipolmoment` line before that step's `FREE ENERGIE` block, i.e. the dipole of
the converged density. The final static (outputs/<job>/static/OUTCAR) is appended as one more frame when present.

Convention (validated on package 55, PBE, 504-atom slab, 55/70/90 A cells):
  - VASP prints the cell dipole along IDIPOL about the DIPOL centre, in e*A; it is independent of the cell length.
  - The vacuum-level step across the slab is  V(high-x side) - V(low-x side) = p * e/(eps0 A) = p * 180.951 / A  eV,
    which reproduced the LOCPOT plateaus to 0.1 meV (p = -0.0774 e*A, A = 369.4 A^2 -> -0.0380 eV).
  - Only meaningful for neutral cells with LDIPOL and no Coulomb truncation (truncated runs print a different value).
Frames whose SCF did not reach a dipole line are skipped and counted, never filled in.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

E_OVER_EPS0 = 180.951   # e/eps0 in eV*A (e^2/(eps0) / e): dV[eV] = p[e*A] * 180.951 / A[A^2]
DIP_RE = re.compile(r"dipolmoment\s+(\S+)\s+(\S+)\s+(\S+)")


def segments(step_dir: Path) -> list[Path]:
    segs = sorted(step_dir.glob("OUTCAR.seg*"), key=lambda p: int(re.sub(r"\D", "", p.suffix) or 0))
    return segs + ([step_dir / "OUTCAR"] if (step_dir / "OUTCAR").exists() else [])


def dipoles_per_step(outcar: Path) -> list[np.ndarray | None]:
    out, last = [], None
    with open(outcar, errors="replace") as fh:
        for ln in fh:
            m = DIP_RE.search(ln)
            if m:
                last = np.array([float(x) for x in m.groups()])
            elif "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" in ln:
                out.append(last)
                last = None
    return out


def incar_tags(outcar: Path) -> dict:
    tags = {}
    with open(outcar, errors="replace") as fh:
        for i, ln in enumerate(fh):
            for key in ("IDIPOL", "LDIPOL", "NELECT", "ISPIN"):
                m = re.match(rf"\s+{key}\s*=\s*(\S+)", ln)
                if m and key not in tags:
                    tags[key] = m.group(1)
            if i > 4000:
                break
    return tags


def read_step(job: str, step: str, outcars: list[Path]) -> tuple[list, dict]:
    frames, info = [], {"segments": [p.name for p in outcars], "skipped_no_dipole": 0}
    for seg in outcars:
        dips = dipoles_per_step(seg)
        try:
            images = read(seg, index=":", format="vasp-out")
        except Exception as exc:          # truncated segment: keep what was parsed before, report it
            info.setdefault("errors", []).append(f"{seg.name}: {exc}")
            continue
        if len(images) != len(dips):
            info.setdefault("errors", []).append(f"{seg.name}: {len(images)} images vs {len(dips)} energy blocks")
        tags = incar_tags(seg)
        for k, (atoms, d) in enumerate(zip(images, dips)):
            if d is None:
                info["skipped_no_dipole"] += 1
                continue
            axis = int(tags.get("IDIPOL", "0")) - 1
            area = float(np.linalg.norm(np.cross(*[atoms.cell[i] for i in range(3) if i != axis])))
            atoms.info.update(job=job, step=step, source=seg.name, frame=k, idipol=axis + 1,
                              dipole_eA=float(d[axis]), area_A2=area,
                              dV_eV=float(d[axis] * E_OVER_EPS0 / area),
                              energy_eV=float(atoms.get_potential_energy()))
            atoms.calc = None
            frames.append(atoms)
    info["frames"] = len(frames)
    return frames, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--jobs", nargs="*", help="default: every outputs/<job> with a relax step and IDIPOL")
    a = ap.parse_args()

    outputs = a.package / "outputs"
    jobs = a.jobs or sorted(p.name for p in outputs.iterdir() if (p / "relax").is_dir())
    all_frames, summary = [], {"package": str(a.package), "convention": "dV = V(high) - V(low) = p*180.951/A eV",
                               "jobs": {}}
    for job in jobs:
        rec = {}
        for step in ("relax", "static"):
            d = outputs / job / step
            outcars = segments(d) if d.is_dir() else []
            if not outcars or "IDIPOL" not in incar_tags(outcars[-1]):
                continue
            frames, rec[step] = read_step(job, step, outcars)
            all_frames += frames
            if frames:
                rec[step]["dipole_first_last_eA"] = [frames[0].info["dipole_eA"], frames[-1].info["dipole_eA"]]
                rec[step]["dV_first_last_eV"] = [frames[0].info["dV_eV"], frames[-1].info["dV_eV"]]
        if rec:
            summary["jobs"][job] = rec
            print(job, json.dumps({s: {k: v for k, v in r.items() if k != "segments"} for s, r in rec.items()}))
    write(a.out, all_frames, format="extxyz")
    summary["frames"] = len(all_frames)
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1))
    print(f"{len(all_frames)} frames -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
