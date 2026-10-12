"""Stage a PBE test of 2D Coulomb kernel truncation (VASP >= 6.5.0) against the dipole correction on a charged slab.

One fixed slab geometry, static PBE only, written straight into a cluster package (inputs, slurm, README):

    python examples/scripts/prepare_truncation_test.py \
        --slab data/labels_r2scan_v1/package/slab_FAI_Schot/relax/POSCAR \
        --cluster data/labels_r2scan_v1/cluster.json --out <staging dir>/<NN_name>

Jobs: method x charge x cell length along the normal.
  dip    3D periodic + LDIPOL/IDIPOL, neutral only (the label-set setting)
  plain  3D periodic, charged, uniform background, no dipole correction (VASP 6.6.1 refuses LDIPOL on a charged
         non-cubic cell: LONI package 48, job 1085072); the baseline truncation has to beat
  trunc  KERNEL_TRUNCATION LTRUNCATE / IDIMENSIONALITY = 2 / LCOARSEN, no dipole correction
  tdip   both at once, neutral, middle length only (does VASP accept the combination, and does it change E?)
The slab is re-centred in each cell, so only the vacuum changes. Expected (Vijay et al., PRB 112, 045409 (2025)):
charged trunc energies stay flat with vacuum, plain charged energies drift; neutral energies agree between methods
(for a slab with a net dipole that also tests whether truncation removes the image dipole interaction).
Analysis: examples/scripts/analyze_truncation_test.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvpassivator.vaspprep import write_job  # noqa: E402
from stage_label_set import LIB, SLURM, lf  # noqa: E402

PROFILE = "pbe_electrostatics_test/static"
MIN_VACUUM_A = 20.0
STAGES = {"smoke": {"nodes": 2, "time": "01:00:00"},
          "static": {"nodes": 2, "time": "08:00:00", "max_parallel": 6}}


def truncation_tags(normal_axis: int) -> dict:
    # ISURFACE is the Cartesian normal (1 = x); the cell must be orthogonal along it (checked below)
    return {"KERNEL_TRUNCATION/LTRUNCATE": ".TRUE.", "KERNEL_TRUNCATION/IDIMENSIONALITY": 2,
            "KERNEL_TRUNCATION/ISURFACE": normal_axis + 1, "KERNEL_TRUNCATION/LCOARSEN": ".TRUE."}


def recentred(atoms, axis: int, length: float):
    """Same slab in a cell of ``length`` along ``axis``, centred, with the in-plane vectors unchanged."""
    a = atoms.copy()
    x = a.positions[:, axis]
    span = x.max() - x.min()
    if span > a.cell.lengths()[axis] - 10:
        raise ValueError("slab seems to wrap across the cell boundary along the normal; unwrap it first")
    vac = length - span
    if vac < MIN_VACUUM_A:
        raise ValueError(f"cell length {length} A leaves {vac:.1f} A of vacuum (< {MIN_VACUUM_A})")
    cell = a.cell.array.copy()
    cell[axis] = cell[axis] / np.linalg.norm(cell[axis]) * length
    a.set_cell(cell, scale_atoms=False)
    a.positions[:, axis] += length / 2 - (x.max() + x.min()) / 2
    return a, span, vac


def qtag(q: int) -> str:
    return "q0" if q == 0 else (f"qp{q}" if q > 0 else f"qm{-q}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab", required=True, help="slab POSCAR (geometry is kept fixed)")
    ap.add_argument("--normal-axis", type=int, default=0)
    ap.add_argument("--lengths", type=float, nargs="+", default=[55.0, 70.0, 90.0])
    ap.add_argument("--charges", type=int, nargs="+", default=[0, 1, -1])
    ap.add_argument("--cluster", required=True, help="cluster JSON (account, partition, module, potcar_dir)")
    ap.add_argument("--out", required=True, help="staging folder, named <number>_<name>")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    ax = a.normal_axis
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise SystemExit(f"{out} is not empty")
    if out.exists():
        shutil.rmtree(out)
    cl = json.loads(Path(a.cluster).read_text())
    slab = read(a.slab, format="vasp")
    c = slab.cell.array
    others = [k for k in range(3) if k != ax]
    if any(abs(c[ax, k]) > 1e-6 or abs(c[k, ax]) > 1e-6 for k in others):
        raise ValueError("the normal must be a Cartesian axis orthogonal to the in-plane vectors (ISURFACE)")
    lengths = sorted(a.lengths)
    mid = lengths[len(lengths) // 2]

    plan = [("dip" if q == 0 else "plain", q, L) for q in a.charges for L in lengths] + \
           [("trunc", q, L) for q in a.charges for L in lengths] + [("tdip", 0, mid)]
    jobs = {}
    for method, q, L in plan:
        name = f"{method}_{qtag(q)}_L{L:g}"
        at, span, vac = recentred(slab, ax, L)
        tags = truncation_tags(ax) if method in ("trunc", "tdip") else {}
        d = out / "inputs" / name / "static"
        # an odd charge leaves one unpaired electron spread over the slab: pin the doublet (NUPDOWN = 1) rather than
        # start ISPIN = 2 from VASP's default 1 muB on every atom
        spec = write_job(at, d, PROFILE, "bare_slab", cell_charge=q, multiplicity=2 if q % 2 else None,
                         kpoints=[1, 1, 1], job_id=name,
                         normal_axis=ax if method in ("dip", "tdip") else None, overrides={"NCORE": 16, "KPAR": 1, **tags})
        (d / "POTCAR.spec").rename(d / "POTCAR.names")
        lf(d / "EXE", "vasp_gam\n")
        jobs[name] = {"method": method, "charge": q, "length_A": L, "vacuum_A": round(vac, 2),
                      "slab_thickness_A": round(span, 2), "nelect": spec.nelect,
                      "ispin": spec.incar.get("ISPIN", 1), "inplane_area_A2": float(np.linalg.norm(
                          np.cross(at.cell.array[others[0]], at.cell.array[others[1]])))}

    smoke_src = f"trunc_{qtag(max(a.charges, key=abs))}_L{mid:g}"
    sd = out / "inputs" / "smoke" / "static"
    shutil.copytree(out / "inputs" / smoke_src / "static", sd)
    keep = [ln for ln in (sd / "INCAR").read_text().splitlines() if ln.split("=")[0].strip() not in ("NELM", "LVHAR")]
    lf(sd / "INCAR", "\n".join(keep + ["NELM = 4", "LVHAR = .FALSE."]) + "\n")

    # the electrostatic potential is the point of the test: keep LOCPOT (gzipped) next to the other outputs
    lib = LIB.replace("__MODULE__", cl["module"]).replace("__POTDIR__", cl["potcar_dir"])
    lib = lib.replace('  [ -f "$1/vasprun.xml" ]',
                      '  [ -f "$1/LOCPOT" ] && gzip -c "$1/LOCPOT" > "$2/LOCPOT.gz"\n  [ -f "$1/vasprun.xml" ]')
    lib = lib.replace('rm -f "$work/WAVECAR"', 'rm -f "$work/LOCPOT" "$work/WAVECAR"')
    lf(out / "lib_pvpass.sh", lib)
    names = list(jobs)
    lf(out / "lists" / "static.txt", "".join(f"{n} \n" for n in names))
    pkg = out.name
    row = 'read -r job geo < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" lists/static.txt)\n'
    smoke_body = (
        "run_step smoke static smoke\n"
        "# desk 2026-10-12: DONE stands only if OUTCAR mentions the truncation; unknown/ignored-tag messages are kept\n"
        "# for the desk, which reads both files before run_trunc_static.slurm is submitted\n"
        "o=outputs/smoke/static\n"
        "grep -i -m 20 'trunc' runs/smoke/static/OUTCAR > \"$o/truncation_echo.txt\" || true\n"
        "grep -i -E 'unknown|not recogni|ignor|kernel_trunc' runs/smoke/static/vasp.out runs/smoke/static/OUTCAR "
        "> \"$o/tag_messages.txt\" || true\n"
        "if [ ! -s \"$o/truncation_echo.txt\" ]; then\n"
        "  rm -f \"$o/DONE\"; echo 'OUTCAR does not mention truncation: 6.6.1 did not take the KERNEL_TRUNCATION tags; "
        "report to the owner' >&2; exit 1\n"
        "fi")
    for fname, key, purpose, body, n in (
            ("run_trunc_smoke.slurm", "smoke", f"{smoke_src} with NELM=4: tags accepted, SCF timed", smoke_body, 0),
            ("run_trunc_static.slurm", "static", "all static jobs (fixed geometry)",
             row + 'run_step "$job" static static', len(names))):
        st = STAGES[key]
        array = f"#SBATCH --array=0-{n - 1}%{min(n, st['max_parallel'])}\n" if n else ""
        lf(out / fname, SLURM.format(name=f"{pkg.split('_')[0]}-{key}", account=cl["account"],
                                     partition=cl["partition"], array=array, nodes=st["nodes"],
                                     ntasks=cl["ntasks_per_node"], time=st["time"],
                                     output="%x_%A_%a.out" if n else "%x_%j.out", package=pkg, purpose=purpose,
                                     body=body).replace("stage_label_set.py", "prepare_truncation_test.py"))
    lf(out / "export.list", "outputs/*\nlogs/*\n")
    lf(out / "logs" / ".keep", "")
    manifest = {"schema": "truncation_test/1", "profile": PROFILE, "source_slab": str(a.slab), "normal_axis": ax,
                "lengths_A": lengths, "charges": a.charges, "smoke_from": smoke_src, "jobs": jobs,
                "expected": {
                    "trunc_charged": "E flat with vacuum (spread < 0.02 eV); vacuum potential linear with "
                                     "|slope| = |q| e / (2 eps0 A)",
                    "plain_charged": "E drifts with vacuum (uniform background, no dipole correction)",
                    "neutral": "dip and trunc flat and equal to within ~0.02 eV (slab with a net dipole: tests both the image dipole removal and the method agreement)",
                    "tdip": "if VASP accepts it, E equal to dip/trunc neutral at the same length"}}
    lf(out / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    n_spin = sum(1 for j in jobs.values() if j["ispin"] == 2)
    readme = [f"# {pkg}", "", "PBE test of 2D Coulomb kernel truncation (VASP KERNEL_TRUNCATION, >= 6.5.0) against "
              "the dipole correction for a charged perovskite slab. Fixed geometry, static only; nothing was "
              "submitted.", "", f"Slab: `{a.slab}` ({len(slab)} atoms), re-centred in cells of "
              f"{', '.join(f'{L:g}' for L in lengths)} Ã… along axis {ax}. Charges {a.charges}. "
              f"{len(jobs)} statics, Î“-only `vasp_gam`, {n_spin} spin-polarised doublets (odd electron count, NUPDOWN = 1).", "",
              "## Commands, in order (from this directory)", "",
              f"1. `sbatch run_trunc_smoke.slurm` (1 job, 2 nodes, â‰¤ 1 h; est. 1â€“2 node-h). `{smoke_src}` with NELM=4. "
              "DONE only if two SCF steps ran and OUTCAR mentions the truncation (otherwise exit 1). Read `timing.txt`, "
              "`truncation_echo.txt` and `tag_messages.txt` before the array.",
              f"2. `sbatch run_trunc_static.slurm` ({len(jobs)} jobs, 2 nodes each, â‰¤ 8 h, at most 6 at once; "
              "est. 40â€“100 node-h). Resubmitting skips finished jobs.", "",
              "Outputs per job: OUTCAR, OSZICAR, vasprun.xml.gz, LOCPOT.gz (Hartree + ionic, LVHAR). "
              "Exports: `export.list` (outputs/*, logs/*).", "",
              "Analysis (locally): `python examples/scripts/analyze_truncation_test.py <this package>`."]
    lf(out / "README.md", "\n".join(readme) + "\n")
    print(f"staged {pkg}: {len(jobs)} statics + smoke -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
