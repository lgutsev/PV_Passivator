"""Stage the LONI GPU package for the chirality production pilot: HPC throughput benchmark + short NPT pilot (ft2 + D3).

    python examples/scripts/make_chirality_pilot_package.py --pkg105 <105 export> --pkg137 <137 export> \
        --pkg140 <140 export> --model-local D:/MLIP_Custom_Models/MACE/mba2pbi4_chirality_ft2.model \
        --model-cluster /work/lgutsev/loni_smoke_tests/133_mace_chiral_ft2_mba2pbi4/results/mba2pbi4_chirality_ft2_stagetwo.model \
        --out <LONI_staging>/chiral_pilot1_mace_mba2pbi4

Owner decision 2026-11-19: the first production package is a throughput benchmark and NPT pilot (plan:
"benchmark the HPC throughput before any request"). Units (array index):
  0 bench        float32 vs float64 accuracy and s/step with and without D3, on 20 thermal frames each of R (LONI 140)
                 and rac (LONI 137, first submission only);
  1 npt_R_358K   5 ps MTK NPT (P = 0) from the last 140 frame (positions + momenta);
  2 npt_rac_358K 5 ps MTK NPT from the 137 rac frame at step 54 000 (before the duplicated tail).
Runner: examples/chirality/production/bench_npt.py. Nothing is submitted.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ase.io import iread, read, write

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
from make_chirality_stability_package import D3_ENV, SLURM, lf, sha256  # noqa: E402

STAB = ROOT / "examples" / "chirality" / "stability"
PROD = ROOT / "examples" / "chirality" / "production"
COMMON = {"dt_fs": 0.5, "d3_cutoff_bohr": 40.0, "d3_cnthr_bohr": 30.0}
BENCH = {**COMMON, "bench_warmup": 20, "bench_steps": 200, "n_frames": 20}
NPT = {**COMMON, "T_K": 358, "npt_ps": 5.0, "tdamp_steps": 100, "pdamp_steps": 1000, "sample_every": 20,
       "check_every": 200, "dtype": "float64", "mx_cut_A": 4.6, "covalent_scale": 1.3, "proton_transfer_HI_A": 1.9}


def frames(traj_gz: Path, max_step: int | None = None):
    """Saved frames by step, first occurrence only (137 rac has a duplicated tail from a second submission)."""
    seen, out = set(), []
    with gzip.open(traj_gz, "rt") as fh:
        for a in iread(fh, index=":", format="extxyz"):
            s = int(a.info["step"])
            if s in seen or (max_step is not None and s > max_step):
                continue
            seen.add(s)
            out.append(a)
    return out


def lf_extxyz(path: Path, images) -> None:
    write(path, images, format="extxyz")
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def build(a) -> dict:
    out = a.out
    r140 = frames(a.pkg140 / "outputs" / "R_358K" / "traj.extxyz.gz")
    rac = frames(a.pkg137 / "outputs" / "rac_358K" / "traj.extxyz.gz", max_step=54000)
    units = {}
    pick = lambda fr: fr[:: max(1, len(fr) // BENCH["n_frames"])][: BENCH["n_frames"]]  # noqa: E731
    units["bench"] = ({"frames_R.extxyz": pick(r140), "frames_rac.extxyz": pick(rac)},
                      {"name": "bench", "kind": "bench", "systems": ["R", "rac"], "protocol": BENCH,
                       "sources": {"R": "LONI 140 R_358K traj", "rac": "LONI 137 rac_358K traj, steps <= 54000"}})
    for key, start in (("R", r140[-1]), ("rac", rac[-1])):
        full = read(a.pkg105 / "outputs" / f"{key}_fullcell" / "relax" / "CONTCAR", format="vasp").repeat((2, 2, 1))
        assert start.get_momenta().any(), f"{key}: starting frame has no momenta"
        name = f"npt_{key}_358K"
        units[name] = ({"structure.extxyz": [start]},
                       {"name": name, "kind": "npt", "system": key, "natoms": len(start), "protocol": NPT,
                        "start_step": int(start.info["step"]),
                        "source": f"LONI {'140' if key == 'R' else '137'} {key}_358K traj step {start.info['step']}",
                        "dft_freecell_volume_A3": float(full.get_volume()),
                        "dft_freecell_abc_A": [float(x) for x in full.cell.cellpar()[:3]]})
    for name, (files, meta) in units.items():
        d = out / "inputs" / name
        d.mkdir(parents=True)
        for f, imgs in files.items():
            lf_extxyz(d / f, imgs)
        meta["model_path"] = a.model_cluster
        lf(d / "unit.json", json.dumps(meta, indent=1) + "\n")
    return units


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("pkg105", "pkg137", "pkg140", "model_local", "out"):
        ap.add_argument("--" + k.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--model-cluster", required=True)
    ap.add_argument("--account", default="loni_perovsk27")
    ap.add_argument("--partition", default="gpu2")
    ap.add_argument("--env", default="/project/lgutsev/env/mace_env")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    for p in (a.model_cluster, a.env):
        if not p.startswith("/") or ":" in p:
            raise SystemExit(f"cluster path must be absolute POSIX: {p} (in Git Bash set MSYS_NO_PATHCONV=1)")
    if a.out.exists():
        if not a.overwrite:
            raise SystemExit(f"{a.out} exists (use --overwrite)")
        shutil.rmtree(a.out)
    msha = sha256(a.model_local)
    units = build(a)
    names = list(units)
    out = a.out
    vend = (("bench_npt.py", PROD / "bench_npt.py"), ("run_stability.py", STAB / "run_stability.py"),
            ("d3_client.py", STAB / "d3_client.py"), ("chirality.py", ROOT / "src" / "pvpassivator" / "chirality.py"))
    for f, src in vend:
        lf(out / f, src.read_text(encoding="utf-8"))
    slurm = SLURM.format(account=a.account, partition=a.partition, last=len(names) - 1, package=out.name,
                         model_name=a.model_local.name, hours="(short)", units=" ".join(names), env=a.env,
                         model_sha=msha, d3_env=D3_ENV)
    slurm = slurm.replace("#SBATCH --job-name=chiral-stab", "#SBATCH --job-name=chiral-pilot")
    slurm = slurm.replace("#SBATCH --time=24:00:00", "#SBATCH --time=08:00:00")
    slurm = slurm.replace("MD-stability check of", "throughput benchmark + NPT pilot of")
    slurm = slurm.replace("# The runner stops inside its own (short) h budget; an INCOMPLETE unit continues from its last frame when resubmitted.\n", "")
    old_run = 'port=$((31800 + SLURM_ARRAY_TASK_ID))\npython -u run_stability.py "inputs/$unit" "outputs/$unit" "runs/$unit" "$port" "{}"\ncat "outputs/$unit/DONE" "outputs/$unit/INCOMPLETE" 2>/dev/null || true\n'.format(D3_ENV)
    assert old_run in slurm
    slurm = slurm.replace(old_run, 'port=$((31900 + 10 * SLURM_ARRAY_TASK_ID))\npython -u bench_npt.py "inputs/$unit" "outputs/$unit" "$port" "{}"\ncat "outputs/$unit/DONE" 2>/dev/null || true\n'.format(D3_ENV))
    lf(out / "run_chiral_pilot.slurm", slurm)
    lf(out / ".gitattributes", "inputs/** -text\n")
    sums = [f"{sha256(out / f)}  {f}" for f, _ in vend]
    sums += [f"{sha256(p)}  {p.relative_to(out).as_posix()}" for p in sorted((out / "inputs").rglob("*")) if p.is_file()]
    lf(out / "SHA256SUMS", "\n".join(sums) + "\n")
    lf(out / "export.list", "outputs/*\nlogs/*\n")
    lf(out / "README.md", f"""# {out.name}: throughput benchmark + NPT pilot for the chirality production MD

Owner decision 2026-11-19: before any production request, measure the A100 cost per ps and check ft2 + D3 under MTK
NPT. Units (array index): {", ".join(f"{i}: {n}" for i, n in enumerate(names))}.
- bench: float32 vs float64 differences (E, F, stress) on 20 thermal frames each of R (752 atoms) and rac (376);
  s/step of float64 / float32 MACE with and without D3. Output bench.json.
- npt_*: {NPT['npt_ps']} ps MTK NPT at {NPT['T_K']} K, P = 0, tdamp {NPT['tdamp_steps']} / pdamp {NPT['pdamp_steps']}
  steps, {NPT['dtype']}, from a 358 K thermal frame; stability stop conditions every {NPT['check_every']} steps
  (Pb six-coordinate within {NPT['mx_cut_A']} A). Output npt.csv, result.json (cell vs start and vs DFT free cell).
Model: `{a.model_cluster}` (sha256 `{msha}`, checked at job start). The desk gives the submit command.
**Done when** every outputs/<unit>/DONE exists. FAILED = crash or setup problem.
""")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        commit = None
    meta = {"package": out.name, "status": "staged, not submitted", "generator": "examples/scripts/make_chirality_pilot_package.py",
            "generator_commit": commit, "model_local": str(a.model_local), "model_cluster": a.model_cluster,
            "model_sha256": msha, "units": {n: m for n, (_, m) in units.items()},
            "files_sha256": {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    lf(out / "package.json", json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"package": str(out), "units": names}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
