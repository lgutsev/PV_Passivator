"""Stage the LONI GPU package for the chirality D3 speed-up check (owner, 2026-11-21: "Speed up D3 first").

    python examples/scripts/make_chirality_d3_package.py --pkg140 <140 export> --pkg146 <146 export> \
        --params data/cache/d3/dftd3_parameters.pt --model-local D:/MLIP_Custom_Models/MACE/mba2pbi4_chirality_ft2.model \
        --model-cluster /work/lgutsev/loni_smoke_tests/133_mace_chiral_ft2_mba2pbi4/results/mba2pbi4_chirality_ft2_stagetwo.model \
        --out <LONI_staging>/chiral_d3speed_mace_mba2pbi4

LONI 146 showed the torch-dftd socket client is ~85 % of every MD step. This package tests the in-process GPU D3 of
the ALCHEMI toolkit (nvalchemiops Warp kernels, env /project/lgutsev/env/alchemi_check, as package 129): accuracy
against torch-dftd on the 146 bench frames, MACE parity between the two envs, MD s/step, and a 2 ps NVE drift test.
Runner: examples/chirality/production/d3_bench.py (+ warp_d3.py). Nothing is submitted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
from make_chirality_pilot_package import frames, lf_extxyz  # noqa: E402
from make_chirality_stability_package import D3_ENV, lf, sha256  # noqa: E402

PROD = ROOT / "examples" / "chirality" / "production"
BOHR = 0.529177210544
UNIT = {"systems": ["R", "rac"], "n_ref_frames": 10, "time_repeats": 10,
        "protocol": {"time_repeats": 20, "md_steps": 200, "md_warm": 20, "dt_fs": 0.5},
        # 95 bohr sharp = the torch-dftd default used for the labels; 40 bohr sharp ~ the 137/140/146 cutoff
        "variants": [{"cutoff_A": round(95 * BOHR, 2), "smoothing": 0.0}, {"cutoff_A": round(40 * BOHR, 2), "smoothing": 0.0},
                     {"cutoff_A": 25.0, "smoothing": 0.2}, {"cutoff_A": 20.0, "smoothing": 0.2},
                     {"cutoff_A": 15.0, "smoothing": 0.2}],
        "nve": {"cutoff_A": 20.0, "smoothing": 0.2, "ps": 2.0, "sample_every": 10}}

SLURM = """#!/bin/bash
#SBATCH --job-name=chiral-d3speed
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# {package}: in-process GPU D3 (nvalchemiops Warp) vs the torch-dftd socket, for ft2 MD on MBA2PbI4.
# Three stages in three envs, in order: torch-dftd references (d3_torchdftd), ft2 in mace_env, then the
# alchemi_check env (MACE parity, Warp D3 variants, MD s/step, 2 ps NVE drift). Nothing here submits anything else.
set -eo pipefail
cd "${{SLURM_SUBMIT_DIR:?submit from the package directory}}"
mkdir -p logs outputs
[ -s outputs/DONE ] && {{ echo "outputs/DONE exists: not rerunning"; cat outputs/DONE; exit 0; }}
rm -f outputs/FAILED
sha256sum -c --quiet SHA256SUMS || {{ echo "package files changed (SHA256SUMS)" > outputs/FAILED; exit 2; }}
echo "{model_sha}  {model}" | sha256sum -c --quiet - || {{ echo "model missing or hash differs: {model}" > outputs/FAILED; exit 2; }}
{{ hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv; }} > outputs/info.txt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TORCHDYNAMO_DISABLE=1 XDG_CACHE_HOME=/project/lgutsev/cache
# 1) torch-dftd references, in the D3 env of packages 67/137/140 (user site off, as the 137 client)
[ -x "{d3_env}" ] || {{ echo "no D3 env {d3_env}" > outputs/FAILED; exit 2; }}
PYTHONNOUSERSITE=1 "{d3_env}" -u d3_bench.py ref inputs outputs > outputs/stage_ref.log 2>&1 || {{ echo "stage ref failed" > outputs/FAILED; exit 1; }}
# 2) ft2 float64 in mace_env (the env of 133/137/140/146)
( module purge; source /home/lgutsev/miniforge3/etc/profile.d/conda.sh; conda activate {mace_env}
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:-}}" PYTHONPATH="/project/lgutsev/env/ase329_overlay${{PYTHONPATH:+:$PYTHONPATH}}"
  python -u d3_bench.py maceref inputs outputs ) > outputs/stage_maceref.log 2>&1 || {{ echo "stage maceref failed" > outputs/FAILED; exit 1; }}
# 3) alchemi_check env (shared; takes numpy from ~/.local, so no PYTHONNOUSERSITE: package 129 rev1)
PY={alchemi_env}/bin/python
[ -x "$PY" ] || {{ echo "no env at $PY" > outputs/FAILED; exit 2; }}
"$PY" -u d3_bench.py main inputs outputs > outputs/stage_main.log 2>&1 || true
[ -s outputs/DONE ] || {{ [ -s outputs/FAILED ] || echo "stage main failed" > outputs/FAILED; exit 1; }}
cat outputs/DONE
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("pkg140", "pkg146", "params", "model_local", "out"):
        ap.add_argument("--" + k.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--model-cluster", required=True)
    ap.add_argument("--account", default="loni_perovsk27")
    ap.add_argument("--partition", default="gpu2")
    ap.add_argument("--mace-env", default="/project/lgutsev/env/mace_env")
    ap.add_argument("--alchemi-env", default="/project/lgutsev/env/alchemi_check")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    for p in (a.model_cluster, a.mace_env, a.alchemi_env):
        if not p.startswith("/") or ":" in p:
            raise SystemExit(f"cluster path must be absolute POSIX: {p} (in Git Bash set MSYS_NO_PATHCONV=1)")
    out = a.out
    if out.exists():
        if not a.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    inp = out / "inputs"
    inp.mkdir(parents=True)
    src146 = a.pkg146 / "inputs" / "bench"
    for s in UNIT["systems"]:
        shutil.copyfile(src146 / f"frames_{s}.extxyz", inp / f"frames_{s}.extxyz")
    nve = frames(a.pkg140 / "outputs" / "R_358K" / "traj.extxyz.gz")[-1]
    assert nve.get_momenta().any()
    lf_extxyz(inp / "nve_start.extxyz", [nve])
    shutil.copyfile(a.params, inp / "dftd3_parameters.pt")
    msha = sha256(a.model_local)
    unit = {**UNIT, "model_path": a.model_cluster, "frames_source": "LONI 146 inputs/bench (R from 140, rac from 137 <= step 54000)",
            "nve_start": "LONI 140 R_358K last frame (step 60000, with momenta)",
            "params_source": "nvalchemi extract_dftd3_parameters (Grimme dftd3.tgz, MD5-checked), cached 2026-10-23",
            "params_sha256": sha256(a.params)}
    lf(inp / "unit.json", json.dumps(unit, indent=1) + "\n")
    for f in ("d3_bench.py", "warp_d3.py"):
        lf(out / f, (PROD / f).read_text(encoding="utf-8"))
    lf(out / "run_chiral_d3speed.slurm", SLURM.format(account=a.account, partition=a.partition, package=out.name,
                                                      model_sha=msha, model=a.model_cluster, d3_env=D3_ENV,
                                                      mace_env=a.mace_env, alchemi_env=a.alchemi_env))
    lf(out / ".gitattributes", "inputs/** -text\n")
    files = ["d3_bench.py", "warp_d3.py"] + [p.relative_to(out).as_posix() for p in sorted(inp.rglob("*")) if p.is_file()]
    lf(out / "SHA256SUMS", "".join(f"{sha256(out / f)}  {f}\n" for f in files))
    lf(out / "export.list", "outputs/*\nlogs/*\n")
    lf(out / "logs" / ".keep", "")   # sbatch --output=logs/... needs logs/ at submit time (desk note on LONI 149)
    lf(out / "README.md", f"""# {out.name}: GPU D3 speed-up check for the chirality production MD

LONI 146: the torch-dftd socket D3 is ~85 % of each ft2 MD step. This tests the ALCHEMI toolkit's in-process Warp
D3(BJ, r2SCAN) in /project/lgutsev/env/alchemi_check: accuracy vs torch-dftd (labels: 95 bohr; MD so far: 40 bohr)
on 10 thermal frames each of R and rac, ft2 parity between mace_env and alchemi_check, MD s/step for each cutoff,
and a {UNIT['nve']['ps']} ps NVE drift test with {UNIT['nve']['cutoff_A']} A / smoothing {UNIT['nve']['smoothing']}.
One GPU task, three stages (see the slurm script). Model `{a.model_cluster}` (sha256 checked). The desk gives the
submit command. Done when outputs/DONE exists; outputs/FAILED names the stage.
""")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        commit = None
    meta = {"package": out.name, "status": "staged, not submitted", "generator": "examples/scripts/make_chirality_d3_package.py",
            "generator_commit": commit, "model_cluster": a.model_cluster, "model_sha256": msha, "unit": unit,
            "files_sha256": {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    lf(out / "package.json", json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"package": str(out), "files": len(meta["files_sha256"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
