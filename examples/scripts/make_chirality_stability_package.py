"""Stage the LONI GPU package for the MD-stability check of a fine-tuned chirality MACE model (ft2) + D3(BJ, r2SCAN).

    python examples/scripts/make_chirality_stability_package.py --pkg105 <exported 105> \
        --model-local D:/MLIP_Custom_Models/MACE/mba2pbi4_chirality_ft2.model \
        --model-cluster /work/lgutsev/loni_smoke_tests/133_mace_chiral_ft2_mba2pbi4/results/mba2pbi4_chirality_ft2_stagetwo.model \
        --out <LONI_staging>/chiral_stab1_mace_mba2pbi4

Units: R and rac in the production cell (2x2x1 of the 105 rev1 fixed-cell minima: 752 / 376 atoms) at the top of
the planned temperature range. Each unit runs examples/chirality/stability/run_stability.py: NVT then NVE with the
plan's stop conditions and the NVE drift test. D3 runs in the d3_torchdftd env over ASE's socket, as in packages
97/119/125. pvpassivator/chirality.py is vendored (numpy + ase only) with its sha256. Nothing is submitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from ase.io import read, write

ROOT = Path(__file__).resolve().parents[2]
STAB = ROOT / "examples" / "chirality" / "stability"
D3_ENV = "/project/lgutsev/env/d3_torchdftd/bin/python"
PROTOCOL = {"T_K": 358, "nvt_ps": 20.0, "nve_ps": 10.0, "dt_fs": 0.5, "friction_per_fs": 0.01, "seed": 7,
            "dtype": "float64", "sample_every": 20, "check_every": 200, "hours": 22.0,
            "d3_cutoff_bohr": 40.0, "d3_cnthr_bohr": 30.0, "mx_cut_A": 4.3, "covalent_scale": 1.3,
            "proton_transfer_HI_A": 1.9, "max_drift_meV_atom_ps": 1.0}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


SLURM = """#!/bin/bash
#SBATCH --job-name=chiral-stab
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --array=0-{last}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
# {package}: MD-stability check of {model_name} + D3(BJ, r2SCAN) on MBA2PbI4, one unit per array task.
# The runner stops inside its own {hours} h budget; an INCOMPLETE unit continues from its last frame when resubmitted.
set -eo pipefail
UNITS=({units})
unit=${{UNITS[$SLURM_ARRAY_TASK_ID]}}
cd "${{SLURM_SUBMIT_DIR:?submit from the package directory}}"
module purge
source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
conda activate {env}
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:-}}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TORCHDYNAMO_DISABLE=1 XDG_CACHE_HOME=/project/lgutsev/cache
export PYTHONPATH="/project/lgutsev/env/ase329_overlay${{PYTHONPATH:+:$PYTHONPATH}}"
mkdir -p logs "outputs/$unit" "runs/$unit"
sha256sum -c SHA256SUMS > /dev/null || {{ echo "package files changed (SHA256SUMS)" > "outputs/$unit/FAILED"; exit 2; }}
model=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "inputs/$unit/unit.json")
echo "{model_sha}  $model" | sha256sum -c - > /dev/null || {{ echo "model missing or hash differs: $model" > "outputs/$unit/FAILED"; exit 2; }}
[ -x "{d3_env}" ] || {{ echo "no D3 env {d3_env}: run package 67 setup.sh" > "outputs/$unit/FAILED"; exit 2; }}
{{ echo "unit $unit model $model"; hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv; }} >> "outputs/$unit/info.txt"
port=$((31800 + SLURM_ARRAY_TASK_ID))
python -u run_stability.py "inputs/$unit" "outputs/$unit" "runs/$unit" "$port" "{d3_env}"
cat "outputs/$unit/DONE" "outputs/$unit/INCOMPLETE" 2>/dev/null || true
"""


def build_units(pkg105: Path, protocol: dict, model_cluster: str, temps=(358,)):
    units = {}
    for key in ("R", "rac"):
        p = pkg105 / "outputs" / f"{key}_fixcell" / "relax" / "CONTCAR"
        base = read(p, format="vasp")
        sc = base.repeat((2, 2, 1))
        for T in temps:
            name = f"{key}_{T}K"
            units[name] = (sc, {"name": name, "system": key, "supercell": [2, 2, 1], "natoms": len(sc),
                                "source": str(p), "source_sha256": sha256(p), "model_path": model_cluster,
                                "protocol": {**protocol, "T_K": T}})
    return units


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg105", type=Path, required=True)
    ap.add_argument("--model-local", type=Path, required=True, help="local copy of the model (hashed here)")
    ap.add_argument("--model-cluster", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--account", default="loni_perovsk27")
    ap.add_argument("--partition", default="gpu2")
    ap.add_argument("--env", default="/project/lgutsev/env/mace_env")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    for p in (a.model_cluster, a.env):
        if not p.startswith("/") or ":" in p:
            raise SystemExit(f"cluster path must be absolute POSIX: {p} (in Git Bash set MSYS_NO_PATHCONV=1)")
    out = a.out
    if out.exists():
        if not a.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    msha = sha256(a.model_local)
    units = build_units(a.pkg105, PROTOCOL, a.model_cluster)
    for name, (at, meta) in units.items():
        (out / "inputs" / name).mkdir(parents=True)
        write(out / "inputs" / name / "structure.extxyz", at, format="extxyz")
        p = out / "inputs" / name / "structure.extxyz"
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
        lf(out / "inputs" / name / "unit.json", json.dumps(meta, indent=1) + "\n")
    for f, src in (("run_stability.py", STAB / "run_stability.py"), ("d3_client.py", STAB / "d3_client.py"),
                   ("chirality.py", ROOT / "src" / "pvpassivator" / "chirality.py")):
        lf(out / f, src.read_text(encoding="utf-8"))
    names = list(units)
    lf(out / "run_chiral_stab.slurm", SLURM.format(account=a.account, partition=a.partition, last=len(names) - 1,
                                                   package=out.name, model_name=a.model_local.name,
                                                   hours=PROTOCOL["hours"], units=" ".join(names), env=a.env,
                                                   model_sha=msha, d3_env=D3_ENV))
    lf(out / ".gitattributes", "inputs/** -text\n")
    sums = [f"{sha256(out / f)}  {f}" for f in ("run_stability.py", "d3_client.py", "chirality.py")]
    sums += [f"{sha256(out / 'inputs' / n / f)}  inputs/{n}/{f}" for n in names for f in ("structure.extxyz", "unit.json")]
    lf(out / "SHA256SUMS", "\n".join(sums) + "\n")
    lf(out / "export.list", "outputs/*\nlogs/*\n")
    pr = PROTOCOL
    lf(out / "README.md", f"""# {out.name}: MD-stability check of the chirality fine-tune ({a.model_local.name})

**What it is.** The last gate item before any production MD: does the fine-tuned model ft2 (+ D3(BJ, r2SCAN)) run
stably in the production cell at the top of the planned temperature range? It passed the held-out force, rotor and
R - rac criteria (PV_Passivator docs/CHIRALITY_FEASIBILITY.md). Trajectories here are a stability test, not results.

**Units** (array index: unit): {", ".join(f"{i}: {n} ({units[n][1]['natoms']} atoms)" for i, n in enumerate(names))};
2x2x1 of the package 105 rev1 fixed-cell r2SCAN minima.

**Protocol per unit:** Langevin NVT {pr['nvt_ps']} ps at {pr['T_K']} K (friction {pr['friction_per_fs']}/fs), then NVE
{pr['nve_ps']} ps; dt {pr['dt_fs']} fs; MACE {pr['dtype']}; D3 cutoff {pr['d3_cutoff_bohr']} bohr (cn {pr['d3_cnthr_bohr']}).
Checks every {pr['check_every']} steps ({pr['check_every'] * pr['dt_fs']:.0f} fs) = plan stop conditions: stereocentre
inversion; Pb not six-coordinate (Pb-I to {pr['mx_cut_A']} A); NH3+/molecule identity (organic bonds at
{pr['covalent_scale']} x covalent radii); proton transfer (ammonium H...I < {pr['proton_transfer_HI_A']} A); runaway
force > 50 eV/A. PASS also needs |NVE drift| <= {pr['max_drift_meV_atom_ps']} meV/atom/ps.

**Model on LONI:** `{a.model_cluster}` (sha256 `{msha}`; the stage-two model of package 133). The job checks the
hash and exits 2 if it differs. D3 env `{D3_ENV}` (package 67), MACE env `{a.env}` with the ASE 3.29 overlay, as
packages 97/119/125.

**Run:** `mkdir -p logs && sbatch run_chiral_stab.slurm` (array 0-{len(names) - 1}, 1 x {a.partition} GPU each,
<= 24 h; the runner stops at {pr['hours']} h and writes INCOMPLETE; resubmit that index to continue).
**Done when** every `outputs/<unit>/DONE` exists; it reads `PASS ...`, `FAIL ...` (drift) or `STOP ...` (a stop
condition, with step and reason). `outputs/<unit>/FAILED` = crash or setup problem.
**Export:** outputs/* (thermo.csv, traj.extxyz.gz, result.json, run_info.json, info.txt, d3_client.log, markers), logs/*.
""")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        commit = None
    meta = {"package": out.name, "status": "staged, not submitted",
            "generator": "examples/scripts/make_chirality_stability_package.py", "generator_commit": commit,
            "model_local": str(a.model_local), "model_cluster": a.model_cluster, "model_sha256": msha,
            "protocol": PROTOCOL, "units": {n: m for n, (_, m) in units.items()},
            "files_sha256": {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    lf(out / "package.json", json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"package": str(out), "units": {n: m["natoms"] for n, (_, m) in units.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
