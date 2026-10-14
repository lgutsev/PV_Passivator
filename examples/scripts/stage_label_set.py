"""Stage a prepared label set (prepare_label_set.py output) as a cluster package: inputs, slurm scripts, gate, README.

    python examples/scripts/stage_label_set.py --package data/labels/package --cluster data/labels/cluster.json \
        --out <staging dir>/<NN_name>

Cluster specifics (account, partition, module, POTCAR directory, nodes and walltimes per stage) come from the cluster
JSON, not from this script. Everything written is LF; the slurm scripts are plain bash (no Python on the cluster).

Stages: smoke (one slab static, NELM=4) -> bulk relax -> bulk gate -> relaxes (slab-type, references) -> statics
(slab-type, references) -> k-point checks. A relax that hits the walltime continues from its CONTCAR when resubmitted
and keeps the previous segment's OUTCAR/OSZICAR; DONE is written only on real completion.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def lf(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


LIB = r'''# Shared functions for the run_*.slurm scripts of this package (plain bash).
set -o pipefail

setup_vasp() {   # $1 = vasp_gam | vasp_std ; sets $vasp, refuses a binary from ~/bin, waits for the module wrapper
  set +u                      # module scripts are not written for nounset
  module purge
  module load __MODULE__
  set -u
  export SINGULARITYENV_OMP_NUM_THREADS=1
  exe=$1
  vasp=$(command -v "$exe" || true)
  case "$vasp" in
    "") echo "no $exe after module load __MODULE__" >&2; exit 2 ;;
    "$HOME"/bin/*) echo "$exe resolves to $vasp, not the module; check PATH" >&2; exit 2 ;;
  esac
  for try in 1 2 3 4 5 6 7 8 9 10 11 12; do wrapper_ok && break; echo "$exe wrapper not ready (try $try)" >&2; sleep 15; done
  wrapper_ok || { echo "$exe at $vasp is not a complete script after 3 min" >&2; exit 2; }
}
wrapper_ok() { [ -s "$vasp" ] && [ -x "$vasp" ] && [ "$(head -c 2 "$vasp")" = "#!" ]; }

build_potcar() {  # $1 = input step dir, $2 = work dir, $3 = output dir
  : > "$2/POTCAR"
  for name in $(cat "$1/POTCAR.names"); do
    [ -s "__POTDIR__/$name/POTCAR" ] || { echo "missing POTCAR $name" >&2; exit 2; }
    cat "__POTDIR__/$name/POTCAR" >> "$2/POTCAR"
  done
  grep TITEL "$2/POTCAR" | awk '{print $4}' > "$3/POTCAR.used"
}

save_outputs() {  # $1 = work dir, $2 = output dir
  for f in OUTCAR OSZICAR CONTCAR vasp.out POSCAR INCAR KPOINTS; do
    [ -f "$1/$f" ] && cp "$1/$f" "$2/"
  done
  [ -f "$1/vasprun.xml" ] && gzip -c "$1/vasprun.xml" > "$2/vasprun.xml.gz"
  return 0
}

# run_step JOB STEP KIND [GEOMETRY_FROM]
#   KIND relax  : continue from CONTCAR if a previous segment exists; DONE only on "reached required accuracy"
#   KIND static : POSCAR from outputs/GEOMETRY_FROM/relax/CONTCAR (must be DONE); DONE when EDIFF is reached
#   KIND smoke  : static with NELM=4; DONE when two electronic steps were timed
run_step() {
  local job=$1 step=$2 kind=$3 geo=${4:-}
  local inp="inputs/$job/$step" work="runs/$job/$step" out="outputs/$job/$step"
  [ -f "$out/DONE" ] && { echo "$job/$step already done"; return 0; }
  mkdir -p "$work" "$out"
  setup_vasp "$(cat "$inp/EXE")"
  { echo "$(cat "$inp/EXE"): $vasp"; echo "LOADEDMODULES: ${LOADEDMODULES:-}"; date; hostname; } > "$out/vasp_binary.txt"
  cp "$inp/INCAR" "$inp/KPOINTS" "$work/"
  build_potcar "$inp" "$work" "$out"
  if [ "$kind" = relax ]; then
    if [ -s "$work/CONTCAR" ] && [ "$(wc -l < "$work/CONTCAR")" -gt 8 ]; then
      local seg=1
      while [ -f "$out/OUTCAR.seg$seg" ]; do seg=$((seg + 1)); done
      [ -f "$work/OUTCAR" ] && cp "$work/OUTCAR" "$out/OUTCAR.seg$seg"
      [ -f "$work/OSZICAR" ] && cp "$work/OSZICAR" "$out/OSZICAR.seg$seg"
      # Keep energy/force trajectories before the next segment overwrites them.
      [ -f "$work/vasprun.xml" ] && gzip -c "$work/vasprun.xml" > "$out/vasprun.xml.seg$seg.gz"
      echo "$job/$step: continuing from CONTCAR (previous segment kept as seg$seg)"
      cp "$work/CONTCAR" "$work/POSCAR"
    else
      cp "$inp/POSCAR" "$work/POSCAR"
    fi
  elif [ -n "$geo" ]; then
    [ -f "outputs/$geo/relax/DONE" ] || { echo "$geo/relax is not DONE; run the relaxes first" >&2; exit 3; }
    cp "outputs/$geo/relax/CONTCAR" "$work/POSCAR"
  else
    cp "$inp/POSCAR" "$work/POSCAR"
  fi
  (cd "$work" && srun "$vasp" > vasp.out 2>&1) || true
  save_outputs "$work" "$out"
  case "$kind" in
    relax)
      grep -q "reached required accuracy" "$work/OUTCAR" 2>/dev/null \
        || { echo "$job/$step did not reach EDIFFG; resubmit to continue from CONTCAR" >&2; exit 1; } ;;
    static)
      grep -q "EDIFF is reached" "$work/OUTCAR" 2>/dev/null \
        || { echo "$job/$step: SCF did not converge (NELM hit or crash)" >&2; exit 1; }
      zcat "$out/vasprun.xml.gz" | tail -n 3 | grep -q "</modeling>" || { echo "$job/$step: vasprun incomplete" >&2; exit 1; } ;;
    smoke)
      grep "LOOP:" "$work/OUTCAR" > "$out/timing.txt" || true
      [ "$(wc -l < "$out/timing.txt")" -ge 2 ] || { echo "smoke: fewer than 2 timed SCF steps" >&2; exit 1; } ;;
  esac
  rm -f "$work/WAVECAR" "$work/CHG" "$work/CHGCAR"
  date > "$out/DONE"
  echo "$job/$step done"
}
'''

SLURM = '''#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={account}
#SBATCH --partition={partition}
{array}#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --time={time}
#SBATCH --output=logs/{output}
# {package}: {purpose}
# Generated by PV_Passivator examples/scripts/stage_label_set.py; nothing was submitted by the generator.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./lib_pvpass.sh
set -u
{body}
'''

GATE = r'''#!/bin/bash
# Bulk gate: compare the relaxed bulk lattice with the lattice the set was built on (plain bash + awk).
# Exit 0 = within __TOL__ (continue); exit 1 = regenerate the set on the new lattice before any other stage.
set -eo pipefail
cd "$(dirname "$0")"
start=inputs/bulk/relax/POSCAR
final=outputs/bulk/relax/CONTCAR
[ -f outputs/bulk/relax/DONE ] || { echo "bulk/relax is not DONE" >&2; exit 2; }
lengths() { awk 'NR==2{s=$1} NR>=3&&NR<=5{printf "%.6f\n", s*sqrt($1*$1+$2*$2+$3*$3)}' "$1"; }
paste <(lengths "$start") <(lengths "$final") | awk -v tol=__TOL__ '
  { d = ($2 - $1) / $1; if (d < 0) d = -d; printf "axis %d: %.4f -> %.4f A (%.3f %%)\n", NR, $1, $2, 100 * d; if (d > m) m = d }
  END { if (m > tol) { printf "STOP: lattice moved by %.3f %% (> %.2f %%); regenerate the set on the new lattice\n", 100 * m, 100 * tol; exit 1 }
        printf "PASS: max change %.3f %%\n", 100 * m }'
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="prepare_label_set.py output folder")
    ap.add_argument("--cluster", required=True, help="cluster JSON (account, partition, module, potcar_dir, stages)")
    ap.add_argument("--out", required=True, help="staging folder, named <number>_<name>")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    src, out = Path(a.package), Path(a.out)
    cl = json.loads(Path(a.cluster).read_text())
    m = json.loads((src / "manifest.json").read_text())
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise SystemExit(f"{out} is not empty")
    pkg = out.name
    stages = cl["stages"]

    # inputs/<job>/<step>/ with EXE and POTCAR.names
    def stage_inputs(job: str, step_src: Path, step: str, exe: str, incar_edit=None):
        d = out / "inputs" / job / step
        d.mkdir(parents=True, exist_ok=True)
        for f in ("INCAR", "POSCAR", "KPOINTS"):
            t = (step_src / f).read_text(encoding="utf-8")
            if f == "INCAR" and incar_edit:
                t = incar_edit(t)
            lf(d / f, t)
        names = (step_src / "POTCAR.spec").read_text(encoding="utf-8").split()
        lf(d / "POTCAR.names", "\n".join(names) + "\n")
        lf(d / "EXE", exe + "\n")
        shutil.copy(step_src / "job.json", d / "job.json")

    def exe_for(kp):
        return "vasp_gam" if list(kp) == [1, 1, 1] else "vasp_std"

    lists = {"relax_slab": [], "relax_ref": [], "static_slab": [], "static_ref": [], "kcheck": []}
    for job, j in m["jobs"].items():
        e = exe_for(j["kpoints"])
        if job == "bulk":
            stage_inputs("bulk", src / "bulk" / "relax", "relax", e)
            continue
        if job.startswith("kcheck_"):
            stage_inputs(job, src / job, "static", e)
            lists["kcheck"].append((job, j["static_geometry_from"].split("/")[0]))
            continue
        kind = "ref" if j["role"] == "reference" else "slab"
        stage_inputs(job, src / job / "relax", "relax", e)
        stage_inputs(job, src / job / "static", "static", e)
        lists[f"relax_{kind}"].append((job, ""))
        lists[f"static_{kind}"].append((job, job))

    # smoke: the bare-slab static with NELM=4 at the start geometry
    slab_job = next(n for n, j in m["jobs"].items() if j["role"] == "bare_slab")

    def smoke_incar(t):
        keep = [ln for ln in t.splitlines() if not ln.split("=")[0].strip() in ("NELM", "LCHARG", "LVHAR", "LORBIT")]
        return "\n".join(keep + ["NELM = 4", "LCHARG = .FALSE.", "LVHAR = .FALSE."]) + "\n"
    stage_inputs("smoke", src / slab_job / "static", "static", exe_for(m["jobs"][slab_job]["kpoints"]), smoke_incar)
    shutil.copy(src / slab_job / "relax" / "POSCAR", out / "inputs" / "smoke" / "static" / "POSCAR")

    for k, rows in lists.items():
        lf(out / "lists" / f"{k}.txt", "".join(f"{a_} {b_}\n" for a_, b_ in rows))

    lib = LIB.replace("__MODULE__", cl["module"]).replace("__POTDIR__", cl["potcar_dir"])
    lf(out / "lib_pvpass.sh", lib)
    tol = cl.get("bulk_gate_tol", 0.005)
    lf(out / "check_bulk_gate.sh", GATE.replace("__TOL__", str(tol)))

    def slurm(fname, key, purpose, body, list_name=None):
        st = stages[key]
        n = len(lists[list_name]) if list_name else 0
        array = f"#SBATCH --array=0-{n - 1}%{min(n, st.get('max_parallel', n))}\n" if list_name else ""
        output = "%x_%A_%a.out" if list_name else "%x_%j.out"
        lf(out / fname, SLURM.format(name=f"{pkg.split('_')[0]}-{key}", account=cl["account"],
                                     partition=cl["partition"], array=array, nodes=st["nodes"],
                                     ntasks=cl["ntasks_per_node"], time=st["time"], output=output, package=pkg,
                                     purpose=purpose, body=body))

    row = 'read -r job geo < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" lists/{L}.txt)\n'
    slurm("run_smoke.slurm", "smoke", "time one SCF step of the bare slab (NELM=4)", "run_step smoke static smoke")
    slurm("run_bulk.slurm", "bulk", "bulk lattice check (ISIF=3); then run ./check_bulk_gate.sh", "run_step bulk relax relax")
    slurm("run_relax_slabs.slurm", "relax_slab", "relax slab and complexes (ISIF=2)",
          row.format(L="relax_slab") + 'run_step "$job" relax relax', "relax_slab")
    slurm("run_relax_refs.slurm", "relax_ref", "relax salt references in vacuum boxes",
          row.format(L="relax_ref") + 'run_step "$job" relax relax', "relax_ref")
    slurm("run_static_slabs.slurm", "static_slab", "static slab and complexes from their relaxed CONTCAR",
          row.format(L="static_slab") + 'run_step "$job" static static "$geo"', "static_slab")
    slurm("run_static_refs.slurm", "static_ref", "static references from their relaxed CONTCAR",
          row.format(L="static_ref") + 'run_step "$job" static static "$geo"', "static_ref")
    slurm("run_kcheck_r2scan.slurm", "kcheck", "k-point check: static on the denser mesh from the relaxed CONTCAR",
          row.format(L="kcheck") + 'run_step "$job" static static "$geo"', "kcheck")
    lf(out / "export.list", "outputs/*\nlogs/*\n")
    (out / "logs").mkdir(exist_ok=True)
    lf(out / "logs" / ".keep", "")
    shutil.copy(src / "manifest.json", out / "manifest.json")

    cmds = [("smoke", "sbatch run_smoke.slurm", "read outputs/smoke/static/timing.txt; size the stages from it"),
            ("bulk", "sbatch run_bulk.slurm", "then: bash check_bulk_gate.sh  (STOP = regenerate the set; do not go on)"),
            ("relax_slab", "sbatch run_relax_slabs.slurm", "resubmit the same command for any task that hit the walltime"),
            ("relax_ref", "sbatch run_relax_refs.slurm", "can run alongside run_relax_slabs"),
            ("static_slab", "sbatch run_static_slabs.slurm", "after every slab-type relax is DONE"),
            ("static_ref", "sbatch run_static_refs.slurm", "after every reference relax is DONE"),
            ("kcheck", "sbatch run_kcheck_r2scan.slurm", "after the relaxes of the listed jobs are DONE")]
    cost = cl.get("cost_estimate_node_h", {})
    readme = [f"# {pkg}", "", m.get("title", "r2SCAN+D3(BJ) adsorption label set"), "",
              "Prepared by PV_Passivator (`examples/scripts/prepare_label_set.py`, staged by `stage_label_set.py`). "
              "Nothing was submitted.", "", "## Commands, in order (from this directory)", ""]
    for k, c, note in cmds:
        st = stages[k]
        n = len(lists.get(k, [])) if k in lists else 1
        readme.append(f"1. `{c}` ({n} job{'s' if n != 1 else ''}, {st['nodes']} node{'s' if st['nodes'] > 1 else ''} "
                      f"each, ≤ {st['time']}; est. {cost.get(k, '?')} node-h). {note}")
    readme += ["", "Every step writes `outputs/<job>/<step>/DONE` only on real completion (relax: reached required "
               "accuracy; static: EDIFF reached and vasprun complete). Steps already DONE are skipped, so any script "
               "can be resubmitted safely.", "",
               "k-point check: dk(X) = E(kcheck_X) − E(X/static); the Γ-only set stands if "
               "|dk(complex) − dk(slab)| < 0.02 eV.", "",
               "Exports: `export.list` (outputs/*, logs/*)."]
    lf(out / "README.md", "\n".join(readme) + "\n")
    lf(out / "package.json", json.dumps({"package": pkg, "generator": "PV_Passivator stage_label_set.py",
                                         "jobs": len(m["jobs"]), "reactions": [r["name"] for r in m["reactions"]],
                                         "stages": {k: len(v) for k, v in lists.items()},
                                         "cost_estimate_node_h": cost, "submitted": False}, indent=1) + "\n")
    print(f"staged {pkg}: {sum(len(v) for v in lists.values()) + 2} runs in {len(cmds)} stages -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
