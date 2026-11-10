"""Stage a statics-only audit set (prepare_static_audit.py output) as a cluster package. Nothing is submitted.

    python examples/scripts/stage_static_audit.py --package PACKAGE_SRC --cluster cluster.json --out STAGING/<name>

Reuses the run library of ``stage_label_set.py`` unchanged (``lib_pvpass.sh``: module guard, POTCAR assembly, periodic
and walltime-signal saves, DONE only on real completion), so a static here behaves exactly like a label-set static.
Stages: smoke (bare slab, NELM=4, to time one SCF step) -> slab-type statics (array) -> the molecule static.
Script names are package-specific (``run_audit_*.slurm``).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_label_set import LIB, SLURM, lf  # noqa: E402



def signal_margin(walltime: str) -> int:
    """Seconds before the walltime at which USR1 is sent: min(1800, walltime / 4).  A fixed 1800 s on a 1 h job
    fires at the half-way mark and stops a run that is still fine (desk, package 110)."""
    parts = [int(x) for x in walltime.replace("-", ":").split(":")]
    secs = sum(v * m for v, m in zip(reversed(parts), (1, 60, 3600, 86400)))
    return min(1800, secs // 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True), ap.add_argument("--cluster", required=True)
    ap.add_argument("--out", required=True), ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    src, out = Path(a.package), Path(a.out)
    cl = json.loads(Path(a.cluster).read_text())
    m = json.loads((src / "manifest.json").read_text())
    if m.get("schema") != "static_audit/1":
        raise SystemExit("not a static_audit package")
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise SystemExit(f"{out} is not empty")
    pkg, stages = out.name, cl["stages"]

    def exe_for(kp):
        return "vasp_gam" if list(kp) == [1, 1, 1] else "vasp_std"

    def stage_inputs(job, step_src, exe, incar_edit=None):
        d = out / "inputs" / job / "static"
        d.mkdir(parents=True, exist_ok=True)
        for f in ("INCAR", "POSCAR", "KPOINTS"):
            t = (step_src / f).read_text(encoding="utf-8")
            lf(d / f, incar_edit(t) if f == "INCAR" and incar_edit else t)
        lf(d / "POTCAR.names", "\n".join((step_src / "POTCAR.spec").read_text(encoding="utf-8").split()) + "\n")
        lf(d / "EXE", exe + "\n")
        shutil.copy(step_src / "job.json", d / "job.json")

    lists = {"audit_slab": [], "audit_ref": []}
    for job, j in m["jobs"].items():
        stage_inputs(job, src / job / "static", exe_for(j["kpoints"]))
        lists["audit_ref" if j["role"] == "reference" else "audit_slab"].append(job)

    def smoke_incar(t):
        keep = [ln for ln in t.splitlines() if ln.split("=")[0].strip() not in ("NELM", "LCHARG", "LVHAR", "LORBIT")]
        return "\n".join(keep + ["NELM = 4", "LCHARG = .FALSE.", "LVHAR = .FALSE."]) + "\n"
    big = max((j for j in m["jobs"].values() if j["role"] == "complex"), key=lambda j: j["nelect"])
    big_name = next(n for n, j in m["jobs"].items() if j is big)
    stage_inputs("smoke", src / big_name / "static", exe_for(big["kpoints"]), smoke_incar)

    for k, rows in lists.items():
        lf(out / "lists" / f"{k}.txt", "".join(f"{r}\n" for r in rows))
    lf(out / "lib_pvpass.sh", LIB.replace("__MODULE__", cl["module"]).replace("__POTDIR__", cl["potcar_dir"]))

    def slurm(fname, key, purpose, body, list_name=None):
        st = stages[key]
        n = len(lists[list_name]) if list_name else 0
        par = min(n, st.get("max_parallel", n)) if list_name else 0
        array = f"#SBATCH --array=0-{n - 1}%{par}\n" if list_name else ""
        text = SLURM.format(name=f"{pkg.split('_')[0]}-{key}", account=cl["account"], partition=cl["partition"],
                            array=array, nodes=st["nodes"], ntasks=cl["ntasks_per_node"], time=st["time"],
                            output="%x_%A_%a.out" if list_name else "%x_%j.out", package=pkg,
                            signal_s=int(st.get("signal_margin_s", signal_margin(st["time"]))), purpose=purpose,
                            body=body)
        lf(out / fname, text.replace("stage_label_set.py", "stage_static_audit.py"))

    # array tasks that start together must not load the container module at the same instant (desk lesson, 2026-11-01)
    row = ('sleep $(( (SLURM_ARRAY_TASK_ID % {P}) * 45 ))\n'
           'job=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" lists/{L}.txt)\n[ -n "$job" ] || {{ echo "no job for task '
           '$SLURM_ARRAY_TASK_ID" >&2; exit 2; }}\nrun_step "$job" static static')
    slurm("run_audit_smoke.slurm", "smoke", f"time one SCF step of the largest structure ({big_name}, NELM=4)",
          "run_step smoke static smoke")
    slurm("run_audit_statics.slurm", "static_slab", "single points of the slab and every audited structure",
          row.format(L="audit_slab", P=max(1, stages["static_slab"].get("max_parallel", 1))), "audit_slab")
    slurm("run_audit_ref.slurm", "static_ref", "single point of the isolated molecule",
          row.format(L="audit_ref", P=1), "audit_ref")
    lf(out / "export.list", "outputs/*\nlogs/*\n")
    lf(out / "logs" / ".keep", "")
    shutil.copy(src / "manifest.json", out / "manifest.json")

    cost = cl.get("cost_estimate_node_h", {})
    ss = stages["static_slab"]
    readme = [
        f"# {pkg}", "", m["title"], "",
        "Prepared by PV_Passivator (`examples/scripts/prepare_static_audit.py`, staged by `stage_static_audit.py`). "
        "Nothing was submitted.", "",
        f"{len(lists['audit_slab'])} slab-type single points ({big['natoms']} atoms at most, Γ only, `vasp_gam`) and "
        f"{len(lists['audit_ref'])} molecule single point. No relaxation: every POSCAR is the geometry at which an "
        "MLIP's energy and forces were stored, and the DFT numbers are compared with them atom by atom.", "",
        "## Commands, in order (from this directory)", "",
        f"1. `sbatch run_audit_smoke.slurm` (1 job, {stages['smoke']['nodes']} nodes, ≤ {stages['smoke']['time']}; "
        f"est. {cost.get('smoke', '?')} node-h). Read `outputs/smoke/static/timing.txt` and size the next stage from it.",
        f"1. `sbatch run_audit_statics.slurm` ({len(lists['audit_slab'])} jobs, array %{ss.get('max_parallel', '?')}, "
        f"{ss['nodes']} nodes each, ≤ {ss['time']}; est. {cost.get('static_slab', '?')} node-h). Task 0 is the bare slab.",
        f"1. `sbatch run_audit_ref.slurm` (1 job, {stages['static_ref']['nodes']} node, ≤ {stages['static_ref']['time']}; "
        f"est. {cost.get('static_ref', '?')} node-h). Can run alongside.", "",
        "Every step writes `outputs/<job>/static/DONE` only when EDIFF is reached and vasprun.xml is complete. Steps "
        "already DONE are skipped, so any script can be resubmitted safely. A static that hits the walltime saves its "
        "outputs and must simply be resubmitted with more time.", "",
        "Settings: the label profile's static (r2SCAN+D3(BJ), ENCUT 520, EDIFF 1E-6, dipole correction along x with one "
        "pinned centre) with `LREAL = .FALSE.`, because this is a force comparison. Do not combine these energies "
        "with sets run with `LREAL = Auto`.", "",
        "Exports: `export.list` (outputs/*, logs/*). `vasprun.xml.gz` carries the forces the analysis reads.",
    ]
    lf(out / "README.md", "\n".join(readme) + "\n")
    import hashlib
    sha = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(out.rglob("*")) if p.is_file() and p.name != "package.json"}
    lf(out / "package.json", json.dumps({
        "package": pkg, "generator": "PV_Passivator stage_static_audit.py", "jobs": len(m["jobs"]),
        "stages": {"smoke": 1, **{k: len(v) for k, v in lists.items()}}, "cost_estimate_node_h": cost,
        "submitted": False, "files_sha256": sha}, indent=1) + "\n")
    print(f"staged {pkg}: {len(m['jobs']) + 1} runs in 3 stages -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
