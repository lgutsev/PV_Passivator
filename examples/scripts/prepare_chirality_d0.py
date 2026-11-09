"""First DFT tranche (D0-D2) of docs/CHIRALITY_FEASIBILITY.md: inputs only, nothing is submitted.

Two steps:

    # 1. hydrogen-only pre-relax of the X-ray structures (riding H), heavy atoms frozen; needs the maceh env
    D:/MLIP_Work_Folder/envs/maceh/Scripts/python.exe examples/scripts/prepare_chirality_d0.py settle-h \
        --out data/cache/chirality_d0_start
    # 2. VASP package (any env with ase)
    python examples/scripts/prepare_chirality_d0.py stage --config data/chirality_d0/config.json \
        --starts data/cache/chirality_d0_start --out <LONI_staging>/<package>

settle-h: FIRE with the foundation MACE-matpes-r2scan + D3(BJ, r2SCAN) on H atoms only (fmax 0.05 eV/A, <= 300
steps). Pb, I, C and N stay bit-for-bit at the X-ray positions, so every handedness descriptor of
pvpassivator.chirality is unchanged (checked). The foundation model is a pre-relaxer here, never a label.

stage: one r2SCAN+D3(BJ) Hamiltonian (project label profile, bulk: no dipole correction), ISYM = 0 everywhere so the
mirror pair and the SOC pair are computed without symmetry reduction. Arrays, in the order they should be run:

  smoke   R at the reference mesh, NELM = 4 (environment and timing)
  check   statics on the pre-relaxed starts: k-mesh/ENCUT convergence for R and rac (D0a), the reference R static
          doubles as the timing benchmark (D0b), and the exact mirror of R (D2: E and mapped forces must equal R's)
  soc     the two reference statics again with LSORBIT (D0c; vasp_ncl)
  relax   R and rac at the experimental cell (ISIF = 2) and with a free cell (ISIF = 3, ENCUT x 1.3) (D1)
  static  production-ENCUT statics on each relaxed geometry (D1 label energies)

The relax and static arrays should wait for the check results (mesh choice and measured cost).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from pvpassivator import chirality as ch  # noqa: E402

CIFS = {"R": "R-MBA2PbI4_COD7203475.cif", "rac": "rac-MBA2PbI4_COD2201646.cif"}
SDIR = ROOT / "examples" / "structures" / "mba2pbi4"

PHYSICS = {  # vaspprep._COMMON_R2SCAN without the slab-only dipole tags; ENCUT is set per job
    "METAGGA": "R2SCAN", "LASPH": ".TRUE.", "IVDW": 12, "VDW_S8": 0.7898, "VDW_A1": 0.4948, "VDW_A2": 5.7308,
    "PREC": "Accurate", "ISMEAR": 0, "SIGMA": 0.05, "LREAL": "Auto", "EDIFF": "1E-06", "NELM": 200,
    "ISYM": 0, "ISPIN": 1, "LCHARG": ".FALSE.", "LWAVE": ".FALSE.",
}
SOC_TAGS = {"LSORBIT": ".TRUE.", "LMAXMIX": 4, "GGA_COMPAT": ".FALSE."}   # ISPIN/MAGMOM set per job


def _read_cif(name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read(SDIR / name)


def descriptors(atoms) -> dict:
    top = ch.build_topology(atoms)
    d = ch.frame_descriptors(top, atoms)
    return {"stereocentres": top.summary()["stereocentres"],
            **{f"mean_{k}": round(d[f"mean_{k}"], 6) for k in ("ax", "eq", "org")}}


# ----------------------------------------------------------------------------------------------- settle-h

def settle_h(out: Path, fmax: float = 0.05, steps: int = 300, device: str = "cuda") -> dict:
    import local_settle as LS
    calc = LS.make_calc(LS.MODEL, device, "float64")
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": str(LS.MODEL), "dispersion": "D3(BJ, r2scan) torch-dftd", "fmax_target": fmax,
              "note": "H-only pre-relax; heavy atoms at the X-ray positions. A starting geometry, not a label."}
    for key, cif in CIFS.items():
        a = _read_cif(cif)
        mobile = np.array([s == "H" for s in a.get_chemical_symbols()])
        settled, rep = LS.settle(a, mobile, calc, fmax=fmax, steps=steps)
        before, after = descriptors(a), descriptors(settled)
        heavy = np.abs(settled.positions[~mobile] - a.positions[~mobile]).max()
        top = ch.build_topology(settled)
        sym = np.array(settled.get_chemical_symbols())
        xh = []
        for m in top.molecules:
            for i, j in m.bonds:
                if "H" in (sym[i], sym[j]):
                    xh.append(float(np.linalg.norm(ch.mic(settled, settled.positions[j] - settled.positions[i]))))
        rep.update({"descriptors_xray": before, "descriptors_settled": after,
                    "heavy_atoms_max_change_A": float(heavy), "X_H_bond_range_A": [round(min(xh), 3), round(max(xh), 3)]})
        ok = heavy == 0.0 and before == after and 0.95 < min(xh) and max(xh) < 1.15
        rep["status"] = "pass" if ok else "fail"
        write(out / f"POSCAR_{key}", settled, format="vasp", direct=True, sort=False)
        report[key] = rep
        print(key, rep["status"], f"fmax {rep['fmax_start']} -> {rep['fmax_end']} in {rep['steps']} steps,",
              f"X-H {rep['X_H_bond_range_A']} A, heavy change {heavy}")
    (out / "settle.json").write_text(json.dumps(report, indent=1))
    return report


# ----------------------------------------------------------------------------------------------- stage

def build_jobs(starts: dict, kspacing: float, encut: int, encut_factor: float):
    """Job table: (array, job, step, kind, atoms_key, tags_override, kmesh, geo). Pure function for testing."""
    from prepare_bulk_refs import kmesh
    kR, kr = kmesh(starts["R"], kspacing), kmesh(starts["rac"], kspacing)
    up = lambda k: [n + 1 if n > 1 else n for n in k]          # one step denser in-plane (and along c if > 1)
    down = lambda k: [max(1, n - 1) if n > 1 else n for n in k]
    big = round(encut * encut_factor)
    J = []
    J.append(("smoke", "smoke", "static", "smoke", "R", {"ENCUT": encut, "NELM": 4}, kR, ""))
    for name, key, k, e in (("R_kref", "R", kR, encut), ("R_kdown", "R", down(kR), encut), ("R_kup", "R", up(kR), encut),
                            ("R_kref_e600", "R", kR, 600), ("Rmirror_kref", "Rmirror", kR, encut),
                            ("rac_kref", "rac", kr, encut), ("rac_kup", "rac", up(kr), encut)):
        J.append(("check", name, "static", "static", key, {"ENCUT": e, "IBRION": -1, "NSW": 0}, k, ""))
    for name, key, k in (("R_kref_soc", "R", kR), ("rac_kref_soc", "rac", kr)):
        J.append(("soc", name, "static", "static", key, {"ENCUT": encut, "IBRION": -1, "NSW": 0, **SOC_TAGS}, k, ""))
    relax = {"IBRION": 2, "NSW": 200, "EDIFFG": -0.01, "POTIM": 0.3}
    for key, k in (("R", kR), ("rac", kr)):
        J.append(("relax", f"{key}_fixcell", "relax", "relax", key, {**relax, "ISIF": 2, "ENCUT": encut}, k, ""))
        J.append(("relax", f"{key}_fullcell", "relax", "relax", key, {**relax, "ISIF": 3, "ENCUT": big}, k, ""))
        for cellmode in ("fixcell", "fullcell"):
            J.append(("static", f"{key}_{cellmode}", "static", "static", key,
                      {"ENCUT": encut, "IBRION": -1, "NSW": 0}, k, f"{key}_{cellmode}"))
    return J


def build_diag_jobs(starts: dict, kspacing: float):
    """D0e LREAL/ENCUT diagnostic (owner-approved 2026-11-08 after package 105 failed the ENCUT check): four R statics
    on the same start and mesh as 105's R_kref (520, LREAL Auto) and R_kref_e600 (600, LREAL Auto), which serve as
    the baselines. LREAL=.FALSE. at 520/600/700 separates projection error from basis error; 700 with LREAL=Auto
    tests whether Auto converges at a higher cutoff."""
    from prepare_bulk_refs import kmesh
    kR = kmesh(starts["R"], kspacing)
    st = {"IBRION": -1, "NSW": 0}
    return [("diag", "R_e520_lrealF", "static", "static", "R", {**st, "ENCUT": 520, "LREAL": ".FALSE."}, kR, ""),
            ("diag", "R_e600_lrealF", "static", "static", "R", {**st, "ENCUT": 600, "LREAL": ".FALSE."}, kR, ""),
            ("diag", "R_e700_lrealF", "static", "static", "R", {**st, "ENCUT": 700, "LREAL": ".FALSE."}, kR, ""),
            ("diag", "R_e700", "static", "static", "R", {**st, "ENCUT": 700}, kR, "")]


def build_d1_jobs(starts: dict, kspacing: float, encut: int, encut_factor: float):
    """Revision 1 of package 105's held relax/static arrays (owner: "Yes, regenerate", 2026-11-09).

    Package 108 traced the iodide force error to LREAL=Auto; with LREAL=.FALSE. the Pb/I forces are converged at
    520 eV (<= 1.7 meV/A). Same jobs, starts and meshes as 105, with LREAL=.FALSE. in every INCAR. The accepted
    residual on C/H/N is ~12 meV/A (ENCUT-dependent, non-monotonic; see docs/CHIRALITY_FEASIBILITY.md)."""
    return [(arr, job, step, kind, key, {**over, "LREAL": ".FALSE."}, k, geo)
            for arr, job, step, kind, key, over, k, geo in build_jobs(starts, kspacing, encut, encut_factor)
            if arr in ("relax", "static")]


def walltime_s(t: str) -> int:
    """Slurm walltime 'D-HH:MM:SS', 'HH:MM:SS' or 'MM:SS' -> seconds."""
    days, _, hms = t.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def signal_margin(st: dict) -> int:
    """USR1 lead time before the walltime: explicit ``signal_margin_s`` or min(1800, wall/4).

    Desk fix 2026-11-08 (package 105): 1800 s on a 30-min smoke job fires USR1 at start-up, before run_vasp has set
    its trap, and the default action kills the batch shell. The margin must stay well below the walltime."""
    wall = walltime_s(st["time"])
    s = int(st.get("signal_margin_s", min(1800, wall // 4)))
    if not 0 < s <= wall // 4:
        raise SystemExit(f"signal margin {s} s must be in (0, walltime/4 = {wall // 4} s] for time {st['time']}")
    return s


def stage(cfg: dict, starts_dir: Path, out: Path, overwrite: bool = False, mode: str = "d0") -> dict:
    from prepare_bulk_refs import incar_text, ordered, potcar_defs
    from stage_label_set import LIB, SLURM, lf
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise SystemExit(f"{out} is not empty")
    pkg = out.name
    starts = {k: read(starts_dir / f"POSCAR_{k}", format="vasp") for k in ("R", "rac")}
    settle_rep = json.loads((starts_dir / "settle.json").read_text())
    if any(settle_rep[k]["status"] != "pass" for k in ("R", "rac")):
        raise SystemExit("settle-h did not pass for every structure; see settle.json")
    starts["Rmirror"] = ch.invert(starts["R"])
    if descriptors(starts["Rmirror"])["stereocentres"]["R"] != 0:
        raise SystemExit("mirror still has R centres")
    defs = potcar_defs(Path(cfg.get("potcar_defs", ROOT / "src" / "pvpassivator" / "data" / "POTCAR_DEFS.txt")))
    cl = cfg["cluster"]
    jobs = {"d0": lambda: build_jobs(starts, cfg["kspacing"], cfg["encut"], cfg["encut_factor"]),
            "diag": lambda: build_diag_jobs(starts, cfg["kspacing"]),
            "d1": lambda: build_d1_jobs(starts, cfg["kspacing"], cfg["encut"], cfg["encut_factor"])}[mode]()
    lists = {}
    manifest = {"title": cfg["title"], "physics": PHYSICS, "soc_tags": SOC_TAGS, "jobs": [], "starts": {},
                "settle": settle_rep}
    for key, at in starts.items():
        manifest["starts"][key] = {"formula": at.get_chemical_formula(), "natoms": len(at), **descriptors(at)}
    for arr, job, step, kind, key, over, k, geo in jobs:
        d = out / "inputs" / job / step
        d.mkdir(parents=True, exist_ok=True)
        at, species = ordered(starts[key])
        if missing := [s for s in species if s not in defs]:
            raise SystemExit(f"{missing} not in POTCAR_DEFS.txt")
        tags = {**PHYSICS, **over, **cfg["parallel"].get(arr, cfg["parallel"]["default"])}
        exe = "vasp_ncl" if "LSORBIT" in over else ("vasp_gam" if k == [1, 1, 1] else "vasp_std")
        if exe == "vasp_ncl":
            tags.pop("ISPIN", None)
            tags["MAGMOM"] = f"{3 * len(at)}*0.0"
        write(d / "POSCAR", at, format="vasp", direct=True, sort=False)
        lf(d / "POSCAR", (d / "POSCAR").read_text(encoding="utf-8"))
        lf(d / "INCAR", incar_text(f"{cfg['title']}: {job} {step} ({arr})", tags))
        lf(d / "KPOINTS", f"{job}: Gamma-centred, KSPACING {cfg['kspacing']} 1/A rule unless a convergence variant\n"
                          f"0\nGamma\n{k[0]} {k[1]} {k[2]}\n0 0 0\n")
        lf(d / "POTCAR.names", "\n".join(defs[s] for s in species) + "\n")
        lf(d / "EXE", exe + "\n")
        lists.setdefault(arr, []).append((job, geo))
        manifest["jobs"].append({"array": arr, "job": job, "step": step, "start": key, "kpoints": k, "exe": exe,
                                 "natoms": len(at), "ENCUT": tags["ENCUT"], "ISIF": tags.get("ISIF"),
                                 "geometry_from": geo or "inputs POSCAR"})
    for arr, rows in lists.items():
        lf(out / "lists" / f"{arr}.txt", "".join(f"{x} {y}\n" for x, y in rows))
    lf(out / "lib_pvpass.sh", LIB.replace("__MODULE__", cl["module"]).replace("__POTDIR__", cl["potcar_dir"]))
    row = 'read -r job geo < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" lists/{L}.txt)\n'
    bodies = {
        "smoke": ("one SCF-step timing check (NELM=4)", "run_step smoke static smoke", False),
        "check": ("D0a/D0b/D2 statics on the H-settled starts", row.format(L="check") + 'run_step "$job" static static', True),
        "soc": ("D0c SOC statics (vasp_ncl)", row.format(L="soc") + 'run_step "$job" static static', True),
        "relax": ("D1 relaxations (fixed and free cell)", row.format(L="relax") + 'run_step "$job" relax relax', True),
        "static": ("D1 production statics on relaxed geometries",
                   row.format(L="static") + 'run_step "$job" static static "$geo"', True),
        "diag": ("D0e LREAL/ENCUT diagnostic statics on R", row.format(L="diag") + 'run_step "$job" static static', True),
    }
    bodies = {k: v for k, v in bodies.items() if k in lists}
    for arr, (purpose, body, is_array) in bodies.items():
        st = cl["stages"].get(arr, cl["stages"]["check"])
        n = len(lists[arr])
        array = f"#SBATCH --array=0-{n - 1}%{min(n, st.get('max_parallel', n))}\n" if is_array else ""
        lf(out / f"run_{arr}.slurm", SLURM.replace("stage_label_set.py", "prepare_chirality_d0.py").format(
            name=f"chiral-{arr}", account=cl["account"], partition=cl["partition"], array=array, nodes=st["nodes"],
            ntasks=cl["ntasks_per_node"], time=st["time"], output="%x_%A_%a.out" if is_array else "%x_%j.out",
            package=pkg, purpose=purpose, body=body, signal_s=signal_margin(st)))
    lf(out / "export.list", "outputs/*\nlogs/*\n")
    lf(out / "logs" / ".keep", "")
    lf(out / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    cost = cl.get("cost_estimate_node_h", {})
    rd = [f"# {pkg}", "", cfg["title"], "",
          "Prepared by PV_Passivator `examples/scripts/prepare_chirality_d0.py`. **Nothing was submitted.**", "",
          "## Commands, in order (from this directory)", ""]
    notes = {"smoke": "check outputs/smoke/static/timing.txt", "check": "report LOOP+ timings and energies before going on",
             "soc": "needs vasp_ncl in the module; independent of check", "relax": "only after check is reviewed; resubmit "
             "tasks that hit the walltime (continues from CONTCAR)", "static": "after the relaxes are DONE",
             "diag": "compare with package 105 R_kref / R_kref_e600 (same start and mesh)"}
    for arr in bodies:
        st = cl["stages"].get(arr, cl["stages"]["check"])
        n = len(lists[arr])
        rd.append(f"1. `sbatch run_{arr}.slurm` ({n} job{'s' if n != 1 else ''}, {st['nodes']} node(s) each, <= {st['time']}; "
                  f"est. {cost.get(arr, '?')} node-h). {notes[arr]}")
    rd += ["", "DONE is written only on real completion; DONE steps are skipped, so resubmitting is safe.", "",
           "## Jobs", "", "| Array | Job | Start | Atoms | k-mesh | ENCUT | EXE |", "|---|---|---|---|---|---|---|"]
    for j in manifest["jobs"]:
        rd.append(f"| {j['array']} | {j['job']} | {j['start']} | {j['natoms']} | {'x'.join(map(str, j['kpoints']))} | "
                  f"{j['ENCUT']} | {j['exe']} |")
    rd += ["", "Exports: `export.list` (outputs/*, logs/*)."]
    lf(out / "README.md", "\n".join(rd) + "\n")
    # per-file sha256 (desk request 2026-11-08) so a re-stage can be verified byte for byte; package.json itself excluded
    files = {f.relative_to(out).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(out.rglob("*")) if f.is_file() and f.name != "package.json"}
    lf(out / "package.json", json.dumps({"package": pkg, "generator": "PV_Passivator prepare_chirality_d0.py",
                                         "config_sha256_16": cfg.get("_sha16"), "arrays": {k: len(v) for k, v in lists.items()},
                                         "cost_estimate_node_h": cost, "submitted": False,
                                         "files_sha256": files}, indent=1) + "\n")
    print(f"staged {pkg}: " + ", ".join(f"{k} {len(v)}" for k, v in lists.items()) + f" -> {out}")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("settle-h")
    s1.add_argument("--out", type=Path, default=ROOT / "data" / "cache" / "chirality_d0_start")
    s1.add_argument("--device", default="cuda")
    s2 = sub.add_parser("stage")
    s2.add_argument("--config", type=Path, required=True)
    s2.add_argument("--starts", type=Path, default=ROOT / "data" / "cache" / "chirality_d0_start")
    s2.add_argument("--out", type=Path, required=True)
    s2.add_argument("--overwrite", action="store_true")
    s2.add_argument("--mode", choices=("d0", "diag", "d1"), default="d0",
                    help="diag: the D0e LREAL/ENCUT statics only; d1: 105 relax/static revision 1 (LREAL=.FALSE.)")
    a = ap.parse_args(argv)
    if a.cmd == "settle-h":
        rep = settle_h(a.out, device=a.device)
        return 0 if all(rep[k]["status"] == "pass" for k in CIFS) else 1
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    cfg["_sha16"] = hashlib.sha256(a.config.read_bytes()).hexdigest()[:16]
    stage(cfg, a.starts, a.out, a.overwrite, a.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
