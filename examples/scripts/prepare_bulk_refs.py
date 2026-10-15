"""Prepare a bulk-reference package: relaxed bulk phases + an O2 molecule, one Hamiltonian, ready for the cluster.

Use: reaction energies that need bulk reservoirs (e.g. a surface unit leaving as a bulk solid) and an empirical O2
energy correction fitted to formation enthalpies. Every phase gets
  relax   ISIF=3 at ENCUT x encut_factor (no Pulay stress in the cell), and
  static  at the production ENCUT on the relaxed cell (the energy that enters reactions);
the O2 molecule is relaxed with the given molecular INCAR; optional extra statics vary one tag (e.g. LREAL) to
measure its effect. Inputs only: nothing is submitted.

    python examples/scripts/prepare_bulk_refs.py --config data/bulk_refs/config.json --out <staging>/<NN>_<name>

Config (JSON):
{
  "title": "...",
  "physics": {"METAGGA": "SCAN", ...},          # common Hamiltonian tags, copied verbatim to every INCAR
  "encut": 550, "encut_factor": 1.3, "kspacing": 0.20,
  "relax": {"IBRION": 2, "ISIF": 3, "NSW": 200, "EDIFFG": -0.01},
  "parallel": {"NCORE": 8, "KPAR": 4},
  "phases": [{"name": "PbO", "cif": "<path>", "source": "mp-19921"}, ...],
  "molecule": {"name": "O2", "poscar": "<path>", "incar": "<INCAR file>", "source": "..."},
  "variants": [{"name": "PbO_lrealF", "of": "PbO", "set": {"LREAL": ".FALSE."}}],
  "smoke": "PbO",
  "potcar_defs": "<POTCAR_DEFS.txt>",
  "cluster": {"account": ..., "partition": ..., "ntasks_per_node": 64, "module": ..., "potcar_dir": ...,
              "stages": {"smoke": {"nodes": 1, "time": "00:30:00"}, ...}, "cost_estimate_node_h": {...}}
}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_label_set import LIB, SLURM, lf  # noqa: E402  (the tested run_step library of the label sets)


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def potcar_defs(path: Path) -> dict[str, str]:
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if "|" in ln and not ln.startswith("#"):
            e, v = ln.strip().split("|", 1)
            out[e] = v
    return out


def kmesh(atoms, kspacing: float) -> list[int]:
    """VASP KSPACING rule: N_i = max(1, ceil(|b_i| / KSPACING)), |b_i| including 2*pi (Gamma-centred)."""
    rec = 2 * math.pi * np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    return [max(1, math.ceil(r / kspacing - 1e-9)) for r in rec]


def incar_text(title: str, tags: dict) -> str:
    return title + "\n" + "".join(f"{k} = {v}\n" for k, v in tags.items())


def ordered(atoms):
    """Group species in first-appearance order (POSCAR blocks) without reordering within a species."""
    syms = atoms.get_chemical_symbols()
    species = list(dict.fromkeys(syms))
    idx = [i for sp in species for i, s in enumerate(syms) if s == sp]
    return atoms[idx], species


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise SystemExit(f"{out} is not empty")
    pkg = out.name
    defs = potcar_defs(Path(cfg["potcar_defs"]))
    cl = cfg["cluster"]
    phys = cfg["physics"]
    for forbidden in ("ENCUT", "KSPACING", "KGAMMA", "IBRION", "ISIF", "NSW", "NCORE", "KPAR"):
        if forbidden in phys:
            raise SystemExit(f"'physics' must not set {forbidden}; the generator sets it per step")
    manifest = {"title": cfg["title"], "physics": phys, "encut": cfg["encut"], "kspacing": cfg["kspacing"],
                "jobs": {}, "variants": {}}

    def stage(job: str, step: str, atoms, tags: dict, title: str, kpts: list[int]):
        d = out / "inputs" / job / step
        d.mkdir(parents=True, exist_ok=True)
        at, species = ordered(atoms)
        missing = [s for s in species if s not in defs]
        if missing:
            raise SystemExit(f"{missing} not in POTCAR_DEFS.txt")
        write(d / "POSCAR", at, format="vasp", direct=True, sort=False)
        # ASE writes LF already; re-save through lf() for certainty
        lf(d / "POSCAR", (d / "POSCAR").read_text(encoding="utf-8"))
        lf(d / "INCAR", incar_text(title, tags))
        lf(d / "KPOINTS", f"{job} {step}: KSPACING {cfg['kspacing']} 1/A rule, Gamma-centred\n0\nGamma\n"
                          f"{kpts[0]} {kpts[1]} {kpts[2]}\n0 0 0\n")
        lf(d / "POTCAR.names", "\n".join(defs[s] for s in species) + "\n")
        exe = "vasp_gam" if kpts == [1, 1, 1] else "vasp_std"
        lf(d / "EXE", exe + "\n")
        return {"species": species, "counts": [at.get_chemical_symbols().count(s) for s in species],
                "kpoints": kpts, "exe": exe, "natoms": len(at)}

    base = dict(phys)
    par = cfg["parallel"]
    relax_tags = {**base, "ENCUT": round(cfg["encut"] * cfg["encut_factor"]), **cfg["relax"], **par, "LWAVE": ".FALSE."}
    static_tags = {**base, "ENCUT": cfg["encut"], "IBRION": -1, "NSW": 0, **par, "LWAVE": ".FALSE."}
    lists = {"relax": [], "static": [], "variant": []}
    for ph in cfg["phases"]:
        src = Path(ph["cif"])
        at = read(src)
        k = kmesh(at, cfg["kspacing"])
        name = ph["name"]
        r = stage(name, "relax", at, relax_tags, f"{cfg['title']}: {name} relax (ISIF=3, ENCUT x {cfg['encut_factor']})", k)
        stage(name, "static", at, static_tags, f"{cfg['title']}: {name} static (production ENCUT, relaxed cell)", k)
        manifest["jobs"][name] = {**r, "role": "bulk", "source": ph["source"], "file": str(src),
                                  "sha256_16": sha16(src), "formula": at.get_chemical_formula(empirical=True)}
        lists["relax"].append((name, ""))
        lists["static"].append((name, name))
    mol = cfg["molecule"]
    mat = read(mol["poscar"], format="vasp")
    mtags = {}
    for ln in Path(mol["incar"]).read_text(encoding="utf-8").splitlines()[1:]:
        ln = ln.split("#")[0]
        if "=" in ln:
            kk, vv = ln.split("=", 1)
            mtags[kk.strip()] = vv.strip()
    r = stage(mol["name"], "relax", mat, mtags, f"{cfg['title']}: {mol['name']} molecule (INCAR copied from {mol['source']})",
              [1, 1, 1])
    manifest["jobs"][mol["name"]] = {**r, "role": "molecule", "source": mol["source"], "file": str(mol["poscar"]),
                                     "sha256_16": sha16(Path(mol["poscar"]))}
    lists["relax"].append((mol["name"], ""))
    for v in cfg.get("variants", []):
        of = v["of"]
        at = read(out / "inputs" / of / "static" / "POSCAR", format="vasp")
        stage(v["name"], "static", at, {**static_tags, **v["set"]}, f"{cfg['title']}: {of} static with {v['set']}",
              manifest["jobs"][of]["kpoints"])
        manifest["variants"][v["name"]] = {"of": of, "set": v["set"]}
        lists["variant"].append((v["name"], of))
    sm = cfg["smoke"]
    at = read(out / "inputs" / sm / "static" / "POSCAR", format="vasp")
    stage("smoke", "static", at, {**static_tags, "NELM": 4}, f"{cfg['title']}: smoke ({sm} static, NELM=4)",
          manifest["jobs"][sm]["kpoints"])

    for kk, rows in lists.items():
        lf(out / "lists" / f"{kk}.txt", "".join(f"{x} {y}\n" for x, y in rows))
    lf(out / "lib_pvpass.sh", LIB.replace("__MODULE__", cl["module"]).replace("__POTDIR__", cl["potcar_dir"]))
    row = 'read -r job geo < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" lists/{L}.txt)\n'

    def slurm(fname, key, purpose, body, list_name=None):
        st = cl["stages"][key]
        n = len(lists[list_name]) if list_name else 0
        array = f"#SBATCH --array=0-{n - 1}%{min(n, st.get('max_parallel', n))}\n" if list_name else ""
        lf(out / fname, SLURM.replace("stage_label_set.py", "prepare_bulk_refs.py").format(
            name=f"{pkg.split('_')[0]}-{key}", account=cl["account"], partition=cl["partition"], array=array,
            nodes=st["nodes"], ntasks=cl["ntasks_per_node"], time=st["time"],
            output="%x_%A_%a.out" if list_name else "%x_%j.out", package=pkg, purpose=purpose, body=body))

    slurm("run_bulkref_smoke.slurm", "smoke", "one SCF step check (NELM=4)", "run_step smoke static smoke")
    slurm("run_bulkref_relax.slurm", "relax", "relax bulk phases (ISIF=3) and the molecule",
          row.format(L="relax") + 'run_step "$job" relax relax', "relax")
    slurm("run_bulkref_static.slurm", "static", "static at the production ENCUT on each relaxed cell",
          row.format(L="static") + 'run_step "$job" static static "$geo"', "static")
    slurm("run_bulkref_variant.slurm", "variant", "one-tag variations of a static (settings sensitivity)",
          row.format(L="variant") + 'run_step "$job" static static "$geo"', "variant")
    lf(out / "export.list", "outputs/*\nlogs/*\n")
    lf(out / "logs" / ".keep", "")
    lf(out / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    cost = cl.get("cost_estimate_node_h", {})
    steps = [("smoke", "sbatch run_bulkref_smoke.slurm", 1, "check outputs/smoke/static/timing.txt"),
             ("relax", "sbatch run_bulkref_relax.slurm", len(lists["relax"]), "resubmit for any task that hit the walltime"),
             ("static", "sbatch run_bulkref_static.slurm", len(lists["static"]), "after the relaxes are DONE"),
             ("variant", "sbatch run_bulkref_variant.slurm", len(lists["variant"]), "after the relaxes are DONE")]
    rd = [f"# {pkg}", "", cfg["title"], "", "Prepared by PV_Passivator `examples/scripts/prepare_bulk_refs.py`. Nothing was submitted.",
          "", "## Commands, in order (from this directory)", ""]
    for kk, c, n, note in steps:
        st = cl["stages"][kk]
        rd.append(f"1. `{c}` ({n} job{'s' if n != 1 else ''}, {st['nodes']} node each, ≤ {st['time']}; "
                  f"est. {cost.get(kk, '?')} node-h). {note}")
    rd += ["", "Every step writes `outputs/<job>/<step>/DONE` only on real completion (relax: reached required accuracy; "
           "static: EDIFF reached and vasprun complete). DONE steps are skipped, so resubmitting is safe.", "",
           "## Jobs", "", "| Job | Role | Atoms | k-mesh | Source |", "|---|---|---|---|---|"]
    for j, v in manifest["jobs"].items():
        rd.append(f"| {j} | {v['role']} | {v['natoms']} | {'x'.join(map(str, v['kpoints']))} | {v['source']} |")
    for j, v in manifest["variants"].items():
        rd.append(f"| {j} | variant of {v['of']} | | | {v['set']} |")
    rd += ["", "Exports: `export.list` (outputs/*, logs/*)."]
    lf(out / "README.md", "\n".join(rd) + "\n")
    lf(out / "package.json", json.dumps({"package": pkg, "generator": "PV_Passivator prepare_bulk_refs.py",
                                         "config_sha256_16": sha16(Path(a.config)), "jobs": list(manifest["jobs"]),
                                         "variants": list(manifest["variants"]), "cost_estimate_node_h": cost,
                                         "submitted": False}, indent=1) + "\n")
    n = sum(len(v) for v in lists.values()) + 1
    print(f"staged {pkg}: {n} runs ({len(lists['relax'])} relax, {len(lists['static'])} static, "
          f"{len(lists['variant'])} variant, 1 smoke) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
