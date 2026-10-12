"""Read a finished label set (prepare_label_set.py / stage_label_set.py layout) into reaction energies.

Layout: ``manifest.json`` at the package root, inputs in ``inputs/<job>/<step>/``, results in
``outputs/<job>/<step>/`` with OUTCAR and a ``DONE`` marker. Only steps with DONE and an accepted OUTCAR are used;
anything else is reported, never filled in. Reactions are rebuilt from the job compositions and must balance.
"""
from __future__ import annotations

from pathlib import Path
import json

from .ledger import Species, Reaction
from .vasp_io import parse_run

K_TOL = 0.02          # eV, accepted |dk(complex) - dk(slab)| for a Gamma-only set


def _composition(poscar: Path) -> dict[str, int]:
    lines = poscar.read_text(errors="replace").splitlines()
    return {s: int(n) for s, n in zip(lines[5].split(), lines[6].split())}


def read_step(root: Path, job: str, step: str = "static", require_relax: bool = False) -> dict:
    out = root / "outputs" / job / step
    rec = {"job": job, "step": step, "energy_eV": None, "problems": []}
    if not (out / "DONE").exists():
        rec["problems"].append("no DONE marker")
        return rec
    run = parse_run(out)
    rec["problems"] += run.acceptance(require_relax=require_relax)
    rec["energy_eV"] = run.energy_sigma0
    fp = run.fingerprint()
    fp.pop("KPOINTS", None)
    fp.pop("POTCARS", None)
    rec["fingerprint"] = json.dumps(fp, sort_keys=True)          # Hamiltonian settings only
    rec["potcars"] = dict(zip(run.species, run.potcars))          # element -> TITEL
    rec["kpoints"] = run.kpoints
    return rec


def analyze(root: str | Path, k_tol: float = K_TOL) -> dict:
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    jobs = m["jobs"]
    statics = {j: read_step(root, j, "static") for j in jobs if j != "bulk" and not j.startswith("kcheck_")}
    relaxes = {j: read_step(root, j, "relax", require_relax=True) for j in jobs
               if j != "bulk" and not j.startswith("kcheck_")}
    ok = {j: r["energy_eV"] for j, r in statics.items() if r["energy_eV"] is not None and not r["problems"]
          and not relaxes[j]["problems"]}
    # comparable statics: identical Hamiltonian settings, and each element on the same POTCAR everywhere
    fps = {r["fingerprint"] for j, r in statics.items() if j in ok}
    pot: dict[str, set] = {}
    for j, r in statics.items():
        if j in ok:
            for el, t in r["potcars"].items():
                pot.setdefault(el, set()).add(t)
    issues = []
    if len(fps) > 1:
        issues.append(f"{len(fps)} different static Hamiltonian settings")
    issues += [f"{el} uses {len(t)} different POTCARs" for el, t in pot.items() if len(t) > 1]
    settings_problem = "; ".join(issues) or None

    reactions = []
    for r in m["reactions"]:
        lhs, rhs = [side.split(" + ") for side in r["equation"].split(" -> ")]
        names = [n.strip() for n in lhs + rhs]
        missing = [n for n in names if n not in ok]
        row = {"name": r["name"], "equation": r["equation"], "E_ads_eV": None, "missing": missing}
        if not missing:
            sp = {n: Species(n, _composition(root / "inputs" / n / "static" / "POSCAR")) for n in names}
            rx = Reaction([(1, sp[n.strip()]) for n in lhs], [(1, sp[n.strip()]) for n in rhs],
                          energy_kind="adsorption_relaxed", n_units=1, label=r["name"])
            row["E_ads_eV"] = rx.delta_e({n: ok[n] for n in names})["delta_E_per_unit_eV"]
        reactions.append(row)

    kc = None
    if "kpoint_check" in m:
        dk = {}
        for name in m["kpoint_check"]["jobs"]:
            dense = read_step(root, f"kcheck_{name}", "static")
            base = ok.get(name)
            dk[name] = (dense["energy_eV"] - base) if (dense["energy_eV"] is not None and not dense["problems"]
                                                       and base is not None) else None
        slab = next((n for n in dk if jobs[n]["role"] == "bare_slab"), None)
        verdicts = {}
        for n, v in dk.items():
            if n == slab:
                continue
            if v is None or dk.get(slab) is None:
                verdicts[n] = None
            else:
                verdicts[n] = {"k_error_eV": v - dk[slab], "pass": abs(v - dk[slab]) < k_tol}
        kc = {"dk_eV": dk, "verdicts": verdicts, "tolerance_eV": k_tol,
              "pass": bool(verdicts) and all(x is not None and x["pass"] for x in verdicts.values())}

    problems = {f"{j}/{s}": r["problems"] for d, s in ((statics, "static"), (relaxes, "relax"))
                for j, r in d.items() if r["problems"]}
    return {"package": root.name, "reactions": reactions, "kpoint_check": kc, "settings_problem": settings_problem,
            "problems": problems, "n_static_ok": len(ok), "n_static": len(statics)}
