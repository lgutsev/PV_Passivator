"""Analyze a returned bulk-reference package (prepare_bulk_refs.py): formation enthalpies and the O2 correction.

    python examples/scripts/analyze_bulk_refs.py <package dir> --exp exp_enthalpies.json [--fit PbO I2O5] [--json out]

* Energies: energy(sigma->0) of each phase's static (bulk) and of the molecule's relax; per formula unit.
* Elemental references: the phases whose composition is a single element (e.g. fcc Pb, solid I2) and the molecule
  (O2, per atom = E/2).
* dH_f(DFT) per formula unit, compared with experiment (kJ/mol in the JSON, converted to eV):
    {"PbO": {"dHf_kJ_mol": -218.08, "source": "..."}, ...}   every value needs a source; missing ones are refused.
* O2 correction, two options side by side:
    fit      one shift per O atom, least squares over the --fit compounds:  dH_exp = dH_DFT + n_O * delta
    none     delta = 0 (the literature result for SCAN-type functionals; see the package README)
  Compounds not in --fit are held out and reported for both options.
* Checks: the molecule energy against a reference value (--o2-ref, e.g. from an earlier package with the same INCAR),
  and every variant static against its parent (e.g. LREAL), per atom.
Missing or unfinished steps are reported, never filled in.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

KJ_PER_EV = 96.48533212


def energy(outcar: Path) -> float | None:
    e = None
    if outcar.exists():
        for ln in outcar.read_text(errors="replace").splitlines():
            if "energy(sigma->0)" in ln:
                e = float(ln.split("=")[-1])
    return e


def composition(poscar: Path) -> Counter:
    ln = poscar.read_text().splitlines()
    return Counter(dict(zip(ln[5].split(), map(int, ln[6].split()))))


def reduce(c: Counter) -> tuple[Counter, int]:
    from math import gcd
    from functools import reduce as fr
    g = fr(gcd, c.values())
    return Counter({k: v // g for k, v in c.items()}), g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--exp", required=True, help="JSON of experimental dH_f with sources")
    ap.add_argument("--fit", nargs="+", default=["PbO", "I2O5"], help="compounds used to fit the O2 shift")
    ap.add_argument("--o2-ref", type=float, help="reference molecule energy (eV) from an earlier package, same INCAR")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    pkg = Path(a.package)
    m = json.loads((pkg / "manifest.json").read_text())
    exp = json.loads(Path(a.exp).read_text())
    for k, v in exp.items():
        if not v.get("source"):
            raise SystemExit(f"experimental value for {k} has no source")
    rep = {"missing": [], "energies": {}, "per_fu": {}, "checks": {}}

    def step_energy(job, step):
        d = pkg / "outputs" / job / step
        if not (d / "DONE").exists():
            rep["missing"].append(f"{job}/{step}")
            return None
        return energy(d / "OUTCAR")

    comp, e_fu = {}, {}
    for job, j in m["jobs"].items():
        step = "static" if j["role"] == "bulk" else "relax"
        e = step_energy(job, step)
        c = composition(pkg / "inputs" / job / step / "POSCAR")
        red, z = reduce(c)
        comp[job] = red
        if e is not None:
            rep["energies"][job] = e
            e_fu[job] = e / z
    # elemental chemical potentials (eV/atom)
    mu = {}
    for job, red in comp.items():
        if len(red) == 1 and job in e_fu:
            el = next(iter(red))
            mu[el] = e_fu[job] / red[el]
    rep["mu_eV_per_atom"] = mu
    rows = {}
    for job, red in comp.items():
        if len(red) < 2 or job not in e_fu:
            continue
        if any(el not in mu for el in red):
            rep["missing"].append(f"{job}: no elemental reference for {[el for el in red if el not in mu]}")
            continue
        dh = e_fu[job] - sum(n * mu[el] for el, n in red.items())
        rows[job] = {"formula_unit": dict(red), "dHf_DFT_eV": dh, "n_O": red.get("O", 0)}
    # O2 shift fit over the --fit compounds that have experiment
    fit = [k for k in a.fit if k in rows and k in exp]
    num = sum(rows[k]["n_O"] * (exp[k]["dHf_kJ_mol"] / KJ_PER_EV - rows[k]["dHf_DFT_eV"]) for k in fit)
    den = sum(rows[k]["n_O"] ** 2 for k in fit)
    delta = num / den if den else None
    rep["o2_shift"] = {"fit_compounds": fit, "delta_eV_per_O": delta,
                       "delta_eV_per_O2": None if delta is None else 2 * delta}
    for k, r in rows.items():
        if k in exp:
            e_exp = exp[k]["dHf_kJ_mol"] / KJ_PER_EV
            r["dHf_exp_eV"] = e_exp
            r["source"] = exp[k]["source"]
            r["error_none_eV"] = r["dHf_DFT_eV"] - e_exp
            r["error_fit_eV"] = None if delta is None else r["dHf_DFT_eV"] + r["n_O"] * delta - e_exp
            r["role"] = "fit" if k in fit else "held out"
    rep["per_fu"] = rows
    for opt in ("none", "fit"):
        errs = [abs(r[f"error_{opt}_eV"]) for r in rows.values() if r.get(f"error_{opt}_eV") is not None]
        held = [abs(r[f"error_{opt}_eV"]) for r in rows.values() if r.get("role") == "held out"
                and r.get(f"error_{opt}_eV") is not None]
        rep["checks"][f"MAE_{opt}_eV_per_fu"] = sum(errs) / len(errs) if errs else None
        rep["checks"][f"MAE_{opt}_held_out_eV_per_fu"] = sum(held) / len(held) if held else None
    # molecule cross-check and variants
    mol = next((j for j, v in m["jobs"].items() if v["role"] == "molecule"), None)
    if mol and a.o2_ref is not None and mol in rep["energies"]:
        rep["checks"]["molecule_vs_reference_meV"] = 1000 * (rep["energies"][mol] - a.o2_ref)
    for v, info in m.get("variants", {}).items():
        ev, ep = step_energy(v, "static"), rep["energies"].get(info["of"])
        if ev is not None and ep is not None:
            n = sum(composition(pkg / "inputs" / v / "static" / "POSCAR").values())
            rep["checks"][f"{v}_minus_{info['of']}_meV_per_atom"] = 1000 * (ev - ep) / n
    out = json.dumps(rep, indent=1)
    if a.json:
        Path(a.json).write_text(out)
    print(f"missing: {rep['missing'] or 'none'}")
    print(f"O2 shift (fit over {fit}): {delta if delta is None else round(delta, 4)} eV/O")
    for k, r in rows.items():
        line = f"  {k:8s} dHf DFT {r['dHf_DFT_eV']:+.3f} eV/fu"
        if "dHf_exp_eV" in r:
            line += (f" | exp {r['dHf_exp_eV']:+.3f} | err none {r['error_none_eV']:+.3f}"
                     + (f" fit {r['error_fit_eV']:+.3f}" if r['error_fit_eV'] is not None else "") + f" ({r['role']})")
        print(line)
    for k, v in rep["checks"].items():
        print(f"  {k}: {v if v is None else round(v, 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
