"""Probe what ACAT does with a halide-perovskite slab, before any search uses it.

Run in an environment that has ACAT (``pip install --no-deps acat ase-ga``; asap3 is not needed for
the paths probed here and is recorded as absent/present).  Every ACAT path is tried separately and
its outcome recorded, because one working argument does not repair the other modules:

  roles        which atoms ACAT's element rule calls "catalyst" vs "adsorbate", against the
               project's own substrate/passivant roles
  sites_*      SlabAdsorptionSites with a CustomSurface built from the slab itself, in three
               substrate representations (all atoms, Pb+I only, Pb only)
  coverage_*   SlabAdsorbateCoverage with and without explicit adsorbate indices
  operators    whether the GA adsorbate operators accept the project's molecule
  comparators  whether the graph comparator separates / merges known fixtures

Writes one JSON report.  Nothing here is a physical result.

    python examples/scripts/acat_surface_probe.py SLAB_CONTCAR --complex COMPLEX_CONTCAR --out report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pvpassivator.surface import detect_normal, load_surface, find_sites  # noqa: E402
from pvpassivator.adsearch import normal_to_z, assign_roles_from_reference, ROLE_NAMES  # noqa: E402


def attempt(fn):
    t = time.time()
    try:
        out = fn()
        return {"ok": True, "seconds": round(time.time() - t, 2), **(out or {})}
    except Exception as e:  # recorded, never hidden
        return {"ok": False, "seconds": round(time.time() - t, 2), "error": f"{type(e).__name__}: {e}"[:400],
                "where": traceback.format_exc().strip().splitlines()[-3:][0].strip()[:200]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("slab")
    ap.add_argument("--complex", dest="complex_", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", default=None, help="internal: run one step and print its JSON")
    ap.add_argument("--step-timeout", type=float, default=300.0, help="seconds allowed per ACAT step")
    a = ap.parse_args(argv)
    if a.step is None:
        return _drive(a)

    import ase
    import acat
    from acat.settings import CustomSurface, adsorbate_elements
    from acat.adsorption_sites import SlabAdsorptionSites
    from acat.adsorbate_coverage import SlabAdsorbateCoverage
    rep = {"schema": "acat_probe/1", "versions": {"acat": getattr(acat, "__version__", "?"), "ase": ase.__version__,
                                                  "python": sys.version.split()[0]},
           "acat_adsorbate_elements": list(adsorbate_elements)}
    try:
        import asap3  # noqa: F401
        rep["versions"]["asap3"] = "present"
    except Exception as e:
        rep["versions"]["asap3"] = f"absent ({type(e).__name__})"
    try:
        import ase_ga
        rep["versions"]["ase_ga"] = getattr(ase_ga, "__version__", "present")
    except Exception as e:
        rep["versions"]["ase_ga"] = f"absent ({type(e).__name__})"

    slab0, rec = load_surface(a.slab)
    slab = normal_to_z(slab0, rec.normal_axis)
    syms = np.array(slab.get_chemical_symbols())
    rep["slab"] = {"path": str(a.slab), "sha256_16": rec.origin["sha256"][:16], "n_atoms": len(slab),
                   "formula": rec.origin["formula"], "normal_axis_source": rec.normal_axis,
                   "terminations": {f: v["termination"] for f, v in rec.faces.items()}}
    native = [s for s in find_sites(slab0, rec, "top") if s.kind == "undercoordinated_Pb"]
    native_xy = normal_to_z(slab0, rec.normal_axis).positions[[s.atom_index for s in native]]
    rep["native_sites"] = {"n_undercoordinated_Pb_top": len(native)}

    # ---- roles: ACAT's element rule on a slab that is 100 % substrate
    acat_ads = np.isin(syms, adsorbate_elements)
    rep["roles_bare_slab"] = {
        "project_substrate_atoms": len(slab),
        "acat_catalyst_atoms": int((~acat_ads).sum()),
        "acat_adsorbate_atoms": int(acat_ads.sum()),
        "misassigned_by_element": {s: int(((syms == s) & acat_ads).sum()) for s in sorted(set(syms[acat_ads]))},
        "note": "every C/N/H here belongs to lattice formamidinium, i.e. substrate"}

    def sites_for(label, keep):
        sub = slab[np.where(keep)[0]]

        def run():
            cs = CustomSurface(sub, n_layers=None)
            sas = SlabAdsorptionSites(sub, surface=cs, both_sides=False)
            sl = sas.get_sites()
            kinds = {}
            for s in sl:
                kinds[s["site"]] = kinds.get(s["site"], 0) + 1
            top = np.array([s["position"] for s in sl if s["site"] == "ontop"]) if sl else np.zeros((0, 3))
            # does each native undercoordinated Pb have an ACAT ontop site within 0.5 A (minimum image, in-plane)?
            hit = 0
            if len(top):
                inv = np.linalg.inv(slab.cell.array)
                for p in native_xy:
                    d = (top - p) @ inv
                    d[:, :2] -= np.round(d[:, :2])
                    hit += int(np.linalg.norm(d @ slab.cell.array, axis=1).min() < 0.5)
            return {"n_layers_detected": cs.n_layers, "n_surface_atoms": len(sas.surf_ids),
                    "surface_elements": {s: int(sum(sub[i].symbol == s for i in sas.surf_ids))
                                         for s in sorted({sub[i].symbol for i in sas.surf_ids})},
                    "n_sites": len(sl), "site_kinds": kinds,
                    "native_Pb_sites_with_acat_ontop": hit, "native_Pb_sites": len(native)}
        rep[f"sites_{label}"] = {"n_atoms_given": int(keep.sum()), **attempt(run)}

    if a.step == "sites_all_atoms":
        sites_for("all_atoms", np.ones(len(slab), bool))
    if a.step == "sites_Pb_I_only":
        sites_for("Pb_I_only", np.isin(syms, ["Pb", "I"]))
    if a.step == "sites_Pb_only":
        sites_for("Pb_only", syms == "Pb")

    # ---- coverage on the bare slab: nothing should be reported as an adsorbate
    def cov_default():
        cs = CustomSurface(slab[np.where(np.isin(syms, ["Pb", "I"]))[0]])
        sac = SlabAdsorbateCoverage(slab, surface=cs)
        occ = sac.get_sites(occupied=True)
        return {"occupied_sites_reported": len(occ), "expected": 0,
                "fragments_reported": sorted({s.get("fragment", s.get("adsorbate", "?")) for s in occ})[:8]}
    if a.step == "coverage_bare_default":
        rep["coverage_bare_default"] = attempt(cov_default)

    if a.complex_:
        cx0 = read(a.complex_)
        role = assign_roles_from_reference(cx0, slab0)
        cx = normal_to_z(cx0, detect_normal(cx0)[0])
        ads = np.where(role != 0)[0].tolist()
        rep["complex"] = {"path": str(a.complex_), "n_atoms": len(cx),
                          "project_roles": {ROLE_NAMES[k]: int((role == k).sum()) for k in sorted(set(role.tolist()))}}
        csym = np.array(cx.get_chemical_symbols())

        def cov_explicit():
            cs = CustomSurface(cx[np.where(np.isin(csym, ["Pb", "I"]) & (role == 0))[0]])
            sac = SlabAdsorbateCoverage(cx, surface=cs, adsorbate_indices=ads)
            occ = sac.get_sites(occupied=True)
            return {"occupied_sites_reported": len(occ), "adsorbate_atoms_given": len(ads),
                    "fragments_reported": sorted({str(s.get("fragment", "?")) for s in occ})[:8],
                    "n_fragments": len(getattr(sac, "ads_list", []) or [])}
        if a.step == "coverage_complex_explicit_indices":
            rep["coverage_complex_explicit_indices"] = attempt(cov_explicit)

        def operators():
            from acat.ga.adsorbate_operators import MoveAdsorbate
            from acat.settings import adsorbate_molecule
            out = {}
            for name in ("C3H6N6", "melamine", "NH3"):
                try:
                    m = adsorbate_molecule(name)
                    out[name] = None if m is None else m.get_chemical_formula()
                except Exception as e:
                    out[name] = f"{type(e).__name__}: {e}"[:120]
            op = MoveAdsorbate(["NH3"], catalyst_indices=np.where(role == 0)[0].tolist())
            return {"adsorbate_molecule_lookup": out, "operator_constructed": type(op).__name__,
                    "note": "operators take species *names*; an arbitrary ase.Atoms molecule cannot be passed"}
        if a.step == "operators":
            rep["operators"] = attempt(operators)

        def comparators():
            from acat.ga.graph_comparators import WLGraphComparator
            cmp_ = WLGraphComparator(hmax=2, dx=0.5)
            shifted = cx.copy()
            shifted.positions += 0.37 * cx.cell.array[0]
            shifted.wrap()
            return {"identical_vs_translated_wrapped_copy_merged": bool(cmp_.looks_like(cx, shifted))}
        if a.step == "comparators":
            rep["comparators"] = attempt(comparators)

    print("PROBE_JSON " + json.dumps(rep))


STEPS = ["roles", "sites_Pb_only", "sites_Pb_I_only", "sites_all_atoms", "coverage_bare_default",
         "coverage_complex_explicit_indices", "operators", "comparators"]


def _drive(a):
    """Run every step in its own process with a timeout and merge the reports."""
    import subprocess
    rep = {}
    for step in STEPS:
        cmd = [sys.executable, __file__, a.slab, "--out", a.out, "--step", step] +               (["--complex", a.complex_] if a.complex_ else [])
        t = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.step_timeout)
            line = next((x for x in p.stdout.splitlines() if x.startswith("PROBE_JSON ")), None)
            if line is None:
                rep[step] = {"ok": False, "error": (p.stderr or p.stdout).strip().splitlines()[-1][:300]}
            else:
                rep.update(json.loads(line[len("PROBE_JSON "):]))
        except subprocess.TimeoutExpired:
            rep[step] = {"ok": False, "error": f"timeout: no result within {a.step_timeout:.0f} s",
                         "seconds": round(time.time() - t, 1)}
    rep["step_timeout_s"] = a.step_timeout
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
