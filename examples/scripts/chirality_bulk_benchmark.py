"""Structure/source manifest and static chirality descriptors for the dry (R/S/rac-MBA)2PbI4 benchmark.

    python examples/scripts/chirality_bulk_benchmark.py [--out data/cache/chirality_bench] [--manifest docs/CHIRALITY_MBA2PBI4_MANIFEST.json]

Reads the three COD CIFs in examples/structures/mba2pbi4/ (experimental, H atoms riding), checks composition,
protonation, octahedra, layers and stereocentres, builds the EXACT S mirror by inverting R (cell unchanged), compares
it with the separately refined S crystal, computes the static descriptors of pvpassivator.chirality and writes
POSCARs for a later MD/DFT stage. No calculator is run; nothing here is a relaxed or validated structure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
from ase.io import read, write

from pvpassivator import chirality as ch

ROOT = Path(__file__).resolve().parents[2]
SDIR = ROOT / "examples" / "structures" / "mba2pbi4"

SOURCES = {
    "R": {"file": "R-MBA2PbI4_COD7203475.cif", "cod": 7203475, "expect": "R",
          "citation": "Billing & Lemmerer, CrystEngComm 2006, 8, 686 (doi:10.1039/b606987h)"},
    "S": {"file": "S-MBA2PbI4_COD7203476.cif", "cod": 7203476, "expect": "S",
          "citation": "Billing & Lemmerer, CrystEngComm 2006, 8, 686 (doi:10.1039/b606987h)"},
    "rac": {"file": "rac-MBA2PbI4_COD2201646.cif", "cod": 2201646, "expect": "rac",
            "citation": "Billing, Acta Cryst. E 2002, 58, m669 (doi:10.1107/S1600536802019025)"},
}
CIF_FIELDS = {
    "space_group": r"_symmetry_space_group_name_H-M\s+'([^']+)'",
    "T_K": r"_diffrn_ambient_temperature\s+([\d.]+)",
    "flack": r"_refine_ls_abs_structure_Flack\s+(\S+)",
    "R1_obs": r"_refine_ls_R_factor_gt\s+(\S+)",
    "Z": r"_cell_formula_units_Z\s+(\d+)",
    "abs_config": r"_chemical_absolute_configuration\s+(\S+)",
    "melting_point_C": r"_chemical_melting_point\s+(\S+)",
    "H_treatment": r"_refine_ls_hydrogen_treatment\s+(\S+)",
}


def cif_header(text: str) -> dict:
    out = {}
    for k, pat in CIF_FIELDS.items():
        m = re.search(pat, text)
        out[k] = m.group(1) if m else None
    return out


def match_frac(a, b):
    """Best rigid-translation match of b onto a in fractional space (same species), distances in a's cell.

    Returns per-species RMS / max displacement (A) and the translation. Atom order may differ.
    """
    fa, fb = a.get_scaled_positions(), b.get_scaled_positions()
    sa, sb = np.array(a.get_chemical_symbols()), np.array(b.get_chemical_symbols())
    pb_a, pb_b = np.where(sa == "Pb")[0], np.where(sb == "Pb")[0]
    best = None
    for k in pb_b:
        t = fa[pb_a[0]] - fb[k]
        d_all = {}
        for el in sorted(set(sa)):
            A, B = fa[sa == el], (fb[sb == el] + t)
            diff = A[:, None, :] - B[None, :, :]
            diff -= np.round(diff)
            dist = np.linalg.norm(diff @ a.cell.array, axis=2)
            d_all[el] = dist.min(axis=1)
        worst = max(v.max() for v in d_all.values())
        if best is None or worst < best[0]:
            best = (worst, t, d_all)
    _, t, d_all = best
    return {"translation_frac": np.round(t, 4).tolist(),
            "per_species": {el: {"rms_A": round(float(np.sqrt(np.mean(v ** 2))), 4), "max_A": round(float(v.max()), 4)}
                            for el, v in d_all.items()}}


def describe(atoms, top) -> dict:
    d = ch.frame_descriptors(top, atoms)
    od = ch.octahedral_distortion(top, atoms)
    i, j, dist = __import__("ase.neighborlist", fromlist=["neighbor_list"]).neighbor_list("ijd", atoms, 3.8)
    sym = np.array(atoms.get_chemical_symbols())
    sel = (sym[i] == "Pb") & (sym[j] == "I")
    return {
        "topology": top.summary(),
        "Pb_I_range_A": [round(float(dist[sel].min()), 3), round(float(dist[sel].max()), 3)],
        "Pb_I_Pb_deg": sorted(set(np.round(ch.mxm_angles(top, atoms), 1).tolist())),
        "octahedra": {"delta_d_max": float(max(v["delta_d"] for v in od.values())),
                      "sigma2_deg2_range": [round(min(v["sigma2"] for v in od.values()), 2),
                                            round(max(v["sigma2"] for v in od.values()), 2)]},
        "headgroup_tilt_deg": sorted(set(np.round(ch.headgroup_tilt(top, atoms), 1).tolist())),
        "r_HB_A": sorted(set(np.round(ch.cation_hbond_length(top, atoms), 3).tolist())),
        "hbonds_per_NH_(3.2A,120deg)": round(float(ch.hbond_matrix(top, atoms)[0].sum()
                                                   / sum(len(m.nh_h) for m in top.molecules)), 3),
        "tau_links": {k: sorted(set(np.round(d[f"tau_{k}"], 4).tolist())) for k in ("ax", "eq", "org")},
        "tau_mean": {k: round(d[f"mean_{k}"], 5) for k in ("ax", "eq", "org")},
        "tau_layer_mean": {k: {str(L): round(v, 5) for L, v in d[f"layer_{k}"].items()} for k in ("ax", "eq", "org")},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "cache" / "chirality_bench"))
    ap.add_argument("--manifest", default=str(ROOT / "docs" / "CHIRALITY_MBA2PBI4_MANIFEST.json"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    man = {"schema": "pvpass_chirality_manifest/1",
           "note": "experimental X-ray structures, H atoms riding (geometric); not relaxed, not DFT, not MLIP",
           "structures": {}}
    atoms = {}
    for key, src in SOURCES.items():
        path = SDIR / src["file"]
        raw = path.read_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = read(path)
        atoms[key] = a
        top = ch.build_topology(a)
        st = top.summary()["stereocentres"]
        ok = (st["S"] == 0 if src["expect"] == "R" else st["R"] == 0 if src["expect"] == "S" else st["R"] == st["S"])
        if not ok:
            raise SystemExit(f"{key}: stereocentres {st} do not match the CIF label {src['expect']}")
        nfu = a.get_chemical_symbols().count("Pb")
        man["structures"][key] = {
            "source": f"COD {src['cod']} (https://www.crystallography.net/cod/{src['cod']}.cif)",
            "citation": src["citation"], "repo_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(raw).hexdigest(), **cif_header(raw.decode("utf8", "replace")),
            "cell_abc_angles": np.round(a.cell.cellpar(), 4).tolist(), "n_atoms": len(a),
            "formula_units_in_cell": nfu, "composition_per_fu": "(C8H12N)2PbI4 = C16H24I4N2Pb",
            "protonation": "every N carries 3 H (NH3+); charge balance 2(+1) + PbI4(2-) = 0",
            "stereochemistry_check": st, **describe(a, top),
        }
        write(out / f"POSCAR_{key}_exp", a, format="vasp", direct=True, sort=True)

    # exact mirror of R and comparison with the separately refined S crystal
    mir = ch.invert(atoms["R"])
    tm = ch.build_topology(mir)
    if tm.summary()["stereocentres"]["R"] != 0:
        raise SystemExit("inverted R still has R centres")
    write(out / "POSCAR_S_mirror_of_R", mir, format="vasp", direct=True, sort=True)
    man["structures"]["S_mirror"] = {
        "construction": "inversion of the R CIF through the origin (r -> -r), cell unchanged (right-handed)",
        "parent": "R", **describe(mir, tm),
        "vs_S_crystal": {"cell_diff_A": np.round(atoms["S"].cell.cellpar()[:3] - atoms["R"].cell.cellpar()[:3], 4).tolist(),
                         **match_frac(atoms["S"], mir)},
    }
    Path(args.manifest).write_text(json.dumps(man, indent=1))
    print(json.dumps({k: {kk: v.get(kk) for kk in ("space_group", "T_K", "tau_mean", "stereochemistry_check")}
                      for k, v in man["structures"].items()}, indent=1))
    print("vs S crystal:", json.dumps(man["structures"]["S_mirror"]["vs_S_crystal"]))
    print(f"POSCARs in {out}; manifest {args.manifest}")


if __name__ == "__main__":
    main()
