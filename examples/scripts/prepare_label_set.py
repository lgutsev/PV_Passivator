"""Prepare one consistent DFT label set for adsorption energies from archived runs (inputs only; never submits).

Every slab and complex is mapped onto ONE fixed cell derived from the bulk lattice (archived ISIF=4 relaxations let
the cells drift apart), then relaxed (ISIF=2) and re-evaluated statically with one profile. Salt references are
isolated ion pairs in a vacuum box. A bulk re-relaxation (ISIF=3) is included to confirm the lattice.

    python examples/scripts/prepare_label_set.py --config data/labels/config.json --out data/labels/package

Config (JSON):
{
  "normal_axis": 0, "normal_length": 70.0, "inplane_reps": [3, 3], "profile_family": "r2scan_d3bj_v1",
  "ref_box": 25.0, "max_strain": 0.03,
  "bulk_dir": "<run folder with CONTCAR>", "slab_dir": "<run folder>", "slab_name": "FAI_Schot",
  "cases": [
    {"name": "PEAAc", "complex_dir": "<run folder>", "ref": {"kind": "file", "path": "<CONTCAR>"}},
    {"name": "PEACl", "complex_dir": "<run folder>", "ref": {"kind": "cation+halide", "halide": "Cl", "dist": 3.1}}
  ]
}
"cation+halide": the passivator cation as relaxed in the complex plus a halide on the most outward N-H direction
at ``dist`` A (a uniform starting geometry; the relaxation sets the real one).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pvpassivator.ledger import Species, Reaction, formula_key  # noqa: E402
from pvpassivator.poses import fragments  # noqa: E402
from pvpassivator.vaspprep import (write_job, kpoints_for, to_common_cell, check_matched_settings,  # noqa: E402
                                   JobSpec)

LATTICE_FRAGMENTS = {"CH5N2"}          # FA+ belongs to the slab


def comp(a: Atoms) -> dict[str, int]:
    c: dict[str, int] = {}
    for s in a.get_chemical_symbols():
        c[s] = c.get(s, 0) + 1
    return c


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def min_distance(a: Atoms) -> float:
    from ase.neighborlist import neighbor_list
    d = neighbor_list("d", a, 1.5)
    return float(d.min()) if len(d) else float("inf")


def min_h_lattice(a: Atoms, cut: float = 3.5) -> float:
    """Closest H to a lattice Pb/I contact (the first thing a bad mapping squeezes)."""
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", a, cut)
    s = np.array(a.get_chemical_symbols())
    m = (s[i] == "H") & np.isin(s[j], ["Pb", "I", "Br", "Cl"])
    return float(d[m].min()) if m.any() else cut


def passivator_cation(cx: Atoms) -> Atoms:
    """The single organic cation that is not a lattice FA+, made whole."""
    cations = []
    for g in fragments(cx):
        syms = [cx[i].symbol for i in g]
        f = formula_key({s: syms.count(s) for s in set(syms)})
        if "N" in syms and "C" in syms and f not in LATTICE_FRAGMENTS and (syms.count("C") >= 2 or "F" in syms):
            cations.append(g)
    if len(cations) != 1:
        raise ValueError(f"expected one passivator cation, found {len(cations)}")
    g = cations[0]
    v = cx.get_distances(g[0], g, mic=True, vector=True)
    return Atoms([cx[i].symbol for i in g], positions=cx.positions[g[0]] + v)


def add_halide(cation: Atoms, halide: str, dist: float) -> Atoms:
    """Halide on the N-H direction that points most away from the cation centroid."""
    sym = np.array(cation.get_chemical_symbols())
    cen = cation.positions.mean(0)
    best = None
    for n in np.flatnonzero(sym == "N"):
        for h in np.flatnonzero(sym == "H"):
            d = cation.positions[h] - cation.positions[n]
            if np.linalg.norm(d) > 1.15:
                continue
            u = d / np.linalg.norm(d)
            x = cation.positions[n] + dist * u
            clear = np.min(np.linalg.norm(cation.positions - x, axis=1))
            score = np.dot(u, (cation.positions[n] - cen) / (np.linalg.norm(cation.positions[n] - cen) + 1e-9))
            if clear > 1.6 and (best is None or score > best[0]):
                best = (score, x)
    if best is None:
        raise ValueError("no N-H direction with room for the halide")
    return cation + Atoms(halide, positions=[best[1]])


def boxed(a: Atoms, box: float) -> Atoms:
    b = a.copy()
    b.pbc = True
    b.cell = np.diag([box, box + 0.5, box + 1.0])
    b.center()
    return b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    ax = cfg["normal_axis"]
    fam = cfg.get("profile_family", "r2scan_d3bj_v1")
    box = cfg.get("ref_box", 25.0)
    if box < 15:
        raise ValueError("reference box must be at least 15 A")

    bulk_path = Path(cfg["bulk_dir"]) / "CONTCAR"
    bulk = read(bulk_path, format="vasp")
    reps = cfg["inplane_reps"]
    inplane = [k for k in range(3) if k != ax]
    cell = np.zeros((3, 3))
    cell[ax, ax] = cfg["normal_length"]
    for k, r in zip(inplane, reps):
        cell[k, k] = r * np.linalg.norm(bulk.cell.array[k])
    if not np.allclose(bulk.cell.angles(), 90, atol=0.5):
        raise ValueError(f"bulk cell is not orthogonal: {bulk.cell.angles()}")

    manifest = {"schema": "label_set/1", "profile_family": fam, "common_cell_A": cell.tolist(),
                "cell_source": {"bulk": str(bulk_path), "sha256_16": sha(bulk_path),
                                "bulk_lengths_A": bulk.cell.lengths().tolist(), "inplane_reps": reps},
                "structures": {}, "jobs": {}, "reactions": [], "notes": []}
    statics: dict[str, JobSpec] = {}

    def job_pair(atoms, name, role, normal_axis, kpoints, prof_relax, prof_static, extra=None):
        r = write_job(atoms, out / name / "relax", f"{fam}/{prof_relax}", role, normal_axis=normal_axis,
                      kpoints=kpoints, overwrite=a.overwrite, overrides=extra)
        s = write_job(atoms, out / name / "static", f"{fam}/{prof_static}", role, normal_axis=normal_axis,
                      kpoints=kpoints, parents=[r.job_id], overwrite=a.overwrite)
        manifest["jobs"][name] = {"relax": f"{name}/relax", "static": f"{name}/static",
                                  "static_geometry_from": f"{name}/relax/CONTCAR", "role": role,
                                  "kpoints": kpoints, "natoms": len(atoms)}
        statics[name] = s
        return s

    # bulk: confirm the lattice that defines the common cell
    kb = kpoints_for(bulk, None)
    rb = write_job(bulk, out / "bulk" / "relax", f"{fam}/relax", "bulk", kpoints=kb, overwrite=a.overwrite,
                   overrides={"ISIF": 3})
    manifest["jobs"]["bulk"] = {"relax": "bulk/relax", "role": "bulk", "kpoints": kb, "natoms": len(bulk),
                                "purpose": "confirm the lattice; if it moves by > 0.5 %, regenerate the set"}

    def load_slab(d: Path, label: str) -> Atoms:
        src = d / "CONTCAR"
        at = read(src, format="vasp")
        m, rep = to_common_cell(at, ax, cell, cfg.get("max_strain", 0.03), cfg.get("max_shear_deg", 7.0))
        manifest["structures"][label] = {"source": str(src), "sha256_16": sha(src), **rep,
                                         "source_angles": at.cell.angles().tolist(),
                                         "min_interatomic_A": min_distance(m)}
        if min_distance(m) < 0.75:
            raise ValueError(f"{label}: atoms closer than 0.75 A after mapping")
        h0, h1 = min_h_lattice(at), min_h_lattice(m)
        manifest["structures"][label].update({"min_H_lattice_source_A": h0, "min_H_lattice_mapped_A": h1})
        if h1 < h0 - 0.15:
            raise ValueError(f"{label}: closest H...Pb/I contact shrank from {h0:.2f} to {h1:.2f} A in the mapping")
        return m

    slab_name = cfg.get("slab_name", "slab")
    slab = load_slab(Path(cfg["slab_dir"]), slab_name)
    kp = kpoints_for(slab, ax)
    job_pair(slab, f"slab_{slab_name}", "bare_slab", ax, kp, "relax", "static")

    for case in cfg["cases"]:
        name = case["name"]
        cx = load_slab(Path(case["complex_dir"]), f"complex_{name}")
        rs = case["ref"]
        if rs["kind"] == "file":
            ref = read(rs["path"], format="vasp")
            ref_src = {"kind": "file", "path": rs["path"], "sha256_16": sha(Path(rs["path"]))}
            from pvpassivator.poses import make_whole
            ref = make_whole(ref)
        elif rs["kind"] == "cation+halide":
            ref = add_halide(passivator_cation(read(Path(case["complex_dir"]) / "CONTCAR", format="vasp")),
                             rs["halide"], rs["dist"])
            ref_src = {"kind": "cation+halide", "cation_from": str(Path(case["complex_dir"]) / "CONTCAR"),
                       "halide": rs["halide"], "dist_A": rs["dist"]}
        else:
            raise ValueError(f"unknown ref kind {rs['kind']}")
        ref = boxed(ref, box)
        rx = Reaction([(1, Species(f"slab_{slab_name}", comp(slab))), (1, Species(f"ref_{name}", comp(ref)))],
                      [(1, Species(f"complex_{name}", comp(cx)))], energy_kind="adsorption_relaxed", n_units=1,
                      label=name)
        rx.assert_balanced()
        job_pair(cx, f"complex_{name}", "complex", ax, kp, "relax", "static")
        job_pair(ref, f"ref_{name}", "reference", None, [1, 1, 1], "molecule_relax", "molecule_static")
        manifest["structures"][f"ref_{name}"] = {**ref_src, "formula": formula_key(comp(ref)),
                                                 "min_interatomic_A": min_distance(ref)}
        manifest["reactions"].append({
            "name": name, "equation": rx.equation(),
            "E_ads": f"E(complex_{name}/static) - E(slab_{slab_name}/static) - E(ref_{name}/static)",
            "sign": "negative = binding", "units": "eV per passivator"})

    problems = check_matched_settings(statics)
    if problems:
        raise SystemExit("static jobs are not comparable:\n  " + "\n  ".join(problems))
    manifest["notes"] += [
        "All slab-type jobs share one cell (common_cell_A) and one k-mesh; references are Gamma-only in vacuum boxes.",
        "Run order: bulk/relax first; if its lattice moves by more than 0.5 %, regenerate. Then every */relax, then "
        "*/static from the relax CONTCAR.",
        "No energy in this set may be combined with archived runs (different cells, k-mesh and ISIF).",
    ]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(manifest['jobs'])} job groups, {len(manifest['reactions'])} reactions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
