"""Structures for the first DFT audit of an MLIP on the adsorption-search pilot (tranche V1).

    python examples/scripts/adsearch_v1_pool.py CONFIG --bulk-dir DIR --out OUT --model FOUNDATION.model
           [--normal-length 70] [--n-pool 11] [--n-asbuilt 3] [--seed 0] [--legacy NAME=CONTCAR ...]
           [--legacy-slab CONTCAR] [--steps 150] [--device cuda]

Run in the MLIP environment (``maceh``).  Every structure is built on the pilot slab mapped onto the label sets'
common cell (``vaspprep.to_common_cell``), so the DFT single points can be run at EXACTLY the geometry the model saw.
For each structure the model's energy and forces at the final geometry are stored with it (``model_energy``,
``model_forces``); the DFT single points are compared with those numbers, structure by structure.

Groups (all relaxed with one protocol: FIRE, face-wide mobile set, the rest fixed at the slab):
  one_<mode>     one molecule, one start per binding mode                 (single-molecule sanity stage)
  legacy_<name>  archived multi-molecule arrangements, each molecule transplanted in the frame of its nearest Pb
  pool_NN        the first ``--n-pool`` valid, distinct random candidates of the search's own generator.  ALL of them
                 are kept: an unselected sample, so the error estimate is not biased toward what the model likes
  asbuilt_NN     the unrelaxed starting geometry of the first ``--n-asbuilt`` pool members (large forces: a force test
                 away from the model's minima); model single point only

Resumable: a structure whose final file exists is skipped.  Nothing here is a physical result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import adsearch_pilot as P  # noqa: E402
from pvpassivator import adsearch as A  # noqa: E402
from pvpassivator.poses import make_whole, min_image_distances, classify_final_state, fragments  # noqa: E402
from pvpassivator.surface import load_surface, sha256_file  # noqa: E402
from pvpassivator.vaspprep import to_common_cell  # noqa: E402


def common_cell(bulk_dir: Path, ax: int, normal_length: float, reps) -> np.ndarray:
    bulk = read(bulk_dir / "CONTCAR", format="vasp")
    cell = np.zeros((3, 3))
    cell[ax, ax] = normal_length
    for k, r in zip([k for k in range(3) if k != ax], reps):
        cell[k, k] = r * np.linalg.norm(bulk.cell.array[k])
    return cell


def transplant(cx, ref_slab, space: A.SearchSpace) -> list:
    """Molecules of an archived complex, each moved rigidly so that it keeps its position relative to its nearest Pb
    (the k-th Pb of the archive is the k-th Pb of the pilot slab; checked through in-plane fractional coordinates)."""
    role = A.assign_roles_from_reference(cx, ref_slab)
    whole = make_whole(cx)
    sub = space.substrate
    pb_old = np.where(np.array(cx.get_chemical_symbols()) == "Pb")[0]
    pb_new = np.where(np.array(sub.get_chemical_symbols()) == "Pb")[0]
    if len(pb_old) != len(pb_new):
        raise ValueError("archive and pilot slab have different Pb counts")
    inpl = [k for k in range(3) if k != space.rec.normal_axis]
    d = cx.get_scaled_positions()[pb_old][:, inpl] - sub.get_scaled_positions()[pb_new][:, inpl]
    d -= np.round(d)
    if np.abs(d).max() > 0.05:
        raise ValueError("Pb order differs between the archive and the pilot slab")
    ads = np.where(role != 0)[0]
    mols = []
    for g in fragments(whole[ads], pbc=False):
        idx = ads[g]
        m = whole[idx]
        D = min_image_distances(cx.cell.array, cx.pbc, m.positions, cx.positions[pb_old])
        k = int(np.unravel_index(D.argmin(), D.shape)[1])
        c = m.positions.mean(axis=0) - cx.positions[pb_old[k]]
        shift = -np.round(c @ np.linalg.inv(cx.cell.array)) @ cx.cell.array      # molecule image nearest that Pb
        m.positions += shift - cx.positions[pb_old[k]] + sub.positions[pb_new[k]]
        mols.append(m)
    return mols


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"), ap.add_argument("--bulk-dir", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True), ap.add_argument("--device", default="cuda")
    ap.add_argument("--normal-length", type=float, default=70.0), ap.add_argument("--reps", default="3,3")
    ap.add_argument("--n-pool", type=int, default=11), ap.add_argument("--n-asbuilt", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0), ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--legacy", action="append", default=[], help="NAME=path/to/CONTCAR of an archived complex")
    ap.add_argument("--legacy-slab", default=None, help="bare slab the archived complexes were built on")
    ap.add_argument("--dry-run", action="store_true", help="build everything, evaluate nothing (no model needed)")
    ap.add_argument("--prerelax-steps", type=int, default=400,
                    help="relax the bare slab's face region with the model first (0 = off); see the docstring")
    ap.add_argument("--prerelax-fmax", type=float, default=0.05)
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())

    # 1. the pilot slab on the common cell
    src = P._resolve(cfg["slab"]["path"])
    slab0, rec0 = load_surface(src)
    cell = common_cell(P._resolve(a.bulk_dir), rec0.normal_axis, a.normal_length, [int(x) for x in a.reps.split(",")])
    mapped, rep = to_common_cell(slab0, rec0.normal_axis, cell)
    (out / "slab_common").mkdir(exist_ok=True)
    write(out / "slab_common" / "POSCAR", mapped, format="vasp", direct=True, sort=False)
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["slab"]["path"] = str((out / "slab_common" / "POSCAR").resolve())
    two, prov = P.load_space(cfg2)
    calc, pre = None, None
    rx = cfg.get("benchmark", {}).get("relax", {})
    pre_file = out / "slab_common" / "POSCAR_prerelaxed"
    if a.prerelax_steps and not a.dry_run:
        # The model's own minimum for the bare face differs from the DFT slab, so without this every candidate
        # relaxation would spend most of its steps re-relaxing the slab. Done once, with the same face-wide mobile
        # set; the candidates are then built on the result and the far field stays at the mapped DFT slab.
        from local_settle import make_calc
        calc = make_calc(Path(a.model), a.device)
        if pre_file.exists() and (out / "slab_common" / "prerelax.json").exists():
            pre = json.loads((out / "slab_common" / "prerelax.json").read_text())
        else:
            ev0 = A.AseRelaxEvaluator(calc, fmax=a.prerelax_fmax, steps=a.prerelax_steps, radius=rx.get("radius_A", 8.0),
                                      mobile="face", space=two)
            e0 = A.single_point(calc, two.substrate).info["model_energy"]
            res = ev0(two.substrate, ())
            move = np.linalg.norm(res.atoms.positions - two.substrate.positions, axis=1)
            pre = {"steps": res.info["steps"], "converged": res.converged, "fmax_target": a.prerelax_fmax,
                   "fmax_end": res.info["fmax_end"], "n_mobile": res.info["n_mobile"], "n_calls": res.n_calls,
                   "energy_change_eV": round(res.energy - e0, 4), "max_move_A": round(float(move.max()), 3),
                   "mean_move_mobile_A": round(float(move[move > 0].mean()), 3), "seconds": res.info["seconds"]}
            plain = res.atoms.copy()
            for k in list(plain.arrays):
                if k not in ("numbers", "positions"):
                    del plain.arrays[k]
            write(pre_file, plain, format="vasp", direct=True, sort=False)
            (out / "slab_common" / "prerelax.json").write_text(json.dumps(pre, indent=1))
            print("slab pre-relaxation:", json.dumps(pre), flush=True)
        cfg2["slab"]["path"] = str(pre_file.resolve())
        two, prov = P.load_space(cfg2)
        prov["slab_prerelax"] = {**pre, "file": "slab_common/POSCAR_prerelaxed",
                                 "note": "face region relaxed with the audited model; far field = mapped DFT slab"}
    one, _ = P.load_space(cfg2, n_molecules=1)
    prov["slab_mapping"] = {"source": str(src), "source_sha256_16": sha256_file(src)[:16], "common_cell_A": cell.tolist(),
                            **rep}
    write(out / "slab_common" / "slab.extxyz", two.substrate, format="extxyz")

    # 2. what to evaluate
    jobs = []           # (name, group, space, atoms, genes, relax?)
    centre = one.sites[len(one.sites) // 2].site_id
    for mode in one.spec.modes:
        g = (A.Gene(centre, mode.mode_id, "c0", 0.0, 0.0),)
        atoms, reasons, _ = one.build(g)
        if reasons:
            raise SystemExit(f"single-molecule start {mode.mode_id} is invalid: {reasons}")
        jobs.append((f"one_{mode.mode_id}", "single", one, atoms, g, True))
    if a.legacy:
        ref = read(P._resolve(a.legacy_slab))
        for item in a.legacy:
            name, path = item.split("=", 1)
            mols = transplant(read(P._resolve(path)), ref, two)
            sp = two if len(mols) == two.n_molecules else A.SearchSpace(
                two.substrate, two.rec, two.sites, two.spec, len(mols), face=two.face)
            jobs.append((f"legacy_{name}", "legacy", sp, sp.adopt(mols), None, True))
    rng = np.random.default_rng([a.seed, 0])
    seen, pool, tries, rejected = set(), [], 0, {}
    while len(pool) < a.n_pool and tries < 100 * a.n_pool:
        tries += 1
        g = two.random_genes(rng)
        atoms, reasons, _ = two.build(g)
        key = two.canonical_key(g)
        if reasons or key in seen:
            for r in reasons or ["duplicate_initial"]:
                rejected[r] = rejected.get(r, 0) + 1
            continue
        seen.add(key)
        pool.append((g, atoms))
    for k, (g, atoms) in enumerate(pool):
        jobs.append((f"pool_{k:02d}", "pool", two, atoms, g, True))
    for k, (g, atoms) in enumerate(pool[:a.n_asbuilt]):
        jobs.append((f"asbuilt_{k:02d}", "asbuilt", two, atoms, g, False))

    summary = {"schema": "adsearch_v1_pool/1", "provenance": prov, "pool_generation": {"seed": a.seed, "attempts": tries,
               "rejections": rejected}, "structures": []}
    if a.dry_run:
        for name, group, sp, atoms, g, relax in jobs:
            summary["structures"].append({"name": name, "group": group, "n_atoms": len(atoms),
                                          "formula": atoms.get_chemical_formula(), "initial_hash": A.structure_hash(atoms)})
        (out / "pool_dry_run.json").write_text(json.dumps(summary, indent=1, default=A._json))
        print(f"dry run: {len(jobs)} structures, slab strain {rep.get('inplane_strain')}")
        return 0

    from local_settle import make_calc
    import hashlib
    calc = calc or make_calc(Path(a.model), a.device)
    proto = None
    molecule = A.single_point(calc, _boxed(two.spec.passivator.atoms, 25.0))
    write(out / "structures" / "ref_molecule.extxyz", molecule, format="extxyz")
    slab_sp = A.single_point(calc, two.substrate)
    write(out / "structures" / "slab.extxyz", slab_sp, format="extxyz")
    for name, group, sp, atoms, g, relax in jobs:
        f = out / "structures" / f"{name}.extxyz"
        recf = out / "structures" / f"{name}.json"
        if f.exists() and recf.exists():
            summary["structures"].append(json.loads(recf.read_text()))
            continue
        t0 = time.time()
        rec = {"name": name, "group": group, "genes": None if g is None else [A.asdict(x) for x in g],
               "n_atoms": len(atoms), "formula": atoms.get_chemical_formula(), "initial_hash": A.structure_hash(atoms)}
        if relax:
            ev = A.AseRelaxEvaluator(calc, fmax=rx.get("fmax_eV_A", 0.1), steps=a.steps, radius=rx.get("radius_A", 8.0),
                                     mobile="face", space=sp)
            proto = ev.protocol
            write(out / "structures" / f"{name}_initial.extxyz", atoms, format="extxyz")
            res = ev(atoms, g)
            final = res.atoms
            ok, why = A.substrate_integrity(sp.substrate, final, rec=sp.rec)
            rec.update(relaxed=True, converged=res.converged, n_calls=res.n_calls, **res.info, substrate_ok=ok,
                       substrate_note=why, final_state=classify_final_state(atoms, final, sp.rec)["final_state_class"])
        else:
            final = A.single_point(calc, atoms)
            rec.update(relaxed=False, n_calls=1)
        syms = np.array(final.get_chemical_symbols())
        ads = final.arrays["mol_id"] > 0
        dn = min_image_distances(final.cell.array, final.pbc, final.positions[ads & (syms == "N")],
                                 final.positions[syms == "Pb"]).min(axis=1)
        fm = np.linalg.norm(final.arrays["model_forces"], axis=1)
        rec.update(model_energy_eV=final.info["model_energy"], model_fmax_all_eV_A=round(float(fm.max()), 4),
                   N_Pb_min_A=round(float(dn.min()), 3), n_N_within_3p2_A_of_Pb=int((dn < 3.2).sum()),
                   final_hash=A.structure_hash(final), file=f"structures/{name}.extxyz",
                   seconds=round(time.time() - t0, 1))
        write(f, final, format="extxyz")
        recf.write_text(json.dumps(rec, indent=1, default=A._json))
        summary["structures"].append(rec)
        print(json.dumps({k: rec[k] for k in ("name", "group", "model_energy_eV", "n_calls", "N_Pb_min_A", "seconds")
                          } | {k: rec.get(k) for k in ("converged", "final_state", "substrate_ok")}), flush=True)
    summary["provenance"]["evaluator"] = {
        "kind": "mace+d3", "model": str(a.model), "model_sha256_16": hashlib.sha256(Path(a.model).read_bytes()).hexdigest()[:16],
        "device": a.device, "d3": "BJ, xc=r2scan (torch-dftd)", "protocol": proto,
        "status": "foundation model, NOT validated for this interface; these numbers are what the DFT audit tests"}
    summary["references"] = {"slab": {"file": "structures/slab.extxyz", "model_energy_eV": slab_sp.info["model_energy"]},
                             "molecule": {"file": "structures/ref_molecule.extxyz",
                                          "model_energy_eV": molecule.info["model_energy"]}}
    (out / "pool.json").write_text(json.dumps(summary, indent=1, default=A._json))
    print(f"{len(summary['structures'])} structures -> {out}")
    return 0


def _boxed(mol, box: float):
    m = mol.copy()
    m.set_cell([box, box, box])
    m.pbc = True
    m.center()
    return m


if __name__ == "__main__":
    raise SystemExit(main())
