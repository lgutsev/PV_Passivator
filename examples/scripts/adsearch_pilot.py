"""Adsorption-arrangement search pilot: site/role demonstration, candidate set, equal-budget benchmark.

    python examples/scripts/adsearch_pilot.py demo  CONFIG --out DIR [--n-candidates 12] [--seed 0]
    python examples/scripts/adsearch_pilot.py bench CONFIG --out DIR --evaluator synthetic|mace
           [--budget 200] [--seeds 0,1,2] [--n-init 20] [--model FOUNDATION.model] [--device cuda]

``demo`` writes the labelled bare slab (roles, site markers), the single-molecule sanity table (every
site x binding mode x orientation, with rejection reasons) and a small valid fixed-composition
candidate set with a manifest.  It evaluates nothing.

``bench`` runs both arms for every seed under one evaluator and one accounting, and writes
``manifest.json`` + ``summary.json``.  ``--evaluator synthetic`` is a SOFTWARE CHECK of the loop and
the bookkeeping (made-up score, no relaxation); its numbers are not search-performance evidence.
``--evaluator mace`` relaxes with the foundation MACE + D3(BJ) calculator of ``local_settle.py`` (run
in the ``maceh`` environment); that model is not validated for this interface, so its energies are
throughput/plumbing data until the DFT validation in docs/acat_feasibility.md passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from pvpassivator.surface import load_surface, find_sites, sha256_file  # noqa: E402
from pvpassivator.poses import Passivator, Component, make_whole, tag_substrate, fragments  # noqa: E402
from pvpassivator import adsearch as A  # noqa: E402


def _resolve(p: str) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    for base in (ROOT, ROOT.parent / "PV_Passivator"):      # worktrees share the main checkout's data/cache
        if (base / q).exists():
            return base / q
    return ROOT / q


def _anchor(mol, sel) -> int:
    if sel == "centroid":
        return -1
    if isinstance(sel, int):
        return sel
    syms = mol.get_chemical_symbols()
    nb = {i: [] for i in range(len(mol))}
    for p, q in A.bond_list(mol):
        nb[p].append(q)
        nb[q].append(p)
    hits = [i for i in range(len(mol)) if syms[i] == sel["element"]
            and ("bonded_H" not in sel or sum(syms[j] == "H" for j in nb[i]) == sel["bonded_H"])]
    if not hits:
        raise ValueError(f"no atom matches anchor selector {sel}")
    return hits[0]


def load_space(cfg: dict, n_molecules: int | None = None) -> tuple[A.SearchSpace, dict]:
    slab_path = _resolve(cfg["slab"]["path"])
    slab, rec = load_surface(slab_path, surface_id=cfg["name"] + "_slab")
    m = cfg["molecule"]
    if "from_complex" in m:
        cx = read(_resolve(m["from_complex"]))
        ref = read(_resolve(m["from_complex_slab"]))
        role = A.assign_roles_from_reference(cx, ref)
        mol = make_whole(cx)[np.where(role != 0)[0]]
        src = {"from_complex": str(_resolve(m["from_complex"])), "sha256_16": sha256_file(_resolve(m["from_complex"]))[:16]}
    else:
        mol = make_whole(read(_resolve(m["file"])))
        src = {"file": str(_resolve(m["file"])), "sha256_16": sha256_file(_resolve(m["file"]))[:16]}
    mol.pbc = False
    mol.center(vacuum=6.0)
    comps = []
    charges = m.get("component_charges", {})
    for g in fragments(mol, pbc=False):
        f = mol[g].get_chemical_formula(mode="hill")
        q = charges.get(f, 0)
        comps.append(Component(f + ("+" * q if q > 0 else "-" * -q), g, q, f))
    if sum(c.formal_charge for c in comps) != m["total_charge"]:
        raise ValueError("component charges do not sum to the declared total charge")
    pas = Passivator(m["name"], mol, m["total_charge"], m["multiplicity"], comps, m["chemical_form"],
                     m.get("note", ""), source=json.dumps(src))
    modes = [A.BindingMode(mode_id=d["mode_id"], anchor=_anchor(mol, d["anchor"]), height=d["height"],
                           up=d.get("up", "anchor_to_centroid"), tilts=tuple(float(t) for t in d.get("tilts", (0, 15, 30))),
                           anchor_window=tuple(d.get("anchor_window", (2.0, 3.6))),
                           azimuth_period=float(d.get("azimuth_period", 360)),
                           second_anchor=None if "second_anchor" not in d else _anchor(mol, d["second_anchor"]),
                           second_window=tuple(d.get("second_window", (2.0, 3.6))),
                           description=d.get("description", "")) for d in cfg["modes"]]
    spec = A.MoleculeSpec(pas, modes, source=json.dumps(src))
    face = cfg["slab"].get("face", "top")
    sites = [s for s in find_sites(slab, rec, face) if s.kind == cfg["slab"].get("site_kind", "undercoordinated_Pb")]
    space = A.SearchSpace(tag_substrate(slab), rec, sites, spec, n_molecules or cfg["n_molecules"], face=face,
                          azimuth_step=float(cfg.get("azimuth_step", 30)), symmetry=cfg.get("symmetry", "site_lattice"))
    prov = {"config": cfg["name"], "slab": {"path": str(slab_path), "sha256_16": rec.origin["sha256"][:16],
                                            "formal_charge": rec.formal_charge, "formal_units": rec.formal_units,
                                            "vacuum_A": round(rec.vacuum, 2), "thickness_A": round(rec.slab_thickness, 2)},
            "molecule": {**src, "formula": mol.get_chemical_formula(), "total_charge": m["total_charge"],
                         "multiplicity": m["multiplicity"], "chemical_form": m["chemical_form"]},
            "code_commit": _commit()}
    return space, prov


def _commit() -> str:
    try:
        c = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return c + ("+uncommitted" if d else "")
    except Exception:
        return "unknown"


def demo(cfg, out: Path, n_candidates: int, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    space, prov = load_space(cfg)
    sub = space.substrate
    # 1. labelled bare slab: roles + one dummy 'X' marker 2 A above every site (inspection only)
    lab = sub.copy()
    for s in space.sites:
        from ase import Atom
        lab.append(Atom("X", np.asarray(s.position) + 2.0 * space.n_out))
    for key, fill in (("role", -1), ("mol_id", 0), ("atom_id", -1), ("fragment_id", -1), ("origin", 0)):
        arr = lab.arrays[key].copy()
        arr[len(sub):] = fill
        lab.set_array(key, arr)
    write(out / "bare_slab_labelled.extxyz", lab, format="extxyz")
    ok, why = A.substrate_integrity(sub, sub, max_move=0.0, rec=space.rec)
    # 2. single-molecule sanity stage: every site x mode x azimuth x tilt
    one, _ = load_space(cfg, n_molecules=1)
    table = {}
    for mode in one.spec.modes:
        naz = max(1, int(round(mode.azimuth_period / one.azimuth_step)))
        cnt, anchors = {"valid": 0}, []
        for s in one.sites:
            for k in range(naz):
                for t in mode.tilts:
                    _, reasons, det = one.build((A.Gene(s.site_id, mode.mode_id, "c0", k * one.azimuth_step, t),))
                    for r in reasons or ["valid"]:
                        cnt[r] = cnt.get(r, 0) + 1
                    if not reasons and "mol1_anchor_A" in det:
                        anchors.append(det["mol1_anchor_A"])
        table[mode.mode_id] = {"tried": len(one.sites) * naz * len(mode.tilts), **cnt,
                               "anchor_A_range": [min(anchors), max(anchors)] if anchors else None,
                               "denticity": mode.denticity}
    # 3. small valid fixed-composition candidate set (unique under the gene key), nothing evaluated
    rng = np.random.default_rng([seed, 0])
    recs, seen, attempts, rej = [], set(), 0, {}
    d = out / "candidates"
    d.mkdir(exist_ok=True)
    while len(recs) < n_candidates and attempts < 50 * n_candidates:
        attempts += 1
        genes = space.random_genes(rng)
        atoms, reasons, det = space.build(genes)
        key = space.canonical_key(genes)
        if reasons or key in seen:
            for r in reasons or ["duplicate_initial"]:
                rej[r] = rej.get(r, 0) + 1
            continue
        seen.add(key)
        f = d / f"c{len(recs):03d}.extxyz"
        write(f, atoms, format="extxyz")
        roles = atoms.arrays["role"]
        recs.append({"id": f.stem, "attempt": attempts - 1, "seed": seed, "operator": "random_init", "parents": [],
                     "genes": [A.asdict(g) for g in genes], "validity": det, "key": repr(key),
                     "hash": A.structure_hash(atoms), "file": f"candidates/{f.name}",
                     "n_atoms": len(atoms), "roles": {A.ROLE_NAMES[int(k)]: int((roles == k).sum()) for k in np.unique(roles)},
                     "substrate_identical": A.substrate_integrity(sub, atoms, max_move=0.0)[0]})
    man = {"schema": "adsearch_demo/1", "provenance": prov, "space": space.describe(),
           "coverage": space.coverage(tuple(A.Gene(**g) for g in recs[0]["genes"])) if recs else None,
           "bare_slab": {"file": "bare_slab_labelled.extxyz", "integrity_ok": ok, "integrity_note": why,
                         "n_substrate": len(sub), "n_site_markers": len(space.sites),
                         "sites": [{"site_id": s.site_id, "kind": s.kind, "atom_index": s.atom_index,
                                    "coordination": s.coordination} for s in space.sites]},
           "single_molecule_stage": table,
           "candidate_set": {"n": len(recs), "attempts": attempts, "rejections": rej, "records": recs},
           "evaluated": False}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=A._json))
    return man


def bench(cfg, out: Path, evaluator: str, budget: int, seeds: list[int], n_init: int, model: str | None,
          device: str, steps: int | None) -> dict:
    space, prov = load_space(cfg)
    b = cfg.get("benchmark", {})
    if evaluator == "synthetic":
        ev = A.SyntheticEvaluator(space)
        prov["evaluator"] = {"kind": "synthetic", "label": ev.label}
    else:
        from local_settle import make_calc
        rx = b.get("relax", {})
        ev = A.AseRelaxEvaluator(make_calc(Path(model), device), fmax=rx.get("fmax_eV_A", 0.1),
                                 steps=steps or rx.get("max_steps", 150), radius=rx.get("radius_A", 8.0),
                                 mobile=rx.get("mobile", "face"), space=space)
        h = hashlib.sha256(Path(model).read_bytes()).hexdigest()[:16]
        prov["evaluator"] = {"kind": "mace+d3", "model": str(model), "model_sha256_16": h, "device": device,
                             "dtype": "float64 (MACE), float32 (D3)", "d3": "BJ, xc=r2scan (torch-dftd)",
                             "protocol": ev.protocol,
                             "status": "foundation model, NOT validated for this interface: plumbing/throughput only"}
    logs = [A.run_search(space, ev, arm, budget, s, n_init=n_init, pop_size=b.get("pop_size", 10),
                         dup_tol=b.get("dup_tol_A", 0.5)) for s in seeds for arm in ("random", "evolutionary")]
    A.write_manifest(out, space, logs, prov, write_structures="unique" if evaluator != "synthetic" else "none")
    summ = {"provenance": prov, "evaluator": evaluator, "budget_attempts": budget, "seeds": seeds, "n_init": n_init,
            "claim_status": "software check, no performance claim" if evaluator == "synthetic"
            else "unvalidated-model run: accounting and throughput only",
            "runs": [{k: v for k, v in log.items() if k not in ("records", "trace")} |
                     {"diversity": A.low_energy_diversity(space, log, 0.1)} for log in logs]}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=A._json))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("demo")
    d.add_argument("config"), d.add_argument("--out", required=True)
    d.add_argument("--n-candidates", type=int, default=12), d.add_argument("--seed", type=int, default=0)
    b = sp.add_parser("bench")
    b.add_argument("config"), b.add_argument("--out", required=True)
    b.add_argument("--evaluator", choices=("synthetic", "mace"), required=True)
    b.add_argument("--budget", type=int, default=None), b.add_argument("--seeds", default=None)
    b.add_argument("--n-init", type=int, default=None), b.add_argument("--model", default=None)
    b.add_argument("--device", default="cuda"), b.add_argument("--steps", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    if a.cmd == "demo":
        m = demo(cfg, Path(a.out), a.n_candidates, a.seed)
        print(json.dumps({"single_molecule_stage": m["single_molecule_stage"], "coverage": m["coverage"],
                          "candidates": m["candidate_set"]["n"], "attempts": m["candidate_set"]["attempts"],
                          "rejections": m["candidate_set"]["rejections"]}, indent=1))
    else:
        bb = cfg.get("benchmark", {})
        if a.evaluator == "mace" and not a.model:
            ap.error("--model is required with --evaluator mace")
        s = bench(cfg, Path(a.out), a.evaluator, a.budget or bb.get("budget_attempts", 200),
                  [int(x) for x in a.seeds.split(",")] if a.seeds else bb.get("seeds", [0, 1, 2]),
                  a.n_init if a.n_init is not None else bb.get("n_init", 20), a.model, a.device, a.steps)
        for r in s["runs"]:
            print(r["arm"], r["seed"], r["counts"], "calls", r["total_calls"], "unique", r["unique_relaxed"],
                  "best", r["best_energy"], "wall", r["wall_seconds"])


if __name__ == "__main__":
    main()
