"""Label tranche A (D3-D7) for fine-tuning: rotor scans, framework distortions and strain on the D1 DFT minima.

    python examples/scripts/prepare_chirality_labels.py --pkg105 <exported package 105 dir> \
        --config data/chirality_d0/config_labels.json --out <LONI_staging>/<package>

Base structures: the fixed-cell r2SCAN+D3(BJ) minima of package 105 rev1 (``outputs/{R,rac}_fixcell/relax/CONTCAR``,
520 eV, LREAL=.FALSE.). Every label is a single point with that same profile (ISYM=0), so the labels are consistent
with the minima and with the relax trajectories.

* D3 NH3+ rotor: the three N-H hydrogens of one cation rotated rigidly about C*->N, 0-120 deg in 10 deg steps (R, rac).
* D4 ring torsion: the phenyl group (ring C + H) rotated about C*->C_ipso, 0-180 deg in 20 deg steps (R, rac).
* D5 methyl rotor (control-restraint target): CH3 hydrogens about C*->C_methyl, 0-120 deg in 20 deg steps (R).
* D6 framework distortions: Gaussian displacements of Pb and I only (net shift removed), sigma 0.05/0.10/0.15 A,
  4 seeds each (R). A substitute for the symmetry-adapted tilt patterns of the plan, documented as such.
* D7 in-plane biaxial strain: a and b scaled by -2/-1/+1/+2 % with atoms (affine), c unchanged (R, rac).

Rigid rotations can drive a group into a neighbour. A point is dropped (and listed in the manifest) when any moved atom
comes within ``--min-contact`` (1.5 A) of an atom it is not bonded to. Scans stay one-cation, rigid (no constrained
relaxation): they sample the coupling of a single rotor to its frozen environment, not a relaxed barrier.
D8 (finite-T frames) and D9 (held-out set) need MD frames and are not part of this tranche.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from pvpassivator import chirality as ch  # noqa: E402

STATIC = {"ENCUT": 520, "IBRION": -1, "NSW": 0, "LREAL": ".FALSE."}


# ----------------------------------------------------------------------------------------------- geometry

def rotation_matrix(axis, angle_deg):
    k = ch._unit(axis)
    t = np.radians(angle_deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _adj(mol: ch.Molecule) -> dict[int, list[int]]:
    adj = {int(a): [] for a in mol.atoms}
    for a, b in mol.bonds:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def rotor_groups(atoms, mol: ch.Molecule) -> dict[str, tuple[int, int, list[int]]]:
    """(pivot, axis atom, moving atoms) for the NH3, phenyl and CH3 rotors of an MBA-type cation."""
    sym = atoms.get_chemical_symbols()
    adj = _adj(mol)
    if len(mol.stereocentres) != 1:
        raise ch.TopologyError("expected one stereocentre per cation")
    cs = next(iter(mol.stereocentres))
    carb = [w for w in adj[cs] if sym[w] == "C"]
    ipso = next(w for w in carb if sum(sym[x] == "C" for x in adj[w]) >= 3)
    methyl = next(w for w in carb if sum(sym[x] == "H" for x in adj[w]) == 3)
    seen, q = {cs, ipso}, deque([ipso])
    ring = []
    while q:                                   # everything reached from C_ipso without passing C*
        u = q.popleft()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                ring.append(w)
                q.append(w)
    return {"NH3": (cs, mol.n_ammonium, list(mol.nh_h)),
            "ring": (cs, ipso, ring),
            "CH3": (cs, methyl, [w for w in adj[methyl] if sym[w] == "H"])}


def rotate_group(atoms, mol: ch.Molecule, pivot: int, axis_atom: int, moving: list[int], angle_deg: float):
    """Rigid rotation of ``moving`` about the pivot->axis_atom bond (molecule unwrapped first). Returns a copy."""
    out = atoms.copy()
    upos = dict(zip(mol.atoms, ch.unwrap_molecule(atoms, mol)))
    R = rotation_matrix(upos[axis_atom] - upos[pivot], angle_deg)
    o = upos[axis_atom]
    for i in moving:
        out.positions[i] = o + R @ (upos[i] - o)
    out.wrap()
    return out


def min_contact(atoms, mol: ch.Molecule, moving: list[int]) -> float:
    """Smallest minimum-image distance from a moved atom to any atom it is not bonded to (1-2 or 1-3 in its molecule)."""
    adj = _adj(mol)
    near = set()
    for i in moving:
        near.add(i)
        for w in adj[i]:
            near.add(w)
            near.update(adj[w])
    others = np.array([j for j in range(len(atoms)) if j not in near and j not in moving])
    d = np.linalg.norm(ch.mic(atoms, np.concatenate([atoms.positions[others] - atoms.positions[i] for i in moving])), axis=1)
    return float(d.min())


def framework_distortion(atoms, sigma: float, seed: int):
    out = atoms.copy()
    sym = np.array(atoms.get_chemical_symbols())
    idx = np.where(np.isin(sym, ["Pb", "I"]))[0]
    dx = np.random.default_rng(seed).normal(scale=sigma, size=(len(idx), 3))
    out.positions[idx] += dx - dx.mean(axis=0)
    out.wrap()
    return out


def inplane_strain(atoms, eps: float):
    out = atoms.copy()
    cell = out.cell.array.copy()
    cell[0] *= 1 + eps
    cell[1] *= 1 + eps
    out.set_cell(cell, scale_atoms=True)
    return out


# ----------------------------------------------------------------------------------------------- tranche

def build_structures(base: dict, min_contact_A: float = 1.5):
    """name -> Atoms for D3-D7, plus the dropped points. ``base`` = {"R": Atoms, "rac": Atoms} (DFT minima)."""
    structs, dropped, meta = {}, [], {}
    scans = {"NH3": (range(0, 121, 10), ("R", "rac")), "ring": (range(0, 181, 20), ("R", "rac")),
             "CH3": (range(0, 121, 20), ("R",))}
    ids = {"NH3": "D3", "ring": "D4", "CH3": "D5"}
    for g, (angles, systems) in scans.items():
        for key in systems:
            at = base[key]
            top = ch.build_topology(at)
            mol = top.molecules[0]
            pivot, axis_atom, moving = rotor_groups(at, mol)[g]
            for ang in angles:
                s = rotate_group(at, mol, pivot, axis_atom, moving, ang)
                c = min_contact(s, mol, moving)
                name = f"{key}_{ids[g]}_{g}_{ang:03d}"
                meta[name] = {"block": ids[g], "system": key, "group": g, "angle_deg": ang, "molecule": 0,
                              "min_contact_A": round(c, 3)}
                if c < min_contact_A:
                    dropped.append(name)
                else:
                    structs[name] = s
    for sig in (0.05, 0.10, 0.15):
        for seed in range(4):
            name = f"R_D6_fw_s{int(round(sig * 100)):02d}_{seed}"
            structs[name] = framework_distortion(base["R"], sig, seed)
            meta[name] = {"block": "D6", "system": "R", "sigma_A": sig, "seed": seed, "atoms": "Pb, I"}
    for key in ("R", "rac"):
        for e in (-0.02, -0.01, 0.01, 0.02):
            name = f"{key}_D7_strain_{'m' if e < 0 else 'p'}{abs(int(round(e * 100)))}"
            structs[name] = inplane_strain(base[key], e)
            meta[name] = {"block": "D7", "system": key, "inplane_strain": e}
    return structs, dropped, meta


def main(argv=None) -> int:
    from prepare_bulk_refs import kmesh
    from prepare_chirality_d0 import stage
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg105", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-contact", type=float, default=1.5)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    cfg["_sha16"] = hashlib.sha256(a.config.read_bytes()).hexdigest()[:16]
    base, prov = {}, {"base": {}, "profile": STATIC}
    for key in ("R", "rac"):
        p = a.pkg105 / "outputs" / f"{key}_fixcell" / "relax" / "CONTCAR"
        if not (p.parent / "DONE").exists():
            raise SystemExit(f"{p.parent} is not DONE")
        base[key] = read(p, format="vasp")
        prov["base"][key] = {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    structs, dropped, meta = build_structures(base, a.min_contact)
    k = {key: kmesh(base[key], cfg["kspacing"]) for key in base}
    jobs = [("labels", n, "static", "static", n, dict(STATIC), k[meta[n]["system"]], "") for n in structs]
    prov.update({"points": meta, "dropped_for_contact": dropped, "min_contact_A": a.min_contact})
    stage(cfg, None, a.out, a.overwrite, structures=structs, jobs=jobs, provenance=prov)
    by = {}
    for n in structs:
        by[meta[n]["block"]] = by.get(meta[n]["block"], 0) + 1
    print("kept per block:", by, "| dropped:", dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
