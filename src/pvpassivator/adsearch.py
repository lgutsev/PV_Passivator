"""Fixed-composition search over arrangements of N passivant molecules on one slab face.

Project-native candidate generation on top of ``surface.find_sites`` and ``poses``; no ACAT import
(``examples/scripts/acat_surface_probe.py`` records why, see ``docs/acat_feasibility.md``).

A candidate is a tuple of ``Gene`` (one per molecule: site, binding mode, conformer, azimuth, tilt)
plus the atoms built from it.  Every built structure carries, in addition to the ``poses`` arrays
(``atom_id``, ``fragment_id``, ``origin``):

  role     0 substrate, 1 defect, 2 passivant, 3 counterion, 4 solvent      (never rewritten)
  mol_id   0 for the substrate, k for the k-th placed molecule (its counterions included)

Operators act on genes only, so the substrate is copied verbatim into every candidate and no
operator can splice, move or delete a lattice atom.  ``substrate_integrity`` checks this after
construction and again after relaxation.

Budget accounting: one *attempt* is one proposal, whether it is rejected, a duplicate, or evaluated.
Nothing leaves the log.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers

from .poses import (Passivator, fragments, tag_substrate, min_image_distances, classify_final_state, _align, _rot)
from .surface import SurfaceRecord, Site, unwrap_along

ROLE_NAMES = {0: "substrate", 1: "defect", 2: "passivant", 3: "counterion", 4: "solvent"}
ROLE = {v: k for k, v in ROLE_NAMES.items()}
MOL_ID0 = 100000          # atom_id base of molecule k is MOL_ID0 + 1000 * (k - 1), as in poses.place


# ----------------------------------------------------------------------------- frames, roles
def normal_to_z(atoms: Atoms, ax: int) -> Atoms:
    """Cyclic axis permutation putting lattice vector ``ax`` on z (right-handed; arrays kept).
    For tools that assume a z normal; the project's own code never needs it."""
    order = [(ax + 1) % 3, (ax + 2) % 3, ax]
    a = atoms.copy()
    a.set_constraint()
    a.set_cell(atoms.cell.array[order][:, order], scale_atoms=False)
    a.positions = atoms.positions[:, order]
    a.pbc = np.array(atoms.pbc)[order]
    return a


def _formula(symbols) -> str:
    c: dict[str, int] = {}
    for s in symbols:
        c[s] = c.get(s, 0) + 1
    return "".join(f"{k}{v if v > 1 else ''}" for k, v in sorted(c.items()))


def assign_roles_from_reference(cx: Atoms, slab: Atoms) -> np.ndarray:
    """Roles for an existing complex, from the bare slab it was built on (cells may differ slightly).

    Covalent fragments whose formula never occurs in the slab are passivant.  What remains of the
    composition difference is taken fragment-wise from the atoms farthest (fractional minimum image)
    from any slab atom of the same element: monatomic leftovers are counterions, molecular ones
    passivant.  Raises if the complex is not slab + something."""
    cs = np.array(cx.get_chemical_symbols())
    ss = np.array(slab.get_chemical_symbols())
    role = np.zeros(len(cx), int)
    known: dict[str, int] = {}
    for g in fragments(slab):
        f = _formula(ss[g])
        known[f] = known.get(f, 0) + 1
    by_formula: dict[str, list[list[int]]] = {}
    for g in fragments(cx):
        by_formula.setdefault(_formula(cs[g]), []).append(g)
    fs, fc = slab.get_scaled_positions(wrap=True), cx.get_scaled_positions(wrap=True)
    for f, groups in by_formula.items():
        extra = len(groups) - known.get(f, 0)
        if extra < 0:
            raise ValueError(f"complex has fewer {f} units than the slab")
        if extra == 0:
            continue
        if f not in known:
            chosen = groups
        else:
            def far(g):
                e = cs[g[0]]
                d = fc[g][:, None, :] - fs[ss == e][None]
                d -= np.round(d)
                return float(np.linalg.norm(d @ cx.cell.array, axis=-1).min(axis=1).mean())
            chosen = sorted(groups, key=far)[-extra:]
        for g in chosen:
            role[g] = ROLE["counterion"] if len(g) == 1 else ROLE["passivant"]
    return role


# ----------------------------------------------------------------------------- molecule library
@dataclass(frozen=True)
class BindingMode:
    """One documented way a molecule meets a site.

    ``anchor``: atom index placed ``height`` A above the site along the outward normal, or -1 for the
    centroid.  ``up``: which body vector is aligned with the normal before tilt/azimuth:
    'anchor_to_centroid' (end-on) or 'plane_normal' (flat-lying).  ``anchor_window``: accepted
    anchor-site distance (A) at construction.  ``second_anchor`` with ``second_window``: a further atom
    that must sit within the window of *some other* site atom of ``second_site_kind`` (a bidentate
    mode); ``denticity`` is then 2 and the mode occupies two sites."""
    mode_id: str
    anchor: int
    height: float
    up: str = "anchor_to_centroid"
    tilts: tuple = (0.0, 15.0, 30.0)
    anchor_window: tuple = (2.0, 3.6)
    azimuth_period: float = 360.0
    second_anchor: int | None = None
    second_window: tuple = (2.0, 3.6)
    second_site_kind: str = "undercoordinated_Pb"
    description: str = ""

    @property
    def denticity(self) -> int:
        return 1 if self.second_anchor is None else 2


@dataclass
class MoleculeSpec:
    """A passivator in its declared chemical form with a small, documented library of conformers
    (same atom order as ``passivator.atoms``) and binding modes."""
    passivator: Passivator
    modes: list[BindingMode]
    conformers: dict[str, Atoms] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self):
        if not self.conformers:
            self.conformers = {"c0": self.passivator.atoms.copy()}
        ref = self.passivator.atoms
        self.topology = bond_list(ref)
        self.chirality = labelled_chirality(ref, self.topology)
        for cid, c in self.conformers.items():
            if c.get_chemical_symbols() != ref.get_chemical_symbols():
                raise ValueError(f"conformer {cid}: atom order differs from the passivator")
            if bond_list(c) != self.topology:
                raise ValueError(f"conformer {cid}: bonding differs from the declared molecule")
            if not same_chirality(self.chirality, labelled_chirality(c, self.topology)):
                raise ValueError(f"conformer {cid}: a labelled centre is inverted (different stereoisomer); "
                                 "declare it as a separate MoleculeSpec, never as a conformer")
        if len({m.mode_id for m in self.modes}) != len(self.modes):
            raise ValueError("binding-mode ids must be unique")

    @property
    def name(self) -> str:
        return self.passivator.name

    def mode(self, mode_id: str) -> BindingMode:
        return next(m for m in self.modes if m.mode_id == mode_id)

    def roles(self) -> np.ndarray:
        """Role of each molecule atom: monatomic charged components are counterions."""
        r = np.full(len(self.passivator.atoms), ROLE["passivant"])
        for c in self.passivator.components:
            if len(c.indices) == 1 and c.formal_charge != 0:
                r[c.indices] = ROLE["counterion"]
        return r

    def sha256_16(self) -> str:
        h = hashlib.sha256()
        for cid in sorted(self.conformers):
            c = self.conformers[cid]
            h.update(cid.encode() + "".join(c.get_chemical_symbols()).encode() + np.round(c.positions, 6).tobytes())
        h.update(json.dumps([asdict(m) for m in self.modes], sort_keys=True).encode())
        return h.hexdigest()[:16]


def bond_list(mol: Atoms, mult: float = 1.15) -> tuple:
    """Covalent bonds (i < j) of an isolated molecule; ionic atoms carry none (as in ``poses.fragments``)."""
    from ase.neighborlist import natural_cutoffs, neighbor_list
    a = mol.copy()
    a.pbc = False
    ionic = {"Pb", "Sn", "Cs", "Na", "K", "Rb"}
    cut = [0.0 if s in ionic else c for s, c in zip(a.get_chemical_symbols(), natural_cutoffs(a, mult=mult))]
    i, j = neighbor_list("ij", a, cut)
    return tuple(sorted({(int(p), int(q)) for p, q in zip(i, j) if p < q}))


def labelled_chirality(mol: Atoms, bonds: tuple, flat: float = 0.35) -> dict[int, int]:
    """Sign of the signed volume of the first three (index-ordered) neighbours about every centre with
    >= 3 neighbours; 0 where the centre is planar (|normalised volume| < ``flat``).

    With fixed atom labels this is invariant under any motion that does not push a centre through a
    plane, and it flips under inversion of that centre or mirroring of the molecule.  It is stricter
    than CIP stereochemistry (it also sees labelled-H inversion of a CH2), which is what a
    construction check wants: a flipped sign means the geometry was mirrored or broken."""
    nb: dict[int, list[int]] = {}
    for p, q in bonds:
        nb.setdefault(p, []).append(q)
        nb.setdefault(q, []).append(p)
    out = {}
    for c, n in nb.items():
        if len(n) < 3:
            continue
        v = [mol.positions[k] - mol.positions[c] for k in sorted(n)[:3]]
        v = [x / np.linalg.norm(x) for x in v]
        vol = float(np.dot(v[0], np.cross(v[1], v[2])))
        out[c] = 0 if abs(vol) < flat else int(np.sign(vol))
    return out


def same_chirality(ref: dict[int, int], new: dict[int, int]) -> bool:
    """True unless a centre that is non-planar in both has opposite sign."""
    return not any(ref[c] * new.get(c, 0) < 0 for c in ref)


# ----------------------------------------------------------------------------- search space
@dataclass(frozen=True)
class Gene:
    site_id: str
    mode_id: str
    conformer_id: str
    azimuth_deg: float
    tilt_deg: float


@dataclass
class SymOp:
    translation: list            # cartesian, A (in-plane)
    site_map: dict               # site_id -> site_id
    site_residual_A: float
    inorganic_residual_A: float
    organic_residual_A: float


@dataclass
class SearchSpace:
    substrate: Atoms             # tagged (poses.tag_substrate) + role array
    rec: SurfaceRecord
    sites: list[Site]
    spec: MoleculeSpec
    n_molecules: int
    face: str = "top"
    azimuth_step: float = 30.0
    clash_scale_slab: float = 1.0      # x (r_cov,i + r_cov,j) between passivant and substrate
    clash_scale_mol: float = 1.3       # between different molecules and periodic self-images
    min_clearance: float = -0.5        # A; lowest passivant atom relative to the outermost inorganic layer
    vacuum_margin: float = 8.0         # A kept free below the opposite face's periodic image
    symmetry: str = "site_lattice"     # "site_lattice" | "none"
    sym_tol: float = 0.8               # A; site-to-site tolerance for an accepted translation
    az_bin: float = 15.0
    symops: list[SymOp] = field(default_factory=list)

    def __post_init__(self):
        sub = self.substrate if "atom_id" in self.substrate.arrays else tag_substrate(self.substrate)
        if "role" not in sub.arrays:
            sub = sub.copy()
            sub.set_array("role", np.zeros(len(sub), int))
        if "mol_id" not in sub.arrays:
            sub.set_array("mol_id", np.zeros(len(sub), int))
        self.substrate = sub
        self.sites = [s for s in self.sites if s.face == self.face]
        if len({s.site_id for s in self.sites}) != len(self.sites):
            raise ValueError("site ids must be unique")
        self._site = {s.site_id: s for s in self.sites}
        ax = self.rec.normal_axis
        n = sub.cell.array[ax] / np.linalg.norm(sub.cell.array[ax])
        self.n_out = n if self.face == "top" else -n
        self.symops = find_site_translations(sub, self.rec, self.sites, self.sym_tol) \
            if self.symmetry == "site_lattice" else [_identity_op(self.sites)]

    # --- description
    def area_A2(self) -> float:
        c = self.substrate.cell.array
        i, j = [k for k in range(3) if k != self.rec.normal_axis]
        return float(np.linalg.norm(np.cross(c[i], c[j])))

    def coverage(self, genes: tuple) -> dict:
        occ = sum(self.spec.mode(g.mode_id).denticity for g in genes)
        return {"n_molecules": len(genes), "area_A2": round(self.area_A2(), 2),
                "molecules_per_nm2": round(len(genes) / self.area_A2() * 100, 4),
                "occupied_sites": occ, "n_sites": len(self.sites),
                "site_convention": "one site per anchoring atom declared by the binding mode (denticity); "
                                   "sites merely shadowed by the molecule are not counted"}

    def describe(self) -> dict:
        sub = self.substrate
        return {"surface_id": self.rec.surface_id, "slab_sha256_16": self.rec.origin.get("sha256", "")[:16],
                "face": self.face, "termination": self.rec.faces[self.face]["termination"],
                "n_substrate_atoms": len(sub), "n_sites": len(self.sites),
                "site_kinds": sorted({s.kind for s in self.sites}), "molecule": self.spec.name,
                "molecule_sha256_16": self.spec.sha256_16(), "n_molecules": self.n_molecules,
                "modes": [asdict(m) for m in self.spec.modes], "conformers": sorted(self.spec.conformers),
                "azimuth_step": self.azimuth_step, "symmetry": self.symmetry,
                "symops": [{"translation_A": np.round(o.translation, 3).tolist(),
                            "site_residual_A": round(o.site_residual_A, 3),
                            "inorganic_residual_A": round(o.inorganic_residual_A, 3),
                            "organic_residual_A": round(o.organic_residual_A, 3)} for o in self.symops],
                "validity": {"clash_scale_slab": self.clash_scale_slab, "clash_scale_mol": self.clash_scale_mol,
                             "min_clearance_A": self.min_clearance, "vacuum_margin_A": self.vacuum_margin}}

    # --- sampling and operators (genes only)
    def random_genes(self, rng) -> tuple:
        sites = list(rng.choice(len(self.sites), size=self.n_molecules, replace=False))
        return tuple(self._random_gene(rng, self.sites[i].site_id) for i in sites)

    def _random_gene(self, rng, site_id: str, mode: BindingMode | None = None) -> Gene:
        mode = mode or self.spec.modes[int(rng.integers(len(self.spec.modes)))]
        conf = sorted(self.spec.conformers)[int(rng.integers(len(self.spec.conformers)))]
        naz = max(1, int(round(mode.azimuth_period / self.azimuth_step)))
        return Gene(site_id, mode.mode_id, conf, float(rng.integers(naz) * self.azimuth_step),
                    float(mode.tilts[int(rng.integers(len(mode.tilts)))]))

    def mutate(self, genes: tuple, op: str, rng, other: tuple | None = None) -> tuple[tuple | None, dict]:
        """Apply one named operator.  Returns (new genes or None, record).  None means the operator had
        nothing legal to do (counted by the caller as a rejected attempt)."""
        g = list(genes)
        k = int(rng.integers(len(g)))
        rec = {"operator": op, "molecule": k}
        if op == "move":
            free = [s.site_id for s in self.sites if s.site_id not in {x.site_id for x in g}]
            if not free:
                return None, {**rec, "reason": "no_free_site"}
            g[k] = Gene(free[int(rng.integers(len(free)))], g[k].mode_id, g[k].conformer_id,
                        g[k].azimuth_deg, g[k].tilt_deg)
        elif op == "rotate":
            mode = self.spec.mode(g[k].mode_id)
            naz = max(1, int(round(mode.azimuth_period / self.azimuth_step)))
            opts = [(i * self.azimuth_step, float(t)) for i in range(naz) for t in mode.tilts
                    if (i * self.azimuth_step, float(t)) != (g[k].azimuth_deg, g[k].tilt_deg)]
            if not opts:
                return None, {**rec, "reason": "single_orientation"}
            az, tilt = opts[int(rng.integers(len(opts)))]
            g[k] = Gene(g[k].site_id, g[k].mode_id, g[k].conformer_id, az, tilt)
        elif op == "mode":
            opts = [(m.mode_id, c) for m in self.spec.modes for c in sorted(self.spec.conformers)
                    if (m.mode_id, c) != (g[k].mode_id, g[k].conformer_id)]
            if not opts:
                return None, {**rec, "reason": "single_mode_and_conformer"}
            m, c = opts[int(rng.integers(len(opts)))]
            new = self._random_gene(rng, g[k].site_id, self.spec.mode(m))
            g[k] = Gene(g[k].site_id, m, c, new.azimuth_deg, new.tilt_deg)
        elif op == "exchange":
            if len(g) < 2:
                return None, {**rec, "reason": "single_molecule"}
            j = int(rng.choice([x for x in range(len(g)) if x != k]))
            a, b = g[k], g[j]
            g[k] = Gene(b.site_id, a.mode_id, a.conformer_id, a.azimuth_deg, a.tilt_deg)
            g[j] = Gene(a.site_id, b.mode_id, b.conformer_id, b.azimuth_deg, b.tilt_deg)
            rec["partner"] = j
        elif op == "crossover":
            if other is None:
                return None, {**rec, "reason": "no_second_parent"}
            n_a = int(rng.integers(1, len(g))) if len(g) > 1 else 1
            keep = [g[i] for i in sorted(rng.choice(len(g), size=n_a, replace=False))]
            used = {x.site_id for x in keep}
            pool = [x for x in other if x.site_id not in used]
            rng.shuffle(pool)
            for x in pool:
                if len(keep) == len(g):
                    break
                if x.site_id not in {y.site_id for y in keep}:
                    keep.append(x)
            if len(keep) != len(g):
                return None, {**rec, "reason": "crossover_site_conflict"}
            g = keep
            rec["from_first_parent"] = n_a
        else:
            raise ValueError(f"unknown operator {op}")
        return tuple(g), rec

    # --- construction
    def place(self, gene: Gene, k: int) -> Atoms:
        """Molecule ``k`` (1-based) built from ``gene``; tagged like ``poses.place`` plus role/mol_id."""
        mode = self.spec.mode(gene.mode_id)
        m = self.spec.conformers[gene.conformer_id].copy()
        site = self._site[gene.site_id]
        n = self.n_out
        origin = m.positions.mean(axis=0) if mode.anchor < 0 else m.positions[mode.anchor]
        pos = m.positions - origin
        if mode.up == "plane_normal":
            w, v = np.linalg.eigh(np.cov((m.positions - m.positions.mean(axis=0)).T))
            up = v[:, 0]
        else:
            up = pos.mean(axis=0)
        R = _align(up if np.linalg.norm(up) > 1e-6 else n, n)
        perp = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0])
        R = _rot(n, np.radians(gene.azimuth_deg)) @ _rot(perp, np.radians(gene.tilt_deg)) @ R
        m.positions = (R @ pos.T).T + np.asarray(site.position) + mode.height * n
        m.cell, m.pbc = self.substrate.cell, self.substrate.pbc
        m.set_array("atom_id", MOL_ID0 + 1000 * (k - 1) + np.arange(len(m)))
        fbase = int(self.substrate.arrays["fragment_id"].max()) + 1 + 50 * (k - 1)
        fr = np.empty(len(m), int)
        for i, g in enumerate(fragments(m, pbc=False)):
            fr[g] = fbase + i
        m.set_array("fragment_id", fr)
        m.set_array("origin", np.full(len(m), k))
        m.set_array("role", self.spec.roles())
        m.set_array("mol_id", np.full(len(m), k))
        for key in list(m.arrays):
            if key not in ("numbers", "positions", "atom_id", "fragment_id", "origin", "role", "mol_id"):
                del m.arrays[key]
        return m

    def adopt(self, molecules: list[Atoms]) -> Atoms:
        """Substrate (verbatim) + externally positioned molecules, tagged like built candidates. For
        structures that do not come from genes (a transplanted archive arrangement); atom order inside
        each molecule is the caller's, so no library topology/stereo check is made here."""
        full = self.substrate.copy()
        fbase = int(self.substrate.arrays["fragment_id"].max()) + 1
        for k, mol in enumerate(molecules, 1):
            m = Atoms(mol.get_chemical_symbols(), positions=mol.positions, cell=self.substrate.cell,
                      pbc=self.substrate.pbc)
            m.set_array("atom_id", MOL_ID0 + 1000 * (k - 1) + np.arange(len(m)))
            fr = np.empty(len(m), int)
            for i, g in enumerate(fragments(m, pbc=False)):
                fr[g] = fbase + 50 * (k - 1) + i
            m.set_array("fragment_id", fr)
            m.set_array("origin", np.full(len(m), k))
            m.set_array("role", np.full(len(m), ROLE["passivant"]))
            m.set_array("mol_id", np.full(len(m), k))
            full += m
        ok, why = substrate_integrity(self.substrate, full, max_move=0.0)
        if not ok:
            raise ValueError("substrate changed while adopting molecules: " + why)
        return full

    def build(self, genes: tuple) -> tuple[Atoms, list[str], dict]:
        """Atoms for ``genes`` (substrate first, verbatim, then molecules unwrapped), the list of
        rejection reasons (empty = valid) and details."""
        reasons, det = [], {}
        if len(genes) != self.n_molecules:
            reasons.append("wrong_molecule_count")
        if len({g.site_id for g in genes}) != len(genes):
            reasons.append("site_conflict")
        sub = self.substrate
        mols = [self.place(g, k + 1) for k, g in enumerate(genes)]
        full = sub.copy()
        for m in mols:
            full += m
        cell, pbc = sub.cell.array, sub.pbc
        ssym = sub.get_chemical_symbols()
        h = unwrap_along(sub, self.rec.normal_axis)
        sign = 1 if self.face == "top" else -1
        edge = self.rec.faces[self.face]["edge_coordinate_A"]
        sub_out = (h - edge) * sign                      # outward height of substrate atoms
        for k, (g, m) in enumerate(zip(genes, mols), 1):
            mode = self.spec.mode(g.mode_id)
            site = self._site[g.site_id]
            msym = m.get_chemical_symbols()
            # anchoring distance
            if mode.anchor >= 0 and site.atom_index is not None:
                d = float(min_image_distances(cell, pbc, m.positions[[mode.anchor]],
                                              sub.positions[[site.atom_index]])[0, 0])
                det[f"mol{k}_anchor_A"] = round(d, 3)
                if not mode.anchor_window[0] <= d <= mode.anchor_window[1]:
                    reasons.append("anchor_distance")
            if mode.second_anchor is not None:
                others = [s.atom_index for s in self.sites if s.kind == mode.second_site_kind
                          and s.site_id != g.site_id and s.atom_index is not None]
                d2 = min_image_distances(cell, pbc, m.positions[[mode.second_anchor]], sub.positions[others]).min() \
                    if others else np.inf
                det[f"mol{k}_second_anchor_A"] = round(float(d2), 3)
                if not mode.second_window[0] <= d2 <= mode.second_window[1]:
                    reasons.append("second_anchor_distance")
            # species-aware sterics against the substrate (the declared anchor contacts are exempt)
            D = min_image_distances(cell, pbc, m.positions, sub.positions)
            lim = self.clash_scale_slab * (_rcov(msym)[:, None] + _rcov(ssym)[None, :])
            if mode.anchor >= 0 and site.atom_index is not None:
                lim[mode.anchor, site.atom_index] = 0.0
            if (D < lim).any():
                reasons.append("slab_clash")
                det[f"mol{k}_slab_min_A"] = round(float(D.min()), 3)
            # height: not buried, not into the opposite face's image
            ref = site.atom_index if site.atom_index is not None else \
                int(np.linalg.norm(sub.positions - np.asarray(site.position), axis=1).argmin())
            out_edge = sub_out[ref] + (m.positions - sub.positions[ref]) @ self.n_out   # outward, relative to the edge
            if out_edge.min() < self.min_clearance:
                reasons.append("buried")
            if out_edge.max() > self.rec.vacuum - self.vacuum_margin:
                reasons.append("vacuum_overrun")
            det[f"mol{k}_height_A"] = [round(float(out_edge.min()), 2), round(float(out_edge.max()), 2)]
            # molecule-molecule, and the molecule against its own periodic images
            for j in range(k + 1, len(mols) + 1):
                o = mols[j - 1]
                D2 = min_image_distances(cell, pbc, m.positions, o.positions)
                if (D2 < self.clash_scale_mol * (_rcov(msym)[:, None] + _rcov(o.get_chemical_symbols())[None, :])).any():
                    reasons.append("molecule_clash")
                    det[f"mol{k}_mol{j}_min_A"] = round(float(D2.min()), 3)
            if _self_image_clash(cell, pbc, m, self.clash_scale_mol):
                reasons.append("self_image_clash")
            # connectivity and labelled chirality of what was actually built
            if bond_list(m) != self.spec.topology:
                reasons.append("topology_changed")
            if not same_chirality(self.spec.chirality, labelled_chirality(m, self.spec.topology)):
                reasons.append("stereo_inverted")
        ok, why = substrate_integrity(sub, full)
        if not ok:
            reasons.append("substrate_changed:" + why)
        return full, sorted(set(reasons)), det

    # --- identity
    def canonical_key(self, genes: tuple) -> tuple:
        """Smallest gene tuple over the accepted surface translations and molecule order."""
        best = None
        for op in self.symops:
            t = tuple(sorted((op.site_map[g.site_id], g.mode_id, g.conformer_id,
                              int(round((g.azimuth_deg % self.spec.mode(g.mode_id).azimuth_period) / self.az_bin)),
                              int(round(g.tilt_deg / self.az_bin))) for g in genes))
            best = t if best is None or t < best else best
        return best

    def same_structure(self, a: Atoms, b: Atoms, tol: float = 0.5) -> bool:
        """Geometric identity of two built/relaxed candidates of this space: some accepted surface
        translation and some pairing of molecules maps every passivant/counterion atom of ``a`` onto a
        distinct atom of ``b`` of the same element within ``tol`` A (minimum image).  Matching is by
        position, not label, so a molecule rotated onto itself is the same structure; mirror images and
        different binding modes are not, because their atoms do not coincide."""
        ma, mb = _molecules(a), _molecules(b)
        if len(ma) != len(mb):
            return False
        cell, pbc = self.substrate.cell.array, self.substrate.pbc
        ca = np.array([a.positions[m].mean(axis=0) for m in ma])
        cb = np.array([b.positions[m].mean(axis=0) for m in mb])
        for op in self.symops:
            t = np.asarray(op.translation)
            near = min_image_distances(cell, pbc, ca + t, cb) < tol      # matched atoms => matched centroids
            for perm in itertools.permutations(range(len(mb))):
                if all(near[i, p] for i, p in enumerate(perm)) and all(_match(cell, pbc, a[ma[i]], b[mb[p]], t, tol) for i, p in enumerate(perm)):
                    return True
        return False


def _rcov(symbols) -> np.ndarray:
    return np.array([covalent_radii[atomic_numbers[s]] for s in symbols])


def _self_image_clash(cell, pbc, m: Atoms, scale: float) -> bool:
    r = _rcov(m.get_chemical_symbols())
    lim = scale * (r[:, None] + r[None, :])
    for s in itertools.product((-1, 0, 1), repeat=3):
        if not any(s) or any(x and not p for x, p in zip(s, pbc)):
            continue
        d = np.linalg.norm(m.positions[:, None] - (m.positions + np.array(s) @ cell)[None], axis=-1)
        if (d < lim).any():
            return True
    return False


def _molecules(a: Atoms) -> list[np.ndarray]:
    mid = a.arrays["mol_id"]
    return [np.where(mid == k)[0] for k in sorted(set(mid[mid > 0].tolist()))]


def _match(cell, pbc, ma: Atoms, mb: Atoms, t, tol: float) -> bool:
    if sorted(ma.get_chemical_symbols()) != sorted(mb.get_chemical_symbols()):
        return False
    D = min_image_distances(cell, pbc, ma.positions + t, mb.positions)
    sa, sb = np.array(ma.get_chemical_symbols()), np.array(mb.get_chemical_symbols())
    D[sa[:, None] != sb[None, :]] = np.inf
    j = D.argmin(axis=1)
    return bool((D[np.arange(len(ma)), j] < tol).all() and len(set(j.tolist())) == len(ma))


def _identity_op(sites) -> SymOp:
    return SymOp([0.0, 0.0, 0.0], {s.site_id: s.site_id for s in sites}, 0.0, 0.0, 0.0)


def find_site_translations(sub: Atoms, rec: SurfaceRecord, sites: list[Site], tol: float) -> list[SymOp]:
    """In-plane translations that map the site list onto itself within ``tol``; the identity first.

    These are symmetries of the *site lattice*.  How well each also maps the substrate is reported
    (inorganic and organic residuals) rather than assumed: orientationally disordered A-cations break
    them, so two arrangements merged here can still differ in their FA environment."""
    ops = [_identity_op(sites)]
    if len(sites) < 2:
        return ops
    cell, pbc = sub.cell.array, sub.pbc
    P = np.array([s.position for s in sites])
    n = cell[rec.normal_axis] / np.linalg.norm(cell[rec.normal_axis])
    syms = np.array(sub.get_chemical_symbols())
    inorg = np.isin(syms, ["Pb", "Sn", "I", "Br", "Cl", "Cs"])
    seen = [np.zeros(3)]
    for j in range(1, len(sites)):
        if sites[j].kind != sites[0].kind:
            continue
        t = P[j] - P[0]
        t = t - (t @ n) * n
        D = min_image_distances(cell, pbc, P + t, P)
        same_kind = np.array([[a.kind == b.kind for b in sites] for a in sites])
        D[~same_kind] = np.inf
        tgt = D.argmin(axis=1)
        if D[np.arange(len(P)), tgt].max() > tol or len(set(tgt.tolist())) != len(P):
            continue
        if any(min_image_distances(cell, pbc, (t - s)[None], np.zeros((1, 3)))[0, 0] < 1e-3 for s in seen):
            continue
        seen.append(t)

        def resid(mask):
            if not mask.any():
                return 0.0
            out = []
            for e in np.unique(syms[mask]):
                m = mask & (syms == e)
                out.append(min_image_distances(cell, pbc, sub.positions[m] + t, sub.positions[m]).min(axis=1))
            return float(np.sqrt(np.mean(np.concatenate(out) ** 2)))
        ops.append(SymOp(t.tolist(), {sites[i].site_id: sites[int(k)].site_id for i, k in enumerate(tgt)},
                         float(D[np.arange(len(P)), tgt].max()), resid(inorg), resid(~inorg & (syms != "H"))))
    return ops


def substrate_integrity(reference: Atoms, candidate: Atoms, max_move: float | None = None,
                        rec: SurfaceRecord | None = None, extract_A: float = 2.5) -> tuple[bool, str]:
    """Every substrate atom of ``reference`` is present in ``candidate`` with the same id, element
    and role; with ``max_move=0`` also at the same position (construction).  With ``rec``, no
    substrate atom has risen more than ``extract_A`` above the outermost inorganic layer of either
    face (a lattice atom pulled out by an adsorbate)."""
    rid = reference.arrays["atom_id"]
    cid = candidate.arrays["atom_id"]
    crole = candidate.arrays["role"]
    idx = {int(i): k for k, i in enumerate(cid)}
    if len(idx) != len(cid):
        return False, "duplicate_atom_ids"
    if any(int(i) not in idx for i in rid):
        return False, "substrate_atom_missing"
    order = np.array([idx[int(i)] for i in rid])
    if (candidate.numbers[order] != reference.numbers).any():
        return False, "substrate_element_changed"
    if (crole[order] != reference.arrays["role"]).any():
        return False, "substrate_role_changed"
    if int((crole == 0).sum()) != int((reference.arrays["role"] == 0).sum()):
        return False, "substrate_count_changed"
    if max_move is not None:
        d = candidate.positions[order] - reference.positions
        d -= np.round(d @ np.linalg.inv(reference.cell.array)) @ reference.cell.array
        if np.linalg.norm(d, axis=1).max() > max_move + 1e-9:
            return False, "substrate_moved"
    if rec is not None:
        c = candidate[order]
        h = unwrap_along(reference, rec.normal_axis)
        hc = h + ((c.positions - reference.positions) -
                  np.round((c.positions - reference.positions) @ np.linalg.inv(reference.cell.array))
                  @ reference.cell.array) @ (reference.cell.array[rec.normal_axis] /
                                             np.linalg.norm(reference.cell.array[rec.normal_axis]))
        top, bot = rec.faces["top"]["edge_coordinate_A"], rec.faces["bottom"]["edge_coordinate_A"]
        heavy = reference.numbers > 1
        if ((hc - top)[heavy] > extract_A).any() or ((bot - hc)[heavy] > extract_A).any():
            return False, "lattice_atom_extracted"
    return True, ""


# ----------------------------------------------------------------------------- evaluation
@dataclass
class EvalResult:
    atoms: Atoms
    energy: float
    n_calls: int
    converged: bool
    info: dict = field(default_factory=dict)


class SyntheticEvaluator:
    """SOFTWARE CHECK ONLY.  A cheap, deterministic made-up score (per-site term + mode term +
    pairwise term on molecule centroids + orientation term) with no relaxation.  It exists so the
    search loop, accounting and deduplication can be exercised; it says nothing about chemistry."""
    label = "synthetic software-check score (not physical)"

    def __init__(self, space: SearchSpace, seed: int = 12345, calls_per_eval: int = 1):
        rng = np.random.default_rng(seed)
        self.space = space
        self.site_e = {s.site_id: float(e) for s, e in zip(space.sites, rng.normal(0, 0.15, len(space.sites)))}
        self.mode_e = {m.mode_id: float(e) for m, e in zip(space.spec.modes, rng.normal(0, 0.2, len(space.spec.modes)))}
        self.calls_per_eval = calls_per_eval

    def __call__(self, atoms: Atoms, genes: tuple) -> EvalResult:
        sp = self.space
        e = sum(self.site_e[g.site_id] + self.mode_e[g.mode_id] + 0.05 * np.cos(np.radians(2 * g.azimuth_deg))
                + 0.002 * g.tilt_deg for g in genes)
        mols = _molecules(atoms)
        com = np.array([atoms.positions[m].mean(axis=0) for m in mols])
        for i in range(len(com)):
            for j in range(i + 1, len(com)):
                r = float(min_image_distances(sp.substrate.cell.array, sp.substrate.pbc, com[[i]], com[[j]])[0, 0])
                e += 0.4 * ((6.5 / r) ** 8 - 2 * (6.5 / r) ** 4)
        return EvalResult(atoms.copy(), float(e), self.calls_per_eval, True, {"evaluator": self.label})


class AseRelaxEvaluator:
    """Relax with any ASE calculator under one fixed protocol; counts energy/force calls.

    Mobile atoms (lattice molecules always whole), the rest fixed at the bare-slab positions:
      "face"       the molecules + every substrate atom within ``radius`` A (along the normal) of the
                   outer inorganic layer of the searched face.  The fixed set is the SAME ATOMS for
                   every candidate, which a fixed-composition ranking needs; needs ``space``.
      "local"      the molecules + every substrate atom within ``radius`` A of them.  Cheaper, but the
                   fixed set depends on where the molecules sit, so energies of different arrangements
                   are minima under different constraints: use for settling, not for ranking.
      "passivant"  the molecules only."""

    def __init__(self, calc, fmax: float = 0.1, steps: int = 150, radius: float = 8.0, mobile: str = "face",
                 protocol: dict | None = None, space: "SearchSpace | None" = None):
        if mobile == "face" and space is None:
            raise ValueError('mobile="face" needs the SearchSpace')
        self.calc, self.fmax, self.steps, self.radius, self.mobile = calc, fmax, steps, radius, mobile
        self.space = space
        self.protocol = {"optimizer": "FIRE", "fmax_eV_A": fmax, "max_steps": steps, "mobile": mobile,
                         "radius_A": radius, **(protocol or {})}

    def mobile_mask(self, atoms: Atoms) -> np.ndarray:
        ads = atoms.arrays["mol_id"] > 0
        if self.mobile == "passivant":
            return ads
        if self.mobile == "face":
            sp = self.space
            n = len(sp.substrate)
            h = unwrap_along(sp.substrate, sp.rec.normal_axis)
            depth = (sp.rec.faces[sp.face]["edge_coordinate_A"] - h) * (1 if sp.face == "top" else -1)
            mob = ads.copy()
            order = {int(i): k for k, i in enumerate(atoms.arrays["atom_id"])}
            idx = np.array([order[int(i)] for i in sp.substrate.arrays["atom_id"]])
            mob[idx] = depth <= self.radius
            fid = atoms.arrays["fragment_id"]
            for f in np.unique(fid[mob]):
                mob[fid == f] = True
            assert n == len(idx)
            return mob
        d = min_image_distances(atoms.cell.array, atoms.pbc, atoms.positions, atoms.positions[ads]).min(axis=1)
        mob = d <= self.radius
        fid = atoms.arrays["fragment_id"]
        for f in np.unique(fid[mob]):
            mob[fid == f] = True
        return mob

    def __call__(self, atoms: Atoms, genes: tuple) -> EvalResult:
        from ase.constraints import FixAtoms
        from ase.optimize import FIRE
        a = atoms.copy()
        mob = self.mobile_mask(a)
        a.set_constraint(FixAtoms(indices=np.flatnonzero(~mob)))
        n0 = getattr(self.calc, "_adsearch_calls", 0)
        _count_calls(self.calc)
        a.calc = self.calc
        t0 = time.time()
        opt = FIRE(a, logfile=None)
        conv = bool(opt.run(fmax=self.fmax, steps=self.steps))
        e = float(a.get_potential_energy())
        f = float(np.linalg.norm(a.get_forces()[mob], axis=1).max())
        forces = a.get_forces(apply_constraint=False)          # raw model forces on every atom, fixed ones included
        out = a.copy()
        out.set_constraint()
        out.calc = None
        out.set_array("model_forces", forces)
        out.set_array("mobile", mob.astype(int))
        out.info["model_energy"] = e
        return EvalResult(out, e, self.calc._adsearch_calls - n0, conv,
                          {"steps": int(opt.nsteps), "fmax_end": round(f, 4), "n_mobile": int(mob.sum()),
                           "seconds": round(time.time() - t0, 2)})


def single_point(calc, atoms: Atoms) -> Atoms:
    """Copy of ``atoms`` with the calculator's energy (``info['model_energy']``) and forces
    (``arrays['model_forces']``) at exactly this geometry; nothing moves."""
    a = atoms.copy()
    a.set_constraint()
    a.calc = calc
    e, f = float(a.get_potential_energy()), a.get_forces()
    a.calc = None
    a.set_array("model_forces", f)
    a.info["model_energy"] = e
    return a


def _count_calls(calc):
    if getattr(calc, "_adsearch_wrapped", False):
        return
    orig = calc.calculate
    calc._adsearch_calls = 0

    def calculate(*a, **k):
        calc._adsearch_calls += 1
        return orig(*a, **k)
    calc.calculate = calculate
    calc._adsearch_wrapped = True


# ----------------------------------------------------------------------------- search
OPERATORS = {"move": 0.30, "rotate": 0.25, "mode": 0.15, "exchange": 0.10, "crossover": 0.20}
RANKED_STATES = ("intact_adsorption",)


def structure_hash(atoms: Atoms) -> str:
    h = hashlib.sha256()
    order = np.argsort(atoms.arrays["atom_id"])
    def q(x, scale):                       # integers: no signed zeros, no platform-dependent widths
        return np.rint(np.asarray(x, float) * scale).astype(np.int64).tobytes()
    h.update(q(atoms.numbers[order], 1) + q(atoms.positions[order], 1e4) + q(atoms.cell.array, 1e4)
             + q(atoms.arrays["role"][order], 1) + q(atoms.arrays["atom_id"][order], 1))
    return h.hexdigest()[:16]


def run_search(space: SearchSpace, evaluator, arm: str, budget: int, seed: int, n_init: int = 20,
               pop_size: int = 10, operators: dict | None = None, dup_tol: float = 0.5,
               keep_atoms: bool = True) -> dict:
    """One arm, one seed.  ``arm``: 'random' or 'evolutionary'.

    Both arms draw their first ``n_init`` proposals from the same generator with the same stream, so
    for one seed they start from the identical proposals; the arms differ only afterwards.  ``budget``
    counts attempts: rejected and duplicate proposals consume it."""
    if arm not in ("random", "evolutionary"):
        raise ValueError(arm)
    ops = operators or OPERATORS
    names, probs = list(ops), np.array(list(ops.values()), float) / sum(ops.values())
    rng_init = np.random.default_rng([seed, 0])
    rng = np.random.default_rng([seed, 1 if arm == "evolutionary" else 2])
    records, seen_keys, unique = [], {}, []          # unique: record indices of unique relaxed structures
    calls = 0
    t_start = time.time()
    for n in range(budget):
        t0 = time.time()
        r = {"attempt": n, "arm": arm, "seed": seed, "parents": [], "operator": None}
        if n < n_init or arm == "random":
            genes, r["operator"] = space.random_genes(rng_init if n < n_init else rng), "random_init" if n < n_init \
                else "random"
        else:
            pool = sorted((i for i in unique if records[i]["ranked"]), key=lambda i: records[i]["energy"])[:pop_size]
            if not pool:
                genes, r["operator"] = space.random_genes(rng), "random_fallback"
            else:
                w = np.arange(len(pool), 0, -1, dtype=float)
                pa = pool[int(rng.choice(len(pool), p=w / w.sum()))]
                op = names[int(rng.choice(len(names), p=probs))]
                other = None
                if op == "crossover":
                    rest = [i for i in pool if i != pa]
                    pb = rest[int(rng.integers(len(rest)))] if rest else None
                    other = _genes(records[pb]) if pb is not None else None
                    r["parents"] = [pa] + ([pb] if pb is not None else [])
                else:
                    r["parents"] = [pa]
                genes, mrec = space.mutate(_genes(records[pa]), op, rng, other)
                r["operator"], r["mutation"] = op, mrec
        r.update(status="", reasons=[], evaluated=False, ranked=False, n_calls=0, energy=None)
        if genes is None:
            r.update(status="rejected", reasons=[r["mutation"]["reason"]])
        else:
            r["genes"] = [asdict(g) for g in genes]
            atoms, reasons, det = space.build(genes)
            r["validity"] = det
            key = space.canonical_key(genes)
            r["key"] = repr(key)
            if reasons:
                r.update(status="rejected", reasons=reasons)
            elif key in seen_keys:
                r.update(status="duplicate_initial", duplicate_of=seen_keys[key])
            else:
                seen_keys[key] = n
                r["initial_hash"] = structure_hash(atoms)
                res = evaluator(atoms, genes)
                calls += res.n_calls
                r.update(evaluated=True, n_calls=res.n_calls, energy=res.energy, converged=res.converged,
                         eval_info=res.info, final_hash=structure_hash(res.atoms))
                ok, why = substrate_integrity(space.substrate, res.atoms, rec=space.rec)
                state = classify_final_state(atoms, res.atoms, space.rec) if ok else \
                    {"final_state_class": "substrate_failure:" + why}
                r["final_state"] = state["final_state_class"]
                dup = next((i for i in unique if space.same_structure(res.atoms, records[i]["_atoms"], dup_tol)), None)
                if dup is not None:
                    r.update(status="duplicate_relaxed", duplicate_of=dup)
                else:
                    unique.append(n)
                    r["status"] = "unique"
                    r["ranked"] = bool(res.converged and r["final_state"] in RANKED_STATES)
                    if not r["ranked"]:
                        r["reasons"] = ["not_converged"] if not res.converged else ["final_state:" + r["final_state"]]
                r["_atoms"], r["_initial"] = res.atoms, atoms
        r["cumulative_calls"] = calls
        r["seconds"] = round(time.time() - t0, 4)
        records.append(r)
    ranked = [r for r in records if r["ranked"]]
    best, trace = None, []
    for r in records:
        if r["ranked"] and (best is None or r["energy"] < best):
            best = r["energy"]
        trace.append([r["attempt"] + 1, r["cumulative_calls"], best])
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        for x in r["reasons"]:
            reasons[x] = reasons.get(x, 0) + 1
    out = {"arm": arm, "seed": seed, "budget_attempts": budget, "n_init": n_init, "pop_size": pop_size,
           "operators": ops if arm == "evolutionary" else None, "dup_tol_A": dup_tol,
           "counts": counts, "rejection_reasons": reasons, "evaluated": sum(r["evaluated"] for r in records),
           "unique_relaxed": len(unique), "ranked": len(ranked), "total_calls": calls,
           "wall_seconds": round(time.time() - t_start, 2), "best_energy": best, "trace": trace,
           "records": records}
    if not keep_atoms:
        for r in records:
            r.pop("_atoms", None), r.pop("_initial", None)
    return out


def _genes(record: dict) -> tuple:
    return tuple(Gene(**g) for g in record["genes"])


def low_energy_diversity(space: SearchSpace, log: dict, window_eV: float) -> dict:
    """Unique ranked structures within ``window_eV`` of this run's best, and their distinct motifs
    (multiset of binding modes + whether any two anchors sit on neighbouring sites)."""
    ranked = [r for r in log["records"] if r["ranked"]]
    if not ranked:
        return {"n_within_window": 0, "motifs": {}}
    e0 = min(r["energy"] for r in ranked)
    motifs: dict[str, int] = {}
    n = 0
    for r in ranked:
        if r["energy"] - e0 <= window_eV:
            n += 1
            m = "+".join(sorted(g["mode_id"] for g in r["genes"]))
            motifs[m] = motifs.get(m, 0) + 1
    return {"window_eV": window_eV, "n_within_window": n, "motifs": motifs}


def write_manifest(out_dir: str | Path, space: SearchSpace, logs: list[dict], provenance: dict,
                   write_structures: str = "unique") -> dict:
    """``manifest.json`` (space, provenance, per-attempt records with seeds, parentage, mutations,
    validity, hashes) and extended-XYZ files of the initial and final structures, which keep the
    atom_id / role / mol_id arrays.  ``write_structures``: 'unique' | 'evaluated' | 'none'."""
    from ase.io import write
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for log in logs:
        tag = f"{log['arm']}_s{log['seed']}"
        recs = []
        for r in log["records"]:
            q = {k: v for k, v in r.items() if not k.startswith("_")}
            keep = r["evaluated"] and (write_structures == "evaluated" or
                                       (write_structures == "unique" and r["status"] == "unique"))
            if keep and "_atoms" in r:
                d = out / "structures" / tag
                d.mkdir(parents=True, exist_ok=True)
                for kind, a in (("initial", r["_initial"]), ("final", r["_atoms"])):
                    f = d / f"a{r['attempt']:04d}_{kind}.extxyz"
                    write(f, a, format="extxyz")
                    q[f"{kind}_file"] = str(f.relative_to(out)).replace("\\", "/")
            recs.append(q)
        runs.append({**{k: v for k, v in log.items() if k != "records"}, "records": recs})
    man = {"schema": "adsearch_manifest/1", "provenance": provenance, "space": space.describe(),
           "coverage_by_mode": {m.mode_id: space.coverage(tuple(Gene(s.site_id, m.mode_id, "", 0.0, 0.0)
                                                                 for s in space.sites[:space.n_molecules]))
                                for m in space.spec.modes},
           "role_names": ROLE_NAMES, "runs": runs}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=_json))
    return man


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
