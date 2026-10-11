"""Passivator records, defect construction, pose generation and post-relaxation classification.

Every generated structure carries per-atom arrays:
  atom_id      stable integer identity (slab atoms keep their source index; adsorbate atoms
               get ids >= ADSORBATE_ID0); survives sorting for export
  fragment_id  covalent fragment index (slab FA cations, molecules, ions)
  origin       0 = substrate, k>0 = k-th placed passivator component
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers

from .ledger import Species, formula_key, _label_charge
from .surface import SurfaceRecord, Site, unwrap_along

ADSORBATE_ID0 = 100000
DONOR_ELEMENTS = ("N", "O", "S", "P", "F", "Cl", "Br", "I")


# ----------------------------------------------------------------------------- molecules
@dataclass
class Component:
    label: str                 # e.g. "PEA+", "I-", "PEA"
    indices: list[int]         # atoms of the passivator geometry belonging to it
    formal_charge: int
    formula: str


@dataclass
class Passivator:
    """A passivator in a *declared* chemical form. Charge and multiplicity are required:
    a plain XYZ carries neither, and they are never guessed."""
    name: str
    atoms: Atoms
    total_charge: int                     # DFT net charge of the isolated reference cell
    multiplicity: int
    components: list[Component]
    chemical_form: str                    # neutral_donor | ion_pair_salt | cation | anion | zwitterion
    protonation_note: str = ""
    donor_indices: list[int] = field(default_factory=list)
    source: str = ""

    def species(self, cell_charge: int | None = None) -> Species:
        comp = {}
        for s in self.atoms.get_chemical_symbols():
            comp[s] = comp.get(s, 0) + 1
        return Species(self.name, comp, cell_charge=self.total_charge if cell_charge is None else cell_charge,
                       spin_multiplicity=self.multiplicity, kind=self.chemical_form,
                       formal_units={c.label: 1 for c in self.components})


def fragments(atoms: Atoms, mult: float = 1.15, pbc: bool | None = None) -> list[list[int]]:
    """Covalent connected components (ionic Pb/halide/alkali atoms are isolated)."""
    from ase.neighborlist import natural_cutoffs, NeighborList
    a = atoms.copy()
    if pbc is not None:
        a.pbc = pbc
    syms = a.get_chemical_symbols()
    ionic = {"Pb", "Sn", "Cs", "Na", "K", "Rb"}
    cut = [0.0 if s in ionic else c for s, c in zip(syms, natural_cutoffs(a, mult=mult))]
    nl = NeighborList(cut, self_interaction=False, bothways=True, skin=0.0)
    nl.update(a)
    parent = list(range(len(a)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(a)):
        for j in nl.get_neighbors(i)[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(len(a)):
        groups.setdefault(find(i), []).append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def make_whole(atoms: Atoms, mult: float = 1.15) -> Atoms:
    """Unwrap covalent molecules split by periodic boundaries (BFS along bonds using the
    neighbour-list image offsets). Ionic atoms are left in place."""
    from ase.neighborlist import natural_cutoffs, NeighborList
    a = atoms.copy()
    if not a.pbc.any():
        return a
    syms = a.get_chemical_symbols()
    ionic = {"Pb", "Sn", "Cs", "Na", "K", "Rb"}
    cut = [0.0 if s in ionic else c for s, c in zip(syms, natural_cutoffs(a, mult=mult))]
    nl = NeighborList(cut, self_interaction=False, bothways=True, skin=0.0)
    nl.update(a)
    pos = a.positions.copy()
    seen = np.zeros(len(a), bool)
    for start in range(len(a)):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        while stack:
            i = stack.pop()
            idx, offs = nl.get_neighbors(i)
            for j, o in zip(idx, offs):
                if not seen[j]:
                    pos[j] = a.positions[j] + o @ a.cell.array - a.positions[i] + pos[i]
                    seen[j] = True
                    stack.append(int(j))
    a.positions = pos
    # gather disconnected components (e.g. counterions) to the minimum image nearest the
    # largest fragment, so an ion pair is placed as the pair, not as periodic neighbours
    groups = fragments(a.__class__(a.get_chemical_symbols(), positions=pos, cell=a.cell, pbc=False), pbc=False)
    if len(groups) > 1:
        main = max(groups, key=len)
        inv = np.linalg.inv(a.cell.array)
        for g in groups:
            if g is main:
                continue
            d = pos[g].mean(axis=0) - pos[main].mean(axis=0)
            shift = -np.round(d @ inv) @ a.cell.array
            # choose image minimising the closest-contact distance
            best, bs = np.inf, shift
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        t = shift + np.array([i, j, k]) @ a.cell.array
                        dm = np.linalg.norm((pos[g] + t)[:, None] - pos[main][None], axis=-1).min()
                        if dm < best:
                            best, bs = dm, t
            pos[g] += bs
        a.positions = pos
    return a


def load_passivator(path: str | Path, name: str, total_charge: int, multiplicity: int,
                    component_charges: dict[str, int], chemical_form: str,
                    protonation_note: str = "", donor_indices: list[int] | None = None) -> Passivator:
    """``component_charges`` maps fragment formula (Hill, e.g. 'C8H12N') -> formal charge.
    Every disconnected fragment must be declared; sum must equal ``total_charge``.
    Counterions are never dropped."""
    from ase.io import read
    atoms = make_whole(read(path))
    atoms.pbc = False
    comps = []
    for g in fragments(atoms, pbc=False):
        f = formula_key({s: sum(1 for i in g if atoms[i].symbol == s) for s in {atoms[i].symbol for i in g}})
        if f not in component_charges:
            raise ValueError(f"{name}: fragment {f} (atoms {g[:6]}...) has no declared formal charge. "
                             f"Declared: {list(component_charges)}")
        q = component_charges[f]
        lab = f + ("+" * q if q > 0 else "-" * (-q)) if q else f
        comps.append(Component(lab, g, q, f))
    if sum(c.formal_charge for c in comps) != total_charge:
        raise ValueError(f"{name}: component formal charges sum to {sum(c.formal_charge for c in comps)} "
                         f"but total_charge={total_charge}. Declare the compensating species explicitly.")
    nel = sum(atomic_numbers[s] for s in atoms.get_chemical_symbols()) - total_charge
    if (nel % 2 == 0) != (multiplicity % 2 == 1):
        raise ValueError(f"{name}: {nel} electrons inconsistent with multiplicity {multiplicity}")
    donors = donor_indices if donor_indices is not None else \
        [i for i, s in enumerate(atoms.get_chemical_symbols()) if s in DONOR_ELEMENTS]
    return Passivator(name, atoms, total_charge, multiplicity, comps, chemical_form,
                      protonation_note, donors, str(path))


# ----------------------------------------------------------------------------- substrate prep
def tag_substrate(atoms: Atoms) -> Atoms:
    a = atoms.copy()
    a.set_array("atom_id", np.arange(len(a)))
    frag = np.empty(len(a), int)
    for k, g in enumerate(fragments(a)):
        frag[g] = k
    a.set_array("fragment_id", frag)
    a.set_array("origin", np.zeros(len(a), int))
    return a


def remove_fragments(atoms: Atoms, rec: SurfaceRecord, atom_indices: list[int], label: str) -> tuple[Atoms, SurfaceRecord]:
    """Remove the whole covalent fragment containing each listed atom (e.g. one FA + one I
    = FAI Schottky pair). Returns new atoms and a record with the defect declared."""
    a = atoms if "fragment_id" in atoms.arrays else tag_substrate(atoms)
    drop = set()
    removed = []
    for i in atom_indices:
        g = np.where(a.arrays["fragment_id"] == a.arrays["fragment_id"][i])[0]
        drop |= set(g.tolist())
        removed.append(formula_key({s: int(sum(a[j].symbol == s for j in g)) for s in {a[j].symbol for j in g}}))
    keep = [i for i in range(len(a)) if i not in drop]
    new = a[keep]
    import copy
    r = copy.deepcopy(rec)
    r.surface_id = f"{rec.surface_id}__{label}"
    for e in [atoms[i].symbol for i in sorted(drop)]:
        r.composition[e] -= 1
    r.composition = {k: v for k, v in r.composition.items() if v}
    r.defects = rec.defects + [{"label": label, "removed_units": removed,
                                "removed_atom_ids": sorted(int(a.arrays['atom_id'][i]) for i in drop),
                                "positions": [atoms.positions[i].tolist() for i in atom_indices]}]
    from .ledger import identify_units
    units, unknown = identify_units(new)
    r.formal_units, r.unknown_fragments = units, unknown
    r.formal_charge = None if unknown else sum(_label_charge(k) * v for k, v in units.items())
    if r.formal_charge:
        r.notes = r.notes + [f"Defect '{label}' leaves formal charge {r.formal_charge:+d}; a neutral cell "
                             "implies electronic compensation unless NELECT is changed explicitly."]
    return new, r


# ----------------------------------------------------------------------------- geometry helpers
def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _align(u, v):
    """Rotation taking unit vector u onto v."""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    c = np.dot(u, v)
    if c > 1 - 1e-10:
        return np.eye(3)
    if c < -1 + 1e-10:
        p = np.cross(u, [1, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(u, [0, 1, 0])
        return _rot(p, np.pi)
    return _rot(np.cross(u, v), np.arccos(c))


def min_image_distances(cell: np.ndarray, pbc, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """|A| x |B| minimum-image distance matrix (general triclinic, checks 27 images)."""
    inv = np.linalg.inv(cell)
    d = A[:, None, :] - B[None, :, :]
    f = d @ inv
    for k in range(3):
        if pbc[k]:
            f[..., k] -= np.round(f[..., k])
    best = np.full(d.shape[:2], np.inf)
    shifts = [np.array([i, j, k]) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
              if all(pbc[x] or s == 0 for x, s in enumerate((i, j, k)))]
    for s in shifts:
        best = np.minimum(best, np.linalg.norm((f + s) @ cell, axis=-1))
    return best


def clash_report(cell, pbc, symA, posA, symB, posB, scale: float = 1.0) -> list[tuple[int, int, float, float]]:
    D = min_image_distances(cell, pbc, posA, posB)
    rA = np.array([covalent_radii[atomic_numbers[s]] for s in symA])
    rB = np.array([covalent_radii[atomic_numbers[s]] for s in symB])
    lim = scale * (rA[:, None] + rB[None, :])
    bad = np.argwhere(D < lim)
    return [(int(i), int(j), float(D[i, j]), float(lim[i, j])) for i, j in bad]


# ----------------------------------------------------------------------------- poses
@dataclass
class PoseSpec:
    pose_id: str
    passivator: str
    site_id: str
    site_kind: str
    face: str
    anchor_index: int
    height: float
    tilt_deg: float
    azimuth_deg: float
    mode: str                      # rigid | separated_ions
    extra: dict = field(default_factory=dict)


def place(substrate: Atoms, rec: SurfaceRecord, pas: Passivator, site: Site, anchor: int,
          height: float, tilt_deg: float, azimuth_deg: float, origin_tag: int = 1,
          mol_atoms: Atoms | None = None) -> Atoms:
    """Put ``anchor`` atom of the passivator ``height`` A above ``site`` along the outward
    normal of ``site.face``; the anchor->centroid vector points outward, tilted by
    ``tilt_deg`` and rotated about the normal by ``azimuth_deg``."""
    m = (mol_atoms if mol_atoms is not None else pas.atoms).copy()
    ax = rec.normal_axis
    n = substrate.cell.array[ax] / np.linalg.norm(substrate.cell.array[ax])
    if site.face == "bottom":
        n = -n
    pos = m.positions - m.positions[anchor]
    c = pos.mean(axis=0)
    R = _align(c if np.linalg.norm(c) > 1e-6 else n, n)
    perp = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0])
    R = _rot(n, np.radians(azimuth_deg)) @ _rot(perp, np.radians(tilt_deg)) @ R
    new = (R @ pos.T).T + np.asarray(site.position) + height * n
    m.positions = new
    m.cell = substrate.cell
    m.pbc = substrate.pbc
    base = int(substrate.arrays["atom_id"].max()) + 1 if "atom_id" in substrate.arrays else ADSORBATE_ID0
    base = max(base, ADSORBATE_ID0 + 1000 * (origin_tag - 1))
    m.set_array("atom_id", base + np.arange(len(m)))
    fbase = int(substrate.arrays["fragment_id"].max()) + 1
    fr = np.empty(len(m), int)
    for k, g in enumerate(fragments(m, pbc=False)):
        fr[g] = fbase + k
    m.set_array("fragment_id", fr)
    m.set_array("origin", np.full(len(m), origin_tag))
    for k in list(m.arrays):
        if k not in ("numbers", "positions", "atom_id", "fragment_id", "origin"):
            del m.arrays[k]
    return m


def generate_poses(substrate: Atoms, rec: SurfaceRecord, pas: Passivator, sites: list[Site],
                   heights=(2.6, 3.0), tilts=(0.0, 30.0), n_azimuth: int = 4,
                   anchors: list[int] | None = None, max_poses: int = 200, seed: int = 0,
                   clash_scale: float = 1.0, wrap: bool = True) -> list[tuple[Atoms, PoseSpec]]:
    """Enumerate rigid poses (salts stay as the declared ion pair) and keep clash-free ones.
    Sampling is systematic over sites x anchors x heights x tilts x azimuths, then
    shuffled with ``seed`` and truncated to ``max_poses`` (the pose budget is reported)."""
    sub = substrate if "atom_id" in substrate.arrays else tag_substrate(substrate)
    anchors = anchors if anchors is not None else pas.donor_indices
    rng = np.random.default_rng(seed)
    combos = [(s, a, h, t, az) for s in sites for a in anchors for h in heights for t in tilts
              for az in np.arange(n_azimuth) * 360.0 / n_azimuth]
    rng.shuffle(combos)
    out = []
    for s, a, h, t, az in combos:
        m = place(sub, rec, pas, s, a, h, t, float(az))
        if clash_report(sub.cell.array, sub.pbc, sub.get_chemical_symbols(), sub.positions,
                        m.get_chemical_symbols(), m.positions, clash_scale):
            continue
        # vacuum check: adsorbate must stay inside the vacuum gap, not cross the periodic image
        ax = rec.normal_axis
        n = sub.cell.array[ax] / np.linalg.norm(sub.cell.array[ax])
        proj = m.positions @ n
        if proj.max() - proj.min() > rec.vacuum - 8.0:
            continue   # adsorbate would approach the periodic image of the opposite face
        full = sub + m
        if wrap:
            full.wrap()
        spec = PoseSpec(f"{pas.name}@{s.site_id}/a{a}/h{h}/t{t:g}/az{az:g}", pas.name, s.site_id, s.kind,
                        s.face, int(a), float(h), float(t), float(az), "rigid",
                        {"seed": seed, "n_candidates": len(combos)})
        out.append((full, spec))
        if len(out) >= max_poses:
            break
    return out


# ----------------------------------------------------------------------------- after relaxation
def classify_final_state(initial: Atoms, final: Atoms, rec: SurfaceRecord,
                         desorb_dist: float = 4.0) -> dict:
    """Compare covalent fragments of placed components before vs after relaxation.
    Atom correspondence uses ``atom_id`` (never position matching)."""
    def by_id(a):
        return {int(i): k for k, i in enumerate(a.arrays["atom_id"])}
    mi, mf = by_id(initial), by_id(final)
    if set(mi) != set(mf):
        return {"final_state_class": "atom_inventory_changed", "details": "atom ids differ"}
    order = [mf[i] for i in initial.arrays["atom_id"]]
    fin = final[order]
    fin.set_array("origin", initial.arrays["origin"])

    def frag_sets(a):
        return {frozenset(int(a.arrays["atom_id"][i]) for i in g) for g in fragments(a)}
    fi, ff = frag_sets(initial), frag_sets(fin)
    origin = initial.arrays["origin"]
    ads_ids = {int(i) for i, o in zip(initial.arrays["atom_id"], origin) if o > 0}
    events = []
    for g in fi - ff:
        if g & ads_ids or any(g2 & g and g2 != g for g2 in ff):
            events.append(sorted(g))
    syms = dict(zip(initial.arrays["atom_id"].tolist(), initial.get_chemical_symbols()))
    cls = "intact_adsorption"
    details = []
    for gi in fi:
        if not (gi & ads_ids) and not any(g2 & gi and g2 != gi and (g2 & ads_ids) for g2 in ff):
            continue
        partners = [g2 for g2 in ff if g2 & gi]
        if len(partners) > 1 or partners[0] != gi:
            main = max(partners, key=lambda g2: len(g2 & gi))
            moved = [syms[i] for i in (gi - main) | (main - gi)]
            if moved and all(s == "H" for s in moved):
                cls = "proton_transfer"
            else:
                cls = "reactive_rearrangement"
            details.append({"initial_fragment": sorted(gi), "final_fragments": [sorted(p) for p in partners],
                            "moved_atoms": moved})
    # desorption: min distance from each placed component to substrate
    sub_idx = np.where(origin == 0)[0]
    for k in sorted(set(origin[origin > 0])):
        idx = np.where(origin == k)[0]
        d = min_image_distances(fin.cell.array, fin.pbc, fin.positions[idx], fin.positions[sub_idx]).min()
        if d > desorb_dist and cls == "intact_adsorption":
            cls = "desorbed"
            details.append({"component": int(k), "min_distance_A": float(d)})
    return {"final_state_class": cls, "details": details,
            "reaction_accounting": "unchanged" if cls in ("intact_adsorption",) else
            "must be re-declared: final state is not the declared molecular adsorption"}


def fill_schottky(vac_slab: Atoms, vac_rec: SurfaceRecord, pas: Passivator, cation_fragment_site: list[float],
                  anion_site: list[float], face: str = "top", anchor_element: str = "N",
                  tilts=(0, 20, 40), azimuths=range(0, 360, 30), heights=(0.0, 0.5, 1.0),
                  clash_scale: float = 1.0) -> tuple[Atoms, dict]:
    """Salt exchange into an A-site + halide Schottky pair (confirmed convention, 2026-10-11): the organic
    cation's ammonium N goes to the removed FA+ site and the salt's halide (I- or Cl-) to the removed I- site.
    Returns the clash-free placement with the largest closest contact, plus a record of the choice."""
    cats = [c for c in pas.components if c.formal_charge > 0]
    ans = [c for c in pas.components if c.formal_charge < 0]
    if len(cats) != 1 or len(ans) != 1 or len(ans[0].indices) != 1:
        raise ValueError("fill_schottky needs a 1:1 salt of one organic cation and one monatomic halide")
    cidx = cats[0].indices
    cation = pas.atoms[cidx]
    nloc = [k for k, i in enumerate(cidx) if pas.atoms[i].symbol == anchor_element]
    if not nloc:
        raise ValueError(f"cation has no {anchor_element} anchor")
    hal = pas.atoms[ans[0].indices[0]].symbol
    site = Site("V_FA", "FA_vacancy", face, list(cation_fragment_site), None)
    best = None
    for t in tilts:
        for az in azimuths:
            for h in heights:
                m = place(vac_slab, vac_rec, pas, site, nloc[0], h, t, az, mol_atoms=cation)
                x = Atoms(hal, positions=[anion_site], cell=vac_slab.cell, pbc=vac_slab.pbc)
                x.set_array("atom_id", np.array([int(m.arrays["atom_id"].max()) + 1]))
                x.set_array("fragment_id", np.array([int(m.arrays["fragment_id"].max()) + 1]))
                x.set_array("origin", np.array([2]))
                mm = m + x
                if clash_report(vac_slab.cell.array, vac_slab.pbc, vac_slab.get_chemical_symbols(),
                                vac_slab.positions, mm.get_chemical_symbols(), mm.positions, clash_scale):
                    continue
                d = float(min_image_distances(vac_slab.cell.array, vac_slab.pbc, mm.positions, vac_slab.positions).min())
                if best is None or d > best[0]:
                    best = (d, vac_slab + mm, {"tilt_deg": t, "azimuth_deg": float(az), "height_A": h,
                                               "min_contact_A": d, "halide": hal,
                                               "convention": "cation N -> V_FA site; halide -> V_I site"})
    if best is None:
        raise RuntimeError("no clash-free exchange placement; widen tilts/heights or check the vacancy sites")
    return best[1], best[2]
