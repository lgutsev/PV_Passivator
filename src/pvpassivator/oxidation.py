"""Surface halide oxyanions: turn one surface iodide into IO_n- (n = 1-4) and enumerate orientations.

The oxidised product keeps the halide's -1 charge, so the cell stays formally neutral and the
reaction is   slab + n/2 O2 -> slab(IO_n-).   O2 enters only as a thermodynamic reservoir in that
ledger entry: this module builds end states, not O2-surface encounter geometries or paths.

Starting geometries are idealised (fixed X-O length and O-X-O angle per n) and are meant to be
relaxed. Orientation is described by the lone-pair direction L of the oxyanion: the O atoms sit on
the side opposite L, at the angle that reproduces the requested O-X-O angle, rotated by ``phi``
about L. Defaults scan L along the outward normal, the inward normal, and tilted toward four
in-plane directions referenced to the halide's nearest Pb neighbours.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict

import numpy as np
from ase import Atoms

from .ledger import Species, Reaction, register_unit, identify_units, _label_charge
from .surface import SurfaceRecord, Site
from .poses import ADSORBATE_ID0, tag_substrate, clash_report, min_image_distances, _rot

# Idealised starting geometry per number of O atoms: (X-O length A, O-X-O angle deg, label, ox. state)
# Lengths/angles are typical crystal/gas-phase values for iodine oxyanions; relaxation sets the real ones.
IODINE_OXYANIONS = {
    1: (1.99, None, "IO-", +1),      # hypoiodite
    2: (1.87, 102.0, "IO2-", +3),    # iodite
    3: (1.81, 99.0, "IO3-", +5),     # iodate (C3v pyramid)
    4: (1.78, 109.4712, "IO4-", +7),  # periodate (tetrahedral)
}
for _n, (_d, _a, _lab, _ox) in IODINE_OXYANIONS.items():
    register_unit({"I": 1, "O": _n}, _lab, -1)

O2_REFERENCE = Species("O2(g)", {"O": 2}, cell_charge=0, spin_multiplicity=3, kind="molecule",
                       notes="gas-phase triplet O2 used only as the oxygen reservoir")


@dataclass
class OxyanionSpec:
    pose_id: str
    halide_index: int
    halide_atom_id: int
    n_oxygen: int
    label: str
    oxidation_state: int
    lone_pair: str                 # orientation label of the lone-pair direction
    lone_pair_vector: list[float]
    phi_deg: float
    lift: float                    # A, halide displacement along the outward normal
    bond_length: float
    angle_deg: float | None
    nearest_pb: list[int]
    min_O_contact: dict = field(default_factory=dict)   # shortest O-substrate distance

    def to_dict(self) -> dict:
        return asdict(self)


def _outward_normal(atoms: Atoms, rec: SurfaceRecord, face: str) -> np.ndarray:
    n = atoms.cell.array[rec.normal_axis] / np.linalg.norm(atoms.cell.array[rec.normal_axis])
    return n if face == "top" else -n


def _nearest_pb(atoms: Atoms, idx: int, k: int = 2, cutoff: float = 3.8) -> list[int]:
    syms = np.array(atoms.get_chemical_symbols())
    pb = np.where(syms == "Pb")[0]
    if len(pb) == 0:
        return []
    d = min_image_distances(atoms.cell.array, atoms.pbc, atoms.positions[[idx]], atoms.positions[pb])[0]
    order = np.argsort(d)
    return [int(pb[i]) for i in order[:k] if d[i] < cutoff]


def _mic_vector(atoms: Atoms, i: int, j: int) -> np.ndarray:
    v = atoms.positions[j] - atoms.positions[i]
    f = v @ np.linalg.inv(atoms.cell.array)
    for k in range(3):
        if atoms.pbc[k]:
            f[k] -= np.round(f[k])
    return f @ atoms.cell.array


def _perp_unit(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    p = v - np.dot(v, n) * n
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0])
    return p / np.linalg.norm(p)


def oxygen_offsets(n_oxygen: int, lone_pair: np.ndarray, ref: np.ndarray, phi_deg: float = 0.0,
                   bond_length: float | None = None, angle_deg: float | None = None) -> np.ndarray:
    """O positions relative to the halide for an idealised IO_n with lone pair along ``lone_pair``.
    ``ref`` fixes phi = 0 (any vector not parallel to the lone pair)."""
    d0, a0, _, _ = IODINE_OXYANIONS[n_oxygen]
    d = bond_length or d0
    a = angle_deg if angle_deg is not None else a0
    L = np.asarray(lone_pair, float) / np.linalg.norm(lone_pair)
    A = -L                                     # axis on the O side
    e1 = _perp_unit(np.asarray(ref, float), A)
    e2 = np.cross(A, e1)
    phi = np.radians(phi_deg)
    if n_oxygen == 1:
        dirs = [A]
    elif n_oxygen == 2:
        beta = np.radians(a) / 2               # each bond is alpha/2 off the axis, in one plane
        u = np.cos(phi) * e1 + np.sin(phi) * e2
        dirs = [np.cos(beta) * A + np.sin(beta) * u, np.cos(beta) * A - np.sin(beta) * u]
    elif n_oxygen == 3:
        c = (np.cos(np.radians(a)) + 0.5) / 1.5    # cos^2(beta) for C3v bonds at beta from the axis
        beta = np.arccos(np.sqrt(c))
        dirs = []
        for k in range(3):
            g = phi + 2 * np.pi * k / 3
            dirs.append(np.cos(beta) * A + np.sin(beta) * (np.cos(g) * e1 + np.sin(g) * e2))
    elif n_oxygen == 4:
        # one O along the axis, three at the O-X-O angle from it (no lone pair; L is orientation only)
        ta = np.radians(a)
        dirs = [A]
        for k in range(3):
            g = phi + 2 * np.pi * k / 3
            dirs.append(np.cos(ta) * A + np.sin(ta) * (np.cos(g) * e1 + np.sin(g) * e2))
    else:
        raise ValueError("n_oxygen must be 1-4")
    return d * np.array(dirs)


def default_lone_pairs(normal: np.ndarray, ref: np.ndarray, tilt_deg: float = 45.0) -> list[tuple[str, np.ndarray]]:
    """Outward, inward, and tilted toward +/- ref and +/- (normal x ref)."""
    r = _perp_unit(ref, normal)
    s = np.cross(normal, r)
    t = np.radians(tilt_deg)
    out = [("out", normal), ("in", -normal)]
    for lab, u in (("tilt+pb", r), ("tilt-pb", -r), ("tilt+perp", s), ("tilt-perp", -s)):
        out.append((lab, np.cos(t) * normal + np.sin(t) * u))
    return out


def oxyanion_poses(substrate: Atoms, rec: SurfaceRecord, site: Site, n_oxygen: int = 3,
                   lone_pairs: list[tuple[str, np.ndarray]] | None = None, phis=(0.0, 60.0),
                   lifts=(0.0, 0.8), clash_scale: float = 0.85, min_pb_o: float = 2.05,
                   wrap: bool = True) -> list[tuple[Atoms, OxyanionSpec]]:
    """Replace the halide at ``site`` by IO_n- in every requested orientation; keep clash-free ones.

    Clashes are checked between the new O atoms and every substrate atom except the oxidised halide
    (covalent-radius sum x ``clash_scale``); O-Pb contacts down to ``min_pb_o`` are allowed, since an
    O bridging to a surface Pb is an intended motif."""
    if site.kind != "exposed_I" or site.atom_index is None:
        raise ValueError(f"site {site.site_id} is not an exposed halide")
    sub = substrate if "atom_id" in substrate.arrays else tag_substrate(substrate)
    h = int(site.atom_index)
    if sub[h].symbol != "I":
        raise ValueError(f"atom {h} is {sub[h].symbol}; only iodide oxyanions are parameterised")
    n = _outward_normal(sub, rec, site.face)
    pbs = _nearest_pb(sub, h)
    ref = _mic_vector(sub, h, pbs[0]) if pbs else np.cross(n, [1, 0, 0])
    lone_pairs = lone_pairs or default_lone_pairs(n, ref)
    d0, a0, label, ox = IODINE_OXYANIONS[n_oxygen]
    syms = np.array(sub.get_chemical_symbols())
    others = np.array([i for i in range(len(sub)) if i != h])
    phis = (0.0,) if n_oxygen == 1 else phis
    out = []
    for lp_label, L in lone_pairs:
        for phi in phis:
            for lift in lifts:
                xpos = sub.positions[h] + lift * n
                opos = xpos + oxygen_offsets(n_oxygen, L, ref, phi)
                bad = clash_report(sub.cell.array, sub.pbc, ["O"] * n_oxygen, opos,
                                   list(syms[others]), sub.positions[others], clash_scale)
                bad = [b for b in bad if not (syms[others[b[1]]] == "Pb" and b[2] >= min_pb_o)]
                if bad:
                    continue
                pb_idx = others[syms[others] == "Pb"]
                if len(pb_idx) and min_image_distances(sub.cell.array, sub.pbc, opos,
                                                       sub.positions[pb_idx]).min() < min_pb_o:
                    continue
                # also keep the halide itself clear of the substrate after lifting
                if lift and clash_report(sub.cell.array, sub.pbc, ["I"], xpos[None],
                                         list(syms[others]), sub.positions[others], clash_scale):
                    continue
                new = sub.copy()
                new.positions[h] = xpos
                ox_atoms = Atoms(["O"] * n_oxygen, positions=opos, cell=sub.cell, pbc=sub.pbc)
                base = max(int(sub.arrays["atom_id"].max()) + 1, ADSORBATE_ID0)
                ox_atoms.set_array("atom_id", base + np.arange(n_oxygen))
                ox_atoms.set_array("fragment_id", np.full(n_oxygen, int(sub.arrays["fragment_id"][h])))
                ox_atoms.set_array("origin", np.full(n_oxygen, 1))
                for k in list(new.arrays):
                    if k not in ("numbers", "positions", "atom_id", "fragment_id", "origin"):
                        del new.arrays[k]
                full = new + ox_atoms
                if wrap:
                    full.wrap()
                D = min_image_distances(sub.cell.array, sub.pbc, opos, sub.positions[others])
                j = np.unravel_index(np.argmin(D), D.shape)
                spec = OxyanionSpec(
                    f"{label}@{site.site_id}/lp-{lp_label}/phi{phi:g}/lift{lift:g}", h,
                    int(sub.arrays["atom_id"][h]), n_oxygen, label, ox, lp_label,
                    [float(x) for x in L / np.linalg.norm(L)], float(phi), float(lift), d0, a0, pbs,
                    {"element": str(syms[others[j[1]]]), "index": int(others[j[1]]), "distance_A": float(D[j])})
                out.append((full, spec))
    return out


def oxidised_record(rec: SurfaceRecord, atoms: Atoms, spec: OxyanionSpec) -> SurfaceRecord:
    """Copy of the surface record for the oxidised structure, with the defect declared."""
    r = copy.deepcopy(rec)
    r.surface_id = f"{rec.surface_id}__{spec.label}{spec.halide_index}"
    r.composition["O"] = r.composition.get("O", 0) + spec.n_oxygen
    units, unknown = identify_units(atoms)
    r.formal_units, r.unknown_fragments = units, unknown
    r.formal_charge = None if unknown else sum(_label_charge(k) * v for k, v in units.items())
    r.defects = rec.defects + [{"label": f"{spec.label} at halide {spec.halide_index}",
                                "pose_id": spec.pose_id, "oxidation_state": spec.oxidation_state}]
    return r


def oxidation_reaction(slab: Species, product: Species, n_oxygen: int) -> Reaction:
    """slab + n/2 O2 -> slab(IO_n-); checked for atom and charge balance."""
    rx = Reaction([(1, slab), (n_oxygen / 2, O2_REFERENCE)], [(1, product)],
                  label=f"surface iodide -> {IODINE_OXYANIONS[n_oxygen][2]}", energy_kind="reaction")
    rx.assert_balanced()
    return rx
