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
from .poses import ADSORBATE_ID0, tag_substrate, clash_report, min_image_distances, _rot, _align, remove_fragments

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


def _check_site(substrate: Atoms, site: Site) -> tuple[Atoms, int]:
    if site.kind != "exposed_I" or site.atom_index is None:
        raise ValueError(f"site {site.site_id} is not an exposed halide")
    sub = substrate if "atom_id" in substrate.arrays else tag_substrate(substrate)
    h = int(site.atom_index)
    if sub[h].symbol != "I":
        raise ValueError(f"atom {h} is {sub[h].symbol}; only iodide oxyanions are parameterised")
    return sub, h


def _assemble(sub: Atoms, h: int, xpos: np.ndarray, opos: np.ndarray, clash_scale: float, min_pb_o: float,
              wrap: bool) -> tuple[Atoms, dict] | None:
    """Move the halide to ``xpos`` and add O at ``opos``; None if an O or the moved halide clashes.
    O-Pb contacts down to ``min_pb_o`` are allowed (an O bridging to a surface Pb is an intended motif)."""
    syms = np.array(sub.get_chemical_symbols())
    others = np.array([i for i in range(len(sub)) if i != h])
    nO = len(opos)
    bad = clash_report(sub.cell.array, sub.pbc, ["O"] * nO, opos, list(syms[others]), sub.positions[others],
                       clash_scale)
    if [b for b in bad if not (syms[others[b[1]]] == "Pb" and b[2] >= min_pb_o)]:
        return None
    pb_idx = others[syms[others] == "Pb"]
    if len(pb_idx) and min_image_distances(sub.cell.array, sub.pbc, opos, sub.positions[pb_idx]).min() < min_pb_o:
        return None
    if np.linalg.norm(xpos - sub.positions[h]) > 1e-9 and clash_report(
            sub.cell.array, sub.pbc, ["I"], xpos[None], list(syms[others]), sub.positions[others], clash_scale):
        return None
    new = sub.copy()
    new.positions[h] = xpos
    ox_atoms = Atoms(["O"] * nO, positions=opos, cell=sub.cell, pbc=sub.pbc)
    base = max(int(sub.arrays["atom_id"].max()) + 1, ADSORBATE_ID0)
    ox_atoms.set_array("atom_id", base + np.arange(nO))
    ox_atoms.set_array("fragment_id", np.full(nO, int(sub.arrays["fragment_id"][h])))
    ox_atoms.set_array("origin", np.full(nO, 1))
    for k in list(new.arrays):
        if k not in ("numbers", "positions", "atom_id", "fragment_id", "origin"):
            del new.arrays[k]
    full = new + ox_atoms
    if wrap:
        full.wrap()
    D = min_image_distances(sub.cell.array, sub.pbc, opos, sub.positions[others])
    j = np.unravel_index(np.argmin(D), D.shape)
    return full, {"element": str(syms[others[j[1]]]), "index": int(others[j[1]]), "distance_A": float(D[j])}


def oxyanion_poses(substrate: Atoms, rec: SurfaceRecord, site: Site, n_oxygen: int = 3,
                   lone_pairs: list[tuple[str, np.ndarray]] | None = None, phis=(0.0, 60.0),
                   lifts=(0.0, 0.8), clash_scale: float = 0.85, min_pb_o: float = 2.05,
                   wrap: bool = True) -> list[tuple[Atoms, OxyanionSpec]]:
    """Replace the halide at ``site`` by IO_n- in every requested orientation; keep clash-free ones.

    Clashes are checked between the new O atoms and every substrate atom except the oxidised halide
    (covalent-radius sum x ``clash_scale``); O-Pb contacts down to ``min_pb_o`` are allowed, since an
    O bridging to a surface Pb is an intended motif."""
    sub, h = _check_site(substrate, site)
    n = _outward_normal(sub, rec, site.face)
    pbs = _nearest_pb(sub, h)
    ref = _mic_vector(sub, h, pbs[0]) if pbs else np.cross(n, [1, 0, 0])
    lone_pairs = lone_pairs or default_lone_pairs(n, ref)
    d0, a0, label, ox = IODINE_OXYANIONS[n_oxygen]
    phis = (0.0,) if n_oxygen == 1 else phis
    out = []
    for lp_label, L in lone_pairs:
        for phi in phis:
            for lift in lifts:
                xpos = sub.positions[h] + lift * n
                opos = xpos + oxygen_offsets(n_oxygen, L, ref, phi)
                got = _assemble(sub, h, xpos, opos, clash_scale, min_pb_o, wrap)
                if got is None:
                    continue
                full, contact = got
                spec = OxyanionSpec(
                    f"{label}@{site.site_id}/lp-{lp_label}/phi{phi:g}/lift{lift:g}", h,
                    int(sub.arrays["atom_id"][h]), n_oxygen, label, ox, lp_label,
                    [float(x) for x in L / np.linalg.norm(L)], float(phi), float(lift), d0, a0, pbs, contact)
                out.append((full, spec))
    return out


def _bridge_direction(sub: Atoms, h: int, n: np.ndarray, pbs: list[int]) -> np.ndarray:
    """In-plane unit vector of the Pb-X-Pb bridge (from one Pb neighbour to the other); falls back to the
    in-plane direction of the nearest Pb, then to any in-plane vector."""
    if len(pbs) >= 2:
        v = _mic_vector(sub, h, pbs[1]) - _mic_vector(sub, h, pbs[0])
    elif pbs:
        v = _mic_vector(sub, h, pbs[0])
    else:
        v = np.cross(n, [1, 0, 0])
    return _perp_unit(v, n)


def oxyanion_preset_poses(substrate: Atoms, rec: SurfaceRecord, site: Site, n_oxygen: int = 3,
                          preset: str = "O-down", psis=None, phis=None, lifts=(0.0,),
                          clash_scale: float = 0.85,
                          min_pb_o: float = 2.05, wrap: bool = True) -> list[tuple[Atoms, OxyanionSpec]]:
    """Named starting orientations for the two bonding limits of a surface oxyanion.

    ``X-down`` (Pb-X-O): the halide stays in its site with the lone pair pointing into the slab, so the
    O atoms point to vacuum; ``phis`` rotates them about the normal (default (90,) for IO2-, which puts the
    O-X-O plane perpendicular to the bridge; along the bridge the O land on the Pb; (0, 60) otherwise). ``lifts`` (A) optionally raises the
    halide: on an in-plane PbI2-face site the near-sideways O of an IO3- pyramid clash with neighbouring
    halides at lift 0, so no X-down IO3- pose survives without a lift (e.g. lifts=(0.0, 0.8)).
    ``O-down`` (Pb-O-X): one O sits in the old halide site and the halide is lifted along the outward
    normal by the X-O length, so that bond points straight down. ``psis`` (deg) rotates the other O atoms
    about the normal; psi = 0 puts the O-X-O plane perpendicular to the Pb-X-Pb bridge (for IO2-, the
    plane holding the normal and normal x bridge). Defaults: (0,) for IO-, (0, 180) for IO2-,
    (0, 90) for IO3- and IO4-.
    """
    sub, h = _check_site(substrate, site)
    n = _outward_normal(sub, rec, site.face)
    if preset == "X-down":
        if phis is None:
            phis = (90.0,) if n_oxygen == 2 else (0.0, 60.0)
        out = oxyanion_poses(substrate, rec, site, n_oxygen, lone_pairs=[("X-down", -n)], phis=phis,
                             lifts=lifts, clash_scale=clash_scale, min_pb_o=min_pb_o, wrap=wrap)
        for _, spec in out:
            spec.pose_id = spec.pose_id.replace("/lp-X-down/", "/X-down/")
        return out
    if preset != "O-down":
        raise ValueError(f"unknown preset {preset!r}; use 'O-down' or 'X-down'")
    pbs = _nearest_pb(sub, h)
    perp = np.cross(n, _bridge_direction(sub, h, n, pbs))     # in-plane, perpendicular to the bridge
    d0, a0, label, ox = IODINE_OXYANIONS[n_oxygen]
    if psis is None:
        psis = {1: (0.0,), 2: (0.0, 180.0)}.get(n_oxygen, (0.0, 90.0))
    # idealised offsets in a local frame, rotated so bond 0 points along -n
    off = oxygen_offsets(n_oxygen, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    off = off @ _align(off[0], -n).T
    out = []
    for psi in psis:
        o = off
        if n_oxygen > 1:
            # psi = 0 where the in-plane part of bond 1 lies along +perp
            p1 = o[1] - np.dot(o[1], n) * n
            ang = np.arctan2(np.dot(np.cross(p1, perp), n), np.dot(p1, perp)) + np.radians(psi)
            o = o @ _rot(n, ang).T
        xpos = sub.positions[h] + d0 * n
        got = _assemble(sub, h, xpos, xpos + o, clash_scale, min_pb_o, wrap)
        if got is None:
            continue
        full, contact = got
        L = -o.mean(axis=0)
        L = L / np.linalg.norm(L) if n_oxygen < 4 and np.linalg.norm(L) > 1e-9 else n
        spec = OxyanionSpec(f"{label}@{site.site_id}/O-down/psi{psi:g}", h, int(sub.arrays["atom_id"][h]),
                            n_oxygen, label, ox, "O-down", [float(x) for x in L], float(psi), float(d0),
                            d0, a0, pbs, contact)
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


# ----------------------------------------------------------------------------- after relaxation
# Fingerprint channels: (centre, neighbour element, number of nearest distances kept). O distances are
# pooled over all O atoms, so the fingerprint does not depend on O numbering. Only the first shell is
# used: outer Pb and the cation H-bond network vary between relaxations of the same motif.
HEAVY_CHANNELS = (("I", "O", 4), ("O", "Pb", 3), ("I", "Pb", 2))
H_CHANNELS = (("O", "H", 3),)
PB_O_BOND = 2.8     # A, O-Pb contact counted as a bond in the motif key


def oxyanion_atoms(atoms: Atoms, halide_atom_id: int) -> tuple[int, list[int]]:
    """Index of the oxidised halide and of the O atoms placed with it (same ``fragment_id``, ``origin`` > 0).
    Uses the arrays written by ``oxyanion_poses``; a structure read back without them is refused."""
    for k in ("atom_id", "fragment_id", "origin"):
        if k not in atoms.arrays:
            raise ValueError(f"structure has no '{k}' array; carry the pose arrays through relaxation")
    hits = np.where(atoms.arrays["atom_id"] == halide_atom_id)[0]
    if len(hits) != 1:
        raise ValueError(f"atom_id {halide_atom_id} found {len(hits)} times")
    h = int(hits[0])
    syms = np.array(atoms.get_chemical_symbols())
    o = np.where((syms == "O") & (atoms.arrays["origin"] > 0)
                 & (atoms.arrays["fragment_id"] == atoms.arrays["fragment_id"][h]))[0]
    if len(o) == 0:
        raise ValueError(f"no oxyanion O atoms attached to halide atom_id {halide_atom_id}")
    return h, [int(i) for i in o]


def oxyanion_fingerprint(atoms: Atoms, halide_atom_id: int,
                         channels=HEAVY_CHANNELS + H_CHANNELS) -> dict[str, np.ndarray]:
    """Sorted nearest-neighbour distances around the oxyanion per channel (missing neighbours pad with
    inf), plus ``o_pb_bonds``: per-O count of Pb within ``PB_O_BOND``, sorted."""
    h, o = oxyanion_atoms(atoms, halide_atom_id)
    syms = np.array(atoms.get_chemical_symbols())
    cell, pbc, pos = atoms.cell.array, atoms.pbc, atoms.positions
    out = {}
    for centre, elem, k in channels:
        cidx = [h] if centre == "I" else o
        nidx = np.array(o) if (centre, elem) == ("I", "O") else np.where(syms == elem)[0]
        nidx = nidx[~np.isin(nidx, cidx)]
        d = np.sort(min_image_distances(cell, pbc, pos[cidx], pos[nidx]).ravel()) if len(nidx) else np.array([])
        v = np.full(k, np.inf)
        v[:min(k, len(d))] = d[:k]
        out[f"{centre}-{elem}"] = v
    pb = np.where(syms == "Pb")[0]
    nb = (min_image_distances(cell, pbc, pos[o], pos[pb]) < PB_O_BOND).sum(1) if len(pb) else np.zeros(len(o))
    out["o_pb_bonds"] = np.sort(nb).astype(float)
    return out


def _max_dev(a: dict, b: dict, keys) -> float:
    worst = 0.0
    for key in keys:
        x, y = a[key], b[key]
        if np.any(np.isfinite(x) != np.isfinite(y)):
            return np.inf
        fin = np.isfinite(x)
        if fin.any():
            worst = max(worst, float(np.abs(x[fin] - y[fin]).max()))
    return worst


def group_relaxed_poses(structures: list[Atoms], halide_atom_id: int, energies: list[float] | None = None,
                        tol: float = 0.10, h_tol: float | None = None, energy_tol: float | None = None,
                        labels: list[str] | None = None) -> list[dict]:
    """Group relaxed oxyanion structures that reached the same local minimum.

    Same group = identical O-Pb bond counts and every heavy-atom first-shell distance (I-O, O-Pb, I-Pb)
    within ``tol`` (A) of the group's representative. O-H distances and energies are reported; they
    split groups only when ``h_tol`` / ``energy_tol`` are given (useful once relaxations are converged).
    Structures are visited in energy order (input order without energies) and join the closest
    matching representative, so each representative is its group's lowest-energy member."""
    n = len(structures)
    labels = labels or [str(i) for i in range(n)]
    if (energies is not None and len(energies) != n) or len(labels) != n:
        raise ValueError("energies/labels must match structures")
    fps = [oxyanion_fingerprint(a, halide_atom_id) for a in structures]
    heavy = [f"{c}-{e}" for c, e, _ in HEAVY_CHANNELS]
    hkeys = [f"{c}-{e}" for c, e, _ in H_CHANNELS]
    order = list(np.argsort(energies, kind="stable")) if energies is not None else list(range(n))
    groups: list[dict] = []
    for i in order:
        best, best_d = None, np.inf
        for g in groups:
            r = g["rep"]
            if not np.array_equal(fps[r]["o_pb_bonds"], fps[i]["o_pb_bonds"]):
                continue
            d = _max_dev(fps[r], fps[i], heavy)
            if d > tol or (h_tol is not None and _max_dev(fps[r], fps[i], hkeys) > h_tol):
                continue
            if energy_tol is not None and energies is not None and abs(energies[i] - energies[r]) > energy_tol:
                continue
            if d < best_d:
                best, best_d = g, d
        if best is None:
            groups.append({"rep": int(i), "members": [int(i)], "dev": 0.0})
        else:
            best["members"].append(int(i))
            best["dev"] = max(best["dev"], best_d)

    def clean(v):
        return [float(x) if np.isfinite(x) else None for x in v]
    out = []
    for g in groups:
        m, r = g["members"], g["rep"]
        rec = {"representative": labels[r], "members": [labels[i] for i in m], "multiplicity": len(m),
               "o_pb_bonds": [int(x) for x in fps[r]["o_pb_bonds"]], "max_heavy_dev_A": g["dev"],
               "max_h_dev_A": max(_max_dev(fps[r], fps[i], hkeys) for i in m),
               "fingerprint": {k: clean(v) for k, v in fps[r].items() if k != "o_pb_bonds"}}
        if energies is not None:
            es = [energies[i] for i in m]
            rec["energy"] = float(energies[r])
            rec["energy_spread"] = float(max(es) - min(es))
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- final-state classification
IO_BOND = 2.3          # A, halide-O distance still counted as an oxyanion bond (IO3- ~1.8, IO- ~2.0)
OH_BOND = 1.15         # A, O-H distance counted as a covalent bond (proton on an oxyanion O)
O_TRANSFER_PB = 2.4    # A, a cleaved O this close to a Pb is counted as transferred to the metal
DESORB_DIST = 4.0      # A, oxyanion-to-substrate distance beyond which it is desorbed


def _outward_height(atoms: Atoms, rec: SurfaceRecord, h: int) -> tuple[float, int]:
    """Height (A) of atom ``h`` above the outermost metal layer of its own face, and the face sign (+1 top)."""
    from .surface import unwrap_along, LAYER_TOL
    z = unwrap_along(atoms, rec.normal_axis)
    syms = np.array(atoms.get_chemical_symbols())
    metal = np.where(np.isin(syms, ["Pb", "Sn"]))[0]
    mid = 0.5 * (z[metal].max() + z[metal].min())
    sign = 1 if z[h] >= mid else -1
    edge = z[metal].max() if sign > 0 else z[metal].min()
    outer = metal[np.abs(z[metal] - edge) < LAYER_TOL]
    return float(sign * (z[h] - z[outer].mean())), sign


def oxyanion_state(initial: Atoms, final: Atoms, rec: SurfaceRecord, halide_atom_id: int) -> dict:
    """What happened to a placed oxyanion during relaxation. Atom correspondence by ``atom_id``.

    Class (first that applies): ``desorbed`` > ``o_transfer_to_metal`` (an O left the halide and sits on a Pb)
    > ``x_o_cleavage`` (an O left the halide otherwise) > ``protonated`` (an oxyanion O gained a covalent H)
    > ``intact_bound`` (all X-O bonds kept, at least one O on Pb) > ``intact_unbound``.
    Flags: ``below_metal_plane`` (halide below the outer Pb plane of its face), ``lifted_out`` (halide more than
    2 A above that plane with no halide-Pb contact). Also reported: O-Pb bond count per O, the angle of the mean
    X->O direction to the outward normal (0 deg = O point to vacuum, 180 = into the slab), halide height."""
    h0, o0 = oxyanion_atoms(initial, halide_atom_id)
    ids = final.arrays["atom_id"]
    pos_of = {int(i): k for k, i in enumerate(ids)}
    h = pos_of[int(initial.arrays["atom_id"][h0])]
    o = [pos_of[int(initial.arrays["atom_id"][j])] for j in o0]
    syms = np.array(final.get_chemical_symbols())
    cell, pbc, P = final.cell.array, final.pbc, final.positions
    d_xo = min_image_distances(cell, pbc, P[[h]], P[o])[0]
    pb = np.where(syms == "Pb")[0]
    d_opb = min_image_distances(cell, pbc, P[o], P[pb]) if len(pb) else np.full((len(o), 1), np.inf)
    hyd = np.where(syms == "H")[0]
    d_oh = min_image_distances(cell, pbc, P[o], P[hyd]) if len(hyd) else np.full((len(o), 1), np.inf)
    d_oh0 = min_image_distances(initial.cell.array, initial.pbc, initial.positions[o0],
                                initial.positions[np.where(np.array(initial.get_chemical_symbols()) == "H")[0]]) \
        if len(hyd) else d_oh
    cleaved = [k for k in range(len(o)) if d_xo[k] > IO_BOND]
    to_metal = [k for k in cleaved if d_opb[k].min() < O_TRANSFER_PB]
    protonated = [k for k in range(len(o)) if d_oh[k].min() < OH_BOND and d_oh0[k].min() >= OH_BOND]
    sub = np.where(final.arrays["origin"] == 0)[0]
    sub = sub[sub != h]
    d_sub = min_image_distances(cell, pbc, P[[h] + o], P[sub]).min()
    height, sign = _outward_height(final, rec, h)
    normal = cell[rec.normal_axis] / np.linalg.norm(cell[rec.normal_axis]) * sign
    vecs = []
    for k, j in enumerate(o):
        v = P[j] - P[h]
        f = v @ np.linalg.inv(cell)
        f -= np.round(f) * pbc
        vecs.append(f @ cell)
    m = np.sum([v / np.linalg.norm(v) for v in vecs], axis=0)
    angle = float(np.degrees(np.arccos(np.clip(m @ normal / np.linalg.norm(m), -1, 1)))) if np.linalg.norm(m) > 1e-6 else None
    d_xpb = min_image_distances(cell, pbc, P[[h]], P[pb]).min() if len(pb) else np.inf
    n_bound = int(sum(d_opb[k].min() < PB_O_BOND for k in range(len(o)) if k not in cleaved))
    if d_sub > DESORB_DIST:
        cls = "desorbed"
    elif to_metal:
        cls = "o_transfer_to_metal"
    elif cleaved:
        cls = "x_o_cleavage"
    elif protonated:
        cls = "protonated"
    else:
        cls = "intact_bound" if n_bound else "intact_unbound"
    flags = []
    if height < 0:
        flags.append("below_metal_plane")
    if height > 2.0 and d_xpb > 3.6:
        flags.append("lifted_out")
    return {"oxyanion_class": cls, "flags": flags, "n_oxygen": len(o),
            "x_o_A": [float(x) for x in d_xo], "o_pb_bonds": [int((d_opb[k] < PB_O_BOND).sum()) for k in range(len(o))],
            "o_bound_to_metal": n_bound, "min_o_pb_A": [float(d_opb[k].min()) for k in range(len(o))],
            "cleaved_o": cleaved, "o_on_metal_after_cleavage": to_metal, "protonated_o": protonated,
            "o_orientation_deg": angle, "halide_height_A": height, "min_x_pb_A": float(d_xpb),
            "reaction_accounting": "unchanged" if cls.startswith("intact") else
            "must be re-declared: the oxyanion is not the declared end state"}


# ----------------------------------------------------------------------------- Pb(IO3)2 surface unit
LEAD_IODATE = {"Pb": 1, "I": 2, "O": 6}


@dataclass
class LeadIodateSpec:
    pose_id: str
    pb_index: int
    pb_atom_id: int
    halide_indices: tuple[int, int]
    halide_atom_ids: tuple[int, int]
    oxygen_atom_ids: list[int]
    x_pb_x_angle_deg: float
    oxyanions: list[OxyanionSpec]

    def unit_atom_ids(self) -> list[int]:
        return [self.pb_atom_id, *self.halide_atom_ids, *self.oxygen_atom_ids]

    def to_dict(self) -> dict:
        return asdict(self)


def shared_lead_pairs(atoms: Atoms, rec: SurfaceRecord, face: str = "top") -> list[dict]:
    """Pairs of in-plane halides bonded to the same outer Pb of ``face``: {pb, halides, angle_deg}.
    angle ~ 90 is a cis pair (adjacent edges of the PbX4 square), ~ 180 trans."""
    from itertools import combinations
    from .surface import classify_halides
    by_pb: dict[int, set[int]] = {}
    for c in classify_halides(atoms, rec, face):
        if c["class"] == "in_plane":
            for p in c["pb"]:
                by_pb.setdefault(p, set()).add(c["index"])
    out = []
    for pb, xs in sorted(by_pb.items()):
        for x1, x2 in combinations(sorted(xs), 2):
            v1, v2 = _mic_vector(atoms, pb, x1), _mic_vector(atoms, pb, x2)
            ang = float(np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))
            out.append({"pb": int(pb), "halides": (int(x1), int(x2)), "angle_deg": ang})
    return out


def lead_iodate_poses(substrate: Atoms, rec: SurfaceRecord, pb: int, halides: tuple[int, int],
                      n_oxygen: int = 3, preset: str | None = "O-down", max_poses: int | None = None,
                      **pose_kwargs) -> list[tuple[Atoms, LeadIodateSpec]]:
    """Oxidise two halides that share the surface Pb ``pb``; every clash-free combination of their
    orientations. The second oxyanion is placed with the first already present, so the clash filter sees
    both. ``preset`` selects ``oxyanion_preset_poses``; None uses ``oxyanion_poses`` (``pose_kwargs`` go to
    either). Only the end state is built; see ``lead_iodate_record`` and ``remove_lead_iodate``."""
    from .surface import find_sites
    sub = substrate if "atom_id" in substrate.arrays else tag_substrate(substrate)
    x1, x2 = halides
    sites = {s.atom_index: s for face in ("top", "bottom") for s in find_sites(sub, rec, face)
             if s.kind == "exposed_I"}
    for x in (x1, x2):
        if x not in sites:
            raise ValueError(f"atom {x} is not an exposed halide")
        if pb not in _nearest_pb(sub, x, k=6):
            raise ValueError(f"halide {x} is not bonded to Pb {pb}")
    if sites[x1].face != sites[x2].face:
        raise ValueError("the two halides are on different faces")

    def make(at, site):
        if preset is None:
            return oxyanion_poses(at, rec, site, n_oxygen, **pose_kwargs)
        return oxyanion_preset_poses(at, rec, site, n_oxygen, preset, **pose_kwargs)
    v1, v2 = _mic_vector(sub, pb, x1), _mic_vector(sub, pb, x2)
    ang = float(np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))
    ids = sub.arrays["atom_id"]
    out = []
    for a1, s1 in make(sub, sites[x1]):
        for a2, s2 in make(a1, sites[x2]):
            o_ids = [int(i) for i in a2.arrays["atom_id"][len(sub):]]
            spec = LeadIodateSpec(f"Pb(IO{n_oxygen})2@Pb{pb}/{s1.pose_id.split('/', 1)[1]}+{s2.pose_id.split('/', 1)[1]}",
                                  int(pb), int(ids[pb]), (int(x1), int(x2)), (int(ids[x1]), int(ids[x2])),
                                  o_ids, ang, [s1, s2])
            out.append((a2, spec))
            if max_poses and len(out) >= max_poses:
                return out
    return out


def lead_iodate_record(rec: SurfaceRecord, atoms: Atoms, spec: LeadIodateSpec) -> SurfaceRecord:
    """Surface record for the structure with both halides oxidised, declaring the Pb(IO_n)2 unit."""
    r = rec
    for s in spec.oxyanions:
        r = oxidised_record(r, atoms, s)
    r.surface_id = f"{rec.surface_id}__Pb(IO{spec.oxyanions[0].n_oxygen})2@Pb{spec.pb_index}"
    r.defects = r.defects + [{"label": f"Pb(IO{spec.oxyanions[0].n_oxygen})2 unit at Pb {spec.pb_index}",
                              "pose_id": spec.pose_id, "unit_atom_ids": spec.unit_atom_ids(),
                              "x_pb_x_angle_deg": spec.x_pb_x_angle_deg}]
    return r


def remove_lead_iodate(atoms: Atoms, rec: SurfaceRecord, spec: LeadIodateSpec,
                       io_bond: float | None = None) -> tuple[Atoms, SurfaceRecord]:
    """Remove the declared Pb(IO_n)2 unit, leaving a PbX2 vacancy (``remove_fragments`` conventions).
    Atoms are found by ``atom_id``. Refuses if any unit O is farther than ``io_bond`` from its halide in
    ``atoms`` (e.g. after relaxation), because the removed group would then not be the declared unit."""
    io_bond = IO_BOND if io_bond is None else io_bond
    ids = {int(i): k for k, i in enumerate(atoms.arrays["atom_id"])}
    missing = [i for i in spec.unit_atom_ids() if i not in ids]
    if missing:
        raise ValueError(f"atom ids {missing} of the unit are not in the structure")
    for s, xid in zip(spec.oxyanions, spec.halide_atom_ids):
        x = ids[xid]
        o = [k for k in range(len(atoms)) if atoms.arrays["fragment_id"][k] == atoms.arrays["fragment_id"][x]
             and atoms[k].symbol == "O"]
        if len(o) != s.n_oxygen:
            raise ValueError(f"halide atom_id {xid} carries {len(o)} O, expected {s.n_oxygen}")
        d = atoms.get_distances(x, o, mic=True)
        if d.max() > io_bond:
            raise ValueError(f"oxyanion at halide atom_id {xid} is not intact (X-O up to {d.max():.2f} A); "
                             "the removed group would not be the declared unit")
    new, r = remove_fragments(atoms, rec, [ids[spec.pb_atom_id], *(ids[i] for i in spec.halide_atom_ids)],
                              f"V_PbI2(Pb(IO{spec.oxyanions[0].n_oxygen})2 removed)@Pb{spec.pb_index}")
    removed = {e: rec.composition.get(e, 0) - r.composition.get(e, 0) for e in rec.composition}
    want = {"Pb": 1, "I": 2, "O": 2 * spec.oxyanions[0].n_oxygen}
    if {k: v for k, v in removed.items() if v} != want:
        raise ValueError(f"removed {removed}, expected {want}")
    return new, r


def lead_iodate_extraction_reaction(unit_slab: Species, vacancy_slab: Species, bulk: Species) -> Reaction:
    """slab(Pb(IO3)2) -> slab(V_PbI2) + (1/Z) Pb(IO3)2(bulk, Z formula units per cell).
    ``bulk`` must be a declared Species whose composition is Z x PbI2O6; its energy is never defaulted
    (``Reaction.delta_e`` refuses to run without it)."""
    z = bulk.composition.get("Pb", 0)
    if z < 1 or {e: n for e, n in bulk.composition.items() if n} != {e: z * n for e, n in LEAD_IODATE.items()}:
        raise ValueError(f"bulk reference {bulk.name} has composition {dict(bulk.composition)}, "
                         "not an integer multiple of Pb(IO3)2")
    rx = Reaction([(1, unit_slab)], [(1, vacancy_slab), (1 / z, bulk)],
                  label="surface Pb(IO3)2 unit -> PbI2 vacancy + bulk Pb(IO3)2", energy_kind="reaction")
    rx.assert_balanced()
    return rx


# ----------------------------------------------------------------------------- water next to an oxyanion
WATER_OH_A, WATER_HOH_DEG = 0.9572, 104.52


def water_passivator():
    """Neutral, closed-shell H2O as a declared Passivator (charge 0, singlet; donor = O)."""
    from .poses import Passivator, Component
    t = np.radians(WATER_HOH_DEG / 2)
    w = Atoms("OH2", positions=[[0, 0, 0], [WATER_OH_A * np.sin(t), 0, WATER_OH_A * np.cos(t)],
                                [-WATER_OH_A * np.sin(t), 0, WATER_OH_A * np.cos(t)]])
    return Passivator("H2O", w, 0, 1, [Component("H2O", [0, 1, 2], 0, "H2O")], "neutral_donor",
                      donor_indices=[0], source="idealised gas-phase geometry")


def oxyanion_water_poses(atoms: Atoms, rec: SurfaceRecord, halide_atom_id: int, hbond_heights=(1.8, 2.1),
                         pb_heights=(2.5, 2.8), pb_radius: float = 5.0, tilts=(0.0, 30.0), n_azimuth: int = 6,
                         max_poses: int = 100, seed: int = 0, origin_tag: int = 2):
    """One water next to the oxyanion of ``halide_atom_id`` in an oxidised structure, two motifs:
      hbond  - a water H ``hbond_heights`` A above each oxyanion O (O-H...O donor bond);
      Pb     - the water O ``pb_heights`` A above each surface Pb within ``pb_radius`` of the halide.
    Uses ``generate_poses`` (rigid water, clash filter); the water gets ``origin_tag`` so it stays
    distinct from the oxyanion O (origin 1)."""
    from .poses import generate_poses
    from .surface import find_sites, unwrap_along
    h, o = oxyanion_atoms(atoms, halide_atom_id)
    z = unwrap_along(atoms, rec.normal_axis)
    face = "top" if abs(z[h] - rec.faces["top"]["edge_coordinate_A"]) <= \
        abs(z[h] - rec.faces["bottom"]["edge_coordinate_A"]) else "bottom"
    w = water_passivator()
    o_sites = [Site(f"{face}-oxyO{k}", "oxyanion_O", face, atoms.positions[k].tolist(), int(k)) for k in o]
    out = [(a, sp) for a, sp in generate_poses(atoms, rec, w, o_sites, heights=hbond_heights, tilts=tilts,
                                                n_azimuth=n_azimuth, anchors=[1], max_poses=max_poses, seed=seed,
                                                origin_tag=origin_tag)]
    pb_sites = [s for s in find_sites(atoms, rec, face) if s.kind == "undercoordinated_Pb"
                and atoms.get_distance(h, s.atom_index, mic=True) <= pb_radius]
    if pb_sites and len(out) < max_poses:
        out += generate_poses(atoms, rec, w, pb_sites, heights=pb_heights, tilts=tilts, n_azimuth=n_azimuth,
                              anchors=[0], max_poses=max_poses - len(out), seed=seed, origin_tag=origin_tag)
    return out

