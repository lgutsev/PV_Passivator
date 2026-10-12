"""Surface records: import, geometry analysis (normal, vacuum, faces, terminations), sites.

Nothing here assumes the surface normal is z: the recovered FAPbI3 slabs use the a/x axis.
Termination labels are *measured* from the outermost inorganic layer of each face, and the
formal-unit inventory is recorded separately, because a "FAI-rich" label alone does not
establish stoichiometry or neutrality.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from ase import Atoms

from .ledger import identify_units, formula_key

PB_I_BOND = 3.6        # A, Pb-I first-shell cutoff (bulk ~3.2)
LAYER_TOL = 1.6        # A, thickness of an "outermost layer" along the normal


@dataclass
class Site:
    site_id: str
    kind: str                 # undercoordinated_Pb | exposed_I | FA | hollow
    face: str                 # top | bottom
    position: list[float]     # cartesian, A
    atom_index: int | None
    coordination: int | None = None
    notes: str = ""
    halide_class: str | None = None   # in_plane | apical | subsurface | adatom (halide sites only)


@dataclass
class SurfaceRecord:
    surface_id: str
    composition: dict
    formal_units: dict
    unknown_fragments: list
    formal_charge: int | None
    phase: str
    facet: str
    normal_axis: int
    cell: list
    slab_thickness: float
    vacuum: float
    faces: dict                   # face -> termination info
    symmetric_terminations: bool
    defects: list = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    reconstruction: str = "none declared"
    origin: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_normal(atoms: Atoms) -> tuple[int, float]:
    """Axis with the largest empty gap along a lattice vector, and that gap (A)."""
    best = (2, -1.0)
    sp = atoms.get_scaled_positions(wrap=True)
    for ax in range(3):
        f = np.sort(sp[:, ax])
        gaps = np.diff(np.concatenate([f, [f[0] + 1]]))
        gap = gaps.max() * atoms.cell.lengths()[ax]
        if gap > best[1]:
            best = (ax, gap)
    return best


def unwrap_along(atoms: Atoms, ax: int) -> np.ndarray:
    """Normal coordinate (A) of each atom with the slab made contiguous (vacuum at edges)."""
    sp = atoms.get_scaled_positions(wrap=True)[:, ax]
    f = np.sort(sp)
    gaps = np.diff(np.concatenate([f, [f[0] + 1]]))
    i = int(np.argmax(gaps))
    cut = (f[i] + gaps[i] / 2) % 1.0
    s = (sp - cut) % 1.0
    return s * atoms.cell.lengths()[ax]


def _pb_coordination(atoms: Atoms, cutoff: float = PB_I_BOND) -> dict[int, int]:
    from ase.neighborlist import neighbor_list
    i, j = neighbor_list("ij", atoms, cutoff)
    syms = np.array(atoms.get_chemical_symbols())
    out = {}
    for p in np.where(syms == "Pb")[0]:
        out[int(p)] = int(np.sum((i == p) & np.isin(syms[j], ["I", "Br", "Cl"])))
    return out


def analyse_faces(atoms: Atoms, ax: int) -> tuple[dict, np.ndarray]:
    h = unwrap_along(atoms, ax)
    syms = np.array(atoms.get_chemical_symbols())
    faces = {}
    for face, sign in (("top", 1), ("bottom", -1)):
        inorg = np.where(np.isin(syms, ["Pb", "I", "Br", "Cl"]))[0]
        edge = h[inorg].max() if sign > 0 else h[inorg].min()
        outer = inorg[np.abs(h[inorg] - edge) < LAYER_TOL]
        n_pb = int(np.sum(syms[outer] == "Pb"))
        n_x = int(np.sum(np.isin(syms[outer], ["I", "Br", "Cl"])))
        # organic cations outside / level with the outermost inorganic layer
        org = np.where(syms == "N")[0]
        beyond = org[(h[org] - edge) * sign > -LAYER_TOL]
        if n_pb > 0:
            term = "PbX2-terminated (Pb in outermost inorganic layer)"
            short = "PbI2"
        else:
            term = "AX-terminated (outermost inorganic layer is halide only; A-cations at surface)"
            short = "FAI"
        faces[face] = {"termination": short, "description": term, "outer_Pb": n_pb,
                       "outer_halide": n_x, "outer_layer_N_atoms": int(len(beyond)),
                       "edge_coordinate_A": float(edge)}
    return faces, h


def load_surface(path: str | Path, surface_id: str | None = None, phase: str = "alpha-FAPbI3 (cubic)",
                 facet: str = "(100)", normal_axis: int | None = None, defects: list | None = None,
                 fmt: str | None = None) -> tuple[Atoms, SurfaceRecord]:
    from ase.io import read
    path = Path(path)
    atoms = read(path, format=fmt) if fmt else read(path)
    ax, gap = detect_normal(atoms) if normal_axis is None else (normal_axis, None)
    faces, h = analyse_faces(atoms, ax)
    thickness = float(h.max() - h.min())
    vac = float(atoms.cell.lengths()[ax] - thickness)
    units, unknown = identify_units(atoms)
    from .ledger import _label_charge
    formal = None if unknown else sum(_label_charge(k) * v for k, v in units.items())
    comp = {}
    for s in atoms.get_chemical_symbols():
        comp[s] = comp.get(s, 0) + 1
    notes = []
    sym = faces["top"]["termination"] == faces["bottom"]["termination"]
    if not sym:
        notes.append("Asymmetric slab: the two faces have different terminations; a net dipole is "
                     "expected and a dipole correction (IDIPOL) should be declared.")
    if formal not in (0, None):
        notes.append(f"Formal ionic charge {formal:+d}: stoichiometry is not formally neutral.")
    fixed = []
    if atoms.constraints:
        for c in atoms.constraints:
            fixed += list(getattr(c, "index", []))
    rec = SurfaceRecord(
        surface_id=surface_id or path.parent.name or path.stem,
        composition=comp, formal_units=units, unknown_fragments=unknown, formal_charge=formal,
        phase=phase, facet=facet, normal_axis=ax, cell=atoms.cell.array.tolist(),
        slab_thickness=thickness, vacuum=vac, faces=faces, symmetric_terminations=sym,
        defects=defects or [], constraints={"fixed_atoms": sorted(map(int, fixed))},
        origin={"path": str(path), "sha256": sha256_file(path), "formula": formula_key(comp)},
        notes=notes)
    return atoms, rec


def find_sites(atoms: Atoms, rec: SurfaceRecord, face: str = "top",
               depth: float = 2.5) -> list[Site]:
    """Candidate anchoring sites on one face: undercoordinated Pb, exposed halide, FA cations.
    ``depth`` (A) is how far below the outermost inorganic layer an atom may sit."""
    ax = rec.normal_axis
    h = unwrap_along(atoms, ax)
    edge = rec.faces[face]["edge_coordinate_A"]
    sign = 1 if face == "top" else -1
    syms = np.array(atoms.get_chemical_symbols())
    near = (edge - h) * sign < depth
    coord = _pb_coordination(atoms)
    sites = []
    for i in np.where(near & (syms == "Pb"))[0]:
        if coord[int(i)] < 6:
            sites.append(Site(f"{face}-Pb{i}", "undercoordinated_Pb", face,
                              atoms.positions[i].tolist(), int(i), coord[int(i)]))
    cls = {c["index"]: c for c in classify_halides(atoms, rec, face, depth=depth)}
    for i in np.where(near & np.isin(syms, ["I", "Br", "Cl"]))[0]:
        c = cls.get(int(i), {})
        sites.append(Site(f"{face}-X{i}", "exposed_I", face, atoms.positions[i].tolist(), int(i),
                          c.get("n_pb"), c.get("geometry", ""), c.get("class")))
    # FA cations: carbon atoms of CH5N2 units near the face
    for i in np.where(((edge - h) * sign < depth + 1.5) & (syms == "C"))[0]:
        sites.append(Site(f"{face}-FA{i}", "FA", face, atoms.positions[i].tolist(), int(i)))
    return sites


def _outward(rec: SurfaceRecord, atoms: Atoms, face: str) -> np.ndarray:
    n = atoms.cell.array[rec.normal_axis]
    n = n / np.linalg.norm(n)
    return n if face == "top" else -n


def classify_halides(atoms: Atoms, rec: SurfaceRecord, face: str = "top", depth: float = 6.0,
                     cutoff: float = PB_I_BOND, axial_cos: float = 0.7, plane_cos: float = 0.5) -> list[dict]:
    """Classify every halide within ``depth`` (A) of the outermost inorganic layer of ``face``.

    Uses the Pb neighbours within ``cutoff`` and the angle of each Pb->X bond to the outward normal (cos > ``axial_cos``: X points out of its Pb; |cos| < ``plane_cos``: bond in the plane).
      in_plane   - in the outer layer, bridging >= 2 Pb with in-plane bonds (the PbI2-face bridge)
      apical     - one Pb neighbour, bond pointing outward (FAI-face iodide, or a terminal X on a PbI2 face)
      subsurface - below the outer layer; ``geometry`` says bridging_in_plane / axial / other
      adatom     - no Pb neighbour
    Returns dicts with index, class, geometry, n_pb, pb (indices), depth_A (below the outer layer edge),
    bond_cos (Pb->X . n_out per Pb)."""
    from ase.neighborlist import neighbor_list
    syms = np.array(atoms.get_chemical_symbols())
    h = unwrap_along(atoms, rec.normal_axis)
    sign = 1 if face == "top" else -1
    edge = rec.faces[face]["edge_coordinate_A"]
    n_out = _outward(rec, atoms, face)
    i_, j_, D = neighbor_list("ijD", atoms, cutoff)
    out = []
    for x in np.where(np.isin(syms, ["I", "Br", "Cl"]))[0]:
        dep = float((edge - h[x]) * sign)
        if dep > depth:
            continue
        m = (i_ == x) & (syms[j_] == "Pb")
        pb = [int(k) for k in j_[m]]
        cos = [float(-d @ n_out / np.linalg.norm(d)) for d in D[m]]     # Pb->X direction = -(X->Pb)
        n = len(pb)
        inplane = n >= 2 and all(abs(c) < plane_cos for c in cos)
        if n == 0:
            cls, geo = "adatom", "no Pb within cutoff"
        elif dep > LAYER_TOL:
            cls = "subsurface"
            geo = "bridging_in_plane" if inplane else ("axial" if any(abs(c) > axial_cos for c in cos) else "other")
        elif n == 1 and cos[0] > axial_cos:
            cls, geo = "apical", "terminal, pointing out"
        elif inplane:
            cls, geo = "in_plane", f"bridging {n} Pb"
        elif n >= 2 and max(cos) > axial_cos:
            cls, geo = "apical", f"axial over {n} Pb"
        else:
            cls, geo = "subsurface" if dep > LAYER_TOL / 2 else "in_plane", "irregular"
        out.append({"index": int(x), "class": cls, "geometry": geo, "n_pb": n, "pb": pb,
                    "depth_A": dep, "bond_cos": cos})
    return out
