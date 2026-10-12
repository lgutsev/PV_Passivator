"""Publication renders of slab structures in the house style (OVITO + Tachyon). See docs/FIGURE_STYLE.md.

House style: light semi-transparent PbX6 octahedra, Pb-X sticks and covalent bonds drawn as thin uniform grey
cylinders, organic cations ball-and-stick with element colours (C grey, N blue, H white), ambient occlusion with soft
shadows, white background. One focus atom per figure in magenta; optional translucent gold halo; any Pb bonded to an
adsorbate donor (O, N, S, ...) is drawn without an octahedron so the new bond reads.

Display only: atoms are never moved. The slab is rotated so the outward surface normal points up (+z), replicated
3x3 in the surface plane and cut to a disc around the focus, so a periodic adsorbate appears once.

The geometry part (``display_cutout``) is pure numpy and tested; ``render`` needs OVITO (``pip install ovito``),
imported only when called. Run OVITO and matplotlib in separate processes (they clash on exit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers

# ----------------------------------------------------------------------------- style
COLORS = {
    "Pb": (0.30, 0.31, 0.34), "Sn": (0.40, 0.45, 0.50), "I": (0.55, 0.45, 0.68), "Br": (0.65, 0.33, 0.22),
    "Cl": (0.35, 0.70, 0.35), "C": (0.42, 0.44, 0.48), "N": (0.22, 0.42, 0.86), "H": (0.96, 0.96, 0.96),
    "O": (0.85, 0.20, 0.16), "S": (0.93, 0.78, 0.20), "F": (0.55, 0.85, 0.45), "P": (0.95, 0.55, 0.15),
    "Cs": (0.35, 0.65, 0.60), "K": (0.55, 0.35, 0.80), "Na": (0.60, 0.45, 0.85), "Rb": (0.45, 0.25, 0.65),
}
RADII = {"Pb": 0.62, "Sn": 0.62, "I": 0.80, "Br": 0.72, "Cl": 0.66, "C": 0.40, "N": 0.40, "H": 0.24, "O": 0.50,
         "S": 0.58, "F": 0.40, "P": 0.55, "Cs": 0.95, "K": 0.85, "Na": 0.75, "Rb": 0.90}
FOCUS_COLOR, FOCUS_SCALE = (0.86, 0.12, 0.47), 1.2
HALO_COLOR, HALO_RADIUS, HALO_TRANSPARENCY = (1.00, 0.76, 0.18), 1.25, 0.55
HIGHLIGHT_SCALE, HIGHLIGHT_TINT = 1.12, 0.35      # whole-fragment highlight: slightly larger, tinted toward gold
BOND_COLOR, BOND_WIDTH = (0.55, 0.55, 0.58), 0.22
OCTA_COLOR, OCTA_TRANSPARENCY = (0.78, 0.80, 0.86), 0.55
TACHYON = dict(ambient_occlusion=True, ambient_occlusion_brightness=0.8, ambient_occlusion_samples=24,
               antialiasing_samples=24, shadows=True, direct_light_intensity=0.9)

METALS = ("Pb", "Sn")
HALIDES = ("I", "Br", "Cl")
METAL_HALIDE_BOND = 3.6           # A
METAL_DONOR_BOND = {"O": 2.8, "N": 2.9, "S": 3.2, "F": 2.7}
COVALENT_MULT = 1.15


@dataclass
class Cutout:
    positions: np.ndarray          # (n, 3), display frame: outward normal = +z, focus at x = y = 0
    symbols: np.ndarray            # element symbols
    source_index: np.ndarray       # index into the input Atoms
    image: np.ndarray              # (n, 2) lateral periodic image of each atom
    focus: int | None              # index into positions of the focus atom (central image)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    no_octahedron: list[int] = field(default_factory=list)   # metals bonded to an adsorbate donor
    incomplete: list[int] = field(default_factory=list)      # metals missing a halide that the cutout cannot show


def _frame(atoms: Atoms, normal_axis: int, face: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation to the display frame: lattice vector ``normal_axis`` (outward for ``face``) -> +z.
    Returns (R, in-plane lattice vector 1, in-plane lattice vector 2) in display coordinates."""
    cell = atoms.cell.array
    n = cell[normal_axis] / np.linalg.norm(cell[normal_axis])
    if face == "bottom":
        n = -n
    a, b = [cell[k] for k in range(3) if k != normal_axis]
    e1 = a - np.dot(a, n) * n; e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    R = np.array([e1, e2, n])
    return R, R @ a, R @ b


def _bond_cutoff(s1: str, s2: str) -> float:
    pair = {s1, s2}
    if pair & set(METALS) and pair & set(HALIDES) and len(pair) == 2:
        return METAL_HALIDE_BOND
    for m in METALS:
        if m in pair:
            other = (pair - {m}) or {m}
            o = next(iter(other))
            return METAL_DONOR_BOND.get(o, 0.0)
    if pair <= set(HALIDES) | {"Cs", "K", "Na", "Rb"} and not (pair & {"O"}):
        return 0.0                 # ionic: no halide-halide or alkali bonds
    return COVALENT_MULT * (covalent_radii[atomic_numbers[s1]] + covalent_radii[atomic_numbers[s2]])


def display_cutout(atoms: Atoms, focus: int, normal_axis: int = 2, face: str = "top", radius: float = 9.0,
                   depth: float = 8.0, keep: list[int] | None = None) -> Cutout:
    """Disc of the slab around atom ``focus`` for rendering.

    Keeps, from a 3x3 lateral replication: metals within ``radius`` (laterally) of the focus and within ``depth`` of
    the outermost metal layer; halides bonded to a kept metal; every covalent fragment with an atom inside the disc
    (organic cations, adsorbates) in the central image only if it contains the focus or a ``keep`` atom, otherwise
    in whichever image lies inside the disc. Periodic copies of the focus fragment are dropped.
    """
    from ase.neighborlist import natural_cutoffs, NeighborList
    R, A, B = _frame(atoms, normal_axis, face)
    sym = np.array(atoms.get_chemical_symbols())
    # covalent fragments in the periodic cell (metals and halides ionic, except halides bonded to O)
    cut = natural_cutoffs(atoms, mult=COVALENT_MULT)
    ionic = set(METALS) | {"Cs", "K", "Na", "Rb"}
    cut = [0.0 if s in ionic else c for s, c in zip(sym, cut)]
    nl = NeighborList(cut, self_interaction=False, bothways=True, skin=0.0); nl.update(atoms)
    frag = -np.ones(len(atoms), int); k = 0
    for i in range(len(atoms)):
        if frag[i] >= 0 or sym[i] in ionic:
            continue
        stack = [i]; frag[i] = k
        while stack:
            j = stack.pop()
            for m in nl.get_neighbors(j)[0]:
                if frag[m] < 0 and sym[m] not in ionic:
                    frag[m] = k; stack.append(m)
        k += 1
    # a lone halide is a lattice anion, not a fragment (an oxyanion keeps its O and stays a fragment)
    for f in range(k):
        idx = np.flatnonzero(frag == f)
        if len(idx) == 1 and sym[idx[0]] in HALIDES:
            frag[idx] = -1
    # unwrap fragments around their first atom (minimum image) so molecules stay whole
    pos = atoms.positions.copy()
    inv = np.linalg.inv(atoms.cell.array)
    for f in range(k):
        idx = np.flatnonzero(frag == f)
        if not len(idx):
            continue
        d = pos[idx] - pos[idx[0]]
        s = d @ inv
        for ax in range(3):
            if atoms.pbc[ax]:
                s[:, ax] -= np.round(s[:, ax])
        pos[idx] = pos[idx[0]] + s @ atoms.cell.array
    P0 = (R @ (pos - pos[focus]).T).T            # focus at origin, normal up
    zmetal = P0[np.isin(sym, METALS), 2].max() if np.isin(sym, METALS).any() else P0[:, 2].max()
    # lateral wrap into the cell around the focus
    M = np.array([A[:2], B[:2]]).T
    f2 = np.linalg.solve(M, P0[:, :2].T).T
    shift = np.round(f2)
    for f in range(k):                           # shift fragments as a unit (by their first atom)
        idx = np.flatnonzero(frag == f)
        if not len(idx):
            continue
        shift[idx] = shift[idx[0]]
    P0[:, :2] -= (shift @ np.array([A[:2], B[:2]]))
    focus_frag = frag[focus]
    keep_frags = {frag[i] for i in (keep or []) if frag[i] >= 0} | ({focus_frag} if focus_frag >= 0 else set())
    out_p, out_s, out_i, out_img = [], [], [], []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            off = ix * A + iy * B
            Q = P0 + off
            lat = np.linalg.norm(Q[:, :2], axis=1)
            sel = np.zeros(len(atoms), bool)
            sel |= np.isin(sym, METALS) & (lat < radius) & (Q[:, 2] > zmetal - depth)
            for f in range(k):
                idx = np.flatnonzero(frag == f)
                if not len(idx):
                    continue
                if f in keep_frags:
                    if ix == 0 and iy == 0:
                        sel[idx] = True
                    continue
                if np.any(lat[idx] < radius) and np.any(Q[idx, 2] > zmetal - depth):
                    sel[idx] = True
            if ix == 0 and iy == 0 and focus_frag < 0:
                sel[focus] = True
            out_p.append(Q[sel]); out_s.append(sym[sel]); out_i.append(np.flatnonzero(sel))
            out_img.append(np.tile([ix, iy], (sel.sum(), 1)))
    P = np.concatenate(out_p); S = np.concatenate(out_s); I = np.concatenate(out_i); IMG = np.concatenate(out_img)
    # halides bonded to a kept metal (ionic halides not in a fragment)
    metals = np.flatnonzero(np.isin(S, METALS))
    extra_p, extra_s, extra_i, extra_img = [], [], [], []
    have = {(int(i), int(a), int(b)) for i, (a, b) in zip(I, IMG)}
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            off = ix * A + iy * B
            for h in np.flatnonzero(np.isin(sym, HALIDES) & (frag < 0)):
                if (int(h), ix, iy) in have:
                    continue
                q = P0[h] + off
                if len(metals) and np.min(np.linalg.norm(P[metals] - q, axis=1)) < METAL_HALIDE_BOND:
                    extra_p.append(q); extra_s.append(sym[h]); extra_i.append(h); extra_img.append([ix, iy])
                    have.add((int(h), ix, iy))
    if extra_p:
        P = np.r_[P, np.array(extra_p)]; S = np.r_[S, np.array(extra_s)]
        I = np.r_[I, np.array(extra_i)]; IMG = np.r_[IMG, np.array(extra_img)]
    centre = np.flatnonzero((I == focus) & (IMG[:, 0] == 0) & (IMG[:, 1] == 0))
    fidx = int(centre[0]) if len(centre) else None
    # bonds in the display set (no periodicity needed: the cutout is explicit)
    bonds = []
    for i in range(len(P)):
        d = np.linalg.norm(P[i + 1:] - P[i], axis=1)
        for kk in np.flatnonzero(d < METAL_HALIDE_BOND):
            j = i + 1 + kk
            c = _bond_cutoff(S[i], S[j])
            if c and d[kk] < c:
                # halide-O bonds only inside a fragment (oxyanion), never iodide-to-random-O contacts
                if {S[i], S[j]} & set(HALIDES) and {S[i], S[j]} & {"O"} and frag[I[i]] != frag[I[j]]:
                    continue
                bonds.append((i, int(j)))
    no_oct = {a for i, j in bonds for a, b in ((i, j), (j, i))
              if S[a] in METALS and S[b] in METAL_DONOR_BOND}
    # metals whose full halide shell cannot be shown (typically a neighbour image bonded to the dropped periodic
    # copy of the focus fragment's halide) get no octahedron either: a partial polyhedron reads as a defect
    from ase.neighborlist import neighbor_list
    ii, jj = neighbor_list("ij", atoms, METAL_HALIDE_BOND)
    true_cn = {int(m): int(np.sum((ii == m) & np.isin(sym[jj], HALIDES))) for m in np.flatnonzero(np.isin(sym, METALS))}
    shown = {m: 0 for m in range(len(P)) if S[m] in METALS}
    for i, j in bonds:
        for a, b in ((i, j), (j, i)):
            if a in shown and S[b] in HALIDES:
                shown[a] += 1
    incomplete = sorted(m for m, n in shown.items() if m not in no_oct and n < true_cn[int(I[m])])
    return Cutout(P, S, I, IMG, fidx, bonds, sorted(no_oct | set(incomplete)), incomplete)


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(255 * c)):02x}" for c in rgb)


def cutout_to_web(c: Cutout, highlight: list[int] | None = None) -> dict:
    """The cutout as plain data for a web viewer (3Dmol.js, three.js), so it draws the same atoms, bonds and
    octahedra as the OVITO render instead of guessing bonds (3Dmol's guess drops most Pb-I bonds).

    ``atoms``: element, x/y/z (A, display frame: outward normal +z, focus at the origin), hex colour, radius,
    source index, image, role (focus | highlight | atom). ``bonds``: index pairs into ``atoms``.
    ``octahedra``: per metal drawn with a polyhedron, the metal and its halide indices. ``style``: bond/octahedron
    colours and transparencies."""
    hl = set()
    if highlight:
        for h in highlight:
            hl |= set(np.flatnonzero((c.source_index == h) & (c.image == 0).all(1)).tolist())
    atoms = []
    for k, (p, e) in enumerate(zip(c.positions, c.symbols)):
        role = "focus" if k == c.focus else ("highlight" if k in hl else "atom")
        col = FOCUS_COLOR if role == "focus" else COLORS.get(str(e), (0.6, 0.6, 0.6))
        rad = RADII.get(str(e), 0.6) * (FOCUS_SCALE if role == "focus" else HIGHLIGHT_SCALE if role == "highlight" else 1)
        atoms.append({"el": str(e), "x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "color": _hex(col),
                      "radius": float(rad), "source_index": int(c.source_index[k]),
                      "image": [int(v) for v in c.image[k]], "role": role})
    octa = []
    skip = set(c.no_octahedron)
    for m in np.flatnonzero(np.isin(c.symbols, METALS)):
        if int(m) in skip:
            continue
        hal = sorted(int(b[0] if b[1] == m else b[1]) for b in c.bonds
                     if m in b and c.symbols[b[0] if b[1] == m else b[1]] in HALIDES)
        octa.append({"metal": int(m), "halides": hal})
    return {"schema": "pvpassivator.cutout/1", "frame": "display: outward normal +z, focus at origin, A",
            "atoms": atoms, "bonds": [[int(i), int(j)] for i, j in c.bonds], "octahedra": octa,
            "no_octahedron": [int(m) for m in c.no_octahedron],
            "style": {"bond_color": _hex(BOND_COLOR), "bond_radius": BOND_WIDTH / 2,
                      "octahedron_color": _hex(OCTA_COLOR), "octahedron_opacity": 1 - OCTA_TRANSPARENCY}}


def render(atoms: Atoms, out: str | Path, focus: int, normal_axis: int = 2, face: str = "top", view: str = "side",
           halo: list[int] | None = None, radius: float = 9.0, depth: float = 8.0, size=(1200, 900),
           fov: float | None = None, view_dir=None, keep: list[int] | None = None,
           highlight: list[int] | None = None) -> Path:
    """Render one view. ``view``: 'side' (orthographic, slight downward tilt), 'top' (orthographic, along -normal) or
    'persp' (oblique perspective with depth of field). ``halo``: input indices that get a gold halo (central image).
    ``view_dir``: optional in-plane direction (display frame) to look along for 'side'; default +x of the display frame.
    ``highlight``: input indices (e.g. every atom of an adsorbed molecule) drawn slightly larger and tinted toward
    gold, central image only; the focus atom keeps its magenta."""
    from ovito.data import DataCollection, ParticleType
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.modifiers import SelectTypeModifier, CoordinationPolyhedraModifier, ClearSelectionModifier
    from ovito.vis import Viewport, TachyonRenderer, BondsVis

    c = display_cutout(atoms, focus, normal_axis, face, radius, depth, keep)
    label = c.symbols.astype("<U8")
    for m in c.no_octahedron:
        label[m] = label[m] + "_x"
    for h in highlight or []:
        for k in np.flatnonzero((c.source_index == h) & (c.image == 0).all(1)):
            if k != c.focus:
                label[k] = label[k] + "_hl"
    if c.focus is not None:
        label[c.focus] = "focus"
    pos = c.positions
    halo_pos = [pos[np.flatnonzero((c.source_index == h) & (c.image == 0).all(1))[0]] for h in (halo or [])]
    col = {n: COLORS.get(n.split("_")[0], (0.6, 0.6, 0.6)) for n in set(label)}
    rad = {n: RADII.get(n.split("_")[0], 0.6) for n in set(label)}
    for n in set(label):
        if n.endswith("_hl"):
            col[n] = tuple((1 - HIGHLIGHT_TINT) * np.array(col[n]) + HIGHLIGHT_TINT * np.array(HALO_COLOR))
            rad[n] = rad[n] * HIGHLIGHT_SCALE
    if c.focus is not None:
        col["focus"] = FOCUS_COLOR; rad["focus"] = RADII.get(c.symbols[c.focus], 0.7) * FOCUS_SCALE
    if halo_pos:
        pos = np.r_[pos, np.array(halo_pos)]; label = np.r_[label, ["halo"] * len(halo_pos)]
        col["halo"] = HALO_COLOR; rad["halo"] = HALO_RADIUS
    data = DataCollection()
    cell = data.create_cell(np.c_[np.diag([80.0, 80.0, 80.0]), [-40.0, -40.0, -40.0]], pbc=(False, False, False))
    cell.vis.enabled = False
    parts = data.create_particles(count=len(pos))
    parts.create_property("Position", data=pos)
    names = list(dict.fromkeys(label.tolist())); ids = {n: k for k, n in enumerate(names, 1)}
    tp = parts.create_property("Particle Type", data=np.array([ids[n] for n in label]))
    for n, k in ids.items():
        tp.types.append(ParticleType(id=k, name=n, color=col[n], radius=rad[n]))
    parts.create_property("Transparency", data=np.array([HALO_TRANSPARENCY if n == "halo" else 0.0 for n in label]))
    if c.bonds:
        b = parts.create_bonds(count=len(c.bonds))
        b.create_property("Topology", data=np.array(c.bonds, int))
        b.vis.width = BOND_WIDTH
        b.vis.coloring_mode = BondsVis.ColoringMode.Uniform
        b.vis.color = BOND_COLOR
    pipe = Pipeline(source=StaticSource(data=data))
    oct_types = {n for n in names if n in METALS}
    if oct_types:
        pipe.modifiers.append(SelectTypeModifier(types=oct_types))
        cp = CoordinationPolyhedraModifier()
        cp.vis.surface_color = OCTA_COLOR; cp.vis.surface_transparency = OCTA_TRANSPARENCY
        cp.vis.highlight_edges = False
        pipe.modifiers.append(cp); pipe.modifiers.append(ClearSelectionModifier())
    target = pos[c.focus] if c.focus is not None else pos.mean(0)
    if view == "side":
        u = np.array(view_dir if view_dir is not None else (1.0, 0.0, 0.0), float); u[2] = 0; u /= np.linalg.norm(u)
        d = u + [0, 0, -0.22]
        vp = Viewport(type=Viewport.Type.Ortho, camera_dir=tuple(d), camera_up=(0, 0, 1), fov=fov or 8.0)
        vp.camera_pos = tuple(target + [0, 0, -3.4])
        r = TachyonRenderer(**TACHYON)
    elif view == "top":
        vp = Viewport(type=Viewport.Type.Ortho, camera_dir=(0, 0, -1), camera_up=(0, 1, 0), fov=fov or 8.0)
        vp.camera_pos = tuple(target + [0, 0, 30])
        r = TachyonRenderer(**TACHYON)
    elif view == "persp":
        d = np.array([0.62, 0.45, -0.50]); d /= np.linalg.norm(d)
        dist = 19.0
        vp = Viewport(type=Viewport.Type.Perspective, camera_dir=tuple(d), camera_up=(0, 0, 1),
                      fov=np.radians(fov or 34.0))
        vp.camera_pos = tuple(target + [0, 0, -0.8] - d * dist)
        r = TachyonRenderer(**dict(TACHYON, depth_of_field=True, focal_length=dist, aperture=0.02))
    else:
        raise ValueError("view must be side, top or persp")
    pipe.add_to_scene()
    out = Path(out)
    vp.render_image(filename=str(out), size=size, background=(1, 1, 1), renderer=r)
    pipe.remove_from_scene()
    return out
