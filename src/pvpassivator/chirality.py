"""Structural chirality observables for layered (2D) metal-halide perovskites.

Scope: per-frame geometry of A2MX4 Ruddlesden-Popper-type crystals such as (R/S/rac-MBA)2PbI4 and the
time-series statistics needed to analyse finite-temperature trajectories. Frames are plain ASE ``Atoms``
(extxyz, XDATCAR, ASE .traj all work). Nothing here runs a calculator.

Conventions (see docs/CHIRALITY_FEASIBILITY.md, section "Descriptor definitions")
-------------------------------------------------------------------------------
* Topology (molecules, octahedra, links, stereocentres) is built ONCE from a reference frame and then reused,
  so atom identity and mapping are fixed for the whole trajectory. Every bond vector in every frame is a
  minimum-image vector; molecules are unwrapped by walking their own bond graph.
* Handedness descriptors are *link twists*. For two neighbouring sites i, j with oriented unit vectors
  v_i, v_j and minimum-image separation unit vector e_ij (i -> j)::

      tau_ij = (v_i x v_j) . e_ij

  Swapping i and j changes nothing (both factors change sign), so atom ordering cannot reverse the result.
  Proper rotations and lattice translations leave tau unchanged; any improper operation (mirror, inversion)
  reverses it. tau is the sine of the twist of v about the link; its sum along a periodic chain is the net
  helical winding per period. A planar zig-zag (achiral tilt) gives alternating +/- links whose chain sum is
  zero, so per-link values are local fluctuations and the chain / site / cell means carry net handedness.
* Site vectors:
    - ``ax``  (out-of-plane framework): octahedral axial trans axis, unit(X_ax(+) - X_ax(-)), oriented along
      +n (layer normal). Flipping the sign of n flips both v_i and v_j, so tau is unchanged.
    - ``eq``  (in-plane framework): for link i-j through bridging halide X_b, the equatorial trans axis of
      each octahedron that does NOT contain X_b, oriented along m = n x e_ij. Proven invariant to the sign
      of n and to i<->j (tests).
    - ``org`` (organic packing): cation axis unit(N - centroid of the cation's C atoms); intrinsically
      oriented, no convention needed. Links join cations on the same face of the same inorganic layer.
  These are our implementation. They follow the vector-chirality idea of Pols et al. (JPCL 2024, 15, 8057;
  eps = sum of cross products of neighbouring unit vectors) but project on the *link* direction, which is a
  pseudoscalar by construction. Projections perpendicular to a chain direction (e.g. on the layer normal)
  reverse with the walking direction and are polar, not chiral; they are deliberately not offered here.
  A numerical reconciliation with the published values needs SI Note 2 of Pols et al. (not accessed).
* Stereocentres are assigned R/S with a CIP-lite ranking (atomic numbers, breadth-first to depth 4, no
  duplicate atoms for multiple bonds). Ties raise; validated only for 1-phenylethylammonium-type centres.
* Hydrogen bonds: D-H...A with d(H...A) <= r_max and angle(D-H...A) >= angle_min. Defaults (3.2 A, 120 deg)
  are a choice, not a published standard; run ``hbond_sensitivity`` before quoting occupancies or lifetimes.
* Statistics: frames are correlated. Use ``block_average`` (Flyvbjerg-Petersen blocking) or
  ``integrated_autocorr_time``; never treat consecutive frames as independent samples.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import find_mic
from ase.neighborlist import natural_cutoffs, neighbor_list


class TopologyError(ValueError):
    pass


# --------------------------------------------------------------------------------------------- geometry helpers

def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("zero-length vector")
    return v / n


def mic(atoms: Atoms, vectors) -> np.ndarray:
    """Minimum-image version of displacement vectors (N,3) under ``atoms``' cell and pbc."""
    v = np.atleast_2d(np.asarray(vectors, float))
    out, _ = find_mic(v, atoms.cell, atoms.pbc)
    return out


def _max_safe_cutoff(atoms: Atoms) -> float:
    """Half of the smallest perpendicular cell width: bonds shorter than this have a unique minimum image."""
    cell = atoms.cell.complete()
    vol = abs(np.linalg.det(cell))
    widths = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
    return 0.5 * min(w for w, p in zip(widths, atoms.pbc) if p) if any(atoms.pbc) else np.inf


def invert(atoms: Atoms) -> Atoms:
    """Exact enantiomer by inversion through the origin (cell unchanged, still right-handed)."""
    out = atoms.copy()
    out.positions = -atoms.positions
    out.wrap()
    return out


# --------------------------------------------------------------------------------------------- topology

@dataclass
class Octahedron:
    metal: int
    axial: tuple[int, int]                 # (X_ax-, X_ax+) ordered along +n in the reference frame
    eq_pairs: tuple[tuple[int, int], tuple[int, int]]   # trans equatorial pairs


@dataclass
class Link:
    i: int        # site index (metal atom or molecule index)
    j: int
    via: int = -1 # bridging halide atom index (framework links), -1 for organic links
    ref_frac: np.ndarray | None = None  # organic links: reference separation in fractional coordinates (image choice)


@dataclass
class Molecule:
    atoms: np.ndarray            # atom indices
    bonds: list[tuple[int, int]] # atom-index bonds (BFS tree + ring closures)
    n_ammonium: int              # ammonium N atom index (N with >= 3 H)
    nh_h: list[int]              # H atoms bonded to that N
    carbons: list[int]
    stereocentres: dict[int, str] = field(default_factory=dict)   # C index -> 'R'/'S' (reference frame)
    layer: int = -1
    face: int = 0                # +1 / -1 relative to the reference layer normal


@dataclass
class Topology:
    n_atoms: int
    symbols: list[str]
    normal: np.ndarray                  # unit layer normal in the reference frame (sign arbitrary)
    normal_coeff: np.ndarray            # n . a_k for the reference cell vectors: n follows the lattice plane per frame
    octahedra: dict[int, Octahedron]
    metal_links: list[Link]
    metal_layer: dict[int, int]
    molecules: list[Molecule]
    org_links: list[Link]
    halide_role: dict[int, str]         # 'axial' | 'bridging'

    def summary(self) -> dict:
        conf = [c for m in self.molecules for c in m.stereocentres.values()]
        return {
            "n_atoms": self.n_atoms,
            "n_metal": len(self.octahedra),
            "n_layers": len(set(self.metal_layer.values())),
            "n_molecules": len(self.molecules),
            "n_metal_links": len(self.metal_links),
            "n_org_links": len(self.org_links),
            "stereocentres": {"R": conf.count("R"), "S": conf.count("S")},
            "faces": {"+": sum(m.face > 0 for m in self.molecules), "-": sum(m.face < 0 for m in self.molecules)},
        }


def _components(n, edges):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, comps = [False] * n, []
    for s in range(n):
        if seen[s]:
            continue
        comp, q = [], deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            comp.append(u)
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    q.append(w)
        comps.append(sorted(comp))
    return comps, adj


def unwrap_molecule(atoms: Atoms, mol: Molecule) -> np.ndarray:
    """Positions of ``mol`` made whole by walking its bond graph with minimum-image bond vectors."""
    idx = list(mol.atoms)
    pos = {idx[0]: atoms.positions[idx[0]].copy()}
    adj = {a: [] for a in idx}
    for a, b in mol.bonds:
        adj[a].append(b)
        adj[b].append(a)
    q = deque([idx[0]])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if w not in pos:
                pos[w] = pos[u] + mic(atoms, atoms.positions[w] - atoms.positions[u])[0]
                q.append(w)
    if len(pos) != len(idx):
        raise TopologyError("molecule bond graph is disconnected")
    return np.array([pos[a] for a in idx])


def _cip_rank_key(atoms_z, adj, start, parent, depth=4):
    """Hierarchical digraph key: tuple of sorted-descending atomic-number shells (CIP-lite)."""
    key, frontier = [], [(start, parent)]
    for _ in range(depth):
        shell, nxt = [], []
        for node, par in frontier:
            for w in adj[node]:
                if w == par:
                    continue
                shell.append(atoms_z[w])
                nxt.append((w, node))
        key.append(tuple(sorted(shell, reverse=True)))
        frontier = nxt
    return (atoms_z[start], tuple(key))


def stereo_descriptor(pos_c, pos_by_priority) -> str:
    """'R' or 'S' from the stereocentre and its three highest-priority substituent positions (1 > 2 > 3).

    Looking with priority 4 away from the viewer, 1 -> 2 -> 3 clockwise is R, which is equivalent to
    (v1 x v2) . v3 < 0 for v_k = r_k - r_C. No hydrogen position is needed.
    """
    v1, v2, v3 = (np.asarray(p, float) - pos_c for p in pos_by_priority)
    s = np.dot(np.cross(v1, v2), v3)
    if abs(s) < 1e-3:
        raise TopologyError("near-planar stereocentre; configuration undefined")
    return "R" if s < 0 else "S"


def build_topology(ref: Atoms, metal="Pb", halide="I", mx_cut=3.8, covalent_scale=1.15,
                   org_cut=6.5, normal=None) -> Topology:
    """Build the fixed topology from a reference frame (an experimental CIF or an equilibrated snapshot)."""
    sym = np.array(ref.get_chemical_symbols())
    n = len(ref)
    if mx_cut >= _max_safe_cutoff(ref):
        raise TopologyError(f"metal-halide cutoff {mx_cut} A exceeds half the cell width; enlarge the cell")
    is_m, is_x = sym == metal, sym == halide

    # --- octahedra and halide roles
    i, j, D = neighbor_list("ijD", ref, mx_cut)
    sel = is_m[i] & is_x[j]
    mx = {}
    for a, b, d in zip(i[sel], j[sel], D[sel]):
        mx.setdefault(int(a), []).append((int(b), d))
    x_count = {}
    for a in mx:
        if len(mx[a]) != 6:
            raise TopologyError(f"{metal} {a} has {len(mx[a])} halides within {mx_cut} A (expected 6)")
        for b, _ in mx[a]:
            x_count[b] = x_count.get(b, 0) + 1
    role = {b: ("axial" if c == 1 else "bridging") for b, c in x_count.items()}
    if any(c > 2 for c in x_count.values()):
        raise TopologyError("halide bonded to >2 metals; not a corner-sharing layer")

    # --- links between metals through bridging halides; layers = connected components
    metals = sorted(mx)
    pos_of = {m: k for k, m in enumerate(metals)}
    by_x = {}
    for a in metals:
        for b, d in mx[a]:
            if role[b] == "bridging":
                by_x.setdefault(b, []).append(a)
    links = [Link(*sorted(v), via=x) for x, v in sorted(by_x.items()) if len(v) == 2]
    comps, _ = _components(len(metals), [(pos_of[l.i], pos_of[l.j]) for l in links])
    metal_layer = {metals[k]: c for c, comp in enumerate(comps) for k in comp}

    # --- layer normal: least-variance direction of in-plane M-X_b vectors
    if normal is None:
        vecs = np.array([d for a in metals for b, d in mx[a] if role[b] == "bridging"])
        normal = np.linalg.svd(vecs, full_matrices=False)[2][-1]
    normal = _unit(normal)

    octa = {}
    for a in metals:
        ax = [(b, d) for b, d in mx[a] if role[b] == "axial"]
        eq = [(b, d) for b, d in mx[a] if role[b] == "bridging"]
        if len(ax) != 2 or len(eq) != 4:
            raise TopologyError(f"{metal} {a}: {len(ax)} axial / {len(eq)} bridging halides (expected 2/4)")
        ax.sort(key=lambda t: np.dot(t[1], normal))
        e0 = eq[0]
        trans = min(eq[1:], key=lambda t: np.dot(_unit(t[1]), _unit(e0[1])))
        rest = [t[0] for t in eq[1:] if t[0] != trans[0]]
        octa[a] = Octahedron(a, (ax[0][0], ax[1][0]), ((e0[0], trans[0]), (rest[0], rest[1])))

    # --- molecules
    org = np.where(~is_m & ~is_x)[0]
    cut = np.array(natural_cutoffs(ref, mult=covalent_scale))
    cut[is_m | is_x] = 0.0
    ii, jj = neighbor_list("ij", ref, cut)
    edges = [(a, b) for a, b in zip(ii, jj) if a < b and not (is_m[a] or is_x[a] or is_m[b] or is_x[b])]
    loc = {a: k for k, a in enumerate(org)}
    comps, adj_local = _components(len(org), [(loc[a], loc[b]) for a, b in edges])
    adj = {int(a): [] for a in org}
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    z = {int(a): atomic_numbers[sym[a]] for a in org}
    mols = []
    for comp in comps:
        idx = np.array([int(org[k]) for k in comp])
        ns = [a for a in idx if sym[a] == "N" and sum(sym[w] == "H" for w in adj[a]) >= 3]
        if len(ns) != 1:
            raise TopologyError(f"molecule {idx[:5]}... has {len(ns)} ammonium N (expected 1)")
        nN = int(ns[0])
        m = Molecule(idx, [(int(a), int(b)) for a, b in edges if a in set(idx)], nN,
                     [w for w in adj[nN] if sym[w] == "H"], [int(a) for a in idx if sym[a] == "C"])
        upos = dict(zip(idx, unwrap_molecule(ref, m)))
        for c in m.carbons:
            nb = adj[c]
            if len(nb) != 4:
                continue
            keys = {w: _cip_rank_key(z, adj, w, c) for w in nb}
            ranked = sorted(nb, key=lambda w: keys[w], reverse=True)
            if len({keys[w] for w in nb}) < 4:
                continue  # not a stereocentre (two identical branches to depth 4)
            m.stereocentres[c] = stereo_descriptor(upos[c], [upos[w] for w in ranked[:3]])
        # layer and face from the nearest metal
        dv = mic(ref, ref.positions[nN] - ref.positions[metals])
        k = int(np.argmin(np.linalg.norm(dv, axis=1)))
        m.layer = metal_layer[metals[k]]
        m.face = 1 if np.dot(dv[k], normal) > 0 else -1
        mols.append(m)

    # --- organic links: same layer, same face, in-plane N-N separation < org_cut, EVERY periodic image
    nidx = np.array([m.n_ammonium for m in mols])
    sub = ref[nidx]
    pi, pj, PD = neighbor_list("ijD", sub, org_cut + 3.0)
    olinks = []
    for p, q, d in zip(pi, pj, PD):
        if p > q:
            continue  # neighbor_list returns (p,q,D) and (q,p,-D); keep one
        if mols[p].layer != mols[q].layer or mols[p].face != mols[q].face:
            continue
        if np.linalg.norm(d - np.dot(d, normal) * normal) >= org_cut:
            continue
        if p == q:  # self-image: keep one of the +/-T pair
            if any(l.i == p and l.j == p and np.allclose(l.ref_frac @ ref.cell.array, -d, atol=1e-3)
                   for l in olinks):
                continue
        olinks.append(Link(int(p), int(q), ref_frac=np.linalg.solve(ref.cell.array.T, d)))
    return Topology(n, list(sym), normal, ref.cell.array @ normal, octa, links, metal_layer, mols, olinks, role)


def layer_normal(top: Topology, atoms: Atoms) -> np.ndarray:
    """Unit layer normal for this frame: the same lattice plane as in the reference (rotates with the cell)."""
    return _unit(top.normal_coeff @ atoms.cell.reciprocal())


# --------------------------------------------------------------------------------------------- per-frame descriptors

def _check_frame(top: Topology, atoms: Atoms):
    if len(atoms) != top.n_atoms or atoms.get_chemical_symbols() != top.symbols:
        raise TopologyError("frame atoms/order differ from the topology reference")


def axial_vectors(top: Topology, atoms: Atoms) -> dict[int, np.ndarray]:
    out = {}
    for m, o in top.octahedra.items():
        lo, hi = o.axial
        v = mic(atoms, atoms.positions[[hi, lo]] - atoms.positions[m])
        out[m] = _unit(v[0] - v[1])   # X_ax(+) - X_ax(-) through the metal, minimum image per bond
    # orient along +n (per frame, so a flipped octahedron is not silently counted with the wrong sign)
    n = layer_normal(top, atoms)
    return {m: (v if np.dot(v, n) >= 0 else -v) for m, v in out.items()}


def framework_twists(top: Topology, atoms: Atoms) -> dict[str, np.ndarray]:
    """Per-link twists for the out-of-plane (``ax``) and in-plane (``eq``) framework descriptors."""
    _check_frame(top, atoms)
    P = atoms.positions
    ax = axial_vectors(top, atoms)
    normal = layer_normal(top, atoms)
    t_ax, t_eq = [], []
    for l in top.metal_links:
        e = _unit(mic(atoms, P[l.via] - P[l.i])[0] + mic(atoms, P[l.j] - P[l.via])[0])
        t_ax.append(np.dot(np.cross(ax[l.i], ax[l.j]), e))
        mvec = np.cross(normal, e)
        vs = []
        for site in (l.i, l.j):
            pairs = top.octahedra[site].eq_pairs
            a, b = pairs[1] if l.via in pairs[0] else pairs[0]
            d = mic(atoms, P[[a, b]] - P[site])
            v = _unit(d[0] - d[1])
            vs.append(v if np.dot(v, mvec) >= 0 else -v)
        t_eq.append(np.dot(np.cross(vs[0], vs[1]), e))
    return {"ax": np.array(t_ax), "eq": np.array(t_eq)}


def cation_axes(top: Topology, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    """(N positions, unit N - C-centroid axes) per molecule, molecules unwrapped."""
    pos, axes = [], []
    for m in top.molecules:
        u = dict(zip(m.atoms, unwrap_molecule(atoms, m)))
        cc = np.mean([u[c] for c in m.carbons], axis=0)
        pos.append(u[m.n_ammonium])
        axes.append(_unit(u[m.n_ammonium] - cc))
    return np.array(pos), np.array(axes)


def organic_twists(top: Topology, atoms: Atoms) -> np.ndarray:
    _check_frame(top, atoms)
    pos, axes = cation_axes(top, atoms)
    out = []
    for l in top.org_links:
        # the image of j closest to the reference separation (valid while displacements < half a cell)
        ref_d = l.ref_frac @ atoms.cell.array
        e = _unit(mic(atoms, pos[l.j] - pos[l.i] - ref_d)[0] + ref_d)
        out.append(np.dot(np.cross(axes[l.i], axes[l.j]), e))
    return np.array(out)


def site_means(links: list[Link], tau: np.ndarray, n_sites_keys) -> dict:
    """Mean link twist per site (each link counted for both ends): local, chain-free handedness."""
    acc = {k: [] for k in n_sites_keys}
    for l, t in zip(links, tau):
        acc[l.i].append(t)
        acc[l.j].append(t)
    return {k: float(np.mean(v)) if v else np.nan for k, v in acc.items()}


def stereocentres(top: Topology, atoms: Atoms) -> list[str]:
    """Current R/S labels (detects inversion of a stereocentre during a trajectory, which should never happen)."""
    _check_frame(top, atoms)
    sym = top.symbols
    z = {int(a): atomic_numbers[sym[a]] for m in top.molecules for a in m.atoms}
    out = []
    for m in top.molecules:
        u = dict(zip(m.atoms, unwrap_molecule(atoms, m)))
        adj = {int(a): [] for a in m.atoms}
        for a, b in m.bonds:
            adj[a].append(b)
            adj[b].append(a)
        for c in m.stereocentres:
            ranked = sorted(adj[c], key=lambda w: _cip_rank_key(z, adj, w, c), reverse=True)
            out.append(stereo_descriptor(u[c], [u[w] for w in ranked[:3]]))
    return out


def frame_descriptors(top: Topology, atoms: Atoms) -> dict:
    """All handedness descriptors for one frame: per-link arrays, per-site means and cell means."""
    fw = framework_twists(top, atoms)
    org = organic_twists(top, atoms)
    res = {"tau_ax": fw["ax"], "tau_eq": fw["eq"], "tau_org": org}
    for k, links, keys in (("ax", top.metal_links, top.octahedra), ("eq", top.metal_links, top.octahedra),
                           ("org", top.org_links, range(len(top.molecules)))):
        tau = res[f"tau_{k}"]
        res[f"site_{k}"] = site_means(links, tau, keys)
        res[f"mean_{k}"] = float(np.mean(tau)) if len(tau) else np.nan
    # per-layer means (domain cancellation check across layers)
    for k in ("ax", "eq"):
        lay = {}
        for l, t in zip(top.metal_links, res[f"tau_{k}"]):
            lay.setdefault(top.metal_layer[l.i], []).append(t)
        res[f"layer_{k}"] = {L: float(np.mean(v)) for L, v in sorted(lay.items())}
    lay = {}
    for l, t in zip(top.org_links, org):
        m = top.molecules[l.i]
        lay.setdefault((m.layer, m.face), []).append(t)
    res["layer_org"] = {L: float(np.mean(v)) for L, v in sorted(lay.items())}
    return res


# --------------------------------------------------------------------------------------------- distortions

def mxm_angles(top: Topology, atoms: Atoms) -> np.ndarray:
    """M-X_b-M bridging angles (degrees), one per link."""
    P = atoms.positions
    out = []
    for l in top.metal_links:
        a = mic(atoms, P[l.i] - P[l.via])[0]
        b = mic(atoms, P[l.j] - P[l.via])[0]
        out.append(np.degrees(np.arccos(np.clip(np.dot(_unit(a), _unit(b)), -1, 1))))
    return np.array(out)


def octahedral_distortion(top: Topology, atoms: Atoms) -> dict[int, dict]:
    """Bond-length distortion Delta_d = <((d-dbar)/dbar)^2> and cis-angle variance sigma^2 (deg^2, 12 angles)."""
    P = atoms.positions
    out = {}
    for m, o in top.octahedra.items():
        xs = [*o.axial, *o.eq_pairs[0], *o.eq_pairs[1]]
        v = mic(atoms, P[xs] - P[m])
        d = np.linalg.norm(v, axis=1)
        u = v / d[:, None]
        cosm = u @ u.T
        iu = np.triu_indices(6, 1)
        ang = np.degrees(np.arccos(np.clip(cosm[iu], -1, 1)))
        cis = np.sort(ang)[:12]
        out[m] = {"delta_d": float(np.mean(((d - d.mean()) / d.mean()) ** 2)),
                  "sigma2": float(np.sum((cis - 90.0) ** 2) / 11.0), "mean_bond": float(d.mean())}
    return out


def headgroup_vectors(top: Topology, atoms: Atoms, which="CN") -> np.ndarray:
    """Unit vectors per molecule: 'CN' = C(alpha)->N, or 'NH' = all N-H bonds (n_mol*3, 3) for rotational ACFs."""
    out = []
    for m in top.molecules:
        u = dict(zip(m.atoms, unwrap_molecule(atoms, m)))
        if which == "CN":
            ca = min(m.carbons, key=lambda c: np.linalg.norm(u[c] - u[m.n_ammonium]))
            out.append(_unit(u[m.n_ammonium] - u[ca]))
        elif which == "NH":
            out.extend(_unit(u[h] - u[m.n_ammonium]) for h in m.nh_h)
        else:
            raise ValueError(which)
    return np.array(out)


def headgroup_tilt(top: Topology, atoms: Atoms) -> np.ndarray:
    """Angle (deg) between C->N and the outward layer normal of each cation's face."""
    v = headgroup_vectors(top, atoms, "CN")
    s = np.array([m.face for m in top.molecules])
    return np.degrees(np.arccos(np.clip((v @ layer_normal(top, atoms)) * s, -1, 1)))


# --------------------------------------------------------------------------------------------- hydrogen bonds

@dataclass
class HBondCriteria:
    r_max: float = 3.2         # H...I (A)
    angle_min: float = 120.0   # N-H...I (deg)


def hbond_matrix(top: Topology, atoms: Atoms, crit: HBondCriteria = HBondCriteria(),
                 acceptors=None) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Boolean presence and H...A distances for every (N-H hydrogen, acceptor) pair (fixed ordering).

    Returns (present[n_pairs], r_HA[n_pairs], pairs) where pairs = [(H index, acceptor index)].
    """
    P = atoms.positions
    if acceptors is None:
        acceptors = sorted(top.halide_role)
    acc = np.array(acceptors)
    pres, dist, pairs = [], [], []
    for m in top.molecules:
        for h in m.nh_h:
            nh = mic(atoms, P[h] - P[m.n_ammonium])[0]
            ha = mic(atoms, P[acc] - P[h])
            r = np.linalg.norm(ha, axis=1)
            cosang = (ha @ -nh) / (r * np.linalg.norm(nh))     # angle at H between H->N and H->A
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            pres.append((r <= crit.r_max) & (ang >= crit.angle_min))
            dist.append(r)
            pairs.extend((h, int(a)) for a in acc)
    return np.concatenate(pres), np.concatenate(dist), pairs


def cation_hbond_length(top: Topology, atoms: Atoms) -> np.ndarray:
    """Per cation: mean of its two shortest N-H...X distances (H...X), after Pols et al.'s r_HB."""
    P = atoms.positions
    acc = np.array(sorted(top.halide_role))
    out = []
    for m in top.molecules:
        best = []
        for h in m.nh_h:
            best.append(np.min(np.linalg.norm(mic(atoms, P[acc] - P[h]), axis=1)))
        out.append(float(np.mean(sorted(best)[:2])))
    return np.array(out)


def hbond_face_asymmetry(top: Topology, atoms: Atoms) -> dict:
    """Per layer: mean r_HB on the + face minus the - face (signed by the reference normal; a polar
    quantity, reported to compare with the published static asymmetry, not a handedness)."""
    r = cation_hbond_length(top, atoms)
    out = {}
    for L in sorted({m.layer for m in top.molecules}):
        up = [r[k] for k, m in enumerate(top.molecules) if m.layer == L and m.face > 0]
        dn = [r[k] for k, m in enumerate(top.molecules) if m.layer == L and m.face < 0]
        if up and dn:
            out[L] = float(np.mean(up) - np.mean(dn))
    return out


def hbond_lifetimes(h: np.ndarray, max_lag: int) -> dict[str, np.ndarray]:
    """Continuous S(t) and intermittent C(t) correlation functions from a presence array h[time, pair].

    C(t) = <h(0) h(t)> / <h>;  S(t) = <h(0) H(0..t)> / <h>, H = 1 only if present at every frame in [0, t].
    Averaged over all time origins and pairs. Frame spacing sets the resolution: transient breaks shorter
    than one saved frame are invisible, so S(t) depends on the save interval (report it).
    """
    h = np.asarray(h, bool)
    T = h.shape[0]
    if max_lag >= T:
        raise ValueError("max_lag must be < number of frames")
    mean_h = h.mean()
    if mean_h == 0:
        return {"C": np.full(max_lag + 1, np.nan), "S": np.full(max_lag + 1, np.nan), "occupancy": 0.0}
    C, S = np.empty(max_lag + 1), np.empty(max_lag + 1)
    run = h.copy()
    for t in range(max_lag + 1):
        if t > 0:
            run = run[:-1] & h[t:]
        C[t] = np.mean(h[: T - t] & h[t:]) / mean_h
        S[t] = np.mean(run) / mean_h
    return {"C": C, "S": S, "occupancy": float(mean_h)}


def hbond_sensitivity(top: Topology, frames, r_values=(3.0, 3.2, 3.4), angle_values=(110.0, 120.0, 135.0)):
    """Mean H-bond count per N-H for a grid of criteria. Results that move strongly with the criteria
    should not be quoted as material properties."""
    out = {}
    for r in r_values:
        for a in angle_values:
            c = HBondCriteria(r, a)
            n = [hbond_matrix(top, f, c)[0].sum() for f in frames]
            nh = sum(len(m.nh_h) for m in top.molecules)
            out[(r, a)] = float(np.mean(n) / nh)
    return out


# --------------------------------------------------------------------------------------------- statistics

def autocorr(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalised autocorrelation of a scalar series (mean removed)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    max_lag = n - 1 if max_lag is None else max_lag
    var = np.dot(x, x) / n
    if var == 0:
        return np.ones(max_lag + 1)
    return np.array([np.dot(x[: n - k], x[k:]) / (n - k) / var for k in range(max_lag + 1)])


def integrated_autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    """Sokal automatic-window estimate of tau_int (in frames); N_eff = N / (2 tau_int)."""
    rho = autocorr(x)
    tau = 0.5
    for M in range(1, len(rho)):
        tau += rho[M]
        if M >= c * tau:
            break
    return float(max(tau, 0.5))


def block_average(x: np.ndarray, min_blocks: int = 8) -> dict:
    """Flyvbjerg-Petersen blocking. Returns the mean, the plateau standard error and the full curve.

    The plateau is taken as the maximum SE over levels that keep >= min_blocks blocks (conservative).
    """
    x = np.asarray(x, float)
    mean = float(x.mean())
    levels, y = [], x.copy()
    while len(y) >= min_blocks:
        n = len(y)
        se = np.sqrt(np.var(y, ddof=1) / n) if n > 1 else np.nan
        levels.append((len(x) // n, n, float(se)))
        y = 0.5 * (y[: n // 2 * 2 : 2] + y[1 : n // 2 * 2 : 2])
    se = max(l[2] for l in levels) if levels else np.nan
    naive = float(np.sqrt(np.var(x, ddof=1) / len(x)))
    return {"mean": mean, "se": float(se), "naive_se": naive, "levels": levels,
            "ci95": (mean - 1.96 * se, mean + 1.96 * se)}


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation r_xy(k) = <dx(t) dy(t+k)> / (s_x s_y) for k in [-max_lag, max_lag].
    Positive k: x leads y. Correlation and lag are supporting evidence only, not causality."""
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    n = len(x)
    sx, sy = x.std(), y.std()
    lags = np.arange(-max_lag, max_lag + 1)
    r = []
    for k in lags:
        if k >= 0:
            r.append(np.mean(x[: n - k] * y[k:]))
        else:
            r.append(np.mean(x[-k:] * y[: n + k]))
    return lags, np.array(r) / (sx * sy if sx * sy > 0 else np.nan)


def rotational_acf(vectors: np.ndarray, max_lag: int, order: int = 1) -> np.ndarray:
    """<P_l(u(t0) . u(t0+t))> over origins and vectors; vectors shape (T, n, 3), unit length."""
    v = np.asarray(vectors, float)
    out = np.empty(max_lag + 1)
    for t in range(max_lag + 1):
        c = np.einsum("tij,tij->ti", v[: len(v) - t], v[t:])
        out[t] = np.mean(c if order == 1 else 1.5 * c ** 2 - 0.5)
    return out


def degree_of_chirality(values: np.ndarray, dead_band: float = 0.0) -> float:
    """chi = (n_+ - n_-) / (n_+ + n_-) over samples with |value| > dead_band; in [-1, 1].

    Our operational definition (the published Eq. 3 of Pols et al. was not verified). It is a sign census,
    insensitive to magnitude, so report it together with the signed distribution and its mean.
    """
    v = np.asarray(values, float)
    p, m = np.sum(v > dead_band), np.sum(v < -dead_band)
    return float((p - m) / (p + m)) if p + m else 0.0


def spatial_sign_correlation(positions: np.ndarray, values: np.ndarray, atoms: Atoms, bins) -> dict:
    """<sign(v_i) sign(v_j)> of signed site values vs minimum-image separation (domain behaviour)."""
    pos = np.asarray(positions, float)
    v = np.asarray(values, float)
    i, j = np.triu_indices(len(v), 1)
    d = np.linalg.norm(mic(atoms, pos[j] - pos[i]), axis=1)
    prod = np.sign(v[i]) * np.sign(v[j])
    idx = np.digitize(d, bins) - 1
    out = {}
    for b in range(len(bins) - 1):
        sel = idx == b
        out[(float(bins[b]), float(bins[b + 1]))] = (float(prod[sel].mean()) if sel.any() else np.nan, int(sel.sum()))
    return out


# --------------------------------------------------------------------------------------------- trajectory driver

def analyze_trajectory(top: Topology, frames, crit: HBondCriteria = HBondCriteria()) -> dict:
    """Time series of the cell/layer means plus per-link arrays (frames x links) and H-bond presence."""
    keys = ("mean_ax", "mean_eq", "mean_org")
    series = {k: [] for k in keys}
    links = {"tau_ax": [], "tau_eq": [], "tau_org": []}
    hb, mxm, tilt, nh = [], [], [], []
    for f in frames:
        d = frame_descriptors(top, f)
        for k in keys:
            series[k].append(d[k])
        for k in links:
            links[k].append(d[k])
        hb.append(hbond_matrix(top, f, crit)[0])
        mxm.append(mxm_angles(top, f))
        tilt.append(headgroup_tilt(top, f))
        nh.append(headgroup_vectors(top, f, "NH"))
    return {**{k: np.array(v) for k, v in series.items()}, **{k: np.array(v) for k, v in links.items()},
            "hbond": np.array(hb), "mxm_angle": np.array(mxm), "headgroup_tilt": np.array(tilt),
            "nh_vectors": np.array(nh)}
