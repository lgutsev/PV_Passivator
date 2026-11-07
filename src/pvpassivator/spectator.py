"""Spectator consistency between a bare slab and a complex (package-53 lesson, docs/PACKAGE53_RELAX_DIAGNOSIS.md).

E_ads = E(complex) - E(slab) - E(salt) assumes the slab part of the complex is the bare slab plus a local response to
the salt. If the two relaxations ended in different tilt/FA patterns far from the salt, E_ads carries that energy
difference too, at any convergence. This module measures it:

1. Best rigid translation of the complex onto the slab from the Pb sublattice (every Pb pairing tried, then refined by
   the mean minimum-image residual). Cells must match (common cell of a label set).
2. One-to-one assignment of slab atoms to complex atoms, per element (Hungarian, minimum-image distances). Complex
   atoms left unassigned are the adsorbate: the composition difference, so an iodide counter-ion is found too.
3. Mismatch = displacement of assigned non-H slab atoms farther than ``cutoff`` from every adsorbate atom.

The verdict uses the mean far-field mismatch (default limit 0.1 A); max and the count above 0.3 A are reported.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment

CUTOFF_A = 10.0
LIMIT_MEAN_A = 0.1
SHELL_A = 2.0              # boundary shell: cutoff - SHELL_A .. cutoff
SHELL_LIMIT_MEAN_A = 0.15  # softer than the far-field limit: the shell may respond to the salt a little


def _dist(A: np.ndarray, B: np.ndarray, cell) -> np.ndarray:
    d = A[:, None, :] - B[None, :, :]
    return np.linalg.norm(find_mic(d.reshape(-1, 3), cell, pbc=True)[0], axis=1).reshape(len(A), len(B))


def _assign_cost(slab: Atoms, cx: Atoms, t: np.ndarray) -> float:
    """Mean one-to-one distance of all non-H slab atoms to the translated complex."""
    ss, cs = np.array(slab.get_chemical_symbols()), np.array(cx.get_chemical_symbols())
    tot, n = 0.0, 0
    for e in np.unique(ss[ss != "H"]):
        C = _dist(slab.positions[ss == e], cx.positions[cs == e] + t, slab.cell)
        r, c = linear_sum_assignment(C)
        tot, n = tot + C[r, c].sum(), n + len(r)
    return tot / n


def _translation(slab: Atoms, cx: Atoms, tie_A: float = 0.05) -> np.ndarray:
    ss, cs = np.array(slab.get_chemical_symbols()), np.array(cx.get_chemical_symbols())
    pa, pb = slab.positions[ss == "Pb"], cx.positions[cs == "Pb"]
    if len(pa) == 0 or len(pa) != len(pb):
        raise ValueError("slab and complex need the same, nonzero number of Pb atoms")
    cands = []
    for j in range(len(pb)):
        t = pa[0] - pb[j]
        for _ in range(2):          # assign, then refine t by the MEDIAN residual: a distorted region does not drag it
            C = _dist(pa, pb + t, slab.cell)
            r, c = linear_sum_assignment(C)
            t = t + np.median(find_mic(pa[r] - (pb[c] + t), slab.cell, pbc=True)[0], axis=0)
        C = _dist(pa, pb + t, slab.cell)
        r, c = linear_sum_assignment(C)
        cands.append((float(np.median(C[r, c])), t))
    best = min(s for s, _ in cands)
    # a nearly periodic Pb sublattice fits several lattice translations equally well: let all non-H atoms decide
    near = [t for s, t in cands if s <= best + tie_A]
    unique = []
    for t in near:
        if not any(np.linalg.norm(find_mic((t - u)[None], slab.cell, pbc=True)[0]) < 1e-3 for u in unique):
            unique.append(t)
    return min(unique, key=lambda t: _assign_cost(slab, cx, t))


def _formula(sym) -> str:
    c = Counter(sym)
    return "".join(f"{e}{c[e]}" for e in sorted(c))


def _salt_fragments(slab: Atoms, cx: Atoms, extra: Counter) -> np.ndarray:
    """Complex atoms in covalent fragments whose formula never occurs in the slab (the salt's cation and molecular or
    foreign anions), if they fit inside the composition difference; else none (the caller falls back to matching).

    Per-element matching alone can trade a salt H or C with a lattice FA atom; whole fragments cannot."""
    from .poses import fragments
    ss, cs = np.array(slab.get_chemical_symbols()), np.array(cx.get_chemical_symbols())
    known = {_formula(ss[g]) for g in fragments(slab)}
    idx = [i for g in fragments(cx) if _formula(cs[g]) not in known for i in g]
    if not idx or Counter(cs[idx]) - extra:
        return np.array([], dtype=int)
    return np.array(sorted(idx), dtype=int)


def align(slab: Atoms, cx: Atoms) -> dict:
    """Rigid translation t (complex frame + t = slab frame), slab->complex atom matching, adsorbate indices, and the
    displacement of every matched slab atom (minimum image).

    The adsorbate is first the complex's fragments with formulas foreign to the slab (``_salt_fragments``); the rest of
    the composition difference (e.g. an iodide counter-ion on an iodide slab) is the complex atoms left over by the
    per-element one-to-one matching."""
    if not np.allclose(slab.cell.array, cx.cell.array, atol=1e-4):
        raise ValueError("slab and complex cells differ; compare structures of one label set only")
    ss, cs = np.array(slab.get_chemical_symbols()), np.array(cx.get_chemical_symbols())
    extra = Counter(cs) - Counter(ss)
    if Counter(ss) - Counter(cs):
        raise ValueError("the complex lacks atoms that the slab has")
    t = _translation(slab, cx)
    P = cx.positions + t
    salt = set(_salt_fragments(slab, cx, extra).tolist())
    match = np.full(len(slab), -1)
    disp = np.full(len(slab), np.nan)
    for e in np.unique(ss):
        a = np.flatnonzero(ss == e)
        b = np.array([i for i in np.flatnonzero(cs == e) if i not in salt], dtype=int)
        C = _dist(slab.positions[a], P[b], slab.cell)
        r, c = linear_sum_assignment(C)
        match[a[r]], disp[a[r]] = b[c], C[r, c]
    ads = np.setdiff1d(np.arange(len(cx)), match)
    if Counter(cs[ads]) != extra:
        raise ValueError("unassigned complex atoms do not equal the composition difference")
    return {"t": t, "match": match, "disp": disp, "ads": ads,
            "d_ads": _dist(slab.positions, P[ads], slab.cell).min(1)}


def spectator_mismatch(slab: Atoms, cx: Atoms, cutoff: float = CUTOFF_A, limit: float = LIMIT_MEAN_A,
                       shell_A: float = SHELL_A, shell_limit: float = SHELL_LIMIT_MEAN_A) -> dict:
    """Far-field mismatch between a bare slab and a complex built on it (see module docstring).

    ``shell``: the non-H slab atoms between cutoff - shell_A and cutoff from the adsorbate. With the far field frozen at
    the slab (``freeze_far_field``) the far-field test passes by construction; the shell then shows whether the
    relaxation's response has decayed at the freeze radius (if not, enlarge it)."""
    al = align(slab, cx)
    ss = np.array(slab.get_chemical_symbols())
    cs = np.array(cx.get_chemical_symbols())
    disp, d_ads = al["disp"], al["d_ads"]
    far = (d_ads > cutoff) & (ss != "H")
    if not far.any():
        raise ValueError(f"no non-H slab atom is farther than {cutoff} A from the adsorbate")
    m = disp[far]
    by_el = {e: round(float(m[ss[far] == e].mean()), 3) for e in np.unique(ss[far])}
    sh = (d_ads > cutoff - shell_A) & (d_ads <= cutoff) & (ss != "H")
    shell = ({"range_A": [cutoff - shell_A, cutoff], "n": int(sh.sum()), "mean_A": round(float(disp[sh].mean()), 4),
              "max_A": round(float(disp[sh].max()), 3), "limit_mean_A": shell_limit,
              "pass": bool(disp[sh].mean() <= shell_limit)} if sh.any()
             else {"range_A": [cutoff - shell_A, cutoff], "n": 0, "pass": None})
    return {"cutoff_A": cutoff, "limit_mean_A": limit, "n_far": int(far.sum()),
            "adsorbate": dict(Counter(cs[al["ads"]].tolist())), "translation_A": [round(float(x), 3) for x in al["t"]],
            "mean_A": round(float(m.mean()), 4), "max_A": round(float(m.max()), 3),
            "n_over_0.3A": int((m > 0.3).sum()), "mean_by_element_A": by_el,
            "pass": bool(m.mean() <= limit), "shell": shell}


def freeze_far_field(slab: Atoms, cx: Atoms, radius: float = CUTOFF_A) -> tuple[Atoms, list[int], dict]:
    """Snap the complex's far field onto the slab and list the atoms to freeze (selective dynamics F F F).

    Atoms farther than ``radius`` from every adsorbate atom are placed exactly at their matched slab position
    (in the complex's own frame) and frozen. Organic cations are frozen whole or not at all: a molecule stays free if
    any of its atoms is within ``radius``. Pb and I are judged per atom. Returns (complex, frozen indices, report);
    the report's snap statistics say how far the frozen atoms had to move (large = the complex was not built on this
    slab: rebuild it first)."""
    from ase.neighborlist import NeighborList, natural_cutoffs
    from scipy.sparse.csgraph import connected_components
    al = align(slab, cx)
    out = cx.copy()
    cs = np.array(cx.get_chemical_symbols())
    ads = set(al["ads"].tolist())
    d = _dist(cx.positions, cx.positions[al["ads"]], cx.cell).min(1)
    far = d > radius
    org = np.flatnonzero(~np.isin(cs, ["Pb", "I"]))
    sub = cx[org]
    nl = NeighborList(natural_cutoffs(sub, mult=1.15), self_interaction=False, bothways=True)
    nl.update(sub)
    _, lab = connected_components(nl.get_connectivity_matrix(sparse=True))
    for c in np.unique(lab):
        idx = org[lab == c]
        if not far[idx].all():
            far[idx] = False
    far[list(ads)] = False
    inv = {int(b): int(a) for a, b in enumerate(al["match"])}
    frozen = [k for k in np.flatnonzero(far) if k in inv]
    snap = []
    for k in frozen:
        target = slab.positions[inv[k]] - al["t"]
        delta = find_mic((target - out.positions[k])[None], cx.cell, pbc=True)[0][0]
        snap.append(np.linalg.norm(delta))
        out.positions[k] = out.positions[k] + delta
    snap = np.array(snap) if snap else np.zeros(1)
    report = {"freeze_beyond_A": radius, "n_frozen": len(frozen), "n_free": len(cx) - len(frozen),
              "n_adsorbate": len(ads), "snap_mean_A": round(float(snap.mean()), 4),
              "snap_max_A": round(float(snap.max()), 3)}
    return out, [int(k) for k in frozen], report
