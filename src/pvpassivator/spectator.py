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


def spectator_mismatch(slab: Atoms, cx: Atoms, cutoff: float = CUTOFF_A, limit: float = LIMIT_MEAN_A) -> dict:
    """Far-field mismatch between a bare slab and a complex built on it (see module docstring)."""
    if not np.allclose(slab.cell.array, cx.cell.array, atol=1e-4):
        raise ValueError("slab and complex cells differ; compare structures of one label set only")
    ss, cs = np.array(slab.get_chemical_symbols()), np.array(cx.get_chemical_symbols())
    extra = Counter(cs) - Counter(ss)
    if Counter(ss) - Counter(cs):
        raise ValueError("the complex lacks atoms that the slab has")
    t = _translation(slab, cx)
    P = cx.positions + t
    match = np.full(len(slab), -1)
    disp = np.full(len(slab), np.nan)
    for e in np.unique(ss):
        a, b = np.flatnonzero(ss == e), np.flatnonzero(cs == e)
        C = _dist(slab.positions[a], P[b], slab.cell)
        r, c = linear_sum_assignment(C)
        match[a[r]], disp[a[r]] = b[c], C[r, c]
    ads = np.setdiff1d(np.arange(len(cx)), match)
    if Counter(cs[ads]) != extra:
        raise ValueError("unassigned complex atoms do not equal the composition difference")
    d_ads = _dist(slab.positions, P[ads], slab.cell).min(1)
    far = (d_ads > cutoff) & (ss != "H")
    if not far.any():
        raise ValueError(f"no non-H slab atom is farther than {cutoff} A from the adsorbate")
    m = disp[far]
    by_el = {e: round(float(m[ss[far] == e].mean()), 3) for e in np.unique(ss[far])}
    return {"cutoff_A": cutoff, "limit_mean_A": limit, "n_far": int(far.sum()),
            "adsorbate": dict(Counter(cs[ads].tolist())), "translation_A": [round(float(x), 3) for x in t],
            "mean_A": round(float(m.mean()), 4), "max_A": round(float(m.max()), 3),
            "n_over_0.3A": int((m > 0.3).sum()), "mean_by_element_A": by_el,
            "pass": bool(m.mean() <= limit)}
