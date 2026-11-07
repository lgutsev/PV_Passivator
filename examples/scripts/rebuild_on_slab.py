"""Transplant each salt of an archived complex onto a (relaxed) bare slab: step A2 of docs/PACKAGE53_REBUILD_PLAN.md.

    python examples/scripts/rebuild_on_slab.py --slab <slab CONTCAR> --case PEAAc=<complex CONTCAR> ... --out <dir>
    python examples/scripts/rebuild_on_slab.py --package <53 folder> --cases PEAAc,PEATFA,... --out <dir>

``--package`` reads ``outputs/slab_FAI_Schot/relax/CONTCAR`` and ``outputs/complex_<name>/relax/CONTCAR``
(``--slab-name`` / ``--source`` change that). Never submits anything; writes inputs and reports only.

Per complex:
1. ``spectator.align``: best rigid translation of the archived complex onto the slab from the Pb sublattice, and the
   salt (fragments foreign to the slab, plus any leftover composition difference such as an iodide counter-ion).
   The complex must be slab + salt by composition, in the same cell.
2. Local frame: the salt keeps its contact geometry with the lattice it was relaxed on. Its translation is the median
   minimum-image offset of the matched Pb/I within ``--local-radius`` of the salt (default 8 A), not the global one:
   the archived far field may differ from the slab by 0.3-0.8 A (diagnosis, Sect. 2), the contact region should not.
   Matching is per lattice unit (Pb, I, whole FA molecule; ``match_units``), not per atom. A unit on another site than
   in the slab (centroid displacement > 2.5 A) whose complex position OR vacated slab site lies within
   ``local_radius`` of the salt is part of the salt's local configuration and moves with it ("relocated").
   Package 53 (dry run 2026-11-07): in 7 of 9 complexes the cation sits on an A-site the slab fills with FA, and
   that FA sits on the slab's V_FA; in 5 the anion sits on a surface I site and that iodide fills V_I.
3. Rebuilt complex = every other matched atom placed exactly at its slab position + the salt (and relocated atoms)
   moved by the local translation.
   The archived complex's atom order is kept (same species blocks, POTCAR order); the cell is the slab's. So the
   spectator mismatch to the slab is 0 by construction (checked, not assumed).

Acceptance (``pass``: all must hold; ``status`` is pass / needs_settle (only the two relative contact checks fail,
which the A3 local settle addresses) / fail):
  min_distance_A        >= 1.0 A between any salt atom and any slab atom (slab-slab pairs are the converged slab's own)
  salt_H_lattice_A      no salt H-Pb/I contact shorter than in the archived complex (tolerance ``--h-tol``, 0.05 A)
  salt_slab_A           no salt-slab contact (H included, e.g. against lattice FA) shorter than archived - h_tol,
                        unless it is still >= 1.8 A (a new contact there is not a clash)
  sites                 cation N and each anion's anchor (its atom closest to Pb) move <= 0.3 A relative to the
                        centroid of the 4 Pb that were nearest to it in the archived complex (the same atoms, by
                        identity): N, the anion and each relocated unit keep their local site. ``shared_Pb`` (how many of
                        those 4 are still the nearest) is reported only: near-ties between Pb make it noisy.
  lattice_sites         no unit farther than ``local_radius`` from the salt (both ends) sits on another site than in
                        the slab: the vacancy pattern must be the slab's away from the salt
  spectator_mean_A      far-field mismatch of the rebuilt complex to the slab < 1e-4 A

A failed case is still written (``POSCAR``), so it can be inspected; ``summary.json`` lists which passed. The local
settle (A3) and the far-field freeze (A4, ``prepare_label_set.py freeze_beyond_A``) come after this step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pvpassivator.poses import fragments  # noqa: E402
from pvpassivator.spectator import align, spectator_mismatch  # noqa: E402

LOCAL_RADIUS_A = 8.0
MIN_DIST_A = 1.0
H_TOL_A = 0.05
NEW_CONTACT_OK_A = 1.8
SITE_K = 4
SITE_SHIFT_MAX_A = 0.3
SITE_JUMP_A = 2.5           # a matched unit (Pb, I, FA) farther than this from its slab partner is on another site
LATTICE = ("Pb", "I")
SOFT = ("salt_H_lattice_A", "salt_slab_A")   # contact shortening only: the local settle (A3) is meant to fix it


def _mic(v: np.ndarray, cell) -> np.ndarray:
    return find_mic(np.atleast_2d(v), cell, pbc=True)[0]


def _dmat(A: np.ndarray, B: np.ndarray, cell) -> np.ndarray:
    d = (A[:, None, :] - B[None, :, :]).reshape(-1, 3)
    return np.linalg.norm(_mic(d, cell), axis=1).reshape(len(A), len(B))


def _min_pair(a: Atoms, I: np.ndarray, J: np.ndarray) -> float:
    if len(I) == 0 or len(J) == 0:
        return float("inf")
    return float(_dmat(a.positions[I], a.positions[J], a.cell).min())


def salt_fragments(cx: Atoms, ads: np.ndarray) -> tuple[list[int], list[list[int]]]:
    """(cation indices, [anion indices, ...]) in complex numbering. The cation is the salt fragment with N and C."""
    sub = cx[ads]
    groups = [[int(ads[i]) for i in g] for g in fragments(sub)]
    sym = np.array(cx.get_chemical_symbols())
    cat = [g for g in groups if "N" in sym[g] and "C" in sym[g]]
    if len(cat) != 1:
        raise ValueError(f"expected one organic cation in the salt, found {len(cat)}")
    return cat[0], [g for g in groups if g is not cat[0]]


def _nearest_pb(a: Atoms, k_idx: int, pb: np.ndarray) -> list[int]:
    v = _mic(a.positions[pb] - a.positions[k_idx], a.cell)
    return [int(pb[i]) for i in np.argsort(np.linalg.norm(v, axis=1))[:SITE_K]]


def _offset(a: Atoms, k_idx: int, nb: list[int]) -> np.ndarray:
    """Anchor position relative to the centroid of the Pb ``nb`` (minimum image)."""
    return -_mic(a.positions[nb] - a.positions[k_idx], a.cell).mean(0)


def _unwrapped(a: Atoms, idx) -> np.ndarray:
    return a.positions[idx[0]] + _mic(a.positions[idx] - a.positions[idx[0]], a.cell)


def match_units(slab: Atoms, cx: Atoms, t: np.ndarray, ads: np.ndarray) -> list[dict]:
    """One-to-one matching of lattice units (covalent fragments: Pb, I, whole FA molecules) of the slab to those of the
    complex without the salt: per formula by centroid (complex + t, minimum image), then atoms within each matched
    pair per element by their offset from the centroid. Atom-level matching alone can split an FA molecule between two
    slab molecules; units cannot."""
    from scipy.optimize import linear_sum_assignment
    rest = np.setdiff1d(np.arange(len(cx)), ads)
    sub = cx[rest]
    su = [np.array(g) for g in fragments(slab)]
    cu = [rest[np.array(g)] for g in fragments(sub)]
    ss, cs = np.array(slab.get_chemical_symbols()), np.array(cx.get_chemical_symbols())
    form = lambda sym: "".join(sorted(sym))
    units = []
    for f in sorted({form(ss[g]) for g in su} | {form(cs[g]) for g in cu}):
        A = [g for g in su if form(ss[g]) == f]
        B = [g for g in cu if form(cs[g]) == f]
        if len(A) != len(B):
            raise ValueError(f"slab and complex (without salt) differ in {f} units: {len(A)} vs {len(B)}")
        ca = np.array([_unwrapped(slab, g).mean(0) for g in A])
        cb = np.array([_unwrapped(cx, g).mean(0) + t for g in B])
        C = _dmat(ca, cb, slab.cell)
        r, c = linear_sum_assignment(C)
        for i, j in zip(r, c):
            ga, gb = A[i], B[j]
            ra = _unwrapped(slab, ga) - ca[i]
            rb = _unwrapped(cx, gb) + t - cb[j]
            amap = {}
            for e in set(ss[ga]):
                ia, ib = np.flatnonzero(ss[ga] == e), np.flatnonzero(cs[gb] == e)
                rr, cc = linear_sum_assignment(np.linalg.norm(ra[ia][:, None] - rb[ib][None], axis=2))
                amap.update({int(ga[ia[x]]): int(gb[ib[y]]) for x, y in zip(rr, cc)})
            units.append({"formula": f, "slab": ga, "cx": np.array([amap[int(k)] for k in ga]), "disp": float(C[i, j])})
    return units


def rebuild(slab: Atoms, cx: Atoms, local_radius: float = LOCAL_RADIUS_A, h_tol: float = H_TOL_A) -> tuple[Atoms, dict]:
    """Rebuilt complex (archived atom order, slab cell) and its acceptance report (see module docstring)."""
    al = align(slab, cx)
    t, ads = al["t"], al["ads"]
    ss = np.array(slab.get_chemical_symbols())
    cs = np.array(cx.get_chemical_symbols())
    units = match_units(slab, cx, t, ads)
    match = np.full(len(slab), -1)
    for u in units:
        match[u["slab"]] = u["cx"]
    for u in units:
        u["d_salt"] = float(_dmat(cx.positions[u["cx"]], cx.positions[ads], cx.cell).min())
        u["jump"] = u["disp"] > SITE_JUMP_A

    # local translation from the on-site Pb/I around the salt
    loc_units = [u for u in units if u["formula"] in LATTICE and not u["jump"] and u["d_salt"] <= local_radius]
    if len(loc_units) < 3:
        raise ValueError(f"fewer than 3 on-site Pb/I within {local_radius} A of the salt; enlarge --local-radius")
    la = np.array([u["slab"][0] for u in loc_units])
    lb = np.array([u["cx"][0] for u in loc_units])
    off = t + _mic(slab.positions[la] - (cx.positions[lb] + t), slab.cell)
    t_loc = np.median(off, axis=0)
    resid = np.linalg.norm(_mic(off - t_loc, slab.cell), axis=1)

    # units on another site than in the slab: part of the salt's local configuration when either end is near the salt
    # (its complex position, or the slab site it left, which the salt may now occupy)
    salt_slab_frame = cx.positions[ads] + t_loc
    for u in units:
        u["d_salt_slab_site"] = float(_dmat(slab.positions[u["slab"]], salt_slab_frame, slab.cell).min())
    reloc_units = [u for u in units if u["jump"] and min(u["d_salt"], u["d_salt_slab_site"]) <= local_radius]
    far_units = [u for u in units if u["jump"] and min(u["d_salt"], u["d_salt_slab_site"]) > local_radius]
    relocated = np.concatenate([u["cx"] for u in reloc_units]).astype(int) if reloc_units else np.array([], int)
    moved = np.concatenate([ads, relocated]).astype(int)

    out = cx.copy()
    out.set_constraint()
    out.set_cell(slab.cell, scale_atoms=False)
    out.pbc = slab.pbc
    out.positions[match] = slab.positions
    out.positions[moved] = cx.positions[moved] + t_loc
    out.wrap()

    # contacts, archived vs rebuilt (same indices: atom order is kept)
    slab_idx = np.setdiff1d(np.arange(len(cx)), ads)
    rest = np.setdiff1d(np.arange(len(cx)), moved)
    lat_cx = rest[np.isin(cs[rest], LATTICE)]
    salt_H = ads[cs[ads] == "H"]
    hl_old, hl_new = _min_pair(cx, salt_H, lat_cx), _min_pair(out, salt_H, lat_cx)
    ss_old, ss_new = _min_pair(cx, moved, rest), _min_pair(out, moved, rest)
    dmin = ss_new

    # sites: anchor keeps its Pb neighbours (by identity) and its offset from them
    cat, anions = salt_fragments(cx, ads)
    pb_cx = slab_idx[cs[slab_idx] == "Pb"]
    anchors = {"cation_N": next(i for i in cat if cs[i] == "N")}
    for n, g in enumerate(anions):
        g = np.array(g)
        anchors[f"anion{n}_{''.join(sorted(set(cs[g])))}"] = int(g[np.argmin(_dmat(cx.positions[g], cx.positions[pb_cx],
                                                                                    cx.cell).min(1))])
    for u in reloc_units:
        k = int(u["cx"][0]) if len(u["cx"]) == 1 else int(next(i for i in u["cx"] if cs[i] == "C"))
        anchors[f"relocated_{u['formula']}_{k}"] = k
    sites = {}
    for name, k in anchors.items():
        nb = _nearest_pb(cx, k, pb_cx)
        shared = len(set(nb) & set(_nearest_pb(out, k, pb_cx)))
        shift = float(np.linalg.norm(_offset(out, k, nb) - _offset(cx, k, nb)))
        sites[name] = {"atom": int(k), "element": str(cs[k]), "shared_Pb": shared, "shift_A": round(shift, 3),
                       "pass": bool(shift <= SITE_SHIFT_MAX_A)}

    stay = np.array([u["disp"] for u in units if not u["jump"]])
    unit_rep = lambda u, key: {key: int(u[key][0]) if key == "slab" else [int(i) for i in u[key]],
                               "formula": u["formula"], "disp_A": round(u["disp"], 2), "d_salt_A": round(u["d_salt"], 2),
                               "d_salt_slab_site_A": round(u["d_salt_slab_site"], 2)}
    spec = spectator_mismatch(slab, out)
    checks = {
        "min_distance_A": {"value": round(dmin, 3), "limit": MIN_DIST_A, "pass": bool(dmin >= MIN_DIST_A)},
        "salt_H_lattice_A": {"archived": round(hl_old, 3), "rebuilt": round(hl_new, 3), "tol": h_tol,
                             "pass": bool(hl_new >= hl_old - h_tol)},
        "salt_slab_A": {"archived": round(ss_old, 3), "rebuilt": round(ss_new, 3), "tol": h_tol,
                        "pass": bool(ss_new >= ss_old - h_tol or ss_new >= NEW_CONTACT_OK_A)},
        "sites": {"pass": all(s["pass"] for s in sites.values()), **sites},
        "lattice_sites": {"n_far_jumps": len(far_units), "far_jumps": [unit_rep(u, "slab") for u in far_units],
                          "relocated": [unit_rep(u, "cx") for u in reloc_units],
                          "on_site_mean_A": round(float(stay.mean()), 3), "on_site_max_A": round(float(stay.max()), 3),
                          "pass": bool(len(far_units) == 0)},
        "spectator_mean_A": {"value": spec["mean_A"], "shell_mean_A": spec["shell"].get("mean_A"),
                             "pass": bool(spec["mean_A"] < 1e-4)},
    }
    report = {
        "n_atoms": len(out), "salt": {e: int((cs[ads] == e).sum()) for e in sorted(set(cs[ads]))},
        "salt_indices": [int(i) for i in ads],
        "translation_global_A": [round(float(x), 3) for x in t],
        "translation_local_A": [round(float(x), 3) for x in t_loc],
        "local_minus_global_A": round(float(np.linalg.norm(_mic(t_loc - t, slab.cell))), 3),
        "local_fit": {"radius_A": local_radius, "n_lattice": int(len(loc_units)),
                      "residual_median_A": round(float(np.median(resid)), 3),
                      "residual_max_A": round(float(resid.max()), 3)},
        "checks": checks,
        "pass": all(c["pass"] for c in checks.values()),
    }
    hard = all(checks[k]["pass"] for k in checks if k not in SOFT)
    report["status"] = "pass" if report["pass"] else ("needs_settle" if hard else "fail")
    return out, report


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slab", type=Path, help="relaxed bare slab (CONTCAR/POSCAR)")
    ap.add_argument("--case", action="append", default=[], metavar="NAME=PATH", help="archived complex; repeatable")
    ap.add_argument("--package", type=Path, help="package folder (outputs/<job>/relax/<source>)")
    ap.add_argument("--cases", default="", help="comma-separated complex names for --package")
    ap.add_argument("--slab-name", default="slab_FAI_Schot")
    ap.add_argument("--source", default="CONTCAR", help="file in each relax folder (CONTCAR or POSCAR)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--local-radius", type=float, default=LOCAL_RADIUS_A)
    ap.add_argument("--h-tol", type=float, default=H_TOL_A)
    a = ap.parse_args(argv)

    cases: dict[str, Path] = {}
    slab_path = a.slab
    if a.package:
        slab_path = slab_path or a.package / "outputs" / a.slab_name / "relax" / a.source
        for n in filter(None, (x.strip() for x in a.cases.split(","))):
            cases[n] = a.package / "outputs" / f"complex_{n}" / "relax" / a.source
    for c in a.case:
        n, p = c.split("=", 1)
        cases[n] = Path(p)
    if slab_path is None or not cases:
        ap.error("give --slab and --case, or --package and --cases")

    slab = read(slab_path, format="vasp")
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {"slab": str(slab_path), "slab_sha256_16": _sha(slab_path), "local_radius_A": a.local_radius,
               "cases": {}}
    for n, p in cases.items():
        d = a.out / n
        d.mkdir(exist_ok=True)
        try:
            out, rep = rebuild(slab, read(p, format="vasp"), a.local_radius, a.h_tol)
        except ValueError as e:
            rep = {"pass": False, "error": str(e)}
        else:
            write(d / "POSCAR", out, format="vasp", direct=True, sort=False)
            rep["poscar_sha256_16"] = _sha(d / "POSCAR")
        rep.update({"name": n, "source": str(p), "source_sha256_16": _sha(p)})
        (d / "rebuild.json").write_text(json.dumps(rep, indent=1))
        rep.setdefault("status", "fail")
        summary["cases"][n] = {"pass": rep["pass"], "status": rep["status"], **({"error": rep["error"]} if "error" in rep else {
            "failed": [k for k, c in rep["checks"].items() if not c["pass"]],
            "local_minus_global_A": rep["local_minus_global_A"],
            "relocated": [r["formula"] for r in rep["checks"]["lattice_sites"]["relocated"]],
            "on_site_mean_A": rep["checks"]["lattice_sites"]["on_site_mean_A"]})}
        print(f"{n:10s} {rep['status']:12s} " + (rep.get("error") or
              " ".join(f"{k}={'ok' if c['pass'] else 'X'}" for k, c in rep["checks"].items())))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0 if all(c["status"] != "fail" for c in summary["cases"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
