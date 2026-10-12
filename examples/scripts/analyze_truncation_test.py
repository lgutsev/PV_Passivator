"""Read a finished truncation test (prepare_truncation_test.py package) and check it against the expectations.

    python examples/scripts/analyze_truncation_test.py <package dir> [--json out.json]

Per job: energy_sigma0, HOMO/LUMO from the last eigenvalue block of OUTCAR, and from LOCPOT.gz the planar-averaged
potential in the vacuum on each side (mean, and the slope of a straight-line fit). Checks:
  1. energy spread over the cell lengths, per method and charge (trunc charged should be flat, plain charged drifts)
  2. neutral energies: trunc vs dip at each length
  3. vacuum slope of charged trunc runs vs the charged-sheet value |q| e / (2 eps0 A)
  4. Delta-SCF vs eigenvalues (trunc): IP = E(+1) - E(0) vs vacuum level - HOMO(0); EA = E(0) - E(-1) vs vacuum - LUMO(0)
Missing or unfinished jobs are reported, never filled in.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pvpassivator.vasp_io import parse_run  # noqa: E402

E_TOL = 0.02          # eV, accepted energy spread over the cell lengths for a converged method
SHEET_EV_PER_A = 1.602176634e-19 / (2 * 8.8541878128e-12) * 1e10   # e/(2 eps0) in eV/A times A^2; divide by area
VAC_MARGIN_A = 6.0    # planar average taken this far from the outermost atom
EDGE_A = 2.0          # and this far from the cell boundary (the dipole-correction step sits there)


def eigen_edges(outcar: Path) -> tuple[float | None, float | None]:
    """HOMO, LUMO over all k-points and spins from the last eigenvalue block (after the last E-fermi line)."""
    lines = outcar.read_text(errors="replace").splitlines()
    start = max((i for i, ln in enumerate(lines) if "E-fermi" in ln), default=None)
    if start is None:
        return None, None
    ispin2 = any("spin component 2" in ln for ln in lines[start:])
    full = 1.0 if ispin2 else 2.0
    occ, emp = [], []
    for ln in lines[start + 1:]:
        if ln.strip().startswith("----") or "soft charge-density" in ln:
            break
        m = re.match(r"^\s*\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", ln)
        if m:
            e, f = float(m.group(1)), float(m.group(2))
            (occ if f > 0.5 * full else emp).append(e)
    return (max(occ) if occ else None), (min(emp) if emp else None)


def planar_average(locpot_gz: Path, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(coordinate along axis in A, planar average in eV, atom coordinates along axis) from a gzipped LOCPOT."""
    with gzip.open(locpot_gz, "rt") as fh:
        txt = fh.read().split("\n")
    scale = float(txt[1].split()[0])
    cell = np.array([[float(x) for x in txt[i].split()] for i in (2, 3, 4)]) * scale
    counts = [int(x) for x in txt[6].split()]
    n = sum(counts)
    direct = txt[7].strip()[0].lower() == "d"
    pos = np.array([[float(x) for x in txt[8 + i].split()[:3]] for i in range(n)])
    coord = (pos[:, axis] * np.linalg.norm(cell[axis])) if direct else pos[:, axis] * scale
    i = 8 + n
    while not txt[i].strip():
        i += 1
    ng = [int(x) for x in txt[i].split()]
    need = ng[0] * ng[1] * ng[2]
    vals = []
    for ln in txt[i + 1:]:
        vals += ln.split()
        if len(vals) >= need:
            break
    v = np.array(vals[:need], float).reshape(ng[2], ng[1], ng[0])       # x fastest
    prof = v.mean(axis=tuple(k for k in (0, 1, 2) if k != 2 - axis))
    L = np.linalg.norm(cell[axis])
    return np.arange(ng[axis]) * L / ng[axis], prof, coord


def vacuum_sides(x, prof, atoms_x, L) -> dict:
    lo, hi = atoms_x.min() - VAC_MARGIN_A, atoms_x.max() + VAC_MARGIN_A
    out = {}
    for side, m in (("low", (x > EDGE_A) & (x < lo)), ("high", (x > hi) & (x < L - EDGE_A))):
        if m.sum() < 3:
            out[side] = None
            continue
        slope, icpt = np.polyfit(x[m], prof[m], 1)
        out[side] = {"mean_eV": float(prof[m].mean()), "slope_eV_per_A": float(slope),
                     "at_slab_edge_eV": float(slope * (lo if side == "low" else hi) + icpt),
                     "n_points": int(m.sum())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    root = Path(a.package)
    m = json.loads((root / "manifest.json").read_text())
    ax = m["normal_axis"]
    res, missing = {}, []
    for name, j in m["jobs"].items():
        o = root / "outputs" / name / "static"
        if not (o / "DONE").exists():
            missing.append(name)
            continue
        run = parse_run(o)
        probs = run.acceptance(require_relax=False)
        homo, lumo = eigen_edges(o / "OUTCAR")
        r = {**j, "energy_eV": run.energy_sigma0, "problems": probs, "homo_eV": homo, "lumo_eV": lumo}
        if (o / "LOCPOT.gz").exists():
            x, prof, ax_x = planar_average(o / "LOCPOT.gz", ax)
            r["vacuum"] = vacuum_sides(x, prof, ax_x, j["length_A"])
        res[name] = r

    report = {"missing": missing, "jobs": res, "checks": []}

    def chk(name, ok, detail):
        report["checks"].append({"check": name, "pass": ok, "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL' if ok is False else 'INFO'}] {name}: {detail}")

    # 1. spread with vacuum
    for method in ("dip", "plain", "trunc"):
        for q in m["charges"]:
            rows = [(r["length_A"], r["energy_eV"]) for r in res.values() if r["method"] == method and r["charge"] == q]
            if len(rows) < 2:
                continue
            rows.sort()
            e = np.array([r[1] for r in rows])
            spread = float(e.max() - e.min())
            slope = float(np.polyfit([r[0] for r in rows], e, 1)[0])
            expect_flat = method == "trunc" or q == 0
            chk(f"{method} q={q:+d} energy vs vacuum", (spread < E_TOL) if expect_flat else None,
                f"spread {spread * 1000:.1f} meV, slope {slope * 1000:+.2f} meV/A over "
                + ", ".join(f"{L:g}: {E:.4f}" for L, E in rows))
    # 2. neutral, method vs method
    for L in m["lengths_A"]:
        d, t = (res.get(f"{k}_q0_L{L:g}") for k in ("dip", "trunc"))
        if d and t:
            diff = t["energy_eV"] - d["energy_eV"]
            chk(f"neutral trunc - dip at L={L:g}", abs(diff) < E_TOL, f"{diff * 1000:+.1f} meV")
    td = next((r for n, r in res.items() if r["method"] == "tdip"), None)
    if td:
        ref = res.get(f"trunc_q0_L{td['length_A']:g}")
        chk("trunc + dipole correction together", None,
            f"E = {td['energy_eV']:.4f}" + (f", {1000 * (td['energy_eV'] - ref['energy_eV']):+.1f} meV vs trunc alone"
                                             if ref else ""))
    # 3. vacuum field of charged truncated slabs
    for n, r in res.items():
        if r["method"] == "trunc" and r["charge"] != 0 and r.get("vacuum"):
            want = abs(r["charge"]) * SHEET_EV_PER_A / r["inplane_area_A2"]
            got = [abs(s["slope_eV_per_A"]) for s in r["vacuum"].values() if s]
            if got:
                dev = max(abs(g - want) / want for g in got)
                chk(f"{n} vacuum field", dev < 0.1, f"|slope| {', '.join(f'{g:.4f}' for g in got)} eV/A, "
                                                   f"charged sheet {want:.4f} eV/A")
    # 4. Delta-SCF vs eigenvalues, truncated runs
    for L in m["lengths_A"]:
        n0, p, mi = (res.get(f"trunc_{t}_L{L:g}") for t in ("q0", "qp1", "qm1"))
        if not n0:
            continue
        vac = [s["mean_eV"] for s in (n0.get("vacuum") or {}).values() if s]
        vref = float(np.mean(vac)) if vac else 0.0
        parts = [f"vacuum level {vref:+.3f} eV (sides {', '.join(f'{v:+.3f}' for v in vac)})"]
        if p and n0["homo_eV"] is not None:
            parts.append(f"IP dSCF {p['energy_eV'] - n0['energy_eV']:.3f} vs vac-HOMO {vref - n0['homo_eV']:.3f}")
        if mi and n0["lumo_eV"] is not None:
            parts.append(f"EA dSCF {n0['energy_eV'] - mi['energy_eV']:.3f} vs vac-LUMO {vref - n0['lumo_eV']:.3f}")
        chk(f"trunc L={L:g} dSCF vs eigenvalues", None, "; ".join(parts))
    for L in m["lengths_A"]:
        d = res.get(f"dip_q0_L{L:g}")
        if d and d.get("vacuum") and d["homo_eV"] is not None:
            chk(f"dip L={L:g} neutral levels", None, ", ".join(
                f"{s}: vac-HOMO {v['mean_eV'] - d['homo_eV']:.3f}" for s, v in d["vacuum"].items() if v))
    if missing:
        print(f"missing/unfinished: {', '.join(missing)}")
    bad = {n: r["problems"] for n, r in res.items() if r["problems"]}
    if bad:
        print(f"jobs with acceptance problems: {bad}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
