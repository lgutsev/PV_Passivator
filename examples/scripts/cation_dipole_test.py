"""Does the dipole of the bound passivant cation rank the surface potential shift dV_ads?

    python examples/scripts/cation_dipole_test.py prepare dipoles.extxyz --out <dir> [--route "..."]
    python examples/scripts/cation_dipole_test.py analyze <dir>

prepare: from the last relax frame of every complex (extract_slab_dipoles.py output), cut the organic cation (the
covalent fragment with N and >= 2 C; charge from the H count on N: NH3 -> +1), keep its bound conformation and the
slab frame (x = slab normal), translate N to the origin and write a Gaussian input <job>.gjf. Also writes
cations.json with dV_ads (complex final minus bare-slab final, eV) and two geometry-only proxies:
  tail_A    x(heavy-atom centroid) - x(N), along the slab normal
  F_out_A   sum over F of (x_F - x(N))   (0 without fluorine)
and their Spearman rank against dV_ads.

Where the salt's anion is identifiable (Cl, acetate, TFA; not iodide) a neutral cation + anion pair in the bound
geometry is also written, <job>_pair.gjf: its dipole is origin-independent and carries the anion's share.

analyze: reads <job>.log (and <job>_pair.log), takes the dipole (field-independent basis, Debye) and the coordinates Gaussian used, moves
the origin to N (p_N = p - q r_N; needed for an ion) and ranks p_N,x against dV_ads.
The cation is gas phase in the bound geometry: no slab, no counter-ion, no screening. A good rank says the cation
dipole is a usable cheap descriptor; it is not a prediction of dV_ads.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

DEBYE_PER_EA = 4.80320
ORGANIC = ["C", "H", "N", "O", "F", "S"]
ROUTE = "#p wB97XD/def2TZVP NoSymm SCF=Tight Pop=(MK,Dipole)"
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16}


def spearman(a, b) -> float:
    """Rank correlation with average ranks for ties (many proxies are 0 for non-fluorinated cations)."""
    return float(spearmanr(a, b).statistic)


def cut_cation(atoms):
    s = np.array(atoms.get_chemical_symbols())
    sub = atoms[np.isin(s, ORGANIC)]
    nl = NeighborList(natural_cutoffs(sub), self_interaction=False, bothways=True)
    nl.update(sub)
    _, lab = connected_components(nl.get_connectivity_matrix())
    ss = np.array(sub.get_chemical_symbols())
    for c in np.unique(lab):
        m = lab == c
        if (ss[m] == "C").sum() >= 2 and (ss[m] == "N").sum() == 1:
            idx = np.where(m)[0]
            n_at = next(i for i in idx if ss[i] == "N")
            n_h = sum(ss[k] == "H" for k in nl.get_neighbors(n_at)[0])
            frag = sub[idx]
            return frag, int(np.where(idx == n_at)[0][0]), n_h - 2      # NH3 -> +1, NH2 -> 0
    raise ValueError("no cation fragment found")


def cut_anion(atoms):
    """The salt's own anion when it is identifiable: Cl (the slab has none) or a carboxylate (O-bearing fragment
    without N). Iodide cannot be told apart from lattice iodide: None."""
    s = np.array(atoms.get_chemical_symbols())
    if (s == "Cl").sum() == 1:
        return atoms[s == "Cl"], -1
    sub = atoms[np.isin(s, ORGANIC)]
    nl = NeighborList(natural_cutoffs(sub), self_interaction=False, bothways=True)
    nl.update(sub)
    _, lab = connected_components(nl.get_connectivity_matrix())
    ss = np.array(sub.get_chemical_symbols())
    for c in np.unique(lab):
        m = lab == c
        if (ss[m] == "O").sum() == 2 and (ss[m] == "N").sum() == 0 and (ss[m] == "C").sum() == 2:
            return sub[np.where(m)[0]], -1
    return None, 0


def gjf(job: str, route: str, title: str, q: int, frag) -> str:
    coords = "\n".join(f"{e:2s} {x:14.8f} {y:14.8f} {z:14.8f}"
                       for e, (x, y, z) in zip(frag.get_chemical_symbols(), frag.positions))
    return f"%chk={job}.chk\n{route}\n\n{title}\n\n{q} 1\n{coords}\n\n"


def prepare(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = [f for f in read(a.extxyz, index=":") if f.info["step"] == "relax"]
    last = {}
    for f in frames:
        last[f.info["job"]] = f
    slab_dv = last[a.slab].info["dV_eV"]
    rows = {}
    for job, at in sorted(last.items()):
        if job == a.slab:
            continue
        frag, n_idx, q = cut_cation(at)
        origin = frag.positions[n_idx].copy()
        frag.positions -= origin
        nelec = sum(Z[e] for e in frag.get_chemical_symbols()) - q
        if nelec % 2:
            raise ValueError(f"{job}: odd electron count {nelec} for charge {q}")
        sym = np.array(frag.get_chemical_symbols())
        heavy = sym != "H"
        rows[job] = {"formula": frag.get_chemical_formula(), "charge": q, "n_index": n_idx,
                     "dV_ads_eV": at.info["dV_eV"] - slab_dv,
                     "tail_A": float(frag.positions[heavy, 0].mean()),
                     "F_out_A": float(frag.positions[sym == "F", 0].sum())}
        write(out / f"{job}.xyz", frag)
        (out / f"{job}.gjf").write_text(gjf(job, a.route, f"{job} bound cation, slab frame, N at origin", q, frag))
        anion, qa = cut_anion(at)
        if anion is not None:
            anion.positions -= origin
            pair = frag + anion
            rows[job]["pair"] = {"formula": pair.get_chemical_formula(), "charge": q + qa}
            write(out / f"{job}_pair.xyz", pair)
            (out / f"{job}_pair.gjf").write_text(gjf(f"{job}_pair", a.route,
                                                     f"{job} bound cation + anion, slab frame", q + qa, pair))
    jobs = sorted(rows)
    dv = [rows[j]["dV_ads_eV"] for j in jobs]
    summary = {"route": a.route, "slab": a.slab, "frame": "x = slab normal; passivated face is low-x (tail -> -x)",
               "cations": rows,
               "spearman": {k: spearman(dv, [rows[j][k] for j in jobs]) for k in ("tail_A", "F_out_A")}}
    (out / "cations.json").write_text(json.dumps(summary, indent=1))
    for j in jobs:
        r = rows[j]
        print(f"{j:15s} {r['formula']:12s} q={r['charge']:+d}  dV_ads {r['dV_ads_eV']:+.3f} eV  "
              f"tail {r['tail_A']:+.2f} A  F_out {r['F_out_A']:+.2f} A")
    print("Spearman vs dV_ads:", summary["spearman"])


def read_log(path: Path):
    txt = path.read_text(errors="replace")
    if "Normal termination" not in txt:
        raise ValueError("not a normal termination")
    blocks = re.findall(r"(?:Input|Standard) orientation:.*?-{5,}\n.*?-{5,}\n(.*?)\n\s*-{5,}", txt, re.S)
    xyz = np.array([[float(v) for v in ln.split()[3:6]] for ln in blocks[-1].strip().splitlines()])
    m = re.findall(r"Dipole moment \(field-independent basis, Debye\):\s*\n\s*X=\s*(\S+)\s+Y=\s*(\S+)\s+Z=\s*(\S+)", txt)
    return xyz, np.array([float(v) for v in m[-1]])


def analyze(a) -> None:
    d = Path(a.dir)
    summary = json.loads((d / "cations.json").read_text())
    got = {}
    for job, r in summary["cations"].items():
        log = d / f"{job}.log"
        if not log.exists():
            print(f"{job}: no log")
            continue
        try:
            xyz, p_debye = read_log(log)
        except ValueError as exc:
            print(f"{job}: {exc}")
            continue
        p_n = p_debye / DEBYE_PER_EA - r["charge"] * xyz[r["n_index"]]       # e*A about N
        got[job] = {"p_N_eA": p_n.tolist(), "p_N_x_eA": float(p_n[0]), "dV_ads_eV": r["dV_ads_eV"]}
        print(f"{job:15s} p_N,x {p_n[0]:+7.3f} eA ({p_n[0]*DEBYE_PER_EA:+6.2f} D)  dV_ads {r['dV_ads_eV']:+.3f} eV")
        pl = d / f"{job}_pair.log"
        if "pair" in r and pl.exists():
            try:
                _, pp = read_log(pl)                              # neutral pair: origin-independent
                got[job]["pair_x_eA"] = float(pp[0] / DEBYE_PER_EA)
                print(f"{'':15s} pair p_x {pp[0] / DEBYE_PER_EA:+7.3f} eA")
            except ValueError as exc:
                print(f"{job}_pair: {exc}")
    if len(got) >= 3:
        jobs = sorted(got)
        rho = spearman([got[j]["dV_ads_eV"] for j in jobs], [got[j]["p_N_x_eA"] for j in jobs])
        print(f"Spearman(p_N,x, dV_ads) over {len(jobs)} cations: {rho:+.2f}")
        summary["dipoles"], summary["spearman"]["p_N_x"] = got, rho
        pj = [j for j in jobs if "pair_x_eA" in got[j]]
        if len(pj) >= 3:
            summary["spearman"]["pair_x"] = spearman([got[j]["dV_ads_eV"] for j in pj], [got[j]["pair_x_eA"] for j in pj])
            print(f"Spearman(pair p_x, dV_ads) over {len(pj)} salts: {summary['spearman']['pair_x']:+.2f}")
        (d / "cations.json").write_text(json.dumps(summary, indent=1))


SLURM = """#!/bin/bash
# Package {pkg}: Gaussian 09 single points of bound passivant cations and cation+anion pairs (dipoles only).
#
#   sbatch run_cation_dipoles.slurm        # from the package directory (array 0-{last} is in the header)
#
# Task i runs line i+1 of lists/jobs.txt. DONE = outputs/<name>/DONE, written on 1 Normal termination;
# resubmitting skips DONE tasks. No optimisation: the geometry is the bound one from the slab.

#SBATCH --job-name={nn}-cation-dip
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --time=02:00:00
#SBATCH --array=0-{last}
#SBATCH --output=logs/cation-dip-%A_%a.out

# module is a shell function that trips -u; load first, tighten after.
set -eo pipefail
module load gaussian/g09-d01
set -u

PKG="${{SLURM_SUBMIT_DIR:?submit with sbatch from the package directory}}"
cd "$PKG"
[ -f lists/jobs.txt ] || {{ echo "lists/jobs.txt missing: submit from the package directory" >&2; exit 1; }}
name=$(sed -n "$(( ${{SLURM_ARRAY_TASK_ID:?array job only}} + 1 ))p" lists/jobs.txt)
[ -n "$name" ] || {{ echo "no job for task ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; }}
out="$PKG/outputs/$name"
mkdir -p "$out"
if [ -f "$out/DONE" ]; then echo "$name already DONE"; exit 0; fi

export GAUSS_SCRDIR="/work/${{USER}}/g09-scr/cation_dip_${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$GAUSS_SCRDIR"
trap 'rm -rf "$GAUSS_SCRDIR"' EXIT
echo "task ${{SLURM_ARRAY_TASK_ID}}: $name on $(hostname)"

cd "$out"
cp "$PKG/inputs/$name.gjf" .
set +e
g09 < "$name.gjf" > "$name.log"
st=$?
# errexit stays off from here: bookkeeping only, and grep -c exits 1 when it counts 0
echo "g09 exit $st"
grep -h "Error termination" "$name.log" | sed 's/^/ERROR: /' || true
have=$(grep -c "Normal termination" "$name.log" 2>/dev/null || true)
echo "normal terminations: ${{have:-0}} of 1"
[ "${{have:-0}}" -ge 1 ] && date > DONE
rm -f ./*.chk
exit 0
"""


def package(a) -> None:
    """Stage the Gaussian inputs as a LONI package (QB4, g09 D.01) in the desk's layout."""
    import hashlib
    src, nn = Path(a.dir), a.nn
    pkg = f"{nn}_g09_pvpass_cation_dipoles"
    dest = Path(a.dest) / pkg
    if dest.exists():
        raise SystemExit(f"{dest} exists: stage a revision instead of overwriting")
    summary = json.loads((src / "cations.json").read_text())
    names = []
    for job, r in sorted(summary["cations"].items()):
        names.append(job)
        if "pair" in r:
            names.append(f"{job}_pair")
    for d in ("inputs", "lists", "logs", "outputs", "geometries"):
        (dest / d).mkdir(parents=True)
    for n in names:
        body = (src / f"{n}.gjf").read_text().split("\n", 1)[1]           # drop the local %chk line
        (dest / "inputs" / f"{n}.gjf").write_bytes(
            f"%chk={n}.chk\n%nprocshared={a.cores}\n%mem={a.mem}\n{body}".encode())
        (dest / "geometries" / f"{n}.xyz").write_bytes((src / f"{n}.xyz").read_bytes())
    (dest / "lists" / "jobs.txt").write_bytes(("\n".join(names) + "\n").encode())
    (dest / "logs" / ".gitkeep").write_bytes(b"")
    (dest / "outputs" / ".gitkeep").write_bytes(b"")
    (dest / "export.list").write_bytes(b"outputs/*\nlogs/*\n")
    (dest / "run_cation_dipoles.slurm").write_bytes(
        SLURM.format(pkg=pkg, nn=nn, last=len(names) - 1, cores=a.cores).encode())
    rows = "\n".join(f"| {i} | `{n}` | {'0 (neutral pair)' if n.endswith('_pair') else '+1'} |"
                     for i, n in enumerate(names))
    (dest / "README.md").write_bytes(f"""# {pkg}

PV_Passivator: dipoles of the bound passivant cations (and neutral cation+anion pairs where the anion is
identifiable), cut from the final r2SCAN+D3(BJ) complexes of package 53. Single points only, Gaussian 09 D.01.
Generated by `examples/scripts/cation_dipole_test.py package`. **Nothing has been submitted.**

Route: `{summary['route']}`; `%nprocshared={a.cores}`, `%mem={a.mem}`. Frame: x = slab normal, ammonium N at
the origin (NoSymm keeps it). The geometry is the bound one on purpose: no optimisation.

| Task | Job | Charge |
|---|---|---|
{rows}

## Run
```bash
cd /work/lgutsev/loni_smoke_tests/{pkg} && mkdir -p logs && sbatch run_cation_dipoles.slurm
```
QB4 `single` partition, {a.cores} cores, 2 h limit per task; each single point should take minutes
(13-29 atoms). Total well under 1 node-hour.

## Done / pass
`outputs/<name>/DONE` after 1 Normal termination. The .chk is deleted (not needed).

## Bring back
`outputs/*` and `logs/*`. On the desktop: copy `outputs/<name>/<name>.log` next to the inputs in
`data/cache/pet_dipole/cation_gaussian/` and run `cation_dipole_test.py analyze` on that folder.
""".encode())
    files = sorted(p for p in dest.rglob("*") if p.is_file())
    manifest = {"package": pkg, "number_status": "assigned by the desk",
                "generator": "examples/scripts/cation_dipole_test.py package",
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files": {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (dest / "manifest.json").write_bytes(json.dumps(manifest, indent=1).encode())
    print(f"{len(names)} tasks -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("extxyz", type=Path)
    p.add_argument("--out", required=True)
    p.add_argument("--slab", default="slab_FAI_Schot")
    p.add_argument("--route", default=ROUTE)
    q = sp.add_parser("analyze")
    q.add_argument("dir")
    k = sp.add_parser("package", help="stage the prepared inputs as a LONI package (QB4, g09)")
    k.add_argument("dir", help="the prepare output folder")
    k.add_argument("--nn", required=True, help="package number from the dispatch desk")
    k.add_argument("--dest", required=True)
    k.add_argument("--cores", type=int, default=16)
    k.add_argument("--mem", default="24GB")
    a = ap.parse_args()
    {"prepare": prepare, "analyze": analyze, "package": package}[a.cmd](a)


if __name__ == "__main__":
    main()
