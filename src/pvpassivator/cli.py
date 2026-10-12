"""pvpass command line: reproduce archived cases, inspect runs, export NAMD frames."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pvpass")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reproduce", help="recompute archived adsorption energies from raw OSZICARs")
    r.add_argument("config")
    r.add_argument("--cache", default="data/cache")
    r.add_argument("--json", help="write full results JSON here")
    i = sub.add_parser("inspect", help="parse one VASP folder (OutPackLite or full)")
    i.add_argument("folder")
    n = sub.add_parser("namd-export", help="FolderPrep.py replacement with validation + manifest")
    n.add_argument("trajectory")
    n.add_argument("out_dir")
    n.add_argument("--n-frames", type=int, default=2000)
    n.add_argument("--stride", type=int, default=1)
    n.add_argument("--start", type=int)
    n.add_argument("--equil-cutoff", type=int, default=0)
    n.add_argument("--timestep-fs", type=float)
    n.add_argument("--save-interval", type=int, default=1)
    n.add_argument("--source-kind", choices=["DFT", "MLIP"], default="DFT")
    n.add_argument("--overwrite", action="store_true")
    o = sub.add_parser("oxidize", help="replace one surface halide by an iodine oxyanion; write start structures")
    o.add_argument("structure", help="slab file readable by ASE (POSCAR, CONTCAR, extxyz)")
    o.add_argument("--out", required=True, help="output folder (must be empty or new)")
    o.add_argument("--n", type=int, default=3, choices=[1, 2, 3, 4], help="number of O (IO-, IO2-, IO3-, IO4-)")
    o.add_argument("--halide", type=int, help="atom index of the halide; default: first site of --site-class")
    o.add_argument("--face", default="top", choices=["top", "bottom"])
    o.add_argument("--site-class", default="in_plane", choices=["in_plane", "apical", "subsurface", "adatom"])
    o.add_argument("--preset", default="scan", choices=["scan", "O-down", "X-down"],
                   help="scan = full lone-pair/phi/lift scan; O-down / X-down = named bonding limits")
    o.add_argument("--format", default="vasp", help="ASE format for the structure file")
    lb = sub.add_parser("labels", help="reaction energies and k-point check from a finished label-set package")
    lb.add_argument("package", help="package folder with manifest.json, inputs/ and outputs/")
    lb.add_argument("--json", help="write the full result here")
    a = ap.parse_args(argv)

    if a.cmd == "reproduce":
        from .recovery import run_config
        rows = [x.row() for x in run_config(a.config, a.cache)]
        for row in rows:
            e = row["E_ads_eV_per_unit"]
            print(f"{row['case_id']:24s} {('%.3f' % e) if e is not None else '   n/a':>8} "
                  f"sheet={row['sheet_eV']}  {row['verdict']}")
            print(f"{'':24s} {row['equation']}")
        if a.json:
            open(a.json, "w").write(json.dumps(rows, indent=1))
    elif a.cmd == "inspect":
        from .vasp_io import parse_lite
        from dataclasses import asdict
        print(json.dumps(asdict(parse_lite(a.folder)), indent=1))
    elif a.cmd == "namd-export":
        from .namd_adapter import NamdExportConfig, export_frames
        cfg = NamdExportConfig(a.trajectory, a.out_dir, n_frames=a.n_frames, stride=a.stride,
                               start=a.start, equilibration_cutoff=a.equil_cutoff,
                               timestep_fs=a.timestep_fs, save_interval=a.save_interval,
                               source_kind=a.source_kind, overwrite=a.overwrite)
        m = export_frames(cfg)
        print(json.dumps({k: m[k] for k in ("n_source_frames", "issues", "status")}, indent=1))
    elif a.cmd == "oxidize":
        return _oxidize(a)
    elif a.cmd == "labels":
        from .labels import analyze
        r = analyze(a.package)
        for row in r["reactions"]:
            e = row["E_ads_eV"]
            print(f"{row['name']:10s} {('%+.3f eV' % e) if e is not None else 'n/a':>11s}  "
                  f"{'missing: ' + ', '.join(row['missing']) if row['missing'] else ''}")
        kc = r["kpoint_check"]
        if kc:
            for n, v in kc["verdicts"].items():
                print(f"k-check {n}: " + ("not available" if v is None else
                      f"k-error {v['k_error_eV']:+.4f} eV -> {'PASS' if v['pass'] else 'FAIL'} (tol {kc['tolerance_eV']} eV)"))
        if r["settings_problem"]:
            print("WARNING:", r["settings_problem"])
        print(f"{r['n_static_ok']}/{r['n_static']} statics usable; {len(r['problems'])} steps with problems")
        for k, v in r["problems"].items():
            print(f"  {k}: {'; '.join(v)}")
        if a.json:
            open(a.json, "w").write(json.dumps(r, indent=1))
        return 0 if not r["problems"] else 1
    return 0


def _oxidize(a) -> int:
    from pathlib import Path
    from ase.io import write
    from .surface import load_surface, find_sites
    from .oxidation import oxyanion_poses, oxyanion_preset_poses, oxidised_record
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} is not empty; refusing to overwrite", file=sys.stderr)
        return 2
    atoms, rec = load_surface(a.structure, fmt=a.format)
    sites = [s for s in find_sites(atoms, rec, a.face) if s.kind == "exposed_I"]
    if a.halide is not None:
        sites = [s for s in sites if s.atom_index == a.halide]
        if not sites:
            print(f"atom {a.halide} is not an exposed halide on the {a.face} face", file=sys.stderr)
            return 2
    else:
        sites = sorted((s for s in sites if s.halide_class == a.site_class), key=lambda s: s.atom_index)
        if not sites:
            print(f"no {a.site_class} halide on the {a.face} face", file=sys.stderr)
            return 2
    site = sites[0]
    poses = (oxyanion_poses(atoms, rec, site, a.n) if a.preset == "scan"
             else oxyanion_preset_poses(atoms, rec, site, a.n, a.preset))
    out.mkdir(parents=True, exist_ok=True)
    specs = []
    for k, (full, spec) in enumerate(poses):
        d = out / f"pose_{k:02d}"
        d.mkdir()
        write(d / "start.extxyz", full, format="extxyz")
        write(d / "POSCAR", full, format="vasp", direct=True)
        r = oxidised_record(rec, full, spec)
        specs.append({"folder": d.name, **spec.to_dict(), "formal_charge": r.formal_charge,
                      "composition": r.composition})
    (out / "specs.json").write_text(json.dumps({"source": str(a.structure), "site": site.site_id,
                                                "halide_class": site.halide_class, "n_oxygen": a.n,
                                                "preset": a.preset, "poses": specs}, indent=1))
    print(f"{len(poses)} poses for {site.site_id} ({site.halide_class}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
