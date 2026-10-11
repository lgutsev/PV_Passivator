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
    return 0


if __name__ == "__main__":
    sys.exit(main())
