"""Pre-relaxer MD that only GENERATES configurations for DFT labels D8 (training pool) and D9 (held-out test).

    D:/MLIP_Work_Folder/envs/maceh/Scripts/python.exe examples/scripts/run_chirality_premd.py \
        --pkg105 <exported package 105 dir> --out data/cache/chirality_premd

The foundation MACE-matpes-r2scan + D3(BJ, r2scan) model failed the zero-shot test against the DFT minima, so these
trajectories are NOT physical results: no energy, force or descriptor from them is reported or used as a label. They
only provide thermally displaced geometries for r2SCAN single points.

Runs (fixed cell = the package 105 rev1 fixed-cell DFT minima, Langevin NVT, dt 0.5 fs, friction 0.01 1/fs):
  D8: R and rac at 173 / 293 / 358 K, seed 1, 2 ps equilibration + 3 ps, a frame every 100 fs -> 30 frames per run
  D9: same conditions, independent seed 101, 2 ps equilibration + 1 ps -> 10 frames per run (held out by trajectory)
D3 uses a 40 bohr cutoff (cn threshold 30 bohr) instead of 95: per-step cost drops ~6x and the forces on the R minimum
change by < 0.1 % in sum |F| (benchmarked 2026-11-10); acceptable for configuration generation only.
Every saved frame is checked: a fresh pvpassivator.chirality topology must build (six-coordinate Pb, intact NH3+,
same molecule count) and the stereocentre labels must be unchanged; otherwise the run stops and is marked BROKEN.
Finished runs write ``<run>.DONE`` and are skipped on restart.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pvpassivator import chirality as ch  # noqa: E402

MODEL = Path("D:/MLIP_Foundational_Models/mace/MACE-matpes-r2scan-omat-ft.model")
TEMPS = (173, 293, 358)
RUNS = {"d8": {"seed": 1, "prod_ps": 3.0}, "d9": {"seed": 101, "prod_ps": 1.0}}
EQUIL_PS, DT_FS, FRAME_FS, FRICTION = 2.0, 0.5, 100.0, 0.01


def make_calc(device="cuda"):
    import torch
    from ase.calculators.mixing import SumCalculator
    from mace.calculators import MACECalculator
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    mace = MACECalculator(model_paths=str(MODEL), device=device, default_dtype="float32")
    d3 = TorchDFTD3Calculator(damping="bj", xc="r2scan", device=device, dtype=torch.float32,
                              cutoff=40.0 * units.Bohr, cnthr=30.0 * units.Bohr)
    return SumCalculator([mace, d3])


def check(frame, ref_summary) -> str | None:
    try:
        s = ch.build_topology(frame).summary()
    except ch.TopologyError as e:
        return f"topology: {e}"
    if s["stereocentres"] != ref_summary["stereocentres"] or s["n_molecules"] != ref_summary["n_molecules"]:
        return f"stereocentres/molecules changed: {s['stereocentres']} {s['n_molecules']}"
    return None


def run_one(base, key, T, kind, out: Path, calc) -> dict:
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    spec = RUNS[kind]
    name = f"{key}_{T}K_{kind}_s{spec['seed']}"
    if (out / f"{name}.DONE").exists():
        return json.loads((out / f"{name}.DONE").read_text())
    atoms = base.copy()
    atoms.calc = calc
    ref = ch.build_topology(base).summary()
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(spec["seed"]))
    Stationary(atoms)
    dyn = Langevin(atoms, DT_FS * units.fs, temperature_K=T, friction=FRICTION / units.fs,
                   rng=np.random.default_rng(spec["seed"] + 1))
    n_eq = int(EQUIL_PS * 1000 / DT_FS)
    n_frame = int(FRAME_FS / DT_FS)
    n_prod = int(spec["prod_ps"] * 1000 / DT_FS)
    traj = out / f"{name}.extxyz"
    traj.unlink(missing_ok=True)
    t0 = time.time()
    dyn.run(n_eq)
    status, temps = "ok", []
    for i in range(n_prod // n_frame):
        dyn.run(n_frame)
        fr = atoms.copy()
        fr.calc = None
        fr.wrap()
        temps.append(float(atoms.get_temperature()))
        err = check(fr, ref)
        fr.info.update({"run": name, "system": key, "T_K": T, "time_fs": (n_eq + (i + 1) * n_frame) * DT_FS,
                        "seed": spec["seed"], "split": "train_pool" if kind == "d8" else "heldout_test",
                        "generator": "foundation MACE-matpes-r2scan + D3(BJ,40 bohr), configurations only"})
        if err:
            status = f"BROKEN at frame {i}: {err}"
            break
        write(traj, fr, format="extxyz", append=True)
    rec = {"run": name, "status": status, "frames": len(temps) if status == "ok" else len(temps) - 1,
           "mean_T_K": round(float(np.mean(temps)), 1) if temps else None, "minutes": round((time.time() - t0) / 60, 1)}
    (out / f"{name}.DONE").write_text(json.dumps(rec))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg105", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "cache" / "chirality_premd")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    calc = make_calc(a.device)
    summary = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind in ("d8", "d9"):
            for key in ("R", "rac"):
                base = read(a.pkg105 / "outputs" / f"{key}_fixcell" / "relax" / "CONTCAR", format="vasp")
                for T in TEMPS:
                    rec = run_one(base, key, T, kind, a.out, calc)
                    summary.append(rec)
                    print(json.dumps(rec), flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
