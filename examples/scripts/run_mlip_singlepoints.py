"""Single-point MLIP and D3 energies on archived DFT final structures (CONTCARs).

Run in an environment with mace-torch and torch-dftd, for example:
    python examples/scripts/run_mlip_singlepoints.py --inventory data/benchmark/runs.json \
        --out data/benchmark/energies.jsonl --models mace-mpa-0-medium mace-matpes-r2scan-0 --device cuda

Writes one JSON line per (run, term): {"run_id", "term", "energy_eV", "seconds"}; ``term`` is a model
name or "d3:<damping>-<xc>". Lines already present are skipped, so the script resumes after a crash.
MACE runs in float64: total energies of ~-9000 eV need it for sub-0.1 eV differences. D3 runs in
float32, which halves GPU memory for the 50 A pair list; the D3 term is only ~-40 eV, and float32
reproduces float64 to 1e-5 eV on a 567-atom slab.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pvpassivator.benchmark import load_runs  # noqa: E402

CACHE = Path.home() / ".cache" / "mace"
MODELS = {
    "mace-mp-0-medium": CACHE / "20231203mace128L1_epoch199model",
    "mace-mpa-0-medium": CACHE / "mace-mpa-0-medium.model",
    "mace-matpes-r2scan-0": CACHE / "MACE-matpes-r2scan-omat-ft.model",
}
# D3 matching each DFT profile (VASP IVDW=11 -> zero damping, IVDW=12 -> BJ)
PROFILE_D3 = {"pbe_d3zero": ("zero", "pbe"), "r2scan_d3bj": ("bj", "r2scan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d3-device", default=None,
                    help="device for D3 (default: --device). A CUDA out-of-memory error leaves the context unusable, "
                         "so use cpu when the largest cells do not fit")
    ap.add_argument("--no-d3", action="store_true")
    ap.add_argument("--only-accepted", action="store_true")
    a = ap.parse_args(argv)
    a.d3_device = a.d3_device or a.device

    runs = [r for r in load_runs(a.inventory) if r.evaluable() and (r.accepted or not a.only_accepted)]
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            done.add((d["run_id"], d["term"]))
    import torch
    torch.set_default_dtype(torch.float64)

    def write(rec):
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    structures = {}

    def atoms_of(r):
        if r.run_id not in structures:
            structures[r.run_id] = read(r.contcar, format="vasp")
        return structures[r.run_id].copy()

    if not a.no_d3:
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        for prof, (damp, xc) in PROFILE_D3.items():
            term = f"d3:{damp}-{xc}"
            todo = [r for r in runs if r.profile == prof and (r.run_id, term) not in done]
            if not todo:
                continue
            calc = TorchDFTD3Calculator(damping=damp, xc=xc, device=a.d3_device, dtype=torch.float32)
            for i, r in enumerate(todo):
                at = atoms_of(r)
                at.calc = calc
                t = time.time()
                e = float(at.get_potential_energy())
                dev = a.d3_device
                write({"run_id": r.run_id, "term": term, "energy_eV": e, "seconds": time.time() - t, "device": dev})
                print(f"[{term}] {i+1}/{len(todo)} {r.run_id} {e:.4f}", flush=True)
            del calc
            torch.cuda.empty_cache() if a.device.startswith("cuda") else None

    from mace.calculators import MACECalculator
    for name in a.models:
        todo = [r for r in runs if (r.run_id, name) not in done]
        if not todo:
            continue
        calc = MACECalculator(model_paths=str(MODELS[name]), device=a.device, default_dtype="float64")
        for i, r in enumerate(todo):
            at = atoms_of(r)
            at.calc = calc
            t = time.time()
            e = float(at.get_potential_energy())
            write({"run_id": r.run_id, "term": name, "energy_eV": e, "seconds": time.time() - t})
            print(f"[{name}] {i+1}/{len(todo)} {r.run_id} {e:.4f}", flush=True)
        del calc
        torch.cuda.empty_cache() if a.device.startswith("cuda") else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
