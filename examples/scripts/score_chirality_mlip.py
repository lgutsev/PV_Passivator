"""Score a MACE model (foundation or fine-tuned) on the chirality fine-tune dataset against the fixed criteria.

    D:/MLIP_Work_Folder/envs/maceh/Scripts/python.exe examples/scripts/score_chirality_mlip.py \
        --model <file.model> --dataset data/training/chirality_v1 --out <scores.json>

The dataset is the one written by export_chirality_training.py. When it was exported with D3 subtracted
(``d3_subtracted``), the model is compared with REF (r2SCAN part) directly; the D3 term is identical on both sides,
so force and relative-energy errors equal those of model + D3 against full DFT.

Reported (criteria from examples/chirality/plan_matrices.json, mlip_acceptance_criteria_fixed_before_production):
* forces per split and per environment (N, C, H_on_N, H_on_C, Pb, I_axial, I_bridging): component RMSE and the
  largest per-atom error; pass = RMSE <= 0.05 eV/A for every environment (the plan's 'none > 0.08' read per environment, owner
  2026-11-16); the largest per-atom error is reported only (judged on test);
* rotor profiles (D3/D4/D5) per system: E(angle) - E(0) for DFT and model, barrier and profile RMSE after removing the
  mean offset; pass = |barrier error| <= max(20 meV, 15 %) and RMSE <= 10 meV;
* R - rac energy per formula unit on the 105 static minima (fixed and free cells); pass = within 10 meV/f.u.
This is a static scoring of DFT geometries. The relaxed-minimum descriptor test (net tau after an MLIP relaxation)
is separate (mlip_vs_dft_chirality.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")   # shared laptop: keep local jobs to a few cores

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mlip_vs_dft_chirality import N_FU, environments  # noqa: E402

ENVS = ("N", "C", "H_on_N", "H_on_C", "Pb", "I_axial", "I_bridging")


def _env_masks(at):
    try:
        m = environments(at, mx_cut=4.3)   # thermal Pb-I stretches reach 4.1 A; the 7th I stays >= 5.4 A
        return {k: m[k] for k in ENVS}, None
    except Exception as e:  # noqa: BLE001  - a heavily displaced frame may not give a clean topology
        sym = np.array(at.get_chemical_symbols())
        m = {k: sym == k for k in ("N", "C", "Pb")}
        m.update({"H_on_N": np.zeros(len(at), bool), "H_on_C": sym == "H", "I_axial": sym == "I",
                  "I_bridging": np.zeros(len(at), bool)})
        return m, str(e)


def predict(frames, calc):
    out = []
    for f in frames:
        a = f.copy()
        a.calc = calc
        out.append((float(a.get_potential_energy()), a.get_forces().copy()))
    return out


def force_table(frames, preds):
    err = {k: [] for k in ENVS}
    notes = []
    for f, (_, F) in zip(frames, preds):
        masks, note = _env_masks(f)
        if note:
            notes.append(f"{f.info.get('job')}: element masks only ({note[:80]})")
        d = F - f.arrays["REF_forces"]
        for k in ENVS:
            if masks[k].any():
                err[k].append(d[masks[k]])
    tab = {}
    for k, v in err.items():
        if not v:
            continue
        v = np.concatenate(v)
        tab[k] = {"rmse_eV_A": round(float(np.sqrt(np.mean(v ** 2))), 4),
                  "max_atom_eV_A": round(float(np.linalg.norm(v, axis=1).max()), 4), "n_atoms": len(v)}
    # "RMSE <= 0.05 and none > 0.08" is read per environment (owner decision 2026-11-16): every environment RMSE
    # <= 0.05 (which already implies none > 0.08). The largest single-atom error is reported, not gated.
    ok = bool(tab) and all(t["rmse_eV_A"] <= 0.05 for t in tab.values())
    return tab, ok, notes


def profiles(frames, preds):
    scans = {}
    for f, (e, _) in zip(frames, preds):
        if f.info.get("block") in ("D3", "D4", "D5"):
            key = f"{f.info['system']}_{f.info['block']}_{f.info.get('group', '')}"
            scans.setdefault(key, []).append((int(f.info["angle_deg"]), f.info["REF_energy"], e, f.info["split"]))
    zero = {}
    for f, (e, _) in zip(frames, preds):   # the 0-degree point is the DFT minimum: the D1 fixed-cell static
        if f.info.get("block") == "D1" and f.info.get("job", "").endswith("fixcell") and f.info.get("step") == "static":
            zero[f.info["system"]] = (f.info["REF_energy"], e)
    out = {}
    for key, pts in scans.items():
        system = key.split("_")[0]
        if not any(p[0] == 0 for p in pts) and system in zero:
            pts.append((0, *zero[system], "train"))
        pts.sort()
        ang = [p[0] for p in pts]
        if 0 not in ang:
            continue
        i0 = ang.index(0)
        dft = 1000 * (np.array([p[1] for p in pts]) - pts[i0][1])
        mod = 1000 * (np.array([p[2] for p in pts]) - pts[i0][2])
        bd, bm = float(dft.max() - dft.min()), float(mod.max() - mod.min())
        r = (dft - dft.mean()) - (mod - mod.mean())
        rmse = float(np.sqrt(np.mean(r ** 2)))
        out[key] = {"angles": ang, "dft_meV": [round(x, 1) for x in dft], "model_meV": [round(x, 1) for x in mod],
                    "barrier_dft_meV": round(bd, 1), "barrier_model_meV": round(bm, 1),
                    "profile_rmse_meV": round(rmse, 1), "splits": sorted({p[3] for p in pts}),
                    "pass": bool(abs(bm - bd) <= max(20.0, 0.15 * bd) and rmse <= 10.0)}
    return out


def r_minus_rac(frames, preds):
    e = {}
    for f, (em, _) in zip(frames, preds):
        if f.info.get("block") == "D1" and f.info.get("step") == "static":
            d3 = float(f.info.get("D3_energy", 0.0))     # report full DFT and model + D3 (same D3 on both sides)
            e[f.info["job"]] = (float(f.info["REF_energy"]) + d3, em + d3, f.info["system"])
    out = {}
    for cell in ("fixcell", "fullcell"):
        r, q = e.get(f"R_{cell}"), e.get(f"rac_{cell}")
        if r and q:
            d = 1000 * (r[0] / N_FU["R"] - q[0] / N_FU["rac"])
            m = 1000 * (r[1] / N_FU["R"] - q[1] / N_FU["rac"])
            out[cell] = {"dft_meV_fu": round(d, 1), "model_meV_fu": round(m, 1), "pass": bool(abs(m - d) <= 10.0)}
    return out


def score(model, dataset, device="cuda"):
    from mace.calculators import MACECalculator
    dataset = Path(dataset)
    rep = json.loads((dataset / "dataset.json").read_text())
    calc = MACECalculator(model_paths=str(model), device=device, default_dtype="float64")
    frames, preds, res = [], [], {"model": str(model), "dataset": str(dataset), "target": rep["target"], "forces": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for split in ("train", "valid", "test"):
            if not rep["files"][split]["frames"]:
                continue
            fr = read(dataset / rep["files"][split]["file"], ":")
            pr = predict(fr, calc)
            tab, ok, notes = force_table(fr, pr)
            res["forces"][split] = {"frames": len(fr), "per_environment": tab, "pass": ok, "notes": notes}
            frames += fr
            preds += pr
    if not rep["d3_subtracted"]:
        raise SystemExit("dataset keeps D3 in the target: score with model + D3 instead (not implemented)")
    res["rotor_profiles"] = profiles(frames, preds)
    res["r_minus_rac"] = r_minus_rac(frames, preds)
    res["summary"] = {"forces_test_pass": res["forces"].get("test", {}).get("pass"),
                      "profiles_pass": {k: v["pass"] for k, v in res["rotor_profiles"].items()},
                      "r_minus_rac_pass": {k: v["pass"] for k, v in res["r_minus_rac"].items()}}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    res = score(a.model, a.dataset, a.device)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["summary"], indent=1))
    for s, v in res["forces"].items():
        print(s, {k: t["rmse_eV_A"] for k, t in v["per_environment"].items()})
    for k, v in res["rotor_profiles"].items():
        print(k, "barrier DFT/model", v["barrier_dft_meV"], v["barrier_model_meV"], "rmse", v["profile_rmse_meV"])
    print("R-rac", res["r_minus_rac"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
