"""Zero-shot test of a MACE model against the D1 DFT minima of package 105 rev1 (local, no labels written).

    D:/MLIP_Work_Folder/envs/maceh/Scripts/python.exe examples/scripts/mlip_vs_dft_chirality.py <package 105 dir> \
        [--model <.model>] [--stride 4] [--out data/cache/chirality_d0_results/mlip_zero_shot.json]

Uses the fixed-cell relax trajectories (r2SCAN+D3(BJ), 520 eV, LREAL=.FALSE.; the free-cell ones run at 676 eV and
are not mixed in). Model = MACE + D3(BJ, r2scan) [torch-dftd], as in local_settle.py.

1. Forces on every ``stride``-th ionic frame of R_fixcell and rac_fixcell: RMSE and max error per element and per
   environment (I axial / I bridging / N-H hydrogen / C-H hydrogen). The frames of one trajectory are correlated
   and the model never saw them, so this is a zero-shot check, not a held-out test of a fitted model.
2. R - rac energy per formula unit at the two DFT fixed-cell minima (DFT: final relax energy).
3. MLIP fixed-cell relax starting from each DFT minimum (FIRE, fmax 0.01 eV/A, <= 600 steps): RMS displacement
   from the DFT minimum and the handedness descriptors against DFT.
Criteria from examples/chirality/plan_matrices.json (``mlip_acceptance_criteria_fixed_before_production``) are
evaluated and reported per item. The model is a candidate, never a label source.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from pvpassivator import chirality as ch  # noqa: E402

N_FU = {"R": 4, "rac": 2}


def environments(atoms, mx_cut: float = 3.8) -> dict[str, np.ndarray]:
    """Boolean masks per chemical environment (from a fresh topology of this cell)."""
    top = ch.build_topology(atoms, mx_cut=mx_cut)
    sym = np.array(atoms.get_chemical_symbols())
    n = len(atoms)
    m = {el: sym == el for el in sorted(set(sym))}
    ax = np.zeros(n, bool)
    br = np.zeros(n, bool)
    for i, r in top.halide_role.items():
        (ax if r == "axial" else br)[i] = True
    nh = np.zeros(n, bool)
    for mol in top.molecules:
        nh[mol.nh_h] = True
    m.update({"I_axial": ax, "I_bridging": br, "H_on_N": nh, "H_on_C": (sym == "H") & ~nh})
    return m


def force_errors(frames, calc) -> dict:
    env = environments(frames[0])
    err = {k: [] for k in env}
    for f in frames:
        dft = f.get_forces()
        a = f.copy()
        a.calc = calc
        d = a.get_forces() - dft
        for k, msk in env.items():
            err[k].append(d[msk])
    out = {}
    for k, v in err.items():
        v = np.concatenate(v)
        out[k] = {"rmse_eV_A": round(float(np.sqrt(np.mean(v ** 2))), 4),
                  "max_atom_err_eV_A": round(float(np.linalg.norm(v, axis=1).max()), 4)}
    return out


def descriptors(atoms) -> dict:
    top = ch.build_topology(atoms)
    d = ch.frame_descriptors(top, atoms)
    return {"stereocentres": top.summary()["stereocentres"],
            "mean": {k: float(d[f"mean_{k}"]) for k in ("ax", "eq", "org")},
            "links": {k: np.sort(d[f"tau_{k}"]).tolist() for k in ("ax", "eq", "org")}}


def main(argv=None) -> int:
    import local_settle as LS
    from ase.constraints import FixAtoms  # noqa: F401  (kept for parity with local_settle)
    from ase.optimize import FIRE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=Path)
    ap.add_argument("--model", type=Path, default=LS.MODEL)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "cache" / "chirality_d0_results" / "mlip_zero_shot.json")
    a = ap.parse_args(argv)
    calc = LS.make_calc(a.model, a.device, "float64")
    crit = json.loads((ROOT / "examples" / "chirality" / "plan_matrices.json").read_text())
    res = {"model": str(a.model), "dispersion": "D3(BJ, r2scan) torch-dftd", "stride": a.stride, "systems": {}}
    E_dft, E_ml, mins = {}, {}, {}
    for key in ("R", "rac"):
        d = a.package / "outputs" / f"{key}_fixcell" / "relax"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traj = read(d / "vasprun.xml.gz", index=":", format="vasp-xml")
        frames = traj[:: a.stride]
        if traj[-1] is not frames[-1]:
            frames.append(traj[-1])
        t0 = time.time()
        fe = force_errors(frames, calc)
        dft_min = read(d / "CONTCAR", format="vasp")
        mins[key] = dft_min
        E_dft[key] = float(traj[-1].get_potential_energy())
        ml = dft_min.copy()
        ml.calc = calc
        E_ml[key] = float(ml.get_potential_energy())
        f0 = float(np.linalg.norm(ml.get_forces(), axis=1).max())
        opt = FIRE(ml, logfile=None)
        conv = bool(opt.run(fmax=0.01, steps=600))
        disp = ml.positions - dft_min.positions
        disp = ch.mic(dft_min, disp)
        dd, dm = descriptors(dft_min), descriptors(ml)
        res["systems"][key] = {
            "n_frames": len(frames), "n_relax_steps_dft": len(traj), "force_errors": fe,
            "mlip_fmax_at_dft_min_eV_A": round(f0, 4),
            "mlip_relax": {"converged": conv, "steps": int(opt.nsteps),
                           "rms_disp_A": round(float(np.sqrt((np.linalg.norm(disp, axis=1) ** 2).mean())), 4),
                           "max_disp_A": round(float(np.linalg.norm(disp, axis=1).max()), 4)},
            "descriptors_dft": {"stereocentres": dd["stereocentres"], "mean": {k: round(v, 5) for k, v in dd["mean"].items()}},
            "descriptors_mlip": {"stereocentres": dm["stereocentres"], "mean": {k: round(v, 5) for k, v in dm["mean"].items()}},
            "max_link_diff": {k: round(float(np.max(np.abs(np.array(dd["links"][k]) - np.array(dm["links"][k])))), 4)
                              for k in ("ax", "eq", "org")},
            "seconds": round(time.time() - t0, 1),
        }
        print(key, json.dumps(res["systems"][key]["force_errors"]["I_axial"]), res["systems"][key]["mlip_relax"], flush=True)
    de_dft = E_dft["R"] / N_FU["R"] - E_dft["rac"] / N_FU["rac"]
    de_ml = E_ml["R"] / N_FU["R"] - E_ml["rac"] / N_FU["rac"]
    res["dE_R_minus_rac_meV_fu"] = {"dft": round(1000 * de_dft, 1), "mlip_at_dft_minima": round(1000 * de_ml, 1)}

    # criteria (plan_matrices.json); tau_ax net is at the 1e-4 noise level, so its sign test is reported but not decisive
    c = {}
    worst = max(v["rmse_eV_A"] for s in res["systems"].values() for v in s["force_errors"].values())
    c["forces_rmse_per_environment_le_0.05"] = worst <= 0.05
    c["forces_none_gt_0.08"] = worst <= 0.08
    c["worst_environment_rmse_eV_A"] = worst
    R = res["systems"]["R"]
    sign_ok, rel_ok = {}, {}
    for k in ("ax", "eq", "org"):
        dv, mv = R["descriptors_dft"]["mean"][k], R["descriptors_mlip"]["mean"][k]
        sign_ok[k] = bool(np.sign(dv) == np.sign(mv))
        rel_ok[k] = bool(abs(mv - dv) <= 0.25 * abs(dv))
    c["net_tau_same_sign_R"] = sign_ok
    c["net_tau_within_25pct_R"] = rel_ok
    c["per_link_within_0.02"] = {key: all(v <= 0.02 for v in s["max_link_diff"].values()) for key, s in res["systems"].items()}
    c["R_rac_energy_within_10meV_fu"] = abs(de_ml - de_dft) <= 0.010
    c["note"] = ("zero-shot on DFT relax trajectories of the same cells; tau_ax net (~2e-4) is below its resolution, so "
                 "its sign/25% items are not decisive; rotor barriers, stress and the held-out set D9 are not tested here")
    res["criteria"] = c
    res["criteria_source"] = crit["mlip_acceptance_criteria_fixed_before_production"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"dE": res["dE_R_minus_rac_meV_fu"], "criteria": c}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
