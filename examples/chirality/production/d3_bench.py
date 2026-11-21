"""D3 speed-up check for the chirality production MD: in-process GPU D3 (nvalchemiops Warp, warp_d3.py) against the
torch-dftd socket used so far. Three stages, each in its own env (the slurm script runs them in order):

  python d3_bench.py ref     inputs out      # d3_torchdftd env: torch-dftd references + per-call time
  python d3_bench.py maceref inputs out      # mace_env: ft2 float64 energies/forces (the env of 133/137/140/146)
  python d3_bench.py main    inputs out      # alchemi_check env: MACE parity, Warp D3 variants, MD timing, NVE drift

References (torch-dftd, BJ, r2SCAN set, float64):
  label = torch-dftd defaults (cutoff 95 bohr, CN 40 bohr): how D3 was subtracted from the ft2 training labels;
  md    = cutoff 40 bohr, CN 30 bohr: what the 137/140/146 runs added back.
Warp variants (unit.json "variants"): cutoff in A, smoothing fraction (0 = sharp, as the Fortran reference).
Outputs: ref_<sys>.npz, maceref_<sys>.npz, d3_bench.json; marker DONE / FAILED (main writes them).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GPA = 160.21766208   # eV/A^3 -> GPa


def frames(inp, sysname):
    from ase.io import read
    return read(inp / f"frames_{sysname}.extxyz", index=":")


def efs(atoms_list, calc, repeats=0):
    e, f, s = [], [], []
    for a in atoms_list:
        b = a.copy()
        b.calc = calc
        e.append(b.get_potential_energy())
        f.append(b.get_forces().copy())
        s.append(b.get_stress(voigt=True).copy())
    t = None
    if repeats:
        b = atoms_list[0].copy()
        b.calc = calc
        b.get_forces()
        t0 = time.time()
        for _ in range(repeats):
            b.positions += 1e-4          # force a new evaluation
            b.get_forces()
        t = (time.time() - t0) / repeats
    return np.array(e), np.array(f), np.array(s), t


def stage_ref(inp, out, unit):
    import torch
    from ase import units
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for sysname in unit["systems"]:
        fr = frames(inp, sysname)[: unit["n_ref_frames"]]
        res = {}
        for tag, (cut, cn) in {"label": (95.0, 40.0), "md": (40.0, 30.0)}.items():
            calc = TorchDFTD3Calculator(damping="bj", xc="r2scan", cutoff=cut * units.Bohr, cnthr=cn * units.Bohr,
                                        dtype=torch.float64, device=dev)
            e, f, s, t = efs(fr, calc, repeats=unit["time_repeats"])
            res.update({f"{tag}_E": e, f"{tag}_F": f, f"{tag}_S": s, f"{tag}_t": t})
            print(sysname, tag, "s/call", round(t, 4), flush=True)
        np.savez(out / f"ref_{sysname}.npz", device=dev, **res)


def stage_maceref(inp, out, unit):
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=unit["model_path"], device="cuda", default_dtype="float64")
    for sysname in unit["systems"]:
        e, f, s, _ = efs(frames(inp, sysname)[: unit["n_ref_frames"]], calc)
        np.savez(out / f"maceref_{sysname}.npz", E=e, F=f, S=s)


def err(e, f, s, E, F, S, n):
    dE = (e - E) / n * 1000
    return {"max_dF_eV_A": float(np.abs(f - F).max()), "rms_dF_eV_A": float(np.sqrt(((f - F) ** 2).mean())),
            "offset_meV_atom": float(dE.mean()), "relative_dE_max_meV_atom": float(np.abs(dE - dE.mean()).max()),
            "max_dstress_GPa": float(np.abs(s - S).max() * GPA)}


def md_time(start, calc, steps, warm, dt_fs):
    from ase import units
    from ase.md.verlet import VelocityVerlet
    a = start.copy()
    a.calc = calc
    dyn = VelocityVerlet(a, dt_fs * units.fs)
    dyn.run(warm)
    t0 = time.time()
    dyn.run(steps)
    return round((time.time() - t0) / steps, 4)


def stage_main(inp, out, unit):
    import torch
    from ase import units
    from ase.calculators.mixing import SumCalculator
    from ase.md.verlet import VelocityVerlet
    from mace.calculators import MACECalculator
    from warp_d3 import WarpD3
    p = unit["protocol"]
    res = {"versions": {}}
    import ase, mace, nvalchemiops, warp
    res["versions"] = {"torch": torch.__version__, "mace": mace.__version__, "ase": ase.__version__,
                       "nvalchemiops": getattr(nvalchemiops, "__version__", "?"), "warp": warp.__version__,
                       "gpu": torch.cuda.get_device_name(0)}
    torch.set_default_dtype(torch.float64)
    mace64 = MACECalculator(model_paths=unit["model_path"], device="cuda", default_dtype="float64")
    pfile = inp / "dftd3_parameters.pt"
    for sysname in unit["systems"]:
        fr = frames(inp, sysname)
        n = len(fr[0])
        ref = np.load(out / f"ref_{sysname}.npz")
        mref = np.load(out / f"maceref_{sysname}.npz")
        k = len(mref["E"])
        r = {"natoms": n, "ref_device": str(ref["device"]),
             "torchdftd_s_per_call": {"label_95bohr": float(ref["label_t"]), "md_40bohr": float(ref["md_t"])},
             "md40_vs_label": err(ref["md_E"], ref["md_F"], ref["md_S"], ref["label_E"], ref["label_F"], ref["label_S"], n)}
        e, f, s, _ = efs(fr[:k], mace64)
        r["mace_alchemi_env_vs_mace_env"] = err(e, f, s, mref["E"], mref["F"], mref["S"], n)
        r["warp"] = {}
        for v in unit["variants"]:
            name = f"{v['cutoff_A']:g}A_s{v['smoothing']:g}"
            calc = WarpD3(pfile, cutoff_A=v["cutoff_A"], smoothing_fraction=v["smoothing"])
            e, f, s, t = efs(fr[:k], calc, repeats=p["time_repeats"])
            r["warp"][name] = {"s_per_call": round(t, 5), "vs_label": err(e, f, s, ref["label_E"], ref["label_F"], ref["label_S"], n),
                               "vs_md40": err(e, f, s, ref["md_E"], ref["md_F"], ref["md_S"], n)}
            print(sysname, name, json.dumps(r["warp"][name]), flush=True)
        start = fr[0].copy()
        r["md_s_per_step"] = {"mace_f64_alone": md_time(start, mace64, p["md_steps"], p["md_warm"], p["dt_fs"])}
        for v in unit["variants"]:
            name = f"{v['cutoff_A']:g}A_s{v['smoothing']:g}"
            calc = SumCalculator([mace64, WarpD3(pfile, cutoff_A=v["cutoff_A"], smoothing_fraction=v["smoothing"])])
            r["md_s_per_step"][f"mace_f64+warp_{name}"] = md_time(start, calc, p["md_steps"], p["md_warm"], p["dt_fs"])
        r["ps_per_gpu_day"] = {kk: round(86400 / vv * p["dt_fs"] / 1000, 1) for kk, vv in r["md_s_per_step"].items()}
        res[sysname] = r
        print(sysname, json.dumps(r["md_s_per_step"]), flush=True)
    # NVE drift with the candidate production setting, from the last 140 frame (positions + momenta)
    from ase.io import read
    nv = unit["nve"]
    a = read(inp / "nve_start.extxyz")
    assert a.get_momenta().any()
    a.calc = SumCalculator([mace64, WarpD3(pfile, cutoff_A=nv["cutoff_A"], smoothing_fraction=nv["smoothing"])])
    dyn = VelocityVerlet(a, p["dt_fs"] * units.fs)
    rows, step, total = [], 0, int(round(nv["ps"] * 1000 / p["dt_fs"]))
    t0 = time.time()
    while step < total:
        dyn.run(nv["sample_every"])
        step += nv["sample_every"]
        ep = a.get_potential_energy()
        rows.append((step * p["dt_fs"] / 1000, ep + a.get_kinetic_energy(), a.get_temperature()))
    r_ = np.array(rows)
    with open(out / "nve.csv", "w") as fh:
        fh.write("time_ps,etot_ev,temperature_k\n" + "".join(f"{x:.4f},{y:.6f},{z:.2f}\n" for x, y, z in r_))
    drift = float(np.polyfit(r_[:, 0], r_[:, 1] / len(a), 1)[0]) * 1000
    res["nve"] = {**nv, "natoms": len(a), "drift_meV_atom_ps": round(drift, 5),
                  "etot_std_meV_atom": round(float(r_[:, 1].std() / len(a) * 1000), 4),
                  "mean_T_K": round(float(r_[:, 2].mean()), 1), "s_per_step": round((time.time() - t0) / total, 4)}
    (out / "d3_bench.json").write_text(json.dumps(res, indent=1) + "\n")
    best = unit["nve"]
    name = f"{best['cutoff_A']:g}A_s{best['smoothing']:g}"
    return ("DONE " + "; ".join(f"{s}: f64+warp {name} {res[s]['md_s_per_step'][f'mace_f64+warp_{name}']} s/step "
                                f"(max dF vs label {res[s]['warp'][name]['vs_label']['max_dF_eV_A']:.2e})"
                                for s in unit["systems"]) + f"; NVE drift {res['nve']['drift_meV_atom_ps']} meV/atom/ps")


def main():
    stage, inp, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    unit = json.loads((inp / "unit.json").read_text())
    if stage == "ref":
        stage_ref(inp, out, unit)
    elif stage == "maceref":
        stage_maceref(inp, out, unit)
    else:
        (out / "FAILED").unlink(missing_ok=True)
        try:
            msg = stage_main(inp, out, unit)
            (out / "DONE").write_text(msg + "\n")
        except Exception:
            import traceback
            (out / "FAILED").write_text(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
