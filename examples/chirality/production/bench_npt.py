"""HPC throughput benchmark and NPT pilot for the chirality production MD (ft2 + D3(BJ, r2SCAN)), one task per unit:
    python -u bench_npt.py inputs/<unit> outputs/<unit> PORT D3_PYTHON

unit.json "kind":
  * "bench": for each system (R 752 atoms, rac 376 atoms) on frames from the 358 K stability runs (LONI 140 / 137):
      - float32 vs float64 MACE: energy, force and stress differences on the saved thermal frames;
      - seconds per MD step for float64 / float32 MACE, each with and without D3 (D3 cost = the difference).
    Output: bench.json. Decides the production precision and gives the cost per ps.
  * "npt": MTK NPT (anisotropic cell, P = 0) at T from a 358 K thermal frame (positions + momenta) for npt_ps,
    with the stability stop conditions (examples/chirality/stability/run_stability.py: checks) every check_every
    steps. Output: npt.csv (T, E, volume, cell lengths and angles, pressure), traj.extxyz.gz, result.json with the
    mean cell over the last third vs the starting cell and vs the DFT free-cell minimum (2x2x1). A stability probe of
    the barostat with ft2, not a thermal-expansion result.
Markers: DONE ("PASS ..." / "STOP ..." for npt, "DONE ..." for bench) or FAILED.
"""
import gzip
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def d3_client(d3_python, structure, port, p, log):
    return subprocess.Popen([d3_python, str(HERE / "d3_client.py"), str(structure), str(port), str(p["d3_cutoff_bohr"]),
                             str(p["d3_cnthr_bohr"]), "stress"], env={**os.environ, "PYTHONNOUSERSITE": "1"},
                            stdout=open(log, "a"), stderr=subprocess.STDOUT)


def cellpar_row(atoms):
    return [float(x) for x in atoms.cell.cellpar()]


def bench(unit, unit_dir, out_dir, port, d3_python, info):
    import numpy as np
    from ase import units
    from ase.calculators.mixing import SumCalculator
    from ase.calculators.socketio import SocketIOCalculator
    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.md.verlet import VelocityVerlet
    from mace.calculators import MACECalculator
    p = unit["protocol"]
    res = {}
    for k, sysname in enumerate(unit["systems"]):
        frames = read(unit_dir / f"frames_{sysname}.extxyz", index=":")
        m = {dt: MACECalculator(model_paths=unit["model_path"], device=os.environ.get("STAB_DEVICE", "cuda"), default_dtype=dt) for dt in ("float64", "float32")}
        props = {}
        for dt, calc in m.items():
            e, f, s = [], [], []
            for a in frames:
                b = a.copy()
                b.calc = calc
                e.append(b.get_potential_energy())
                f.append(b.get_forces().copy())
                s.append(b.get_stress(voigt=True).copy())
            props[dt] = (np.array(e), np.array(f), np.array(s))
        n = len(frames[0])
        dF = props["float32"][1] - props["float64"][1]
        acc = {"frames": len(frames), "max_dE_meV_atom": float(np.abs(props["float32"][0] - props["float64"][0]).max() / n * 1000),
               "max_dF_eV_A": float(np.abs(dF).max()), "rms_dF_eV_A": float(np.sqrt((dF ** 2).mean())),
               "max_dstress_GPa": float(np.abs(props["float32"][2] - props["float64"][2]).max() / units.GPa)}
        d3 = SocketIOCalculator(port=port + k)
        cl = d3_client(d3_python, unit_dir / f"frames_{sysname}.extxyz", port + k, p, out_dir / "d3_client.log")
        timing = {}
        try:
            for dt, calc in m.items():
                for with_d3 in (False, True):
                    a = frames[0].copy()
                    MaxwellBoltzmannDistribution(a, temperature_K=358, rng=np.random.default_rng(1))
                    Stationary(a)
                    a.calc = SumCalculator([calc, d3]) if with_d3 else calc
                    dyn = VelocityVerlet(a, p["dt_fs"] * units.fs)
                    dyn.run(p["bench_warmup"])
                    t = time.time()
                    dyn.run(p["bench_steps"])
                    timing[f"{dt}{'+D3' if with_d3 else ''}"] = round((time.time() - t) / p["bench_steps"], 4)
        finally:
            d3.close()
            try:
                cl.wait(timeout=60)
            except subprocess.TimeoutExpired:
                cl.kill()
        timing["D3_share_float64"] = round(timing["float64+D3"] - timing["float64"], 4)
        ps_per_day = {k2: round(86400 / v * p["dt_fs"] / 1000, 1) for k2, v in timing.items() if "+D3" in k2}
        res[sysname] = {"natoms": n, "float32_vs_float64": acc, "s_per_step": timing, "ps_per_gpu_day": ps_per_day}
        print(sysname, json.dumps(res[sysname]), flush=True)
    (out_dir / "bench.json").write_text(json.dumps(res, indent=1) + "\n")
    return ("DONE", "DONE bench: " + "; ".join(f"{s} f64+D3 {r['s_per_step']['float64+D3']} s/step, f32+D3 "
                                               f"{r['s_per_step']['float32+D3']} s/step" for s, r in res.items()))


def npt(unit, unit_dir, out_dir, work, port, d3_python, info):
    import numpy as np
    from ase import units
    from ase.calculators.mixing import SumCalculator
    from ase.calculators.socketio import SocketIOCalculator
    from ase.io import read, write
    from ase.md.nose_hoover_chain import MTKNPT
    from mace.calculators import MACECalculator
    import chirality as ch
    import run_stability as RS
    p = unit["protocol"]
    s0 = read(unit_dir / "structure.extxyz")
    ref_top = ch.build_topology(s0, mx_cut=p["mx_cut_A"], covalent_scale=p["covalent_scale"])
    ref_labels = ch.stereocentres(ref_top, s0)
    n_mol = len(ref_top.molecules)
    atoms = s0.copy()
    assert atoms.get_momenta().any(), "the NPT pilot starts from a thermal frame with momenta"
    model = MACECalculator(model_paths=unit["model_path"], device=os.environ.get("STAB_DEVICE", "cuda"), default_dtype=p["dtype"])
    d3 = SocketIOCalculator(port=port)
    cl = d3_client(d3_python, unit_dir / "structure.extxyz", port, p, out_dir / "d3_client.log")
    traj, csvp = work / "traj.extxyz", out_dir / "npt.csv"
    try:
        atoms.calc = SumCalculator([model, d3])
        dt = p["dt_fs"] * units.fs
        dyn = MTKNPT(atoms, dt, temperature_K=p["T_K"], pressure_au=0.0, tdamp=p["tdamp_steps"] * dt,
                     pdamp=p["pdamp_steps"] * dt)
        total = int(round(p["npt_ps"] * 1000 / p["dt_fs"]))
        csvp.write_text("step,time_ps,temperature_k,epot_ev,volume_A3,a,b,c,alpha,beta,gamma,pressure_GPa,s_per_step\n")
        step, marker, rows = 0, None, []
        while step < total:
            t = time.time()
            dyn.run(p["sample_every"])
            step += p["sample_every"]
            per = (time.time() - t) / p["sample_every"]
            pres = float(-atoms.get_stress(voigt=True)[:3].mean() / units.GPa)
            row = [step, step * p["dt_fs"] / 1000, atoms.get_temperature(), atoms.get_potential_energy(),
                   atoms.get_volume(), *cellpar_row(atoms), pres, per]
            rows.append(row)
            with csvp.open("a") as fh:
                fh.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            if fmax > 50.0:
                marker = ("DONE", f"STOP at step {step}: runaway force {fmax:.1f} eV/A")
                break
            if step % p["check_every"] == 0:
                snap = atoms.copy()
                snap.calc = None
                snap.info.update({"step": step})
                write(traj, snap, format="extxyz", append=True)
                why = RS.checks(atoms, ref_top, ref_labels, n_mol, ch, p, np)
                if why:
                    marker = ("DONE", f"STOP at step {step} ({step * p['dt_fs'] / 1000:.2f} ps): {why}")
                    break
    finally:
        d3.close()
        try:
            cl.wait(timeout=60)
        except subprocess.TimeoutExpired:
            cl.kill()
    r = np.array(rows, dtype=float)
    last = r[-max(1, len(r) // 3):]
    mean = {"volume_A3": float(last[:, 4].mean()), "abc": last[:, 5:8].mean(0).round(4).tolist(),
            "angles": last[:, 8:11].mean(0).round(3).tolist(), "T_K": float(last[:, 2].mean()),
            "pressure_GPa": float(last[:, 11].mean())}
    v_start, v_dft = s0.get_volume(), unit["dft_freecell_volume_A3"]
    slope = float(np.polyfit(last[:, 1], last[:, 4], 1)[0]) if len(last) > 2 else float("nan")
    res = {"unit": unit["name"], "steps_done": step, "steps_total": total, "mean_last_third": mean,
           "volume_vs_start_pct": round(100 * (mean["volume_A3"] / v_start - 1), 2),
           "volume_vs_dft_freecell_pct": round(100 * (mean["volume_A3"] / v_dft - 1), 2),
           "abc_vs_dft_freecell_pct": [round(100 * (x / y - 1), 2) for x, y in zip(mean["abc"], unit["dft_freecell_abc_A"])],
           "volume_slope_last_third_pct_per_ps": round(100 * slope / v_start, 3),
           "pb_i_shell_A": RS.pb_i_shell(read(traj, index=":"), np) if traj.is_file() else None}
    if marker is None:
        res["pass"] = True
        marker = ("DONE", f"PASS {p['npt_ps']} ps NPT; V {res['volume_vs_dft_freecell_pct']:+.2f}% vs DFT free cell, "
                          f"slope {res['volume_slope_last_third_pct_per_ps']:+.3f} %/ps")
    else:
        res.update({"pass": False, "stop": marker[1]})
    (out_dir / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    if traj.is_file():
        with traj.open("rb") as src, gzip.open(out_dir / "traj.extxyz.gz", "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
    return marker


def main():
    unit_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    port, d3_python = int(sys.argv[3]), sys.argv[4]
    work = Path("runs") / unit_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").is_file():
        print(f"unit already has a verdict, not rerunning: {(out_dir / 'DONE').read_text().strip()}")
        return
    (out_dir / "FAILED").unlink(missing_ok=True)
    unit = json.loads((unit_dir / "unit.json").read_text())
    start = time.time()
    info = {"unit": unit, "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "start": time.ctime(start)}
    marker = ("FAILED", "did not finish")
    try:
        import ase
        import mace
        import torch
        info["versions"] = {"ase": ase.__version__, "mace": mace.__version__, "torch": torch.__version__,
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if unit["kind"] == "bench":
            marker = bench(unit, unit_dir, out_dir, port, d3_python, info)
        else:
            marker = npt(unit, unit_dir, out_dir, work, port, d3_python, info)
    except Exception:  # noqa: BLE001
        marker = ("FAILED", traceback.format_exc())
        raise
    finally:
        info["wall_s"] = round(time.time() - start, 1)
        (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
        (out_dir / marker[0]).write_text(marker[1] + "\n")


if __name__ == "__main__":
    main()
