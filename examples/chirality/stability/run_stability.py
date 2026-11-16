"""MD-stability check of a fine-tuned MACE model (+ D3(BJ, r2SCAN)) on MBA2PbI4, one unit per array task:
    python -u run_stability.py inputs/<unit> outputs/<unit> runs/<unit> PORT D3_PYTHON

Protocol (unit.json): Langevin NVT at T for nvt_ps, then NVE (velocity Verlet) for nve_ps, in the production cell.
Plan stop conditions (examples/chirality/plan_matrices.json, md stop_conditions), checked every check_every steps:
  * any stereocentre inverts (labels vs the starting structure, pvpassivator.chirality.stereocentres);
  * any Pb not six-coordinate (Pb-I counted to mx_cut_A; thermal stretches reach ~4.1 A, the 7th I stays >= 5.4 A);
  * NH3+ / molecule identity lost (organic bonds at covalent_scale x natural radii);
  * proton transfer: an ammonium H closer than 1.9 A to any I;
  * runaway forces (> 50 eV/A).
After the NVE segment: energy drift (linear fit of the total energy per atom) must be <= 1 meV/atom/ps.
The working trajectory (with momenta) is in runs/; a resubmission continues from its last frame.
Outputs: thermo.csv, traj.extxyz.gz (one frame per check), result.json, run_info.json, marker DONE / INCOMPLETE / FAILED.
DONE holds "PASS ..." or "STOP ..." / "FAIL ..."; the trajectory is a configuration record, never a physical result
on its own.
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


def checks(atoms, ref_top, ref_labels, n_mol, ch, p, np):
    """None if every stop condition is clear, else the reason."""
    try:
        top = ch.build_topology(atoms, mx_cut=p["mx_cut_A"], covalent_scale=p["covalent_scale"])
    except ch.TopologyError as e:
        return f"topology: {e}"
    if len(top.molecules) != n_mol:
        return f"molecule count {len(top.molecules)} != {n_mol}"
    labels = ch.stereocentres(ref_top, atoms)
    if labels != ref_labels:
        return f"stereocentre labels changed: {labels.count('R')}R/{labels.count('S')}S"
    sym = np.array(atoms.get_chemical_symbols())
    hs = np.array([h for m in ref_top.molecules for h in m.nh_h])
    iod = np.where(sym == "I")[0]
    dmin = min(float(atoms.get_distances(int(h), iod, mic=True).min()) for h in hs)
    if dmin < p["proton_transfer_HI_A"]:
        return f"proton transfer: ammonium H...I {dmin:.2f} A < {p['proton_transfer_HI_A']}"
    return None


def main():
    unit_dir, out_dir, work = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    port, d3_python = int(sys.argv[4]), sys.argv[5]
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    for name in ("DONE", "FAILED", "INCOMPLETE"):
        (out_dir / name).unlink(missing_ok=True)
    unit = json.loads((unit_dir / "unit.json").read_text())
    p = unit["protocol"]
    start = time.time()
    deadline = start + p["hours"] * 3600
    traj, thermo = work / "traj.extxyz", out_dir / "thermo.csv"
    info_path = out_dir / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {"unit": unit, "jobs": []}
    job = {"host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "start": time.ctime(start)}
    info["jobs"].append(job)
    d3 = client = None
    marker = ("FAILED", "did not finish")
    try:
        import numpy as np
        from ase import units
        from ase.calculators.mixing import SumCalculator
        from ase.calculators.socketio import SocketIOCalculator
        from ase.io import read, write
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
        from ase.md.verlet import VelocityVerlet
        from mace.calculators import MACECalculator
        import ase
        import mace
        import torch
        import chirality as ch
        job["versions"] = {"ase": ase.__version__, "mace": mace.__version__, "torch": torch.__version__,
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        model = MACECalculator(model_paths=unit["model_path"], device=os.environ.get("STAB_DEVICE", "cuda"),
                               default_dtype=p["dtype"])
        d3 = SocketIOCalculator(port=port)
        client = subprocess.Popen([d3_python, str(HERE / "d3_client.py"), str(unit_dir / "structure.extxyz"), str(port),
                                   str(p["d3_cutoff_bohr"]), str(p["d3_cnthr_bohr"])],
                                  env={**os.environ, "PYTHONNOUSERSITE": "1"},
                                  stdout=open(out_dir / "d3_client.log", "a"), stderr=subprocess.STDOUT)
        calc = SumCalculator([model, d3])
        s0 = read(unit_dir / "structure.extxyz")
        ref_top = ch.build_topology(s0, mx_cut=p["mx_cut_A"], covalent_scale=p["covalent_scale"])
        ref_labels = ch.stereocentres(ref_top, s0)
        n_mol = len(ref_top.molecules)
        dt = p["dt_fs"]
        n_nvt, n_nve = int(round(p["nvt_ps"] * 1000 / dt)), int(round(p["nve_ps"] * 1000 / dt))
        total = n_nvt + n_nve
        if traj.is_file():
            atoms = read(traj, index=-1)
            step = int(atoms.info["step"])
            job["restart_from_step"] = step
        else:
            atoms = s0.copy()
            MaxwellBoltzmannDistribution(atoms, temperature_K=p["T_K"], rng=np.random.default_rng(p["seed"]))
            Stationary(atoms)
            step = 0
        atoms.calc = calc
        if not thermo.is_file():
            thermo.write_text("step,time_ps,phase,temperature_k,epot_ev,etot_ev,max_force_ev_a,s_per_step\n")
        nve_dyn = VelocityVerlet(atoms, dt * units.fs)
        nvt_dyn = Langevin(atoms, dt * units.fs, temperature_K=p["T_K"], friction=p["friction_per_fs"] / units.fs,
                           fixcm=True, rng=np.random.default_rng(p["seed"] + 1 + step))
        marker = None
        while step < total and time.time() < deadline:
            phase = "nvt" if step < n_nvt else "nve"
            dyn = nvt_dyn if phase == "nvt" else nve_dyn
            n = min(p["sample_every"], (n_nvt if phase == "nvt" else total) - step)
            t = time.time()
            dyn.run(n)
            step += n
            per = (time.time() - t) / n
            f = atoms.get_forces()
            fmax = float(np.linalg.norm(f, axis=1).max())
            epot = atoms.get_potential_energy()
            with thermo.open("a") as fh:
                fh.write(f"{step},{step * dt / 1000:.4f},{phase},{atoms.get_temperature():.2f},{epot:.6f},"
                         f"{epot + atoms.get_kinetic_energy():.6f},{fmax:.3f},{per:.4f}\n")
            if fmax > 50.0:
                marker = ("DONE", f"STOP at step {step}: runaway force {fmax:.1f} eV/A")
                break
            if step % p["check_every"] == 0 or step == total:
                snap = atoms.copy()
                snap.calc = None
                snap.info.update({"step": step, "phase": phase})
                write(traj, snap, format="extxyz", append=True)
                why = checks(atoms, ref_top, ref_labels, n_mol, ch, p, np)
                if why:
                    marker = ("DONE", f"STOP at step {step} ({step * dt / 1000:.2f} ps, {phase}): {why}")
                    break
        res = {"unit": unit["name"], "T_K": p["T_K"], "steps_done": step, "steps_total": total}
        if marker is None and step >= total:
            import csv
            rows = [r for r in csv.DictReader(thermo.open()) if r["phase"] == "nve"]
            t_ps = np.array([float(r["time_ps"]) for r in rows])
            e = np.array([float(r["etot_ev"]) for r in rows]) / len(atoms)
            drift = float(np.polyfit(t_ps, e, 1)[0]) * 1000 if len(rows) > 2 else float("nan")
            temps = [float(r["temperature_k"]) for r in csv.DictReader(thermo.open()) if r["phase"] == "nvt"]
            res.update({"nve_drift_meV_atom_ps": round(drift, 4), "nve_frames": len(rows),
                        "nvt_mean_T_K": round(float(np.mean(temps[len(temps) // 5:])), 1),
                        "stop_conditions": "none triggered"})
            ok = abs(drift) <= p["max_drift_meV_atom_ps"]
            res["pass"] = bool(ok)
            marker = ("DONE", ("PASS" if ok else "FAIL") + f" {total * dt / 1000:.1f} ps; NVE drift {drift:.4f} meV/atom/ps")
        elif marker is None:
            marker = ("INCOMPLETE", f"{step * dt / 1000:.2f} of {total * dt / 1000:.1f} ps: resubmit this array task")
        else:
            res.update({"pass": False, "stop": marker[1]})
        (out_dir / "result.json").write_text(json.dumps(res, indent=1) + "\n")
        job["end_step"] = step
    except Exception:  # noqa: BLE001
        marker = ("FAILED", traceback.format_exc())
        raise
    finally:
        if d3 is not None:
            d3.close()
        if client is not None:
            try:
                client.wait(timeout=60)
            except subprocess.TimeoutExpired:
                client.kill()
        if traj.is_file():
            with traj.open("rb") as src, gzip.open(out_dir / "traj.extxyz.gz", "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
        job["wall_s"] = round(time.time() - start, 1)
        info_path.write_text(json.dumps(info, indent=2, default=str) + "\n")
        (out_dir / marker[0]).write_text(marker[1] + "\n")


if __name__ == "__main__":
    main()
