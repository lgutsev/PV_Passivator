"""Native MACE vs the ALCHEMI backend on real slab + passivant structures (run in the ALCHEMI environment).

    python examples/scripts/alchemi_benchmark.py singlepoint POOL --model M --out DIR [--dtype float64] [--max-batch 4]
    python examples/scripts/alchemi_benchmark.py steptime    POOL --model M --out DIR [--dtypes float64,float32]
    python examples/scripts/alchemi_benchmark.py relax       POOL --model M --out DIR [--names a,b,...] [--batch 2]

POOL is an ``adsearch_v1_pool.py`` output folder: real structures on one slab, with atom ids, roles and the
native workflow's mobile mask.  The model is the MACE checkpoint alone on both sides (no dispersion term), so the
comparison is the same function through two interfaces.

  singlepoint  identical coordinates: native calculator one structure at a time, ALCHEMI batch 1, ALCHEMI multi-
               system batches; then batch independence (reordered, repeated, different companions) and the slab
               with periodicity switched off along the normal.
  steptime     seconds per energy/force evaluation and peak VRAM against batch size and precision, warm, repeated.
               An out-of-memory batch is recorded as such, not skipped.
  relax        same starts, same frozen atoms, same force criterion over the same free atoms.  Native = ASE FIRE;
               ALCHEMI = FIRE2 (the algorithms differ, so batch-1 vs batched ALCHEMI is reported separately from
               native vs ALCHEMI).  Cold setup and warm run are timed separately; every input gets a record.

Nothing here says the model is right for the chemistry; that is the DFT audit's job.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from pvpassivator import alchemi_backend as B  # noqa: E402

TOL = {"energy_abs_eV": 1e-6, "force_abs_eV_A": 1e-6,
       "note": "float64, set before the GPU runs from the CPU toy check (1e-13 eV, 7e-12 eV/A); float32 is reported, "
               "not gated: its error is compared with the 0.10 eV energy resolution the search needs"}


def load_panel(pool: Path, names=None, initial=False):
    from ase.io import read
    summ = json.loads((pool / "pool.json").read_text()) if (pool / "pool.json").exists() else None
    files = sorted(p for p in (pool / "structures").glob("*.extxyz") if not p.stem.endswith("_initial")
                   and p.stem not in ("ref_molecule",))
    mob_sub = None
    out = []
    for f in files:
        a = read(f)
        if "mobile" in a.arrays and "mol_id" in a.arrays and (a.arrays["mol_id"] > 0).any() and mob_sub is None:
            sub = a.arrays["mol_id"] == 0
            mob_sub = set(a.arrays["atom_id"][sub & (a.arrays["mobile"] > 0)].tolist())
        out.append((f.stem, a))
    panel = []
    for name, a in out:
        if names and name not in names:
            continue
        if initial:
            fi = pool / "structures" / f"{name}_initial.extxyz"
            if not fi.exists():
                continue
            a = read(fi)
        ids = a.arrays["atom_id"] if "atom_id" in a.arrays else np.arange(len(a))
        mol = a.arrays["mol_id"] > 0 if "mol_id" in a.arrays else np.zeros(len(a), bool)
        mobile = mol | np.isin(ids, list(mob_sub or []))
        for k in ("model_forces", "mobile"):
            if k in a.arrays:
                del a.arrays[k]
        a.calc = None
        panel.append({"name": name, "atoms": a, "frozen": ~mobile})
    return panel, summ


def gpu_info():
    import torch
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    p = torch.cuda.get_device_properties(0)
    return {"device": p.name, "vram_MiB": round(p.total_memory / 2 ** 20), "cuda": torch.version.cuda}


def native_calc(model, device, dtype):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=str(model), device=device, default_dtype=dtype)


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _try(fn):
    import torch
    try:
        return fn(), None
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"[:200]
        try:                                      # a CUDA out-of-memory error can leave the context unusable
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            msg += " [CUDA context lost]"
        return None, ("out_of_memory" if "out of memory" in msg.lower() else "error") + ": " + msg


def cmd_singlepoint(a):
    panel, _ = load_panel(Path(a.pool), a.names)
    atoms = [p["atoms"] for p in panel]
    names = [p["name"] for p in panel]
    rep = {"schema": "alchemi_singlepoint/1", "versions": B.versions(), "gpu": gpu_info(), "dtype": a.dtype,
           "tolerances": TOL, "structures": {n: {"n_atoms": len(x), "formula": x.get_chemical_formula()}
                                             for n, x in zip(names, atoms)}}
    calc = native_calc(a.model, a.device, a.dtype)
    nat = []
    for x in atoms:
        c = x.copy()
        c.calc = calc
        nat.append((float(c.get_potential_energy()), c.get_forces().copy()))
    c = atoms[0].copy()
    c.calc = calc
    e_rep = float(c.get_potential_energy())
    rep["native_repeatability"] = {"dE_eV": abs(e_rep - nat[0][0])}
    del calc
    model = B.load_mace(a.model, a.device, a.dtype)

    def compare(res, ref):
        de = [r["energy"] - e for r, (e, f) in zip(res, ref)]
        df = [float(np.abs(r["forces"] - f).max()) for r, (e, f) in zip(res, ref)]
        rms = [float(np.sqrt(((r["forces"] - f) ** 2).mean())) for r, (e, f) in zip(res, ref)]
        # relative energies inside one composition (what the search ranks)
        return {"max_abs_dE_eV": float(np.abs(de).max()), "max_abs_dE_per_atom_eV": float(max(abs(d) / len(x) for d, x in zip(de, atoms))),
                "max_abs_dF_eV_A": max(df), "max_rms_dF_eV_A": max(rms), "dE_eV": dict(zip(names, de))}
    one, err = _try(lambda: B.single_points(model, atoms, 1, a.dtype, a.device, names))
    if one is None:
        raise SystemExit(f"ALCHEMI batch 1 failed on the panel: {err}")
    rep["alchemi_batch1_vs_native"] = compare(one, nat)
    ref_one = [(r["energy"], r["forces"]) for r in one]
    rep["batches"] = {}
    bs = 2
    while bs <= a.max_batch and bs <= len(atoms):
        res, err = _try(lambda: B.single_points(model, atoms, bs, a.dtype, a.device, names))
        rep["batches"][str(bs)] = {"vs_batch1": compare(res, ref_one), "vs_native": compare(res, nat)} if res else {"failed": err}
        if not res:
            break
        bs *= 2
    k = min(2, a.max_batch)
    order = list(range(len(atoms)))[::-1]
    rev, err = _try(lambda: B.single_points(model, [atoms[i] for i in order], k, a.dtype, a.device))
    if rev:
        back = [None] * len(atoms)
        for pos, i in enumerate(order):
            back[i] = rev[pos]
        rep["independence_reordered"] = compare(back, ref_one)
    else:
        rep["independence_reordered"] = {"failed": err}
    dup, err = _try(lambda: B.single_points(model, [atoms[0], atoms[0]], 2, a.dtype, a.device))
    rep["independence_repeated"] = ({"dE_between_copies_eV": abs(dup[0]["energy"] - dup[1]["energy"]),
                                     "dE_vs_alone_eV": abs(dup[0]["energy"] - one[0]["energy"]),
                                     "dF_vs_alone_eV_A": float(np.abs(dup[0]["forces"] - one[0]["forces"]).max())}
                                    if dup else {"failed": err})
    comp, err = _try(lambda: B.single_points(model, [atoms[0], atoms[-1]], 2, a.dtype, a.device))
    rep["independence_other_companion"] = ({"dE_eV": abs(comp[0]["energy"] - one[0]["energy"]),
                                            "dF_eV_A": float(np.abs(comp[0]["forces"] - one[0]["forces"]).max())}
                                           if comp else {"failed": err})
    again, _ = _try(lambda: B.single_points(model, atoms[:1], 1, a.dtype, a.device))
    rep["alchemi_repeatability"] = {"dE_eV": abs(again[0]["energy"] - one[0]["energy"]),
                                    "dF_eV_A": float(np.abs(again[0]["forces"] - one[0]["forces"]).max())} if again else None
    slab = atoms[0].copy()
    ax = int(np.argmax(slab.cell.lengths()))
    pbc = [True, True, True]
    pbc[ax] = False
    slab.pbc = pbc
    part, err = _try(lambda: B.single_points(model, [slab], 1, a.dtype, a.device))
    rep["partial_pbc"] = ({"pbc": pbc, "structure": names[0], "dE_vs_full_pbc_eV": abs(part[0]["energy"] - one[0]["energy"]),
                           "dF_vs_full_pbc_eV_A": float(np.abs(part[0]["forces"] - one[0]["forces"]).max()),
                           "vacuum_note": "the vacuum exceeds the cutoff, so both conventions must give the same numbers"}
                          if part else {"failed": err})
    g = rep["alchemi_batch1_vs_native"]
    rep["passes_float64_gate"] = bool(a.dtype == "float64" and "failed" not in g and g["max_abs_dE_eV"] <= TOL["energy_abs_eV"]
                                      and g["max_abs_dF_eV_A"] <= TOL["force_abs_eV_A"])
    return rep


def _steptime_point(a, backend, dtype, bs):
    """One (backend, precision, batch size) measurement; run in its own process (see cmd_steptime)."""
    import torch
    panel, _ = load_panel(Path(a.pool), a.names)
    nmax = max(len(q["atoms"]) for q in panel)
    big = [p["atoms"] for p in panel if len(p["atoms"]) == nmax]
    row = {"backend": backend, "dtype": dtype, "batch": bs, "n_atoms": nmax}
    torch.cuda.reset_peak_memory_stats()
    if backend == "native":
        calc = native_calc(a.model, a.device, dtype)
        c = big[0].copy()
        c.calc = calc
        t0 = time.perf_counter(); c.get_potential_energy(); _sync(); cold = time.perf_counter() - t0
        ts = []
        for r in range(a.repeats):
            c.positions[0, 0] += 1e-6
            t0 = time.perf_counter(); c.get_forces(); _sync(); ts.append(time.perf_counter() - t0)
    else:
        model = B.load_mace(a.model, a.device, dtype)
        xs = [big[i % len(big)] for i in range(bs)]
        t0 = time.perf_counter(); B.single_points(model, xs, bs, dtype, a.device); _sync(); cold = time.perf_counter() - t0
        ts = []
        for r in range(a.repeats):
            t0 = time.perf_counter(); B.single_points(model, xs, bs, dtype, a.device); _sync()
            ts.append(time.perf_counter() - t0)
        row["includes"] = "conversion, neighbour list, model call, transfer back"
    row.update(cold_s=round(cold, 2), warm_s_per_call=[round(t, 3) for t in ts],
               s_per_system=round(float(np.median(ts)) / bs, 3),
               peak_allocated_MiB=round(torch.cuda.max_memory_allocated() / 2 ** 20),
               peak_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2 ** 20))
    return row


def cmd_steptime(a):
    """Every point runs in a fresh process: a CUDA out-of-memory error leaves the context unusable, and a fresh
    process also makes each peak-memory figure and each cold start independent of the points before it."""
    import subprocess
    if a.point:
        backend, dtype, bs = a.point.split(":")
        print("ROW " + json.dumps(_steptime_point(a, backend, dtype, int(bs))))
        return None
    rep = {"schema": "alchemi_steptime/2", "versions": B.versions(), "gpu": gpu_info(), "repeats": a.repeats, "rows": []}
    base = [sys.executable, __file__, "steptime", a.pool, "--model", a.model, "--out", a.out, "--device", a.device,
            "--repeats", str(a.repeats)] + (["--names", ",".join(a.names)] if a.names else [])

    def point(backend, dtype, bs):
        try:
            r = subprocess.run(base + ["--point", f"{backend}:{dtype}:{bs}"], capture_output=True, text=True,
                               timeout=a.point_timeout)
        except subprocess.TimeoutExpired:
            # on Windows the driver spills into shared system memory instead of raising, and the point crawls
            return {"backend": backend, "dtype": dtype, "batch": bs,
                    "failed": f"timeout: no result in {a.point_timeout} s (VRAM exhausted, shared-memory fallback)"}
        for line in r.stdout.splitlines():
            if line.startswith("ROW "):
                return json.loads(line[4:])
        tail = (r.stderr or "").strip().splitlines()
        err = next((l for l in reversed(tail) if "Error" in l), tail[-1] if tail else "no output")[:200]
        return {"backend": backend, "dtype": dtype, "batch": bs,
                "failed": ("out_of_memory: " if "out of memory" in (r.stderr or "").lower() else "error: ") + err}
    for dtype in a.dtypes.split(","):
        rep["rows"].append(point("native", dtype, 1))
        print(json.dumps(rep["rows"][-1]), flush=True)
        bs = 1
        while bs <= a.max_batch:
            row = point("alchemi", dtype, bs)
            rep["rows"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "includes"}), flush=True)
            if "failed" in row:
                break
            bs += 1 if bs < 4 else 2
    return rep


def cmd_relax(a):
    import torch
    from ase.constraints import FixAtoms
    from ase.optimize import FIRE
    panel, _ = load_panel(Path(a.pool), a.names, initial=True)
    names = [p["name"] for p in panel]
    rep = {"schema": "alchemi_relax/1", "versions": B.versions(), "gpu": gpu_info(), "dtype": a.dtype, "fmax": a.fmax,
           "steps": a.steps, "pool": names, "pool_size": len(panel),
           "optimizers": {"native": "ASE FIRE (default parameters)", "alchemi": f"nvalchemi FIRE2, dt={a.dt}, defaults"},
           "arms": {}}

    def summarize(recs, secs, extra=None):
        ok = [r for r in recs if r["status"] == "converged"]
        return {"seconds_total": round(secs, 1), "converged": len(ok), "n": len(recs),
                "converged_per_hour": round(len(ok) / secs * 3600, 1) if secs else None,
                "statuses": {s: sum(r["status"] == s for r in recs) for s in sorted({r["status"] for r in recs})},
                "records": recs, **(extra or {})}

    # --- native
    torch.cuda.reset_peak_memory_stats()
    t_cold = time.perf_counter()
    calc = native_calc(a.model, a.device, a.dtype)
    setup = time.perf_counter() - t_cold
    recs, finals_native = [], {}
    t0 = time.perf_counter()
    for p in panel:
        x = p["atoms"].copy()
        x.set_constraint(FixAtoms(mask=p["frozen"]))
        x.calc = calc
        t1 = time.perf_counter()
        opt = FIRE(x, logfile=None)
        conv = bool(opt.run(fmax=a.fmax, steps=a.steps))
        f = x.get_forces(apply_constraint=False)
        fm = float(np.linalg.norm(f[~p["frozen"]], axis=1).max())
        finals_native[p["name"]] = (x.positions.copy(), float(x.get_potential_energy()))
        recs.append({"name": p["name"], "status": "converged" if conv and fm <= a.fmax * 1.02 else "step_limit",
                     "steps": int(opt.nsteps), "energy": finals_native[p["name"]][1], "fmax_free": fm,
                     "frozen_max_move": float(np.abs(x.positions[p["frozen"]] - p["atoms"].positions[p["frozen"]]).max()),
                     "seconds": round(time.perf_counter() - t1, 1)})
    _sync()
    rep["arms"]["native"] = summarize(recs, time.perf_counter() - t0, {
        "seconds_setup": round(setup, 1), "peak_allocated_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20),
        "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2 ** 20)})
    del calc
    torch.cuda.empty_cache()
    # --- ALCHEMI, batch 1 and batched
    t_cold = time.perf_counter()
    model = B.load_mace(a.model, a.device, a.dtype)
    setup = time.perf_counter() - t_cold
    finals = {}
    for bs in [1] + [b for b in (a.batch,) if b > 1]:
        t0 = time.perf_counter()
        out, acct = B.relax(model, [p["atoms"] for p in panel], [p["frozen"] for p in panel], a.fmax, a.steps, bs,
                            names, a.dt, a.dtype, a.device)
        _sync()
        secs = time.perf_counter() - t0
        recs = [{"name": r.candidate_id, "status": r.status, "steps": r.steps_to_convergence, "steps_run": r.steps_run,
                 "energy": r.energy, "fmax_free": r.fmax_free, "frozen_max_move": r.frozen_max_move_any_step,
                 **r.info} for r in out]
        finals[bs] = {r.candidate_id: r for r in out}
        rep["arms"][f"alchemi_batch{bs}"] = summarize(recs, secs, {"seconds_setup": round(setup, 1), **acct})
        if a.save:
            from ase.io import write
            d = Path(a.out) / f"alchemi_batch{bs}"
            d.mkdir(parents=True, exist_ok=True)
            for r in out:
                if r.atoms is not None:
                    write(d / f"{r.candidate_id}.extxyz", r.atoms, format="extxyz")
    # --- agreement
    def agree(fa, fb):
        rows = {}
        for n in names:
            ra, rb = fa.get(n), fb.get(n)
            if ra is None or rb is None or ra[0] is None or rb[0] is None:
                rows[n] = None
                continue
            rows[n] = {"dE_eV": ra[1] - rb[1], "max_dpos_A": float(np.abs(ra[0] - rb[0]).max())}
        return rows
    as_pair = lambda d: {n: ((r.atoms.positions if r.atoms is not None else None), r.energy) for n, r in d.items()}
    rep["agreement"] = {"alchemi_batch1_vs_native": agree(as_pair(finals[1]), finals_native)}
    if a.batch > 1:
        rep["agreement"][f"alchemi_batch{a.batch}_vs_batch1"] = agree(as_pair(finals[a.batch]), as_pair(finals[1]))
    en = lambda d: [d[n][1] for n in names if d.get(n) and d[n][1] is not None]
    two = [n for n in names if not n.startswith("one_")]
    if len(two) >= 3:
        na = np.array([finals_native[n][1] for n in two]); al = np.array([finals[1][n].energy or np.nan for n in two])
        if np.isfinite(al).all():
            rep["agreement"]["ranking_two_molecule"] = {
                "names": two, "native_rel_eV": (na - na.min()).round(4).tolist(), "alchemi_rel_eV": (al - al.min()).round(4).tolist(),
                "same_lowest": bool(na.argmin() == al.argmin()),
                "note": "different optimizers can end in different minima; compare with the 0.10 eV resolution"}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("singlepoint", "steptime", "relax"):
        p = sub.add_parser(name)
        p.add_argument("pool"), p.add_argument("--model", required=True), p.add_argument("--out", required=True)
        p.add_argument("--device", default="cuda"), p.add_argument("--dtype", default="float64")
        p.add_argument("--names", default=None), p.add_argument("--max-batch", type=int, default=4)
        p.add_argument("--dtypes", default="float64,float32"), p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--fmax", type=float, default=0.1), p.add_argument("--steps", type=int, default=150)
        p.add_argument("--batch", type=int, default=2), p.add_argument("--dt", type=float, default=0.1)
        p.add_argument("--save", action="store_true"), p.add_argument("--point", default=None)
        p.add_argument("--point-timeout", type=float, default=300.0)
    a = ap.parse_args(argv)
    a.names = a.names.split(",") if a.names else None
    rep = {"singlepoint": cmd_singlepoint, "steptime": cmd_steptime, "relax": cmd_relax}[a.cmd](a)
    if rep is None:
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.cmd}_{a.dtype if a.cmd != 'steptime' else a.dtypes.replace(',', '_')}.json"
    f.write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("structures", "versions")}, indent=1,
                     default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
