"""Periodic-cell check of the NVIDIA ALCHEMI batched-MACE backend (nvalchemi-toolkit 0.3.0) on our own cells.

Runs on a GPU node inside a package built by ``make_alchemi_periodic_package.py``.  Three subcommands:

    parity  CELLS_MANIFEST --model M --out parity.json [--dtype float64]
        Single points: native MACECalculator twice (the yardstick), then the toolkit as batch of 1, batch of N,
        shuffled order, and two different batch compositions (even / odd halves).  PASS when every toolkit
        comparison is within --tol-e eV and --tol-f eV/A of native (defaults 1e-6, float64).

    timing  POOL_MANIFEST --model M --out DIR [--optimizer LBFGS|FIRE] [--repeats 3] [--dtype float32]
        Fixed-cell relaxations of the pool: native ASE optimizer one structure at a time against the toolkit in
        fixed batches (batch size from a memory probe, or --batch).  Frozen atoms come from the pool's ``frozen``
        array.  Records atoms per cell, cells per batch, peak GPU memory, wall times per repeat, the optimizer on
        both sides, a native re-check of the batched end points (every input returns once, frozen atoms still,
        native fmax at the final geometry) and the end-point agreement between the two backends.

    summary RESULTS_DIR --out summary.json
        Collects the parity and timing reports into one verdict file.

The toolkit adapter is PV_Passivator's ``alchemi_backend.py`` (adsorbate-search branch), copied next to this file
by the package generator.  ``TORCHDYNAMO_DISABLE=1`` is expected (set by the job script).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402


# ----------------------------------------------------------------------------------------------- helpers
def load_cells(manifest: Path):
    from ase.io import read
    man = json.loads(Path(manifest).read_text())
    root = Path(manifest).parent
    cells = []
    for c in man["cells"]:
        a = read(root / c["file"])
        frozen = a.arrays["frozen"].astype(bool) if "frozen" in a.arrays else np.zeros(len(a), bool)
        for k in list(a.arrays):
            if k not in ("numbers", "positions"):
                del a.arrays[k]
        a.set_constraint()
        a.calc = None
        cells.append({"name": c["name"], "atoms": a, "frozen": frozen, "group": c.get("group", "")})
    return cells, man


def gpu_info():
    import torch
    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        return {"device": "cpu"}
    p = torch.cuda.get_device_properties(0)
    return {"device": p.name, "vram_MiB": round(p.total_memory / 2 ** 20), "cuda": torch.version.cuda}


def sync():
    import torch
    if (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        torch.cuda.synchronize()


def is_oom(e: Exception) -> bool:
    return "out of memory" in f"{type(e).__name__} {e}".lower()


def free_cuda():
    import gc
    import torch
    gc.collect()
    if (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        torch.cuda.empty_cache()


def peak_reset():
    import torch
    if (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        torch.cuda.reset_peak_memory_stats()


def peak_mib():
    import torch
    return round(torch.cuda.max_memory_allocated() / 2 ** 20) if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else None


def native_calc(model, device, dtype):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=str(model), device=device, default_dtype=dtype)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------- parity
def cmd_parity(a):
    import torch
    import alchemi_backend as B
    cells, man = load_cells(a.manifest)
    calc = native_calc(a.model, a.device, a.dtype)
    # a fine-tuned model knows only its training elements (ft_v2: C H I N Pb; package 129 rev1 crashed on O):
    # test each model on the cells it can represent, and say which were left out
    z_model = sorted(int(z) for z in calc.models[0].atomic_numbers)
    skipped = [{"name": c["name"], "unsupported_Z": sorted(set(int(z) for z in c["atoms"].numbers) - set(z_model))}
               for c in cells if not set(int(z) for z in c["atoms"].numbers) <= set(z_model)]
    cells = [c for c in cells if set(int(z) for z in c["atoms"].numbers) <= set(z_model)]
    if len(cells) < 2:
        raise SystemExit(f"fewer than 2 cells have only elements in the model's z_table {z_model}")
    atoms = [c["atoms"] for c in cells]
    n = len(atoms)

    def native():
        out = []
        for x in atoms:
            y = x.copy()
            y.calc = calc
            out.append({"energy": float(y.get_potential_energy()), "forces": np.array(y.get_forces())})
        return out

    t0 = time.perf_counter()
    n1 = native()
    n2 = native()
    t_native = time.perf_counter() - t0
    del calc
    free_cuda()
    model = B.load_mace(a.model, a.device, a.dtype)

    def batched(xs, bs_req):
        """single points of xs in batches of bs_req, halving on CUDA out-of-memory; returns (results, bs_used)."""
        bs = bs_req
        while True:
            try:
                return B.single_points(model, xs, batch_size=bs, dtype=a.dtype, device=a.device), bs
            except Exception as e:  # noqa: BLE001
                if not is_oom(e) or bs == 1:
                    raise
                free_cuda()
                bs = max(1, bs // 2)

    b1, _ = batched(atoms, 1)
    peak_reset()
    bn, bn_used = batched(atoms, n)
    peak_bn = peak_mib()
    order = list(np.random.default_rng(129).permutation(n))
    sh, sh_used = batched([atoms[i] for i in order], n)
    shuffled = [sh[order.index(i)] for i in range(n)]
    even_idx, odd_idx = list(range(0, n, 2)), list(range(1, n, 2))
    ev, _ = batched([atoms[i] for i in even_idx], len(even_idx))
    od, _ = batched([atoms[i] for i in odd_idx], len(odd_idx))
    mixed = [None] * n
    for k, i in enumerate(even_idx):
        mixed[i] = ev[k]
    for k, i in enumerate(odd_idx):
        mixed[i] = od[k]

    def d(x, y):
        return abs(x["energy"] - y["energy"]), float(np.abs(x["forces"] - y["forces"]).max())

    def worst(xs, ys):
        dd = [d(x, y) for x, y in zip(xs, ys)]
        return {"max_abs_dE_eV": max(e for e, _ in dd), "max_abs_dF_eV_per_A": max(f for _, f in dd)}

    comps = {"native_vs_native": worst(n1, n2), "native_vs_batch1": worst(n1, b1), "native_vs_batchN": worst(n1, bn),
             "native_vs_shuffled": worst(n1, shuffled), "native_vs_half_batches": worst(n1, mixed),
             "batchN_vs_shuffled": worst(bn, shuffled), "batchN_vs_half_batches": worst(bn, mixed)}
    toolkit = [k for k in comps if k != "native_vs_native"]
    ok = all(comps[k]["max_abs_dE_eV"] <= a.tol_e and comps[k]["max_abs_dF_eV_per_A"] <= a.tol_f for k in toolkit)
    rep = {"schema": "pvpass_alchemi_parity/1", "verdict": "PASS" if ok else "FAIL",
           "model": Path(a.model).name, "dtype": a.dtype, "tolerance": {"dE_eV": a.tol_e, "dF_eV_per_A": a.tol_f},
           "versions": B.versions(), "gpu": gpu_info(), "model_z_table": z_model, "cells_skipped_unsupported_elements": skipped, "n_cells": n, "n_atoms_total": int(sum(len(x) for x in atoms)),
           "batch_of_N": {"requested": n, "used": bn_used, "shuffled_used": sh_used, "peak_allocated_MiB": peak_bn,
                          "note": "batch size is halved only after a CUDA out-of-memory error"},
           "shuffled_order": [int(i) for i in order], "seconds_native_two_passes": round(t_native, 1),
           "comparisons": comps,
           "cells": [{"name": c["name"], "group": c["group"], "atoms": len(c["atoms"]),
                      "pbc": [bool(p) for p in c["atoms"].pbc], "cell_lengths_A": [round(float(v), 3) for v in c["atoms"].cell.lengths()],
                      "native_energy_eV": n1[i]["energy"], "native_max_abs_force_eV_per_A": float(np.abs(n1[i]["forces"]).max()),
                      "native_vs_native_dE": d(n1[i], n2[i])[0], "native_vs_batch1_dE": d(n1[i], b1[i])[0],
                      "native_vs_batchN_dE": d(n1[i], bn[i])[0], "native_vs_batchN_dF": d(n1[i], bn[i])[1],
                      "native_vs_shuffled_dF": d(n1[i], shuffled[i])[1], "native_vs_half_dF": d(n1[i], mixed[i])[1]}
                     for i, c in enumerate(cells)]}
    write_json(Path(a.out), rep)
    for c in rep["cells"]:
        print(f"{c['name']:34s} {c['atoms']:4d} at  dE(N)={c['native_vs_batchN_dE']:.1e}  dF(N)={c['native_vs_batchN_dF']:.1e}")
    for k, v in comps.items():
        print(f"{k:26s} dE {v['max_abs_dE_eV']:.2e}  dF {v['max_abs_dF_eV_per_A']:.2e}")
    print("PARITY", rep["verdict"], Path(a.model).name)
    return 0 if ok else 3


# ----------------------------------------------------------------------------------------------- batched relax
def relax_batched(model, atoms_list, frozen, *, fmax, steps, batch_size, ids, optimizer, dtype, device):
    """Fixed-cell batched relaxation (static batches), one terminal record per input, in input order.

    Mirrors ``alchemi_backend.relax`` (FIRE2 only there) with a choice of optimizer.  In 0.3.0 a standalone
    optimizer keeps stepping converged systems until the slowest in the batch is done, so each system's geometry is
    taken at the step it FIRST meets fmax, where a one-system run would have stopped.  Frozen-atom displacement is
    measured after every step.  No audit single points here: the caller re-checks end points natively, untimed."""
    import torch
    import alchemi_backend as B
    from nvalchemi.dynamics import FIRE2, LBFGS
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.dynamics.hooks import FreezeAtomsHook
    n = len(atoms_list)
    out = [None] * n
    acct = {"batches": 0, "model_steps": 0, "seconds_setup": 0.0, "seconds_run": 0.0}
    for start in range(0, n, batch_size):
        chunk = list(range(start, min(start + batch_size, n)))
        t0 = time.perf_counter()
        batch, metas = B.to_batch([atoms_list[i] for i in chunk], [frozen[i] for i in chunk],
                                  [ids[i] for i in chunk], dtype, device)
        start_pos = B._split_positions(batch, metas)
        hooks = [FreezeAtomsHook(mask_key="frozen")] + list(model.make_neighbor_hooks())
        conv_hook = ConvergenceHook.from_fmax(fmax, source_status=0, target_status=1)
        if optimizer == "LBFGS":
            opt = LBFGS(model=model, n_steps=steps, hooks=hooks, convergence_hook=conv_hook)
        else:
            opt = FIRE2(model=model, dt=0.1, n_steps=steps, hooks=hooks, convergence_hook=conv_hook)
        sync()
        acct["seconds_setup"] += time.perf_counter() - t0
        t1 = time.perf_counter()
        first = {k: None for k in range(len(chunk))}
        snap, moved, ran, err = {}, {k: 0.0 for k in range(len(chunk))}, 0, None
        try:
            for step in range(1, steps + 1):
                batch, conv = opt.step(batch)
                ran = step
                pos = B._split_positions(batch, metas)
                for k, m in enumerate(metas):
                    if m.frozen.any():
                        moved[k] = max(moved[k], float(np.abs(pos[k][m.frozen] - start_pos[k][m.frozen]).max()))
                done = B._converged_set(batch, conv)
                for k in done:
                    if first[k] is None:
                        first[k] = step
                        snap[k] = pos[k].copy()
                # stop once every system has met fmax at least once; the status column can flip back after
                # convergence, so "all converged at this step" is not a usable test (smoke run, 2026-11-14)
                if all(v is not None for v in first.values()):
                    break
        except Exception as e:  # noqa: BLE001
            err = e
        sync()
        acct["seconds_run"] += time.perf_counter() - t1
        acct["batches"] += 1
        acct["model_steps"] += ran
        if err is not None:
            for k, i in enumerate(chunk):
                out[i] = {"name": ids[i], "status": B._failure_kind(err), "positions": None, "steps": None,
                          "steps_run": ran, "frozen_max_move": moved[k], "error": f"{type(err).__name__}: {err}"[:300]}
            continue
        B._check_identity(batch, metas)
        pos = B._split_positions(batch, metas)
        for k, i in enumerate(chunk):
            p = snap.get(k, pos[k])
            out[i] = {"name": ids[i], "status": "engine_converged" if first[k] is not None else "step_limit",
                      "positions": p, "steps": first[k], "steps_run": ran, "frozen_max_move": moved[k]}
        del batch, opt
    if any(r is None for r in out):
        raise RuntimeError("an input has no terminal record")
    return out, acct


def probe_batch(model, atoms, dtype, device, sizes):
    """Peak GPU memory of one batched single point (energy + forces) at each batch size; stops at the first OOM."""
    import torch
    import alchemi_backend as B
    rows = []
    for bs in sizes:
        free_cuda()
        peak_reset()
        t0 = time.perf_counter()
        try:
            B.single_points(model, atoms[:bs], batch_size=bs, dtype=dtype, device=device)
            sync()
            rows.append({"cells": bs, "atoms": int(sum(len(x) for x in atoms[:bs])),
                         "peak_allocated_MiB": peak_mib(),
                         "seconds": round(time.perf_counter() - t0, 3)})
        except Exception as e:  # noqa: BLE001
            if not is_oom(e):
                raise
            rows.append({"cells": bs, "oom": True})
            free_cuda()
            break
    return rows


def cmd_timing(a):
    import torch
    from ase.constraints import FixAtoms
    from ase.optimize import FIRE as ASE_FIRE
    from ase.optimize import LBFGS as ASE_LBFGS
    import alchemi_backend as B
    cells, man = load_cells(a.manifest)
    names = [c["name"] for c in cells]
    atoms = [c["atoms"] for c in cells]
    frozen = [c["frozen"] for c in cells]
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    gi = gpu_info()
    vram = gi.get("vram_MiB") or 0
    native_opt = {"LBFGS": "ASE LBFGS (defaults: maxstep 0.2, memory 100, damping 1, alpha 70)",
                  "FIRE": "ASE FIRE (defaults: dt 0.1, maxstep 0.2)"}[a.optimizer]
    toolkit_opt = {"LBFGS": "nvalchemi LBFGS (defaults: history_size 6, maxstep 0.2)",
                   "FIRE": "nvalchemi FIRE2 (dt 0.1, other defaults)"}[a.optimizer]
    rep = {"schema": "pvpass_alchemi_timing/1", "model": Path(a.model).name, "dtype": a.dtype, "fmax_eV_per_A": a.fmax,
           "step_cap": a.steps, "optimizer": {"family": a.optimizer, "native": native_opt, "toolkit": toolkit_opt},
           "criterion": "max force norm over free atoms; frozen atoms fixed (ASE FixAtoms / toolkit FreezeAtomsHook)",
           "versions": B.versions(), "gpu": gi, "pool": [{"name": c["name"], "group": c["group"], "atoms": len(c["atoms"]),
                                                       "free_atoms": int((~c["frozen"]).sum())} for c in cells],
           "repeats": []}

    # ---- memory probe -> cells per batch
    model = B.load_mace(a.model, a.device, a.dtype)
    B.single_points(model, atoms[:1], batch_size=1, dtype=a.dtype, device=a.device)        # warm-up
    sizes = sorted({s for s in (1, 2, 4, 8, 12, 16, 24, 32, len(atoms)) if s <= len(atoms)})
    probe = probe_batch(model, atoms, a.dtype, a.device, sizes)
    good = [r for r in probe if not r.get("oom")]
    if a.batch:
        bs = a.batch
    else:
        fits = [r["cells"] for r in good if r["peak_allocated_MiB"] <= a.mem_frac * vram]
        bs = max(fits) if fits else 1
    est = None
    if len(good) >= 2 and vram:
        x = np.array([r["atoms"] for r in good], float)
        y = np.array([r["peak_allocated_MiB"] for r in good], float)
        slope, icpt = np.polyfit(x, y, 1)
        mean_atoms = float(np.mean([len(t) for t in atoms]))
        est = {"MiB_per_atom": round(float(slope), 3), "MiB_fixed": round(float(icpt)),
               "cells_fit_in_80pct_vram_est": int((0.8 * vram - icpt) / (slope * mean_atoms)) if slope > 0 else None,
               "note": "linear fit of single-point peaks; a relaxation holds a little more (optimizer state)"}
    rep["batch_probe"] = {"rows": probe, "chosen_batch": bs, "rule": f"largest probed size with peak <= {a.mem_frac:.0%} of VRAM"
                          if not a.batch else "fixed by --batch", "estimate": est}
    free_cuda()

    calc = native_calc(a.model, a.device, a.dtype)
    w = atoms[0].copy()
    w.calc = calc
    w.get_potential_energy()                                                                  # warm-up
    Opt = ASE_LBFGS if a.optimizer == "LBFGS" else ASE_FIRE

    def native_arm():
        recs = []
        peak_reset()
        t0 = time.perf_counter()
        for c in cells:
            x = c["atoms"].copy()
            x.set_constraint(FixAtoms(mask=c["frozen"]))
            x.calc = calc
            t1 = time.perf_counter()
            opt = Opt(x, logfile=None)
            conv = bool(opt.run(fmax=a.fmax, steps=a.steps))
            sync()
            secs = time.perf_counter() - t1
            f = x.get_forces(apply_constraint=False)
            fm = float(np.linalg.norm(f[~c["frozen"]], axis=1).max())
            recs.append({"name": c["name"], "status": "converged" if conv else "step_limit", "steps": int(opt.nsteps),
                         "energy": float(x.get_potential_energy()), "fmax_free": fm, "seconds": round(secs, 2),
                         "frozen_max_move": float(np.abs(x.positions[c["frozen"]] - c["atoms"].positions[c["frozen"]]).max())
                         if c["frozen"].any() else 0.0, "positions": x.positions.copy()})
        return recs, time.perf_counter() - t0, peak_mib()

    def toolkit_arm(batch_size):
        peak_reset()
        t0 = time.perf_counter()
        recs, acct = relax_batched(model, atoms, frozen, fmax=a.fmax, steps=a.steps, batch_size=batch_size, ids=names,
                                   optimizer=a.optimizer, dtype=a.dtype, device=a.device)
        secs = time.perf_counter() - t0
        return recs, secs, peak_mib(), acct

    def recheck(recs):
        """native energy and free-atom fmax at the toolkit's end points (untimed)."""
        for r, c in zip(recs, cells):
            if r["positions"] is None:
                r.update(energy_native=None, fmax_free_native=None, status_final=r["status"])
                continue
            x = c["atoms"].copy()
            x.positions = r["positions"]
            x.calc = calc
            f = x.get_forces()
            fm = float(np.linalg.norm(f[~c["frozen"]], axis=1).max())
            r["energy_native"] = float(x.get_potential_energy())
            r["fmax_free_native"] = fm
            r["status_final"] = ("converged" if r["status"] == "engine_converged" and fm <= a.fmax * 1.02
                                 else "engine_converged_but_native_fmax_above" if r["status"] == "engine_converged"
                                 else r["status"])
        return recs

    def strip(recs):
        return [{k: v for k, v in r.items() if k != "positions"} for r in recs]

    last_native = last_toolkit = None
    for rep_i in range(1, a.repeats + 1):
        row = {"repeat": rep_i}
        free_cuda()
        nrec, nsec, npeak = native_arm()
        row["native"] = {"wall_s": round(nsec, 1), "peak_allocated_MiB": npeak,
                         "converged": sum(r["status"] == "converged" for r in nrec), "n": len(nrec),
                         "steps_total": int(sum(r["steps"] for r in nrec)), "records": strip(nrec)}
        free_cuda()
        trec, tsec, tpeak, acct = toolkit_arm(bs)
        trec = recheck(trec)
        row[f"toolkit_batch{bs}"] = {"wall_s": round(tsec, 1), "peak_allocated_MiB": tpeak, "cells_per_batch": bs,
                                     **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in acct.items()},
                                     "converged": sum(r["status_final"] == "converged" for r in trec), "n": len(trec),
                                     "records": strip(trec)}
        row["speedup"] = round(nsec / tsec, 2) if tsec else None
        print(f"repeat {rep_i}: native {nsec:.1f} s, toolkit batch {bs} {tsec:.1f} s, x{row['speedup']}", flush=True)
        rep["repeats"].append(row)
        last_native, last_toolkit = nrec, trec
        write_json(outdir / f"timing_{a.optimizer}.json", rep)

    if a.batch1_once and bs > 1:
        free_cuda()
        brec, bsec, bpeak, acct = toolkit_arm(1)
        rep["toolkit_batch1_once"] = {"wall_s": round(bsec, 1), "peak_allocated_MiB": bpeak,
                                      **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in acct.items()},
                                      "engine_converged": sum(r["status"] == "engine_converged" for r in brec)}

    # ---- reliability and agreement, from the last repeat
    tk = last_toolkit
    rel = {"every_input_once": sorted(r["name"] for r in tk) == sorted(names) and len(tk) == len(names),
           "statuses": {s: sum(r["status_final"] == s for r in tk) for s in sorted({r["status_final"] for r in tk})},
           "frozen_max_move_A": max(r["frozen_max_move"] for r in tk),
           "native_fmax_at_toolkit_end_max": max((r["fmax_free_native"] or 0.0) for r in tk),
           "native_statuses": {s: sum(r["status"] == s for r in last_native) for s in sorted({r["status"] for r in last_native})}}
    rel["pass"] = bool(rel["every_input_once"] and rel["frozen_max_move_A"] < 1e-6
                       and all(r["status_final"] in ("converged", "step_limit") for r in tk))
    agree = []
    for rn, rt, c in zip(last_native, tk, cells):
        if rt["positions"] is None:
            agree.append({"name": c["name"], "dE_eV": None})
            continue
        dpos = np.linalg.norm(rt["positions"] - rn["positions"], axis=1)
        agree.append({"name": c["name"], "dE_toolkit_minus_native_eV": round(rt["energy_native"] - rn["energy"], 5),
                      "max_dpos_A": round(float(dpos.max()), 3), "steps_native": rn["steps"], "steps_toolkit": rt["steps"]})
    de = np.array([r["dE_toolkit_minus_native_eV"] for r in agree if r.get("dE_toolkit_minus_native_eV") is not None])
    en_n = np.array([r["energy"] for r in last_native])
    en_t = np.array([r["energy_native"] if r["energy_native"] is not None else np.nan for r in tk])
    rep["reliability"] = rel
    rep["agreement"] = {"rows": agree, "median_abs_dE_eV": round(float(np.median(np.abs(de))), 4) if de.size else None,
                        "max_abs_dE_eV": round(float(np.abs(de).max()), 4) if de.size else None,
                        "n_moved_more_than_0p15_A": int(sum(1 for r in agree if (r.get("max_dpos_A") or 0) > 0.15)),
                        "same_lowest": bool(np.nanargmin(en_n) == np.nanargmin(en_t)) if np.isfinite(en_t).any() else None,
                        "note": "different optimizer implementations can end in different minima (ACAT: LBFGS vs FIRE "
                                "0.1-1.3 eV); compare end points, do not assume they coincide"}
    wn = [r["native"]["wall_s"] for r in rep["repeats"]]
    wt = [r[f"toolkit_batch{bs}"]["wall_s"] for r in rep["repeats"]]
    rep["median"] = {"native_wall_s": float(np.median(wn)), "toolkit_wall_s": float(np.median(wt)),
                     "speedup": round(float(np.median(wn)) / float(np.median(wt)), 2), "cells_per_batch": bs,
                     "atoms_per_cell": sorted({len(x) for x in atoms})}
    write_json(outdir / f"timing_{a.optimizer}.json", rep)
    print("TIMING", a.optimizer, json.dumps(rep["median"]), "reliability", "PASS" if rel["pass"] else "FAIL")
    return 0


# ----------------------------------------------------------------------------------------------- summary
def cmd_summary(a):
    res = Path(a.results)
    out = {"schema": "pvpass_alchemi_summary/1", "parity": {}, "timing": {}}
    for p in sorted(res.glob("parity_*.json")):
        r = json.loads(p.read_text())
        out["parity"][r["model"]] = {"verdict": r["verdict"], "n_cells": r["n_cells"], "cells_skipped": [c["name"] for c in r.get("cells_skipped_unsupported_elements", [])], "batch_of_N_used": r["batch_of_N"]["used"],
                                     "worst_dE_eV": max(v["max_abs_dE_eV"] for k, v in r["comparisons"].items() if k != "native_vs_native"),
                                     "worst_dF_eV_per_A": max(v["max_abs_dF_eV_per_A"] for k, v in r["comparisons"].items() if k != "native_vs_native"),
                                     "native_vs_native": r["comparisons"]["native_vs_native"]}
    for p in sorted(res.glob("timing/timing_*.json")):
        r = json.loads(p.read_text())
        out["timing"][p.stem] = {"model": r["model"], "dtype": r["dtype"], "optimizer": r["optimizer"],
                                 "median": r.get("median"), "reliability": r.get("reliability"),
                                 "agreement_median_abs_dE_eV": (r.get("agreement") or {}).get("median_abs_dE_eV"),
                                 "repeats_done": len(r["repeats"])}
    write_json(Path(a.out), out)
    print(json.dumps(out, indent=1, default=float))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity")
    p.add_argument("manifest", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--tol-e", type=float, default=1e-6)
    p.add_argument("--tol-f", type=float, default=1e-6)
    t = sub.add_parser("timing")
    t.add_argument("manifest", type=Path)
    t.add_argument("--model", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--optimizer", default="LBFGS", choices=["LBFGS", "FIRE"])
    t.add_argument("--repeats", type=int, default=3)
    t.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    t.add_argument("--device", default="cuda")
    t.add_argument("--fmax", type=float, default=0.05)
    t.add_argument("--steps", type=int, default=500)
    t.add_argument("--batch", type=int, default=0, help="cells per batch; 0 = from the memory probe")
    t.add_argument("--mem-frac", type=float, default=0.5)
    t.add_argument("--batch1-once", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("results")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"parity": cmd_parity, "timing": cmd_timing, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
