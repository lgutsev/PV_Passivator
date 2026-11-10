"""Optional batched execution backend on NVIDIA ALCHEMI (nvalchemi-toolkit), behind the native path.

Nothing in the package imports this module; the native ASE path (``adsearch.AseRelaxEvaluator``) stays the
default and the reference.  Tested against nvalchemi-toolkit 0.3.0 only (see docs/alchemi_feasibility.md).

What the adapter guarantees, and how:

  identity     Every system carries an integer ``system_id`` into the batch.  Atom ids, roles, molecule ids, the
               frozen mask and the candidate id are kept OUTSIDE the framework, in ``SystemMeta``, and re-attached
               after the run by ``system_id``; atomic numbers and frozen-atom positions are compared with the
               input before anything is returned, so a mix-up cannot pass silently.
  constraints  The frozen mask is the native workflow's (``AseRelaxEvaluator.mobile_mask``), passed as a node
               field to ``FreezeAtomsHook(mask_key=...)``.  The displacement of frozen atoms is measured after
               EVERY step, not only at the end.
  convergence  The same criterion as the native path (max force norm over free atoms).  It is re-evaluated from
               an independent single point at the final geometry; a system that only reached the step limit is
               recorded as ``step_limit``, never as converged.
  records      One terminal record per input, whatever happened to it.

Windows notes for 0.3.0 (found here, not documented upstream):
  * ``MACEWrapper.from_checkpoint`` passes a local path to MACE's downloader, which reads the drive letter as a
    URL scheme; ``load_mace`` loads the checkpoint itself and wraps the module (the documented constructor).
  * the dynamics engine calls ``torch.compile``; without an MSVC compiler that fails, so ``TORCHDYNAMO_DISABLE=1``
    is set on import unless the caller already chose a value.  Compiled and uncompiled timings are therefore not
    comparable, and this build is the uncompiled one.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402

TESTED_VERSION = "0.3.0"
STATUSES = ("converged", "step_limit", "invalid_geometry", "numerical_failure", "resource_failure")


def versions() -> dict:
    import torch
    out = {"torch": torch.__version__, "cuda": torch.version.cuda, "torchdynamo_disabled":
           os.environ.get("TORCHDYNAMO_DISABLE") == "1"}
    for m in ("nvalchemi", "nvalchemiops", "warp", "mace", "e3nn", "ase", "numpy", "tensordict", "physicsnemo"):
        try:
            out[m] = getattr(__import__(m), "__version__", "present")
        except Exception as e:
            out[m] = f"absent ({type(e).__name__})"
    return out


def load_mace(path: str, device: str = "cuda", dtype: str = "float64"):
    """The checkpoint at ``path`` as an ALCHEMI ``MACEWrapper`` (eval mode, frozen weights, no cuEq, no compile)."""
    import torch
    from nvalchemi.models.mace import MACEWrapper
    model = torch.load(str(path), weights_only=False, map_location=torch.device(device))
    # map_location alone leaves some e3nn constant buffers (compiled tensor-product Wigner 3j) on the CPU
    model.to(device=torch.device(device), dtype=getattr(torch, dtype))
    return MACEWrapper(model)


@dataclass
class SystemMeta:
    """Per-system metadata kept outside the framework's tensors."""
    system_id: int
    candidate_id: str
    n_atoms: int
    numbers: np.ndarray
    frozen: np.ndarray
    arrays: dict = field(default_factory=dict)       # atom_id, role, mol_id, fragment_id, origin ...
    info: dict = field(default_factory=dict)
    pbc: tuple = (True, True, True)
    cell: np.ndarray | None = None


def _meta(atoms: Atoms, k: int, cid: str, frozen) -> SystemMeta:
    fz = np.zeros(len(atoms), bool) if frozen is None else np.asarray(frozen, bool)
    if fz.shape != (len(atoms),):
        raise ValueError(f"{cid}: frozen mask has shape {fz.shape}, expected ({len(atoms)},)")
    keep = {n: v.copy() for n, v in atoms.arrays.items() if n not in ("numbers", "positions")}
    return SystemMeta(k, cid, len(atoms), atoms.numbers.copy(), fz, keep, dict(atoms.info),
                      tuple(bool(x) for x in atoms.pbc), atoms.cell.array.copy())


def to_batch(atoms_list, frozen=None, ids=None, dtype: str = "float64", device: str = "cuda"):
    """(Batch, [SystemMeta]).  Positions, cell, every PBC flag, atomic numbers and atom order go into the batch
    unchanged; ``frozen`` (node) and ``system_id`` (system) ride along; everything else stays in the metadata."""
    import torch
    from nvalchemi.data import AtomicData, Batch
    td = getattr(torch, dtype)
    datas, metas = [], []
    for k, a in enumerate(atoms_list):
        cid = ids[k] if ids is not None else f"sys{k}"
        m = _meta(a, k, cid, None if frozen is None else frozen[k])
        bare = Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=a.pbc)   # no stale results, no arrays
        d = AtomicData.from_atoms(bare, dtype=td)
        d.add_node_property("frozen", torch.tensor(m.frozen))
        d.add_system_property("system_id", torch.tensor([[k]], dtype=torch.long))
        datas.append(d)
        metas.append(m)
    return Batch.from_data_list(datas, device=device), metas


def _split(batch, metas, key):
    """Per-system numpy arrays of a node-level field, addressed by ``system_id``, with the atom count checked."""
    sid = batch.system_id.detach().cpu().numpy().ravel()
    ptr = batch.batch_ptr.detach().cpu().numpy()
    val = getattr(batch, key).detach().cpu().numpy()
    out = {}
    for row, s in enumerate(sid[:batch.num_graphs]):
        seg = val[ptr[row]:ptr[row + 1]]
        if len(seg) != metas[int(s)].n_atoms:
            raise RuntimeError(f"system {int(s)}: {len(seg)} atoms came back, {metas[int(s)].n_atoms} went in")
        out[int(s)] = seg
    return out


def _check_identity(batch, metas):
    for s, z in _split(batch, metas, "atomic_numbers").items():
        if not np.array_equal(z.ravel(), metas[s].numbers):
            raise RuntimeError(f"system {s} ({metas[s].candidate_id}): atomic numbers or atom order changed")


def single_points(model, atoms_list, batch_size: int = 1, dtype: str = "float64", device: str = "cuda",
                  ids=None) -> list[dict]:
    """Energy (eV) and forces (eV/A) of each structure at exactly its geometry, ``batch_size`` systems per call."""
    from nvalchemi.neighbors import compute_neighbors
    res = [None] * len(atoms_list)
    for start in range(0, len(atoms_list), batch_size):
        chunk = list(range(start, min(start + batch_size, len(atoms_list))))
        batch, metas = to_batch([atoms_list[i] for i in chunk], ids=None if ids is None else [ids[i] for i in chunk],
                                dtype=dtype, device=device)
        compute_neighbors(batch, config=model.model_config.neighbor_config)
        out = model(batch)
        _check_identity(batch, metas)
        e = out["energy"].detach().cpu().numpy().ravel()
        batch.forces = out["forces"].detach()
        f = _split(batch, metas, "forces")
        sid = batch.system_id.detach().cpu().numpy().ravel()
        for row, s in enumerate(sid):
            res[chunk[int(s)]] = {"energy": float(e[row]), "forces": f[int(s)].astype(float),
                                  "candidate_id": metas[int(s)].candidate_id, "n_edges": None}
        nl = getattr(batch, "neighbor_list", None)
        if nl is not None:                                       # no edge may join two systems
            bi = batch.batch_idx.detach().cpu().numpy()
            e2 = nl.detach().cpu().numpy()
            e2 = e2[(e2 < len(bi)).all(axis=1)]
            if (bi[e2[:, 0]] != bi[e2[:, 1]]).any():
                raise RuntimeError("neighbour list contains an edge between two different systems")
    return res


@dataclass
class RelaxRecord:
    candidate_id: str
    status: str                      # one of STATUSES
    atoms: Atoms | None
    energy: float | None
    fmax_free: float | None          # from an independent single point at the final geometry
    steps_to_convergence: int | None
    steps_run: int
    frozen_max_move_any_step: float
    info: dict = field(default_factory=dict)


def relax(model, atoms_list, frozen, fmax: float = 0.1, steps: int = 150, batch_size: int | None = None,
          ids=None, dt: float = 0.1, dtype: str = "float64", device: str = "cuda", optimizer_kwargs=None,
          audit: bool = True) -> tuple[list[RelaxRecord], dict]:
    """Fixed-cell FIRE2 relaxation of every structure, ``batch_size`` at a time (static batching).

    In 0.3.0 a standalone optimizer keeps stepping systems that have already converged until the slowest one in
    the batch is done (status migration only acts inside a fused stage), which would make a system's final
    geometry depend on its batch companions.  So each system's geometry is taken at the step it FIRST meets
    ``fmax`` over its free atoms, exactly where a one-system run would have stopped.  Returns one record per
    input, in input order, and a timing/accounting dict."""
    import torch
    from nvalchemi.dynamics import FIRE2
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.dynamics.hooks import FreezeAtomsHook
    n = len(atoms_list)
    ids = list(ids) if ids is not None else [f"sys{k}" for k in range(n)]
    batch_size = batch_size or n
    records: list[RelaxRecord | None] = [None] * n
    acct = {"batches": 0, "model_steps": 0, "system_steps": 0, "seconds_setup": 0.0, "seconds_run": 0.0,
            "seconds_audit": 0.0, "peak_allocated_MiB": None, "peak_reserved_MiB": None}
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for start in range(0, n, batch_size):
        chunk = list(range(start, min(start + batch_size, n)))
        t0 = time.perf_counter()
        try:
            batch, metas = to_batch([atoms_list[i] for i in chunk], [frozen[i] for i in chunk],
                                    [ids[i] for i in chunk], dtype, device)
            start_pos = _split_positions(batch, metas)
            hooks = [FreezeAtomsHook(mask_key="frozen")] + list(model.make_neighbor_hooks())
            opt = FIRE2(model=model, dt=dt, n_steps=steps, hooks=hooks,
                        convergence_hook=ConvergenceHook.from_fmax(fmax, source_status=0, target_status=1),
                        **(optimizer_kwargs or {}))
        except Exception as e:                                      # nothing ran: every system of the chunk is recorded
            for i in chunk:
                records[i] = RelaxRecord(ids[i], _failure_kind(e), None, None, None, None, 0, 0.0,
                                         {"error": f"{type(e).__name__}: {e}"[:300]})
            continue
        if cuda:
            torch.cuda.synchronize()
        acct["seconds_setup"] += time.perf_counter() - t0
        t1 = time.perf_counter()
        first = {k: None for k in range(len(chunk))}
        snap = {}
        moved = {k: 0.0 for k in range(len(chunk))}
        ran, err = 0, None
        try:
            opt._open_hooks() if hasattr(opt, "_open_hooks") else None
            for step in range(1, steps + 1):
                batch, conv = opt.step(batch)
                ran = step
                pos = _split_positions(batch, metas)
                for k, m in enumerate(metas):
                    if m.frozen.any():
                        moved[k] = max(moved[k], float(np.abs(pos[k][m.frozen] - start_pos[k][m.frozen]).max()))
                done = _converged_set(batch, conv)
                for k in done:
                    if first[k] is None:
                        first[k] = step
                        snap[k] = pos[k].copy()
                acct["system_steps"] += sum(1 for k in first if first[k] is None or first[k] == step)
                if len(done) == len(chunk):
                    break
        except Exception as e:
            err = e
        if cuda:
            torch.cuda.synchronize()
        acct["seconds_run"] += time.perf_counter() - t1
        acct["batches"] += 1
        acct["model_steps"] += ran
        if err is not None:                                         # the whole chunk is lost, and says so
            for k, i in enumerate(chunk):
                records[i] = RelaxRecord(ids[i], _failure_kind(err), None, None, None, None, ran, moved[k],
                                         {"error": f"{type(err).__name__}: {err}"[:300], "batch_failure": True})
            continue
        _check_identity(batch, metas)
        pos = _split_positions(batch, metas)
        finals = []
        for k, m in enumerate(metas):
            a = Atoms(numbers=m.numbers, positions=snap.get(k, pos[k]), cell=m.cell, pbc=m.pbc)
            for name, v in m.arrays.items():
                a.set_array(name, v)
            a.info.update(m.info)
            finals.append(a)
        t2 = time.perf_counter()
        sp = single_points(model, finals, batch_size=1, dtype=dtype, device=device) if audit else None
        acct["seconds_audit"] += time.perf_counter() - t2
        for k, i in enumerate(chunk):
            m, a = metas[k], finals[k]
            if not np.isfinite(a.positions).all():
                records[i] = RelaxRecord(ids[i], "numerical_failure", None, None, None, first[k], ran, moved[k])
                continue
            e = f = fm = None
            if sp is not None:
                e, f = sp[k]["energy"], sp[k]["forces"]
                fm = float(np.linalg.norm(f[~m.frozen], axis=1).max()) if (~m.frozen).any() else 0.0
                a.set_array("model_forces", f)
                a.info["model_energy"] = e
            a.set_array("mobile", (~m.frozen).astype(int))
            if sp is not None and not (np.isfinite(e) and np.isfinite(f).all()):
                status = "numerical_failure"
            elif first[k] is not None and (fm is None or fm <= fmax * 1.02):
                status = "converged"
            else:
                status = "step_limit"
            records[i] = RelaxRecord(ids[i], status, a, e, fm, first[k], ran, moved[k],
                                     {"engine_reported_converged": first[k] is not None,
                                      "extra_steps_run_after_convergence": 0 if first[k] is None else ran - first[k]})
    if cuda:
        acct["peak_allocated_MiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        acct["peak_reserved_MiB"] = round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)
    missing = [ids[i] for i, r in enumerate(records) if r is None]
    if missing:
        raise RuntimeError(f"no terminal record for {missing}")
    return records, acct


def _split_positions(batch, metas) -> dict:
    by_sid = _split(batch, metas, "positions")
    return {k: by_sid[m.system_id].astype(float) for k, m in enumerate(metas)}


def _converged_set(batch, conv) -> set:
    """Rows that have met the criterion so far: from the status column when present, else from ``step``'s return."""
    status = getattr(batch, "status", None)
    sid = batch.system_id.detach().cpu().numpy().ravel()
    if status is not None:
        st = status.detach().cpu().numpy().ravel()[:batch.num_graphs]
        return {int(sid[row]) for row in np.flatnonzero(st >= 1)}
    if conv is None:
        return set()
    return {int(sid[int(row)]) for row in conv.detach().cpu().numpy().ravel()}


def _failure_kind(e: Exception) -> str:
    s = f"{type(e).__name__} {e}".lower()
    if "out of memory" in s or "cuda error" in s or "alloc" in s:
        return "resource_failure"
    if "nan" in s or "inf" in s or "singular" in s:
        return "numerical_failure"
    return "invalid_geometry" if "cell" in s or "neighbor" in s else "resource_failure"
