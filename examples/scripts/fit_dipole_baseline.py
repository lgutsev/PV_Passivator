"""Offline baseline for the slab-normal dipole: environment-class point charges, fitted to existing labels.

    python examples/scripts/fit_dipole_baseline.py dipoles.extxyz [--test complex_TFPAI complex_THCl] [--json out.json]

Input: extract_slab_dipoles.py output (info: job, dipole_eA, dV_eV, area_A2, idipol). Model:
    p = sum_i q_{c(i)} * (x_i - <x>)       x along the IDIPOL axis, <x> = the frame's mean (uniform compensating charge: neutral)
with an element charge plus a correction per environment class c = element + sorted bonded-neighbour elements (ASE natural cutoffs x 1.2), so
e.g. I bonded to two Pb, I bonded to one Pb and a free iodide are different classes. Ridge fit, no intercept.
This is the physical baseline any learned model (PET or otherwise) has to beat; it is exactly invariant to rotations
about the slab normal and odd under the mirror x -> -x by construction.

Reports, on whole held-out jobs (never on frames of a training job):
  - absolute error of dV per frame (eV), and of the change from each job's first frame;
  - dV_ads = dV(final complex) - dV(final bare slab), predicted vs DFT;
  - leave-one-job-out Spearman rank of the final dV over all jobs;
  - persistence baseline (predict each frame = the job's first-frame DFT value) for the trajectory changes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs

E_OVER_EPS0 = 180.951


def env_classes(atoms) -> list[str]:
    nl = NeighborList(natural_cutoffs(atoms, mult=1.2), self_interaction=False, bothways=True)
    nl.update(atoms)
    sym = atoms.get_chemical_symbols()
    return [f"{s}:{''.join(sorted(sym[j] for j in nl.get_neighbors(i)[0]))}" for i, s in enumerate(sym)]


def with_elements(classes: list[str]) -> list[tuple[str, str]]:
    """Each atom carries an element charge plus an environment correction; an unseen environment falls back to the
    element charge (its correction column is absent from training, ridge keeps unseen weights at 0)."""
    return [(c.split(":")[0], c) for c in classes]


def features(frames, classes_of_job) -> tuple[list[str], np.ndarray]:
    """Per frame, per class: sum of (x - <x>). Measuring x from the frame's mean position is the same as adding a
    uniform compensating charge, so a frame stays neutral (origin-independent) even with unseen or zero charges.
    Classes are fixed per job (taken from its first frame)."""
    rows = []
    for at in frames:
        ax = at.info["idipol"] - 1
        x = at.positions[:, ax] - at.positions[:, ax].mean()
        row = defaultdict(float)
        for (el, c), xi in zip(with_elements(classes_of_job[at.info["job"]]), x):
            row[el] += xi
            row[c] += xi
        rows.append(row)
    names = sorted({k for r in rows for k in r})
    return names, np.array([[r.get(k, 0.0) for k in names] for r in rows])


def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("extxyz", type=Path)
    ap.add_argument("--test", nargs="*", default=["complex_TFPAI", "complex_THCl"])
    ap.add_argument("--slab", default="slab_FAI_Schot")
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    frames = [f for f in read(a.extxyz, index=":") if f.info["step"] == "relax"]
    jobs = sorted({f.info["job"] for f in frames})
    first = {j: next(f for f in frames if f.info["job"] == j) for j in jobs}
    classes_of_job = {j: env_classes(first[j]) for j in jobs}
    names, X = features(frames, classes_of_job)
    y = np.array([f.info["dipole_eA"] for f in frames])
    area = np.array([f.info["area_A2"] for f in frames])
    job = np.array([f.info["job"] for f in frames])
    to_dV = E_OVER_EPS0 / area

    def fit_predict(train_mask):
        seen = np.abs(X[train_mask]).sum(axis=0) > 0      # columns of classes absent from training get weight 0
        w = np.zeros(X.shape[1])
        w[seen] = ridge(X[train_mask][:, seen], y[train_mask], a.lam)
        return X @ w, w

    out = {"n_frames": int(len(y)), "jobs": {j: int((job == j).sum()) for j in jobs}, "classes": names,
           "dft_dV_range_eV": [float((y * to_dV).min()), float((y * to_dV).max())]}

    # 1) production split: whole test jobs held out
    test = np.isin(job, a.test)
    pred, w = fit_predict(~test)
    dV, dVp = y * to_dV, pred * to_dV
    per = {}
    for j in jobs:
        m = job == j
        d0, p0 = dV[m][0], dVp[m][0]
        per[j] = {"split": "test" if j in a.test else "train",
                  "frames": int(m.sum()),
                  "dft_dV_first_last": [float(d0), float(dV[m][-1])],
                  "pred_dV_first_last": [float(p0), float(dVp[m][-1])],
                  "mae_abs_eV": float(np.abs(dVp[m] - dV[m]).mean()),
                  "mae_change_eV": float(np.abs((dVp[m] - p0) - (dV[m] - d0)).mean()),
                  "dft_change_span_eV": float(np.ptp(dV[m] - d0)),
                  "persistence_mae_change_eV": float(np.abs(dV[m] - d0).mean())}
    out["production_split"] = {
        "test_jobs": a.test, "per_job": per,
        "test_mae_abs_eV": float(np.abs(dVp[test] - dV[test]).mean()),
        "train_mae_abs_eV": float(np.abs(dVp[~test] - dV[~test]).mean()),
        "charges_e": dict(zip(names, map(float, w)))}

    # 2) leave-one-job-out: final-frame dV, ordering and dV_ads relative to the bare slab
    final_dft, final_pred = {}, {}
    for j in jobs:
        p, _ = fit_predict(job != j)
        last = np.where(job == j)[0][-1]
        final_dft[j], final_pred[j] = float(dV[last]), float(p[last] * to_dV[last])
    cx = [j for j in jobs if j != a.slab]
    out["lojo"] = {
        "final_dV_eV": {j: {"dft": final_dft[j], "pred": final_pred[j]} for j in jobs},
        "mae_final_dV_eV": float(np.mean([abs(final_pred[j] - final_dft[j]) for j in jobs])),
        "spearman_final_dV": spearman([final_dft[j] for j in jobs], [final_pred[j] for j in jobs]),
        "dV_ads_eV": {j: {"dft": final_dft[j] - final_dft[a.slab], "pred": final_pred[j] - final_dft[a.slab]}
                      for j in cx if a.slab in final_dft},
        "spearman_dV_ads": spearman([final_dft[j] for j in cx], [final_pred[j] for j in cx])}

    # 3) symmetry checks of the model on one test frame: rotation about the normal (invariant), mirror (odd)
    at = next(f for f in frames if f.info["job"] in a.test).copy()
    ax = at.info["idipol"] - 1
    q = np.array([sum(w[names.index(k)] for k in pair if k in names)
                  for pair in with_elements(classes_of_job[at.info["job"]])])
    xm = at.positions[:, ax].mean()
    p0 = float(q @ (at.positions[:, ax] - xm))
    rot = at.copy()
    rot.rotate(37.0, v=np.eye(3)[ax], center=tuple(np.eye(3)[ax] * xm))
    p_rot = float(q @ (rot.positions[:, ax] - rot.positions[:, ax].mean()))
    p_mir = float(q @ (-(at.positions[:, ax] - xm)))
    out["symmetry"] = {"p": p0, "p_rotated_about_normal": p_rot, "p_mirrored": p_mir}

    txt = json.dumps(out, indent=1)
    if a.json:
        a.json.write_text(txt)
    print(json.dumps({k: out[k] for k in ("n_frames", "dft_dV_range_eV")}))
    ps = out["production_split"]
    print(f"production split: train MAE {ps['train_mae_abs_eV']*1e3:.1f} meV, test MAE {ps['test_mae_abs_eV']*1e3:.1f} meV")
    for j, r in ps["per_job"].items():
        print(f"  {j:16s} {r['split']:5s} n={r['frames']:3d} DFT dV {r['dft_dV_first_last'][0]:+.3f}->{r['dft_dV_first_last'][1]:+.3f}"
              f"  pred {r['pred_dV_first_last'][0]:+.3f}->{r['pred_dV_first_last'][1]:+.3f}"
              f"  MAE abs {r['mae_abs_eV']*1e3:6.1f}  change {r['mae_change_eV']*1e3:5.1f} (persist {r['persistence_mae_change_eV']*1e3:5.1f}) meV")
    lj = out["lojo"]
    print(f"LOJO final dV MAE {lj['mae_final_dV_eV']*1e3:.1f} meV, Spearman {lj['spearman_final_dV']:.2f}; "
          f"dV_ads Spearman {lj['spearman_dV_ads']:.2f}")
    print("symmetry", out["symmetry"])


if __name__ == "__main__":
    main()
