"""Scorer for the chirality fine-tune criteria: rotor profiles and R - rac on synthetic records."""
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
import score_chirality_mlip as S  # noqa: E402


def rec(info):
    a = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    a.info.update(info)
    return a


def test_profile_uses_d1_static_as_zero_and_checks_barrier():
    zero = rec({"block": "D1", "job": "R_fixcell", "step": "static", "system": "R", "REF_energy": -10.0, "split": "train"})
    pts = [rec({"block": "D5", "system": "R", "group": "CH3", "angle_deg": a, "split": "valid",
                "REF_energy": -10.0 + 0.1 * np.sin(np.radians(a * 1.5)) ** 2}) for a in (20, 60, 100)]
    frames = [zero] + pts
    exact = [(f.info["REF_energy"], None) for f in frames]
    p = S.profiles(frames, exact)["R_D5_CH3"]
    assert p["angles"] == [0, 20, 60, 100] and p["pass"] and p["profile_rmse_meV"] == 0
    soft = [(f.info["REF_energy"] * 1.0 if f is zero else -10.0 + 0.5 * (f.info["REF_energy"] + 10.0), None) for f in frames]
    q = S.profiles(frames, soft)["R_D5_CH3"]
    assert not q["pass"] and q["barrier_model_meV"] < q["barrier_dft_meV"]


def test_r_minus_rac_adds_d3_back_and_uses_formula_units():
    frames = [rec({"block": "D1", "step": "static", "job": "R_fixcell", "system": "R", "REF_energy": -4.0, "D3_energy": -0.4}),
              rec({"block": "D1", "step": "static", "job": "rac_fixcell", "system": "rac", "REF_energy": -2.1, "D3_energy": -0.2})]
    out = S.r_minus_rac(frames, [(-4.0, None), (-2.0, None)])["fixcell"]
    # DFT per f.u.: R (-4.4)/4 = -1.1, rac (-2.3)/2 = -1.15 -> +50 meV; model: -1.1 vs -1.1 -> 0
    assert out["dft_meV_fu"] == 50.0 and out["model_meV_fu"] == 0.0 and out["pass"] is False
