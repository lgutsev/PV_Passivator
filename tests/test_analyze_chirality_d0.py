"""Decision logic of examples/scripts/analyze_chirality_d0.py on SYNTHETIC energies/forces (no VASP output)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
import analyze_chirality_d0 as A  # noqa: E402


def _at(sym, e, f):
    a = Atoms(sym, positions=np.zeros((len(sym), 3)))
    a.calc = SinglePointCalculator(a, energy=e, forces=np.asarray(f, float))
    return a


@pytest.fixture()
def base():
    rng = np.random.default_rng(0)
    symR, symr = ["Pb", "I", "N", "H"] * 4, ["Pb", "I", "N", "H"] * 2
    fR, fr = rng.normal(scale=0.1, size=(16, 3)), rng.normal(scale=0.1, size=(8, 3))
    jobs = {
        "R_kref": _at(symR, -100.0, fR), "R_kdown": _at(symR, -100.01, fR + 0.01),
        "R_kup": _at(symR, -100.0004, fR + 0.001), "R_kref_e600": _at(symR, -100.02, fR + 0.002),
        "Rmirror_kref": _at(symR, -100.0000001, -fR), "rac_kref": _at(symr, -50.0, fr),
        "rac_kup": _at(symr, -50.0002, fr + 0.001),
        "R_kref_soc": _at(symR, -101.0, fR + 0.005), "rac_kref_soc": _at(symr, -50.5, fr + 0.005),
    }
    return jobs, symR


def test_all_pass_keeps_staged_meshes(base):
    jobs, _ = base
    r = A.analyze(jobs)
    assert r["kmesh_pass"] and r["encut_pass"] and r["mirror"]["pass"]
    assert r["checks"]["R_kdown_vs_kup_fmax_meV_A"] > 5           # the coarse mesh is visibly off
    assert r["checks"]["dE_R_rac_mesh_change_meV_fu"] == pytest.approx(0.0, abs=1e-6)
    assert not r["soc"]["R_kref"]["soc_needed_for_structure"]
    assert r["recommendation"].startswith("keep the staged meshes")


def test_mirror_failure_and_mesh_failure_are_flagged(base):
    jobs, symR = base
    jobs["Rmirror_kref"] = _at(symR, -99.99, -jobs["R_kref"].get_forces() + 0.02)
    jobs["R_kup"] = _at(symR, -100.0, jobs["R_kref"].get_forces() + 0.01)
    r = A.analyze(jobs)
    assert not r["mirror"]["pass"] and not r["kmesh_pass"]
    assert "denser mesh" in r["recommendation"]


def test_soc_flag_on_lead_or_iodide(base):
    jobs, symR = base
    f = jobs["R_kref"].get_forces().copy()
    f[0] += [0.05, 0, 0]                     # atom 0 is Pb
    jobs["R_kref_soc"] = _at(symR, -101.0, f)
    r = A.analyze(jobs)
    assert r["soc"]["R_kref"]["soc_needed_for_structure"]
    assert r["soc"]["R_kref"]["max_dF_meV_A_by_element"]["Pb"] == pytest.approx(50.0, abs=0.1)


def test_missing_jobs_are_reported_not_filled(base):
    jobs, _ = base
    jobs["R_kup"] = None
    r = A.analyze(jobs)
    assert r["missing"] == ["R_kup"] and not r["kmesh_pass"]
    assert r["recommendation"].startswith("incomplete")


def test_timing_parser(tmp_path):
    p = tmp_path / "OUTCAR"
    p.write_text("      LOOP+:  cpu time    250.10: real time    251.20\n"
                 "      LOOP+:  cpu time    10.0: real time    12.5\n"
                 "                  Elapsed time (sec):      601.123\n")
    t = A.timing(p)
    assert t == {"loop_plus_s": [251.2, 12.5], "elapsed_s": 601.123}
