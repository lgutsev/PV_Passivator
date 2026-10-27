"""export_prerelax_frames: complete, SCF-converged OUTCAR steps only; splits by job and trajectory tail; never label grade."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "examples" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import export_prerelax_frames as X  # noqa: E402

CELL = "x\n1.0\n10 0 0\n0 10 0\n0 0 10\nPb I\n1 1\nCartesian\n0 0 0\n2 0 0\n"


def step(x, e, converged=True):
    msg = (" ------------------------ aborting loop because EDIFF is reached ----------------------------------------\n"
           if converged else " ------------------------ aborting loop EDIFF was not reached (unconverged)  ----------------------\n")
    return (msg + " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
            " -----------------------------------------------------------------------------------\n"
            f"      0.00000      0.00000      0.00000         {x:.5f}      0.00000      0.00000\n"
            f"      {2 + x:.5f}      0.00000      0.00000        -{x:.5f}      0.00000      0.00000\n"
            " -----------------------------------------------------------------------------------\n"
            f"  free  energy   TOTEN  =      {e:.8f} eV\n")


def package(tmp_path, jobs):
    root = tmp_path / "pkg"
    manifest = {"common_cell_A": np.diag([10.0, 10, 10]).tolist(),
                "jobs": {j: {"role": "complex" if j != "slab" else "bare_slab"} for j in jobs}}
    manifest["jobs"]["ref_X"] = {"role": "reference"}
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))
    for j, outcars in jobs.items():
        inp = root / "inputs" / j / "relax"
        inp.mkdir(parents=True)
        (inp / "POSCAR").write_text(CELL)
        (inp / "INCAR").write_text("METAGGA = R2SCAN\nIVDW = 12\nVDW_S8 = 0.7898\nVDW_A1 = 0.4948\nVDW_A2 = 5.7308\n")
        out = root / "outputs" / j / "relax"
        out.mkdir(parents=True)
        for name, text in outcars.items():
            (out / name).write_text(text)
    return root


def test_reader_keeps_complete_converged_steps_only(tmp_path):
    text = step(0.1, -10.0) + step(0.2, -10.5, converged=False) + step(0.3, -11.0)
    text += step(0.4, -11.2).split("  free  energy")[0]          # cut off before TOTEN (walltime kill / live save)
    p = tmp_path / "OUTCAR"
    p.write_text(text)
    frames = X.outcar_frames(p, 2)
    assert [f["free_energy"] for f in frames] == [-10.0, -10.5, -11.0]
    assert [f["scf_ok"] for f in frames] == [True, False, True]
    assert np.isclose(frames[2]["forces"][1, 0], -0.3)


def test_export_splits_dedups_and_is_not_label_grade(tmp_path):
    seg1 = "".join(step(0.05 * i, -10 - i) for i in range(1, 9))
    cur = step(0.40, -18.0) + "".join(step(0.05 * i, -18 - i) for i in range(9, 13))   # first step repeats seg1's last geometry
    root = package(tmp_path, {"slab": {"OUTCAR.seg1": seg1, "OUTCAR": cur},
                              "cx_test": {"OUTCAR": "".join(step(0.013 * i, -5 - i) for i in range(1, 7))}})
    r = X.export(root, tmp_path / "out", ["cx_test"], valid_tail=0.15)
    assert r["label_grade"] is False and r["schema"] == "pvpass_prerelax/1"
    f = r["files"]
    assert f["test"]["frames"] == 6
    assert f["train"]["frames"] + f["valid"]["frames"] == 12      # 8 + 5, one restart duplicate dropped
    assert f["valid"]["frames"] == 5                               # tail of the trajectory, at least 5
    assert any("duplicate" in x["reason"] for x in r["rejected"])
    from ase.io import read
    valid = read(tmp_path / "out" / "valid.extxyz", index=":")
    assert all(a.info["job"] == "slab" and a.info["label_grade"] in (False, "False") for a in valid)
    assert min(a.info["REF_energy"] for a in valid) == pytest.approx(-30.0)   # the last, lowest-energy steps


def test_export_refuses_unknown_test_job(tmp_path):
    root = package(tmp_path, {"slab": {"OUTCAR": step(0.1, -1.0)}})
    with pytest.raises(ValueError, match="unknown test jobs"):
        X.export(root, tmp_path / "out", ["nope"])
