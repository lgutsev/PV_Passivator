"""MD-stability package generator: units, hashes, script checks, and the runner's stop-condition checks."""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
sys.path.insert(0, str(ROOT / "examples" / "chirality" / "stability"))
import make_chirality_stability_package as M  # noqa: E402
import run_stability as RS  # noqa: E402

from pvpassivator import chirality as ch  # noqa: E402

PKG105 = Path("D:/MLIP_Work_Folder/hpc_smoke_tests/105_vasp_pvpass_mba2pbi4_chirality_d0")
SDIR = ROOT / "examples" / "structures" / "mba2pbi4"


@pytest.mark.skipif(not (PKG105 / "outputs").exists(), reason="package 105 export not on this machine")
def test_package_layout(tmp_path):
    fake = tmp_path / "m.model"
    fake.write_bytes(b"x")
    out = tmp_path / "pkg"
    M.main(["--pkg105", str(PKG105), "--model-local", str(fake), "--model-cluster", "/work/x/m.model", "--out", str(out)])
    meta = json.loads((out / "package.json").read_text())
    for rel, h in meta["files_sha256"].items():
        assert M.sha256(out / rel) == h
    assert {n: u["natoms"] for n, u in meta["units"].items()} == {"R_358K": 752, "rac_358K": 376}
    slurm = (out / "run_chiral_stab.slurm").read_text(encoding="utf-8")
    assert "\r" not in slurm and "--array=0-1" in slurm and M.sha256(fake) in slurm
    assert subprocess.run(["bash", "-n", str(out / "run_chiral_stab.slurm")]).returncode == 0
    for line in (out / "SHA256SUMS").read_text().splitlines():
        h, f = line.split("  ")
        assert M.sha256(out / f) == h
    assert b"\r" not in (out / "inputs" / "R_358K" / "structure.extxyz").read_bytes()


def test_runner_does_not_rerun_a_unit_with_a_verdict(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "DONE").write_text("STOP at step 41000 (20.50 ps, nve): topology\n")
    monkeypatch.setattr(sys, "argv", ["run_stability.py", str(tmp_path / "in"), str(out), str(tmp_path / "runs"),
                                      "31800", "python"])
    RS.main()   # returns before reading inputs (tmp_path/in does not exist)
    assert (out / "DONE").read_text().startswith("STOP") and not (out / "FAILED").exists()


def test_checks_flag_proton_transfer_and_pass_clean_structure():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at = read(SDIR / "rac-MBA2PbI4_COD2201646.cif")
    p = dict(M.PROTOCOL)
    top = ch.build_topology(at, mx_cut=p["mx_cut_A"], covalent_scale=p["covalent_scale"])
    labels = ch.stereocentres(top, at)
    assert RS.checks(at, top, labels, len(top.molecules), ch, p, np) is None
    bad = at.copy()
    h = top.molecules[0].nh_h[0]
    iod = np.where(np.array(at.get_chemical_symbols()) == "I")[0]
    j = iod[np.argmin(at.get_distances(h, iod, mic=True))]
    v = at.get_distance(h, j, mic=True, vector=True)
    bad.positions[h] += v * (1 - 1.7 / np.linalg.norm(v))       # move this H to 1.7 A from its nearest I
    why = RS.checks(bad, top, labels, len(top.molecules), ch, p, np)
    assert why is not None and ("proton transfer" in why or "topology" in why)
