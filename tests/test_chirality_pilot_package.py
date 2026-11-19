"""Production pilot package (throughput benchmark + NPT): units, starting frames, hashes, script checks."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
import make_chirality_pilot_package as M  # noqa: E402

HPC = Path("D:/MLIP_Work_Folder/hpc_smoke_tests")
PK = {k: HPC / v for k, v in {"105": "105_vasp_pvpass_mba2pbi4_chirality_d0", "137": "137_mace_chiral_stab1_mba2pbi4",
                              "140": "140_mace_chiral_stab2_mba2pbi4"}.items()}


@pytest.mark.skipif(not all((p / "outputs").exists() for p in PK.values()), reason="LONI exports not on this machine")
def test_pilot_package_layout(tmp_path):
    fake = tmp_path / "m.model"
    fake.write_bytes(b"x")
    out = tmp_path / "pkg"
    M.main(["--pkg105", str(PK["105"]), "--pkg137", str(PK["137"]), "--pkg140", str(PK["140"]), "--model-local", str(fake),
            "--model-cluster", "/work/x/m.model", "--out", str(out)])
    meta = json.loads((out / "package.json").read_text())
    assert list(meta["units"]) == ["bench", "npt_R_358K", "npt_rac_358K"]
    for rel, h in meta["files_sha256"].items():
        assert M.sha256(out / rel) == h
    slurm = (out / "run_chiral_pilot.slurm").read_text(encoding="utf-8")
    assert "\r" not in slurm and "--array=0-2" in slurm and "bench_npt.py" in slurm and M.sha256(fake) in slurm
    assert subprocess.run(["bash", "-n", str(out / "run_chiral_pilot.slurm")]).returncode == 0
    r = read(out / "inputs" / "npt_R_358K" / "structure.extxyz")
    assert len(r) == 752 and r.get_momenta().any() and meta["units"]["npt_R_358K"]["start_step"] == 60000
    rac = read(out / "inputs" / "npt_rac_358K" / "structure.extxyz")
    assert len(rac) == 376 and meta["units"]["npt_rac_358K"]["start_step"] == 54000   # before the 137 duplicated tail
    fr = read(out / "inputs" / "bench" / "frames_rac.extxyz", index=":")
    assert len(fr) == 20 and max(int(a.info["step"]) for a in fr) <= 54000
    assert b"\r" not in (out / "inputs" / "bench" / "frames_R.extxyz").read_bytes()
    assert '"stress"' in (out / "bench_npt.py").read_text() and "use_stress=len(sys.argv) > 5" in (out / "d3_client.py").read_text()
