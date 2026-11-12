"""LONI fine-tune package for the chirality labels: layout, hashes, config and script checks."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
import make_chirality_finetune_package as M  # noqa: E402

DS = ROOT / "data" / "training" / "chirality_v1"


@pytest.mark.skipif(not (DS / "dataset.json").exists(), reason="chirality_v1 dataset not exported on this machine")
def test_package_layout(tmp_path):
    fake = tmp_path / "foundation.model"
    fake.write_bytes(b"x")
    out = tmp_path / "pkg"
    M.main(["--dataset", str(DS), "--foundation", str(fake), "--out", str(out)])
    meta = json.loads((out / "package.json").read_text())
    for rel, h in meta["files_sha256"].items():
        assert M.sha256(out / rel) == h
    slurm = (out / "run_chiral_ft1.slurm").read_text(encoding="utf-8")
    assert "\r" not in slurm and "--partition=gpu2" in slurm and M.sha256(fake) in slurm
    assert subprocess.run(["bash", "-n", str(out / "run_chiral_ft1.slurm")]).returncode == 0
    yml = (out / "finetune.yaml").read_text()
    ov = json.loads(M.OVERRIDES.read_text())
    assert f"max_num_epochs: {ov['max_num_epochs']}" in yml and "E0s: foundation" in yml and "foundation_model: /ddnB/" in yml
    sums = (out / "data" / "SHA256SUMS").read_text().split("\n")
    assert len([s for s in sums if s]) == 4


def test_refuses_windows_cluster_path(tmp_path):
    with pytest.raises(SystemExit):
        M.main(["--dataset", str(DS), "--foundation", str(tmp_path), "--out", str(tmp_path / "o"),
                "--foundation-on-cluster", "C:/Program Files/Git/ddnB/x.model"])
