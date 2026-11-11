"""Chirality fine-tuning exporter: split policy and (when the export is on this machine) the 105 relax frames."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
import export_chirality_training as X  # noqa: E402

RULES = json.loads(X.SPLITS.read_text())["rules"]
STAGED_113 = Path("C:/Users/lguts/OneDrive/Desktop/Research/2026_LATech/LONI_staging/chiral_labels_a_vasp_mba2pbi4")
PKG105 = Path("D:/MLIP_Work_Folder/hpc_smoke_tests/105_vasp_pvpass_mba2pbi4_chirality_d0")


@pytest.mark.parametrize("meta,split", [
    ({"block": "D3", "system": "rac", "angle_deg": 40}, "test"),
    ({"block": "D3", "system": "R", "angle_deg": 40}, "train"),
    ({"block": "D9", "system": "R"}, "test"),
    ({"block": "D5", "system": "R", "angle_deg": 60}, "valid"),
    ({"block": "D6", "system": "R", "sigma_A": 0.1, "seed": 3}, "valid"),
    ({"block": "D6", "system": "R", "sigma_A": 0.1, "seed": 0}, "train"),
    ({"block": "D7", "system": "rac", "inplane_strain": -0.01}, "valid"),
    ({"block": "D7", "system": "R", "inplane_strain": 0.02}, "train"),
    ({"block": "D1", "system": "R"}, "train"),
])
def test_split_rules(meta, split):
    assert X.assign_split(meta, RULES) == split


@pytest.mark.skipif(not (STAGED_113 / "manifest.json").exists(), reason="staged package 113 not on this machine")
def test_split_counts_on_package_113_manifest():
    man = json.loads((STAGED_113 / "manifest.json").read_text())
    pts = man["settle"]["points"]
    counts = {}
    for j in man["jobs"]:
        s = X.assign_split(pts[j["job"]], RULES)
        counts[s] = counts.get(s, 0) + 1
    assert counts == {"train": 35, "valid": 14, "test": 13}


@pytest.mark.skipif(not (PKG105 / "manifest.json").exists(), reason="package 105 export not on this machine")
def test_relax_frames_from_105(tmp_path):
    rep = X.export([PKG105], [], tmp_path / "ds", relax_stride=10, subtract_d3=False)
    assert rep["files"]["train"]["frames"] > 10 and rep["files"]["valid"]["frames"] == 0
    assert set(rep["files"]["train"]["by_config_type"]) == {"D1_R", "D1_rac"}
    assert not rep["ready_for_pilot"]
    # the fixed-cell statics supersede their relax endpoints; the free-cell statics (520 eV) are added
    assert sorted(d["kept"].split("/", 1)[1] for d in rep["duplicates"]) == ["R_fixcell/static", "rac_fixcell/static"]
    assert not any("fullcell" in r["job"] for r in rep["rejected"])
    assert rep["reference_fingerprint"]["LREAL"] in ("FALSE", ".FALSE.", "F")
