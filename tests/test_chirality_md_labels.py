"""Label tranche B stager on synthetic MD frames made from the X-ray cells (stand-ins for the 105 minima)."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
import prepare_chirality_md_labels as B  # noqa: E402

SDIR = ROOT / "examples" / "structures" / "mba2pbi4"


@pytest.fixture(scope="module")
def base():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"R": read(SDIR / "R-MBA2PbI4_COD7203475.cif"), "rac": read(SDIR / "rac-MBA2PbI4_COD2201646.cif")}


def frames(base, tmp_path, bad=False):
    d = tmp_path / "md"
    d.mkdir()
    for key, kind, split, n in (("R", "d8", "train_pool", 3), ("rac", "d9", "heldout_test", 2)):
        run = f"{key}_293K_{kind}_s1"
        for i in range(n):
            at = base[key].copy()
            at.positions += np.random.default_rng(i).normal(scale=0.02, size=at.positions.shape)
            if bad and key == "R" and i == 1:
                at.set_cell(at.cell.array * 1.01, scale_atoms=True)
            at.info.update(run=run, system=key, T_K=293, time_fs=2100.0 + 100 * i, seed=1, split=split, generator="test")
            write(d / f"{run}.extxyz", at, format="extxyz", append=True)
        (d / f"{run}.DONE").write_text(json.dumps({"run": run, "status": "ok", "mx_cut_A": 4.3}))
    return d


def test_collect_keeps_runs_apart_and_refuses_bad_frames(base, tmp_path):
    s, m, refused = B.collect(frames(base, tmp_path), base)
    assert len(s) == 5 and not refused
    assert {m[n]["block"] for n in s if n.startswith("R_")} == {"D8"}
    assert {m[n]["split"] for n in s if n.startswith("rac_")} == {"heldout_test"}
    assert all(m[n]["mx_cut_A"] == 4.3 for n in s)
    (tmp_path / "b").mkdir()
    s2, _, refused2 = B.collect(frames(base, tmp_path / "b", bad=True), base)
    assert len(s2) == 4 and refused2[0]["reason"].startswith("cell differs")
