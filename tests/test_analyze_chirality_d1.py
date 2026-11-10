"""analyze_chirality_d1.py on a SYNTHETIC export: the X-ray R/rac cells stand in for start and 'relaxed' geometry."""
import sys
import warnings
from pathlib import Path

import pytest
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
import analyze_chirality_d1 as D  # noqa: E402

SDIR = ROOT / "examples" / "structures" / "mba2pbi4"


def test_relax_report(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R, rac = read(SDIR / "R-MBA2PbI4_COD7203475.cif"), read(SDIR / "rac-MBA2PbI4_COD2201646.cif")
    for job, at in (("R_fixcell", R), ("rac_fullcell", rac)):
        (tmp_path / "inputs" / job / "relax").mkdir(parents=True)
        (tmp_path / "outputs" / job / "relax").mkdir(parents=True)
        write(tmp_path / "inputs" / job / "relax" / "POSCAR", at, format="vasp")
        fin = at.copy()
        fin.set_cell(fin.cell * 0.98, scale_atoms=True)
        write(tmp_path / "outputs" / job / "relax" / "CONTCAR", fin, format="vasp")
        (tmp_path / "outputs" / job / "relax" / "DONE").write_text("x")
    r = D.analyze(tmp_path)
    assert r["R_fullcell"]["status"] == "missing"
    assert r["R_fixcell"]["status"] == "intact" and r["R_fixcell"]["mirror_exact"]
    assert r["R_fixcell"]["volume_change_pct"] == pytest.approx(100 * (0.98 ** 3 - 1), abs=0.01)
    assert r["rac_fullcell"]["relaxed"]["stereocentres"] == {"R": 2, "S": 2}
