"""prepare_bulk_refs.py + analyze_bulk_refs.py on synthetic structures and outputs (no VASP)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from ase.build import bulk
from ase import Atoms
from ase.io import write

SCRIPTS = Path(__file__).resolve().parents[1] / "examples" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_bulk_refs as P  # noqa: E402
import analyze_bulk_refs as A  # noqa: E402

DEFS = Path(__file__).resolve().parents[1] / "src" / "pvpassivator" / "data" / "POTCAR_DEFS.txt"


def make_config(tmp: Path) -> Path:
    write(tmp / "Pb.cif", bulk("Pb", "fcc", a=4.95, cubic=True))
    write(tmp / "PbO.cif", Atoms("Pb2O2", scaled_positions=[[0, .5, .24], [.5, 0, .76], [0, 0, 0], [.5, .5, 0]],
                                 cell=[4.0, 4.0, 5.0], pbc=True))
    write(tmp / "O2.vasp", Atoms("O2", positions=[[5, 5, 5], [5, 5, 6.21]], cell=[10, 11, 12], pbc=True), format="vasp")
    (tmp / "INCAR.o2").write_text("O2 test\nISPIN = 2\nNUPDOWN = 2\nIBRION = 2\nNSW = 60\n")
    cfg = {"title": "test", "physics": {"METAGGA": "SCAN", "ISPIN": 1}, "encut": 550, "encut_factor": 1.3,
           "kspacing": 0.20, "relax": {"IBRION": 2, "ISIF": 3, "NSW": 200, "EDIFFG": -0.01},
           "parallel": {"NCORE": 8, "KPAR": 4},
           "phases": [{"name": "Pb", "cif": str(tmp / "Pb.cif"), "source": "test"},
                      {"name": "PbO", "cif": str(tmp / "PbO.cif"), "source": "test"}],
           "molecule": {"name": "O2", "poscar": str(tmp / "O2.vasp"), "incar": str(tmp / "INCAR.o2"), "source": "test"},
           "variants": [{"name": "PbO_lrealF", "of": "PbO", "set": {"LREAL": ".FALSE."}}],
           "smoke": "PbO", "potcar_defs": str(DEFS),
           "cluster": {"account": "acct", "partition": "workq", "ntasks_per_node": 64, "module": "vasp6/test",
                       "potcar_dir": "/pot", "stages": {k: {"nodes": 1, "time": "01:00:00"}
                                                         for k in ("smoke", "relax", "static", "variant")}}}
    p = tmp / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def test_generator_layout_and_settings(tmp_path):
    out = tmp_path / "99_test"
    assert P.main(["--config", str(make_config(tmp_path)), "--out", str(out)]) == 0
    relax = (out / "inputs/PbO/relax/INCAR").read_text()
    static = (out / "inputs/PbO/static/INCAR").read_text()
    assert "ENCUT = 715" in relax and "ISIF = 3" in relax and "ENCUT = 550" in static and "NSW = 0" in static
    assert "LREAL = .FALSE." in (out / "inputs/PbO_lrealF/static/INCAR").read_text()
    assert "NELM = 4" in (out / "inputs/smoke/static/INCAR").read_text()
    assert (out / "inputs/PbO/relax/POTCAR.names").read_text().split() == ["Pb_d", "O"]
    assert (out / "inputs/O2/relax/EXE").read_text().strip() == "vasp_gam"
    assert "NUPDOWN = 2" in (out / "inputs/O2/relax/INCAR").read_text()
    for f in out.glob("*.slurm"):
        b = f.read_bytes()
        assert b.startswith(b"#!/bin/bash\n") and b"\r" not in b and b"--mem" not in b
    assert (out / "lists/static.txt").read_text() == "Pb Pb\nPbO PbO\n"
    with pytest.raises(SystemExit):           # refuses a non-empty output folder
        P.main(["--config", str(make_config(tmp_path)), "--out", str(out)])


def test_kmesh_follows_kspacing_rule():
    assert P.kmesh(bulk("Pb", "fcc", a=4.95, cubic=True), 0.20) == [7, 7, 7]


def fake_outputs(pkg: Path, energies: dict):
    for (job, step), e in energies.items():
        d = pkg / "outputs" / job / step
        d.mkdir(parents=True, exist_ok=True)
        (d / "OUTCAR").write_text(f"  energy  without entropy= {e:.6f}  energy(sigma->0) = {e:.6f}\n")
        (d / "DONE").write_text("x")


def test_analysis_fit_vs_none(tmp_path):
    out = tmp_path / "99_test"
    P.main(["--config", str(make_config(tmp_path)), "--out", str(out)])
    # Pb: 4 atoms at -3.5 eV; O2: -10.0 eV; PbO cell has 2 fu at E = 2*(-3.5 - 5.0 - 2.0) = -21.0
    # -> dHf(DFT) = -2.0 eV/fu; with exp -2.3 eV the fitted shift is -0.3 eV/O
    fake_outputs(out, {("Pb", "static"): -14.0, ("O2", "relax"): -10.0, ("PbO", "static"): -21.0,
                       ("PbO_lrealF", "static"): -21.004})
    exp = tmp_path / "exp.json"
    exp.write_text(json.dumps({"PbO": {"dHf_kJ_mol": -2.3 * A.KJ_PER_EV, "source": "test"}}))
    rep_file = tmp_path / "rep.json"
    A.main([str(out), "--exp", str(exp), "--fit", "PbO", "--o2-ref", "-9.99", "--json", str(rep_file)])
    rep = json.loads(rep_file.read_text())
    assert rep["per_fu"]["PbO"]["dHf_DFT_eV"] == pytest.approx(-2.0)
    assert rep["o2_shift"]["delta_eV_per_O"] == pytest.approx(-0.3)
    assert rep["per_fu"]["PbO"]["error_fit_eV"] == pytest.approx(0.0, abs=1e-12)
    assert rep["per_fu"]["PbO"]["error_none_eV"] == pytest.approx(0.3)
    assert rep["checks"]["molecule_vs_reference_meV"] == pytest.approx(-10.0)
    assert rep["checks"]["PbO_lrealF_minus_PbO_meV_per_atom"] == pytest.approx(-1.0)


def test_analysis_refuses_unsourced_and_reports_missing(tmp_path):
    out = tmp_path / "99_test"
    P.main(["--config", str(make_config(tmp_path)), "--out", str(out)])
    exp = tmp_path / "exp.json"
    exp.write_text(json.dumps({"PbO": {"dHf_kJ_mol": -218.0}}))
    with pytest.raises(SystemExit, match="no source"):
        A.main([str(out), "--exp", str(exp)])
    exp.write_text(json.dumps({"PbO": {"dHf_kJ_mol": -218.0, "source": "x"}}))
    rep_file = tmp_path / "rep.json"
    A.main([str(out), "--exp", str(exp), "--json", str(rep_file)])
    assert "PbO/static" in json.loads(rep_file.read_text())["missing"]
