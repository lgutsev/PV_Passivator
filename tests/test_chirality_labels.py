"""Label tranche A generator (examples/scripts/prepare_chirality_labels.py) on the X-ray cells as stand-in minima."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
import prepare_chirality_labels as L  # noqa: E402

from pvpassivator import chirality as ch  # noqa: E402

SDIR = ROOT / "examples" / "structures" / "mba2pbi4"


@pytest.fixture(scope="module")
def base():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"R": read(SDIR / "R-MBA2PbI4_COD7203475.cif"), "rac": read(SDIR / "rac-MBA2PbI4_COD2201646.cif")}


def test_rotor_groups_identify_mba(base):
    at = base["R"]
    mol = ch.build_topology(at).molecules[0]
    g = L.rotor_groups(at, mol)
    sym = at.get_chemical_symbols()
    assert len(g["NH3"][2]) == 3 and all(sym[i] == "H" for i in g["NH3"][2])
    assert len(g["CH3"][2]) == 3
    ring = g["ring"][2]
    assert sorted(sym[i] for i in ring).count("C") == 5 and len(ring) == 10   # 5 ring C besides ipso + 5 H


def test_rigid_rotation_keeps_geometry_and_configuration(base):
    at = base["R"]
    top = ch.build_topology(at)
    mol = top.molecules[0]
    for name, (piv, ax, mov) in L.rotor_groups(at, mol).items():
        same = L.rotate_group(at, mol, piv, ax, mov, 0.0)
        np.testing.assert_allclose(ch.mic(at, same.positions - at.positions), 0, atol=1e-9)
        s = L.rotate_group(at, mol, piv, ax, mov, 47.0)
        u0, u1 = ch.unwrap_molecule(at, mol), ch.unwrap_molecule(s, mol)
        idx = {a: k for k, a in enumerate(mol.atoms)}
        # distances from the axis atom are unchanged for every moved atom (rigid rotation)
        for i in mov:
            assert np.linalg.norm(u1[idx[i]] - u1[idx[ax]]) == pytest.approx(np.linalg.norm(u0[idx[i]] - u0[idx[ax]]), abs=1e-9)
        assert set(ch.stereocentres(top, s)) == {"R"}
        moved = np.linalg.norm(ch.mic(at, s.positions - at.positions), axis=1)
        assert np.all(moved[np.setdiff1d(np.arange(len(at)), mov)] < 1e-9) and moved[mov].max() > 0.1


def test_strain_and_distortion(base):
    at = base["rac"]
    s = L.inplane_strain(at, 0.02)
    np.testing.assert_allclose(s.cell.cellpar()[:2], at.cell.cellpar()[:2] * 1.02)
    assert s.cell.cellpar()[2] == pytest.approx(at.cell.cellpar()[2])
    d = L.framework_distortion(at, 0.1, 0)
    sym = np.array(at.get_chemical_symbols())
    moved = np.linalg.norm(ch.mic(at, d.positions - at.positions), axis=1)
    assert np.all(moved[~np.isin(sym, ["Pb", "I"])] < 1e-9) and moved[sym == "I"].mean() > 0.05


def test_build_structures_counts_and_contacts(base):
    structs, dropped, meta = L.build_structures(base)
    blocks = {}
    for n in list(structs) + dropped:
        blocks[meta[n]["block"]] = blocks.get(meta[n]["block"], 0) + 1
    assert blocks == {"D3": 26, "D4": 20, "D5": 7, "D6": 12, "D7": 8}        # before contact filtering
    assert all(meta[n]["min_contact_A"] < 1.5 for n in dropped)
    assert all(meta[n].get("min_contact_A", 9) >= 1.5 for n in structs)
    assert "R_D3_NH3_000" in structs and "R_D5_CH3_000" in structs          # the 0-degree points never clash


def test_stage_writes_label_package(base, tmp_path):
    pkg = tmp_path / "pkg105"
    for key, at in base.items():
        d = pkg / "outputs" / f"{key}_fixcell" / "relax"
        d.mkdir(parents=True)
        write(d / "CONTCAR", at, format="vasp")
        (d / "DONE").write_text("x")
    cfg = {"title": "t", "kspacing": 0.25, "encut": 520, "encut_factor": 1.3,
           "parallel": {"default": {"NCORE": 16, "KPAR": 1}},
           "cluster": {"account": "a", "partition": "p", "ntasks_per_node": 64, "module": "m", "potcar_dir": "/pot",
                       "stages": {"check": {"nodes": 1, "time": "02:00:00"}, "labels": {"nodes": 1, "time": "02:00:00"}}}}
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    out = tmp_path / "labels_pkg"
    L.main(["--pkg105", str(pkg), "--config", str(tmp_path / "c.json"), "--out", str(out)])
    rows = (out / "lists" / "labels.txt").read_text().split("\n")[:-1]
    man = json.loads((out / "manifest.json").read_text())
    assert len(rows) == 73 - len(man["settle"]["dropped_for_contact"])
    t = (out / "inputs" / rows[0].split()[0] / "static" / "INCAR").read_text()
    assert "LREAL = .FALSE." in t and "ENCUT = 520" in t and "ISYM = 0" in t and "NSW = 0" in t
    assert sorted(f.name for f in out.glob("*.slurm")) == ["run_labels.slurm"]
