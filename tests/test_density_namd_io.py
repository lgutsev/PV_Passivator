from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from pvpassivator.density import Grid, charge_displacement, GridMismatch
from pvpassivator.namd_adapter import (NamdExportConfig, export_frames, select_window,
                                       FrameSelectionError, check_nac_alignment)
from pvpassivator.vasp_io import parse_lite

CELL = np.diag([4.0, 5.0, 20.0])


def _gauss(shape, z0, w, q):
    n = shape[2]
    z = np.arange(n) * 20.0 / n
    prof = np.exp(-((z - z0) ** 2) / (2 * w ** 2))
    prof = prof / (prof.sum() * 20.0 / n) * q / (4 * 5)   # electrons/A^3 integrating to q
    return np.broadcast_to(prof, shape).copy()


def test_cd_synthetic_dipole():
    shape = (8, 10, 200)
    base = _gauss(shape, 10, 2, 50.0)
    shift = _gauss(shape, 12, 0.5, 0.3) - _gauss(shape, 8, 0.5, 0.3)  # 0.3 e moved +z
    cplx = Grid(base + shift, CELL, nelect=50.0, label="cplx")
    frag = Grid(base, CELL, nelect=50.0, label="frag")
    r = charge_displacement(cplx, [frag], axis=2, check=False)
    assert r.area == pytest.approx(20.0)
    assert r.integral_total == pytest.approx(0.0, abs=1e-8)
    assert r.cd[np.argmin(abs(r.z - 10))] == pytest.approx(-0.3, abs=1e-3)  # depletion below
    assert r.region(10, 16) == pytest.approx(0.3, abs=1e-3)
    assert not r.warnings
    assert cplx.total_electrons() == pytest.approx(50.0)


def test_cd_flags_electron_exchange_and_grid_mismatch():
    shape = (8, 10, 200)
    cplx = Grid(_gauss(shape, 10, 2, 51.0), CELL, nelect=51.0)
    frag = Grid(_gauss(shape, 10, 2, 50.0), CELL, nelect=50.0)
    r = charge_displacement(cplx, [frag], check=False)
    assert any("electron exchange" in w for w in r.warnings)
    with pytest.raises(GridMismatch):
        charge_displacement(cplx, [Grid(np.zeros((8, 10, 100)), CELL)])


def _traj(n, dup_at=None):
    rng = np.random.default_rng(0)
    base = Atoms("PbIIHC", positions=rng.random((5, 3)) * 5, cell=[6, 6, 6], pbc=True)
    frames = []
    for i in range(n):
        a = base.copy()
        a.positions += 0.01 * i
        if dup_at is not None and i == dup_at:
            a.positions = frames[-1].positions.copy()
        frames.append(a)
    return frames


def test_window_too_short_raises():
    with pytest.raises(FrameSelectionError, match="never repeated"):
        select_window(100, NamdExportConfig("x", "y", n_frames=2000))
    assert select_window(10, NamdExportConfig("x", "y", n_frames=3, stride=2)) == [5, 7, 9]


def test_export_atom_map_and_manifest(tmp_path):
    frames = _traj(10)
    cfg = NamdExportConfig("none", str(tmp_path / "out"), n_frames=4, timestep_fs=1.0)
    m = export_frames(cfg, frames)
    assert [f["source_frame"] for f in m["frames"]] == [6, 7, 8, 9]
    perm = m["atom_map"]["export_index_to_source_index"]
    from ase.io import read
    a = read(tmp_path / "out" / "1" / "POSCAR")
    assert a.get_chemical_symbols() == [frames[0][i].symbol for i in perm]
    assert np.allclose(a.positions, frames[6].positions[perm])
    with pytest.raises(FileExistsError):
        export_frames(cfg, frames)


def test_duplicate_frame_rejected(tmp_path):
    with pytest.raises(FrameSelectionError, match="identical"):
        export_frames(NamdExportConfig("none", str(tmp_path / "o"), n_frames=5), _traj(6, dup_at=3))


def test_nac_alignment():
    assert check_nac_alignment(2000, 2000, 1999) == []
    assert check_nac_alignment(1999, 1999, 1999)  # ambiguous -> reported


def test_parse_lite_oszicar(tmp_path):
    (tmp_path / "INCAR").write_text("IBRION = 2\nNSW = 99\nIVDW = 11\n")
    (tmp_path / "POSCAR").write_text("x\n1\n1 0 0\n0 1 0\n0 0 1\nPb I\n1 2\nDirect\n")
    (tmp_path / "OSZICAR").write_text(
        "       N       E\nDAV:   1 ...\n   1 F= -.10000000E+02 E0= -.10000000E+02  d E =-.1E+02\n"
        "  99 F= -.24180321E+04 E0= -.24180300E+04  d E =-.135047E-02\n")
    r = parse_lite(tmp_path)
    assert r.energy_F == pytest.approx(-2418.0321) and r.status == "hit_nsw"
    assert r.composition == {"Pb": 1, "I": 2}
