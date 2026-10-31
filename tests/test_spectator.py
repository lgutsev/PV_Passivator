"""Spectator consistency (spectator.py + pvpass labels): far from the salt the complex must match the bare slab."""
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from pvpassivator.labels import analyze
from pvpassivator.spectator import spectator_mismatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_labels as L  # noqa: E402

A = 6.4                       # pseudo-cubic lattice constant
CELL = np.diag([40.0, 3 * A, 3 * A])


def slab():
    """3 Pb layers along x (normal), 3x3 in-plane; I on the in-plane Pb-Pb bonds; an N-H 'cation' in each cage."""
    syms, pos = [], []
    for k in range(3):
        x = 10 + k * A
        for i in range(3):
            for j in range(3):
                y, z = i * A, j * A
                syms += ["Pb", "I", "I"]; pos += [(x, y, z), (x, y + A / 2, z), (x, y, z + A / 2)]
                if k < 2:
                    syms += ["N", "H"]; pos += [(x + A / 2, y + A / 2, z + A / 2), (x + A / 2 + 1.0, y + A / 2, z + A / 2)]
    return Atoms(syms, positions=pos, cell=CELL, pbc=True)


def complex_from(s, salt=(("C", 0.0), ("N", 1.5), ("H", 2.5), ("Cl", -2.0)), shift=(0.3, -2.1, 1.7)):
    """The slab + a salt above the top Pb layer (x ~ 25), rigidly shifted, with atoms reordered."""
    top = 10 + 2 * A
    syms = list(s.get_chemical_symbols()) + [e for e, _ in salt]
    pos = list(s.positions) + [(top + 3.0, A + 0.3, A + dz) for _, dz in salt]
    c = Atoms(syms, positions=pos, cell=CELL, pbc=True)
    c.positions += shift
    order = np.random.default_rng(0).permutation(len(c))
    return c[order]


def test_rigid_shift_and_salt_pass():
    s = slab()
    r = spectator_mismatch(s, complex_from(s))
    assert r["pass"] and r["mean_A"] < 1e-6 and r["adsorbate"] == {"C": 1, "N": 1, "H": 1, "Cl": 1}
    d = (np.array(r["translation_A"]) - [-0.3, 2.1, -1.7]) / A              # equal up to a lattice translation
    assert np.allclose(d, np.round(d), atol=1e-3) and r["n_far"] > 10


def test_far_field_distortion_fails():
    s = slab()
    c = complex_from(s, shift=(0, 0, 0))
    bottom = c.positions[:, 0] < 11                       # the layer farthest from the salt
    c.positions[bottom, 1] += 0.3
    r = spectator_mismatch(s, c)
    assert not r["pass"] and r["mean_A"] > 0.1 and r["max_A"] == pytest.approx(0.3, abs=1e-3)   # full shift, not diluted


def test_local_response_near_salt_still_passes():
    s = slab()
    c = complex_from(s, shift=(0, 0, 0))
    near = np.linalg.norm(c.positions - [10 + 2 * A + 3.0, A, A], axis=1) < 5
    c.positions[near & (np.array(c.get_chemical_symbols()) == "I")] += [0.0, 0.5, 0.0]
    assert spectator_mismatch(s, c)["pass"]


def test_iodide_counter_ion_is_found():
    s = slab()
    r = spectator_mismatch(s, complex_from(s, salt=(("C", 0.0), ("N", 1.5), ("H", 2.5), ("I", -2.5))))
    assert r["adsorbate"] == {"C": 1, "N": 1, "H": 1, "I": 1} and r["pass"]


def test_refuses_mismatched_cells_and_missing_atoms():
    s = slab()
    c = complex_from(s); c.set_cell(np.diag([40.0, 3 * A + 0.5, 3 * A]))
    with pytest.raises(ValueError, match="cells differ"):
        spectator_mismatch(s, c)
    with pytest.raises(ValueError, match="lacks atoms"):
        spectator_mismatch(s, s[1:])


@pytest.mark.parametrize("distort,flag", [(False, False), (True, True)])
def test_labels_flag_eads_on_spectator_failure(tmp_path, distort, flag):
    root = L._package(tmp_path)
    s = slab()
    c = complex_from(s, shift=(0, 0, 0))
    if distort:
        c.positions[c.positions[:, 0] < 11, 1] += 0.3
    write(root / "outputs" / "slab_X" / "relax" / "CONTCAR", s, format="vasp")
    write(root / "outputs" / "complex_PEACl" / "relax" / "CONTCAR", c, format="vasp")
    r = analyze(root)
    (row,) = r["reactions"]
    assert row["E_ads_eV"] == pytest.approx(-1.2)                       # the number is kept ...
    assert row["spectator"]["available"] and row["spectator"]["pass"] is (not flag)
    assert row["E_ads_flagged"] is flag and r["spectator_failed"] == (["PEACl"] if flag else [])


def test_labels_report_unavailable_without_contcar(tmp_path):
    (row,) = analyze(L._package(tmp_path))["reactions"]
    assert row["spectator"] == {"available": False, "reason": "slab_X/relax/CONTCAR missing"} and not row["E_ads_flagged"]
