"""Far-field freeze (spectator.freeze_far_field, prepare_label_set freeze_beyond_A) and the boundary-shell check."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from pvpassivator.spectator import freeze_far_field, spectator_mismatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "examples" / "scripts"))
import test_spectator as T  # noqa: E402
import prepare_label_set as P  # noqa: E402

SALT = (("C", 0.0), ("N", 1.5), ("H", 2.5), ("H", 3.5), ("Cl", -2.0))     # 18 valence electrons: closed shell
TOP = 10 + 2 * T.A + 3.0


def cx_with(s, far_shift=0.0, shell_shift=0.0, radius=10.0, far_margin=3.0, far_scramble=0.0):
    c = T.complex_from(s, salt=SALT, shift=(0, 0, 0))
    salt = np.array([(TOP, T.A + 0.3, T.A + dz) for _, dz in SALT])
    # nearest-salt-atom distance (minimum image), the same definition the check uses
    from ase.geometry import find_mic
    v = (c.positions[:, None, :] - salt[None]).reshape(-1, 3)
    d = np.linalg.norm(find_mic(v, c.cell, pbc=True)[0], axis=1).reshape(len(c), len(salt)).min(1)
    sym = np.array(c.get_chemical_symbols())
    heavy = sym != "H"
    c.positions[(d > radius + far_margin) & heavy, 1] += far_shift
    # another slab's far field: atom-by-atom differences (+/- alternating), which no rigid translation absorbs
    idx = np.flatnonzero((d > radius) & heavy)
    c.positions[idx, 1] += far_scramble * np.where(np.arange(len(idx)) % 2 == 0, 1.0, -1.0)
    c.positions[(d > radius - 2) & (d <= radius) & heavy & (d > 0.5), 2] += shell_shift
    return c


def test_freeze_snaps_far_field_onto_slab_and_keeps_molecules_whole():
    s = T.slab()
    c = cx_with(s, far_shift=0.2)
    assert not spectator_mismatch(s, c)["pass"]
    out, frozen, rep = freeze_far_field(s, c, 10.0)
    assert rep["n_frozen"] == len(frozen) > 10 and 0 < rep["snap_mean_A"] <= 0.2 and rep["n_adsorbate"] == len(SALT)
    r = spectator_mismatch(s, out, cutoff=10.0)
    assert r["pass"] and r["mean_A"] < 1e-6                   # beyond the radius the complex IS the slab now
    sym = np.array(out.get_chemical_symbols())
    for i in np.flatnonzero(sym == "N"):                       # each N-H cation frozen whole or free whole
        h = np.argmin([out.get_distance(i, j, mic=True) if sym[j] == "H" else 9e9 for j in range(len(out))])
        assert (i in frozen) == (h in frozen)
    assert not set(frozen) & {k for k in range(len(out)) if np.linalg.norm(out.positions[k] - [TOP, T.A + 0.3, T.A]) < 4}


def test_shell_check_flags_an_undecayed_response():
    s = T.slab()
    ok = spectator_mismatch(s, cx_with(s), cutoff=10.0)
    assert ok["shell"]["n"] > 0 and ok["shell"]["pass"]
    bad = spectator_mismatch(s, cx_with(s, shell_shift=0.4), cutoff=10.0)
    assert bad["pass"] and bad["shell"]["pass"] is False and bad["shell"]["mean_A"] > 0.15


def _label_set(tmp_path, cx_atoms):
    bulk = Atoms("PbI2", positions=[(0, 0, 0), (3.2, 0, 0), (0, 3.2, 0)], cell=np.diag([T.A] * 3), pbc=True)
    s = T.slab()
    ref = Atoms([e for e, _ in SALT], positions=[(0, 0, dz) for _, dz in SALT], cell=np.diag([20.0] * 3), pbc=True)
    for name, at in {"bulk": bulk, "slab": s, "cx": cx_atoms, "ref": ref}.items():
        (tmp_path / name).mkdir()
        write(tmp_path / name / "CONTCAR", at, format="vasp")
    cfg = {"normal_axis": 0, "normal_length": 40.0, "inplane_reps": [3, 3], "profile_family": "r2scan_d3bj_v1",
           "ref_box": 20.0, "max_strain": 0.03, "bulk_dir": str(tmp_path / "bulk"), "slab_dir": str(tmp_path / "slab"),
           "slab_name": "S", "slab_kpoints": [1, 1, 1], "freeze_beyond_A": 10.0,
           "cases": [{"name": "X", "complex_dir": str(tmp_path / "cx"),
                      "ref": {"kind": "file", "path": str(tmp_path / "ref" / "CONTCAR")}}]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path / "config.json"


def test_generator_writes_selective_dynamics_and_records_the_freeze(tmp_path):
    cfg = _label_set(tmp_path, cx_with(T.slab(), far_shift=0.2))
    assert P.main(["--config", str(cfg), "--out", str(tmp_path / "pkg")]) == 0
    m = json.loads((tmp_path / "pkg" / "manifest.json").read_text())
    job, frz = m["jobs"]["complex_X"], m["structures"]["complex_X"]["freeze"]
    assert job["freeze_beyond_A"] == 10.0 and job["n_frozen"] == frz["n_frozen"] > 10 and frz["snap_max_A"] > 0
    relax = (tmp_path / "pkg" / "complex_X" / "relax" / "POSCAR").read_text()
    assert "Selective dynamics" in relax and relax.count(" F   F   F") == frz["n_frozen"]
    assert "freeze_beyond_A" not in m["jobs"]["slab_S"]                         # the bare slab is never frozen
    assert "Selective dynamics" not in (tmp_path / "pkg" / "slab_S" / "relax" / "POSCAR").read_text()
    a = read(tmp_path / "pkg" / "complex_X" / "relax" / "POSCAR", format="vasp")
    assert spectator_mismatch(read(tmp_path / "pkg" / "slab_S" / "relax" / "POSCAR", format="vasp"), a)["mean_A"] < 1e-5


def test_generator_refuses_a_complex_not_built_on_this_slab(tmp_path):
    cfg = _label_set(tmp_path, cx_with(T.slab(), far_scramble=0.9))   # far field 0.9 A off, atom by atom
    with pytest.raises(ValueError, match="not built on this slab"):
        P.main(["--config", str(cfg), "--out", str(tmp_path / "pkg")])


def test_generator_complex_file_dipol_pin_and_freeze_radius_test(tmp_path):
    """A4 inputs: the complex from a settled POSCAR, a pinned DIPOL, and an extra job frozen at the test radius."""
    cfg_path = _label_set(tmp_path, cx_with(T.slab(), far_shift=0.9))           # complex_dir CONTCAR: unusable
    write(tmp_path / "settled.vasp", cx_with(T.slab()), format="vasp")          # the settled one: built on the slab
    cfg = json.loads(cfg_path.read_text())
    cfg["cases"][0]["complex_file"] = str(tmp_path / "settled.vasp")
    cfg["dipol"] = [0.5, 0.25, 0.75]
    cfg["freeze_test"] = {"radius_A": 8.0, "cases": ["X"]}
    cfg_path.write_text(json.dumps(cfg))
    assert P.main(["--config", str(cfg_path), "--out", str(tmp_path / "pkg")]) == 0
    m = json.loads((tmp_path / "pkg" / "manifest.json").read_text())
    assert m["structures"]["complex_X"]["source"] == str(tmp_path / "settled.vasp")
    assert m["structures"]["complex_X"]["freeze"]["snap_max_A"] < 1e-3
    for job in ("slab_S", "complex_X", "complex_X_R8"):
        for kind in ("relax", "static"):
            assert "DIPOL = 0.5000 0.2500 0.7500" in (tmp_path / "pkg" / job / kind / "INCAR").read_text()
    ft = m["freeze_test"]
    assert ft["radius_A"] == 8.0 and ft["cases"]["X"]["job"] == "complex_X_R8"
    assert m["jobs"]["complex_X_R8"]["freeze_beyond_A"] == 8.0 and m["jobs"]["complex_X"]["freeze_beyond_A"] == 10.0
    assert m["jobs"]["complex_X_R8"]["n_frozen"] > m["jobs"]["complex_X"]["n_frozen"]
    assert [r["name"] for r in m["reactions"]] == ["X"]                          # the test job is not a label
    assert (tmp_path / "pkg" / "complex_X_R8" / "relax" / "INCAR").read_text() == \
        (tmp_path / "pkg" / "complex_X" / "relax" / "INCAR").read_text()


def test_freeze_test_requires_cases_in_the_set(tmp_path):
    cfg_path = _label_set(tmp_path, cx_with(T.slab()))
    cfg = json.loads(cfg_path.read_text())
    cfg["freeze_test"] = {"radius_A": 8.0, "cases": ["Y"]}
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="not in the set"):
        P.main(["--config", str(cfg_path), "--out", str(tmp_path / "pkg")])
