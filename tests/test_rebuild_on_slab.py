"""rebuild_on_slab.py (package-53 rebuild plan, step A2): transplant a salt onto a relaxed slab."""
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

from pvpassivator.spectator import spectator_mismatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "examples" / "scripts"))
import test_spectator as T  # noqa: E402
import rebuild_on_slab as R  # noqa: E402

SALT = (("C", 0.0), ("N", 1.5), ("H", 2.5), ("Cl", -2.6))     # Cl beyond the C-Cl bond cutoff: a separate anion
SALT_POS = np.array([(10 + 2 * T.A + 3.0, T.A + 0.3, T.A + dz) for _, dz in SALT])
LOCAL = np.array([0.0, 0.25, -0.15])                            # how the relaxed slab moved around the salt


def archived():
    """Complex built on the UNRELAXED slab, rigidly shifted and reordered (as in a separate archived run)."""
    s = T.slab()
    return s, T.complex_from(s, salt=SALT, shift=(0.3, -2.1, 1.7))


def relaxed(s, rng=0, noise=0.05):
    """The 'converged' slab: small random relaxation everywhere, plus a rigid shift of the region near the salt."""
    r = s.copy()
    r.positions += np.random.default_rng(rng).normal(0, noise, r.positions.shape)
    near = np.linalg.norm(r.positions - SALT_POS[1], axis=1) < 9.0
    r.positions[near] += LOCAL
    return r


def test_rebuild_keeps_order_contacts_and_matches_slab():
    s, cx = archived()
    rs = relaxed(s)
    out, rep = R.rebuild(rs, cx)
    assert rep["pass"] and rep["status"] == "pass", json.dumps(rep["checks"], indent=1)
    assert out.get_chemical_symbols() == cx.get_chemical_symbols()                 # POSCAR blocks / POTCAR order kept
    assert np.allclose(out.cell.array, rs.cell.array)
    assert rep["salt"] == {"C": 1, "Cl": 1, "H": 1, "N": 1}
    assert spectator_mismatch(rs, out)["mean_A"] < 1e-6                             # far field IS the slab
    ads = rep["salt_indices"]
    # the salt keeps its internal geometry ...
    v_old = cx.get_distances(ads[0], ads, mic=True, vector=True)
    v_new = out.get_distances(ads[0], ads, mic=True, vector=True)
    assert np.allclose(v_old, v_new, atol=1e-6)
    # ... and follows the LOCAL lattice shift (its Pb neighbours, by identity), not the global alignment
    assert rep["local_minus_global_A"] > 0.1
    assert all(v["shift_A"] < 0.1 for k, v in rep["checks"]["sites"].items() if k != "pass")
    assert {k for k in rep["checks"]["sites"] if k.startswith("anion")} == {"anion0_Cl"}


def test_clash_with_relaxed_lattice_fails():
    s, cx = archived()
    rs = relaxed(s)
    out, rep = R.rebuild(rs, cx)
    h = out.positions[[i for i in rep["salt_indices"] if cx[i].symbol == "H"][0]]   # salt H, slab frame
    k = int(np.argmin(np.linalg.norm(rs.positions - h, axis=1) + 99 * (np.array(rs.get_chemical_symbols()) != "I")))
    rs.positions[k] = h + [-0.9, 0, 0]                            # an iodide pushed into the salt H
    _, rep = R.rebuild(rs, cx)
    assert not rep["pass"] and rep["status"] == "fail"
    assert not rep["checks"]["min_distance_A"]["pass"] and not rep["checks"]["salt_H_lattice_A"]["pass"]


def test_atom_on_another_lattice_site_fails():
    s, cx = archived()
    rs = relaxed(s)
    far = np.flatnonzero((np.array(rs.get_chemical_symbols()) == "I") & (rs.positions[:, 0] < 11))[0]
    rs.positions[far, 0] -= 3.0                                  # an I on an empty site below the slab, far from the salt
    _, rep = R.rebuild(rs, cx)
    ls = rep["checks"]["lattice_sites"]
    assert not ls["pass"] and ls["n_far_jumps"] == 1 and not rep["pass"]


def test_cli_writes_poscar_and_reports(tmp_path):
    s, cx = archived()
    write(tmp_path / "slab.vasp", relaxed(s), format="vasp")
    write(tmp_path / "cx.vasp", cx, format="vasp")
    rc = R.main(["--slab", str(tmp_path / "slab.vasp"), "--case", f"X={tmp_path / 'cx.vasp'}",
                 "--out", str(tmp_path / "out")])
    assert rc == 0
    summ = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summ["cases"]["X"]["pass"] and summ["cases"]["X"]["failed"] == []
    rep = json.loads((tmp_path / "out" / "X" / "rebuild.json").read_text())
    assert rep["poscar_sha256_16"] and len(read(tmp_path / "out" / "X" / "POSCAR")) == len(cx)


def test_anion_on_lattice_site_with_iodide_in_vacancy_is_relocated():
    """Package-53 pattern: the anion took a surface I site (S1) and that iodide fills the slab's V_I (S2) nearby."""
    from ase import Atoms
    s = T.slab()
    top = 10 + 2 * T.A
    site = lambda p: int(np.argmin(np.linalg.norm(s.positions - p, axis=1)))
    v_i, s1 = site([top, T.A, 1.5 * T.A]), site([top, 1.5 * T.A, T.A])
    slab_v = s[[i for i in range(len(s)) if i != v_i]]              # Schottky-like V_I at S2
    cx = slab_v.copy()
    k = site([top, 1.5 * T.A, T.A]) - (1 if s1 > v_i else 0)
    s1_pos = cx.positions[k].copy()
    cx.positions[k] = s.positions[v_i] + [0.1, 0.0, 0.0]            # the iodide moves into V_I
    cx += Atoms("CNH", positions=[(top + 3.0, T.A + 0.3, T.A + dz) for dz in (0.0, 1.5, 2.5)])
    cx += Atoms("Cl", positions=[s1_pos + [0.8, 0, 0]])             # the anion sits on S1
    cx.positions += [0.3, -2.1, 1.7]
    cx = cx[np.random.default_rng(1).permutation(len(cx))]
    rs = slab_v.copy()
    rs.positions += np.random.default_rng(2).normal(0, 0.04, rs.positions.shape)
    out, rep = R.rebuild(rs, cx)
    ls = rep["checks"]["lattice_sites"]
    assert rep["pass"], json.dumps(rep["checks"], indent=1)
    assert len(ls["relocated"]) == 1 and ls["relocated"][0]["formula"] == "I" and ls["n_far_jumps"] == 0
    assert rep["salt"] == {"C": 1, "Cl": 1, "H": 1, "N": 1}
    assert any(n.startswith("relocated_I") for n in rep["checks"]["sites"])
    # relocated iodide keeps its distance to the anion
    cl = [i for i in rep["salt_indices"] if cx[i].symbol == "Cl"][0]
    j = ls["relocated"][0]["cx"][0]
    assert abs(out.get_distance(cl, j, mic=True) - cx.get_distance(cl, j, mic=True)) < 1e-6
