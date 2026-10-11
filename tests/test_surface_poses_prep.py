import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from pvpassivator.surface import load_surface, find_sites, detect_normal
from pvpassivator.poses import (tag_substrate, remove_fragments, generate_poses, load_passivator,
                                classify_final_state, make_whole, min_image_distances)
from pvpassivator.vaspprep import write_job, adsorption_bundle
from pvpassivator.ledger import UnbalancedReaction

FA = [("C", (0, 0, 0)), ("H", (0, 0, 1.09)), ("N", (1.15, 0, -0.6)), ("N", (-1.15, 0, -0.6)),
      ("H", (2.0, 0, -0.1)), ("H", (1.2, 0, -1.6)), ("H", (-2.0, 0, -0.1)), ("H", (-1.2, 0, -1.6))]


def toy_slab(top="PbI2", nlayers=3, a=6.3, vac=30.0, normal=0):
    """Cubic FAPbI3-like slab, normal along x. PbI2 layers alternate with FAI layers.
    Built as PbI2 | FAI | PbI2 ... so with odd nlayers both faces are PbI2-terminated."""
    syms, pos = [], []
    x = 0.0
    for L in range(nlayers):
        if L % 2 == 0:   # PbI2 layer: Pb at (x, 0,0), I at (x, a/2, 0),(x,0,a/2); apical I between layers handled by FAI layer
            syms += ["Pb", "I", "I"]; pos += [(x, 0, 0), (x, a / 2, 0), (x, 0, a / 2)]
        else:            # FAI layer: I at (x, 0, 0) (apical), FA at (x, a/2, a/2)
            syms += ["I"]; pos += [(x, 0, 0)]
            for s, p in FA:
                syms.append(s); pos.append((x + p[2] * 0.3, a / 2 + p[0] * 0.8, a / 2 + p[1] * 0.8))
        x += a / 2
    cell = [x + vac, a, a]
    at = Atoms(syms, positions=pos, cell=cell, pbc=True)
    at.positions[:, 0] += vac / 2
    return at


def test_detect_normal_and_faces(tmp_path):
    at = toy_slab()
    at.write(tmp_path / "POSCAR", format="vasp")
    a, rec = load_surface(tmp_path / "POSCAR")
    assert rec.normal_axis == 0 and rec.vacuum > 25
    assert rec.faces["top"]["termination"] == "PbI2" and rec.faces["bottom"]["termination"] == "PbI2"
    assert rec.formal_units == {"Pb2+": 2, "I-": 5, "FA+": 1}
    assert rec.formal_charge == 0
    kinds = {s.kind for s in find_sites(a, rec, "top")}
    assert "undercoordinated_Pb" in kinds and "exposed_I" in kinds


def test_schottky_removal_keeps_neutrality(tmp_path):
    at = toy_slab()
    at.write(tmp_path / "POSCAR", format="vasp")
    a, rec = load_surface(tmp_path / "POSCAR")
    a = tag_substrate(a)
    n_idx = [i for i, s in enumerate(a.get_chemical_symbols()) if s == "N"][0]
    i_idx = [i for i, s in enumerate(a.get_chemical_symbols()) if s == "I"][0]
    b, r2 = remove_fragments(a, rec, [n_idx, i_idx], "FAI_Schottky")
    assert len(b) == len(a) - 9
    assert r2.formal_charge == 0 and r2.defects[-1]["removed_units"] == ["CH5N2", "I"]
    c, r3 = remove_fragments(a, rec, [i_idx], "V_I")
    assert r3.formal_charge == 1 and any("electronic compensation" in n for n in r3.notes)


def _pea_i(tmp_path, split=True):
    m = molecule("CH3CH2NH2")              # stand-in amine cation scaffold
    m.append("H")
    n = [i for i, s in enumerate(m.get_chemical_symbols()) if s == "N"][0]
    m.positions[-1] = m.positions[n] + [0, 0, 1.02]
    m.append("I")
    m.positions[-1] = m.positions[n] + [0, 3.3, 0]
    m.cell = [15, 15, 15]
    m.pbc = True
    m.center()
    if split:   # shift so the molecule straddles the boundary like a wrapped CONTCAR
        m.positions += [7.0, 0, 0]
        m.wrap()
        m.positions[-1] += [15.0, 0, 0]     # counterion in a periodic image
        m.wrap()
    p = tmp_path / "salt.vasp"
    m.write(p, format="vasp")
    return p


def test_passivator_declared_charges_and_whole(tmp_path):
    p = _pea_i(tmp_path)
    pas = load_passivator(p, "EAI", 0, 1, {"C2H8N": 1, "I": -1}, "ion_pair_salt")
    assert sorted(c.label for c in pas.components) == ["C2H8N+", "I-"]
    i = pas.atoms.get_chemical_symbols().index("I")
    n = pas.atoms.get_chemical_symbols().index("N")
    assert np.linalg.norm(pas.atoms.positions[i] - pas.atoms.positions[n]) < 4.0
    with pytest.raises(ValueError, match="no declared formal charge"):
        load_passivator(p, "EAI", 0, 1, {"C2H8N": 1}, "ion_pair_salt")
    with pytest.raises(ValueError, match="sum to"):
        load_passivator(p, "EAI", 1, 1, {"C2H8N": 1, "I": -1}, "ion_pair_salt")
    with pytest.raises(ValueError, match="multiplicity"):
        load_passivator(p, "EAI", 0, 2, {"C2H8N": 1, "I": -1}, "ion_pair_salt")


def test_poses_clash_free_and_tagged(tmp_path):
    at = toy_slab()
    at.write(tmp_path / "POSCAR", format="vasp")
    a, rec = load_surface(tmp_path / "POSCAR")
    a = tag_substrate(a)
    pas = load_passivator(_pea_i(tmp_path), "EAI", 0, 1, {"C2H8N": 1, "I": -1}, "ion_pair_salt")
    sites = [s for s in find_sites(a, rec, "top") if s.kind == "undercoordinated_Pb"]
    iod = [i for i, s in enumerate(pas.atoms.get_chemical_symbols()) if s == "I"]
    poses = generate_poses(a, rec, pas, sites, heights=(3.2,), anchors=iod, n_azimuth=4, max_poses=10)
    assert poses
    for full, spec in poses:
        o = full.arrays["origin"]
        D = min_image_distances(full.cell.array, full.pbc, full.positions[o > 0], full.positions[o == 0])
        assert D.min() > 2.0
        assert len(set(full.arrays["atom_id"])) == len(full)


def test_classify_proton_transfer_and_desorption():
    # adsorbate NH4+ near a slab I-; move one H onto... use atom_id correspondence
    slab = Atoms("I", positions=[[5, 5, 5]], cell=[20, 20, 20], pbc=True)
    slab = tag_substrate(slab)
    m = Atoms("NH4", positions=[[5, 5, 8.5], [5, 5, 9.5], [5.9, 5, 8.2], [4.5, 5.8, 8.2], [4.5, 4.2, 8.2]],
              cell=slab.cell, pbc=True)
    m.set_array("atom_id", 100000 + np.arange(5)); m.set_array("fragment_id", np.full(5, 1))
    m.set_array("origin", np.ones(5, int))
    init = slab + m
    rec = type("R", (), {"normal_axis": 2})()
    fin = init.copy()
    assert classify_final_state(init, fin, rec)["final_state_class"] == "intact_adsorption"
    pt = init.copy()
    pt.positions[4] = [5, 5, 5 + 1.6]            # H now bonded to I (H-I 1.6 A)
    assert classify_final_state(init, pt, rec)["final_state_class"] == "proton_transfer"
    de = init.copy()
    de.positions[1:] = init.positions[1:] + [0, 0, 6]
    assert classify_final_state(init, de, rec)["final_state_class"] == "desorbed"
    shuffled = pt[[3, 0, 5, 1, 2, 4]]            # export permutation must not matter
    assert classify_final_state(init, shuffled, rec)["final_state_class"] == "proton_transfer"


def test_write_job_charge_and_spin(tmp_path):
    i2m = Atoms("I2", positions=[[0, 0, 0], [0, 0, 2.9]], cell=[20] * 3, pbc=True)
    j = write_job(i2m, tmp_path / "i2m", "r2scan_d3bj_v1/molecule_static", "reference",
                  cell_charge=-1, multiplicity=2, kpoints=[1, 1, 1])
    incar = (tmp_path / "i2m" / "INCAR").read_text()
    assert j.nelect == 15 and "NELECT = 15" in incar and "ISPIN = 2" in incar and "NUPDOWN = 1" in incar
    assert (tmp_path / "i2m" / "POTCAR.spec").read_text().split() == ["I"]
    with pytest.raises(FileExistsError):
        write_job(i2m, tmp_path / "i2m", "r2scan_d3bj_v1/molecule_static", "reference")


def test_bundle_refuses_unbalanced(tmp_path):
    at = tag_substrate(toy_slab())
    rec = load_surface.__globals__["SurfaceRecord"]  # noqa: just need normal axis
    from types import SimpleNamespace
    r = SimpleNamespace(normal_axis=0)
    cplx = at + Atoms("I", positions=[[at.positions[:, 0].max() + 3, 0, 0]], cell=at.cell, pbc=True)
    with pytest.raises(UnbalancedReaction):
        adsorption_bundle(tmp_path, "bad", at, r, cplx, [("Cl", Atoms("Cl"), -1, 1, 1)], n_units=1)
    # neutral complex from a charged I- reference: charge imbalance must be caught
    with pytest.raises(UnbalancedReaction, match="charge"):
        adsorption_bundle(tmp_path, "chg", at, r, cplx, [("I-", Atoms("I"), -1, 1, 1)], n_units=1)
    plan = adsorption_bundle(tmp_path, "ok", at, r, cplx, [("I", Atoms("I"), 0, 2, 1)], n_units=1)
    assert plan["equation"] == "bare_slab + I -> complex"
    assert "ISPIN = 2" in (tmp_path / "ok" / "ref_I" / "static" / "INCAR").read_text()


def test_fill_schottky_puts_halide_in_iodide_vacancy(tmp_path):
    from pvpassivator.poses import fill_schottky
    at = toy_slab()
    at.write(tmp_path / "POSCAR", format="vasp")
    a, rec = load_surface(tmp_path / "POSCAR")
    a = tag_substrate(a)
    syms = a.get_chemical_symbols()
    c_idx = syms.index("C")
    i_idx = [i for i, s in enumerate(syms) if s == "I"][-1]
    fa_pos, i_pos = a.positions[c_idx].copy(), a.positions[i_idx].copy()
    vac, vrec = remove_fragments(a, rec, [c_idx, i_idx], "FAI_Schottky")
    p = _pea_i(tmp_path, split=False)
    m = load_passivator(p, "EAI", 0, 1, {"C2H8N": 1, "I": -1}, "ion_pair_salt")
    full, info = fill_schottky(vac, vrec, m, fa_pos, i_pos, heights=(0.0, 0.5, 1.0, 1.5), tilts=(0, 20, 40, 60))
    hal = [k for k, o in enumerate(full.arrays["origin"]) if o == 2][0]
    assert np.allclose(full.positions[hal], i_pos) and info["halide"] == "I"
    assert len(full) == len(vac) + len(m.atoms)
