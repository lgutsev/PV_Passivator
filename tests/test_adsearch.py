"""Software checks of the adsorption-arrangement search (pvpassivator.adsearch).

Everything here runs on synthetic fixtures (a toy FAPbI3-like slab, toy molecules, a made-up score).
The tests establish that the machinery keeps atom roles, composition, topology and bookkeeping
straight; none of them is evidence about real adsorption energetics or search performance.
"""
import json

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from pvpassivator.surface import load_surface, find_sites
from pvpassivator.poses import Passivator, Component, tag_substrate
from pvpassivator import adsearch as A
from test_surface_poses_prep import toy_slab


def _pas(mol, name, comps=None, charge=0):
    comps = comps or [Component(name, list(range(len(mol))), 0, name)]
    return Passivator(name, mol, charge, 1, comps, "neutral_donor")


def nh3():
    m = molecule("NH3")
    return m, [i for i, s in enumerate(m.get_chemical_symbols()) if s == "N"][0]


def chiral():
    """N anchor bonded to a C that carries H, F, Cl: a labelled stereocentre (toy molecule)."""
    t = 1.0 / np.sqrt(3)
    d = np.array([[t, t, t], [t, -t, -t], [-t, t, -t], [-t, -t, t]])
    pos = [(0, 0, 0)] + [tuple(L * v) for L, v in zip((1.47, 1.09, 1.35, 1.77), d)]
    m = Atoms("CNHFCl", positions=pos)
    for v in ((0.9, 0.3, 0.3), (0.3, 0.9, 0.3)):                      # two H on N
        m.append("H")
        m.positions[-1] = m.positions[1] + np.array(v) * 1.0
    return m, 1


@pytest.fixture
def space(tmp_path):
    return make_space(tmp_path)


def make_space(tmp_path, mol_fn=nh3, n=2, modes=None, reps=(1, 3, 3), **kw):
    at = toy_slab().repeat(reps)
    at.write(tmp_path / "POSCAR", format="vasp")
    slab, rec = load_surface(tmp_path / "POSCAR")
    sites = [s for s in find_sites(slab, rec, "top") if s.kind == "undercoordinated_Pb"]
    mol, anchor = mol_fn()
    modes = modes or [A.BindingMode("N_down", anchor, 2.6, tilts=(0.0, 20.0), azimuth_period=120.0),
                      A.BindingMode("N_high", anchor, 3.2, tilts=(0.0,), azimuth_period=120.0)]
    return A.SearchSpace(tag_substrate(slab), rec, sites, A.MoleculeSpec(_pas(mol, "mol"), modes), n, **kw)


def g(space, i, mode="N_down", az=0.0, tilt=0.0):
    return A.Gene(space.sites[i].site_id, mode, "c0", az, tilt)


# ----------------------------------------------------------------------------- roles, substrate
def test_substrate_and_roles_survive_construction(space):
    sub = space.substrate
    atoms, reasons, _ = space.build((g(space, 0), g(space, 4)))
    assert reasons == []
    n = len(sub)
    assert np.array_equal(atoms.positions[:n], sub.positions)                 # bit-identical, not "close"
    assert np.array_equal(atoms.arrays["atom_id"][:n], np.arange(n))
    assert (atoms.arrays["role"][:n] == 0).all() and (atoms.arrays["role"][n:] == A.ROLE["passivant"]).all()
    assert sorted(set(atoms.arrays["mol_id"].tolist())) == [0, 1, 2]
    assert len(set(atoms.arrays["atom_id"].tolist())) == len(atoms)           # ids unique
    assert A.substrate_integrity(sub, atoms, max_move=0.0, rec=space.rec) == (True, "")


def test_every_operator_keeps_composition_and_never_touches_the_slab(space):
    rng = np.random.default_rng(3)
    a, b = space.random_genes(rng), space.random_genes(rng)
    formula = space.build(a)[0].get_chemical_formula()
    for op in A.OPERATORS:
        for _ in range(25):
            new, rec = space.mutate(a, op, rng, other=b)
            if new is None:
                assert rec["reason"]
                continue
            assert len(new) == len(a) and len({x.site_id for x in new}) == len(new)
            atoms, _, _ = space.build(new)
            assert atoms.get_chemical_formula() == formula
            assert A.substrate_integrity(space.substrate, atoms, max_move=0.0)[0]
            assert int((atoms.arrays["role"] == 0).sum()) == len(space.substrate)


def test_substrate_integrity_detects_damage(space):
    atoms, _, _ = space.build((g(space, 0), g(space, 4)))
    lost = atoms[1:]
    assert A.substrate_integrity(space.substrate, lost)[1] == "substrate_atom_missing"
    relabelled = atoms.copy()
    relabelled.arrays["role"][5] = A.ROLE["passivant"]                        # an "adsorbate" claims a lattice atom
    assert A.substrate_integrity(space.substrate, relabelled)[1] == "substrate_role_changed"
    pulled = atoms.copy()
    pb = int(space.sites[0].atom_index)
    pulled.positions[pb] += 4.0 * space.n_out
    assert A.substrate_integrity(space.substrate, pulled, rec=space.rec)[1] == "lattice_atom_extracted"
    assert A.substrate_integrity(space.substrate, pulled, max_move=0.0)[1] == "substrate_moved"


def test_roles_from_reference_separates_passivant_counterion_and_lattice_fa(tmp_path):
    slab = toy_slab().repeat((1, 2, 2))
    cx = slab.copy()
    top = slab.positions[:, 0].max()
    m, _ = nh3()
    m.positions += [top + 4.0, 3.0, 3.0] - m.positions.mean(axis=0)
    cx += m
    cx += Atoms("I", positions=[[top + 5.0, 9.0, 9.0]])                       # extra iodide on an iodide slab
    role = A.assign_roles_from_reference(cx, slab)
    assert (role[:len(slab)] == 0).all()                                      # lattice FA C/N/H stay substrate
    assert (role[len(slab):len(slab) + 4] == A.ROLE["passivant"]).all()
    assert role[-1] == A.ROLE["counterion"] and int((role == A.ROLE["counterion"]).sum()) == 1


def test_normal_to_z_is_a_proper_relabelling(space):
    z = A.normal_to_z(space.substrate, space.rec.normal_axis)
    assert np.isclose(z.get_volume(), space.substrate.get_volume())
    assert np.allclose(sorted(z.cell.lengths()), sorted(space.substrate.cell.lengths()))
    assert np.allclose(z.get_all_distances(mic=True)[0], space.substrate.get_all_distances(mic=True)[0])
    assert z.cell.lengths()[2] == pytest.approx(space.substrate.cell.lengths()[space.rec.normal_axis])


# ----------------------------------------------------------------------------- validity filters
def test_rejection_reasons(tmp_path):
    sp = make_space(tmp_path)
    assert "site_conflict" in sp.build((g(sp, 0), g(sp, 0)))[1]
    low = make_space(tmp_path, modes=[A.BindingMode("N_low", 0, 0.9, anchor_window=(2.0, 3.6))])
    r = low.build((g(low, 0, "N_low"), g(low, 4, "N_low")))[1]
    assert "anchor_distance" in r and "slab_clash" in r
    far = make_space(tmp_path, modes=[A.BindingMode("N_far", 0, 5.0)])
    assert far.build((g(far, 0, "N_far"), g(far, 4, "N_far")))[1] == ["anchor_distance"]


def test_molecule_clash_is_seen_through_the_periodic_boundary(tmp_path):
    def rod():                                    # long toy molecule lying in the surface plane
        m = Atoms("NCCCCC", positions=[(0, 0, 0)] + [(1.5 * k, 0, 0) for k in range(1, 6)])
        return m, 0
    flat = [A.BindingMode("lying", 0, 3.0, up="plane_normal", tilts=(0.0,))]
    sp = make_space(tmp_path, rod, modes=flat)
    hits = set()
    for i, j in ((0, 2), (0, 6)):                 # sites 0 and 2 (and 0 and 6) are neighbours only across the cell edge
        for az in range(0, 360, 30):
            hits |= set(sp.build((g(sp, i, "lying", az), g(sp, j, "lying", az)))[1])
    assert "molecule_clash" in hits
    small = make_space(tmp_path, rod, n=1, modes=flat, reps=(1, 1, 1))
    assert "self_image_clash" in small.build((g(small, 0, "lying"),))[1]


def test_buried_and_vacuum_checks(tmp_path):
    def tall():
        return Atoms("N" + "C" * 30, positions=[(0, 0, 1.5 * k) for k in range(31)]), 0
    sp = make_space(tmp_path, tall, n=1, modes=[A.BindingMode("up", 0, 2.6, tilts=(0.0,))])
    assert "vacuum_overrun" in sp.build((g(sp, 0, "up"),))[1]
    sp2 = make_space(tmp_path, n=1, modes=[A.BindingMode("N_in", 0, -1.5, anchor_window=(0.0, 9.0))])
    assert "buried" in sp2.build((g(sp2, 0, "N_in"),))[1]


# ----------------------------------------------------------------------------- topology / stereo
def test_conformer_library_rejects_other_stereoisomer_and_broken_bonds():
    m, anchor = chiral()
    pas = _pas(m, "tox")
    mode = [A.BindingMode("N_down", anchor, 2.6)]
    spec = A.MoleculeSpec(pas, mode)
    assert any(v != 0 for v in spec.chirality.values())
    rot = m.copy()
    rot.rotate(73, "z")
    rot.rotate(31, "x")
    A.MoleculeSpec(pas, mode, conformers={"c0": m, "rotated": rot})           # a rigid rotation is a conformer
    mirror = m.copy()
    mirror.positions[:, 0] *= -1
    with pytest.raises(ValueError, match="stereoisomer"):
        A.MoleculeSpec(pas, mode, conformers={"c0": m, "mirror": mirror})
    broken = m.copy()
    broken.positions[4] += [0, 0, 3.0]
    with pytest.raises(ValueError, match="bonding differs"):
        A.MoleculeSpec(pas, mode, conformers={"c0": m, "broken": broken})


def test_built_molecules_keep_topology_and_labelled_chirality(tmp_path):
    sp = make_space(tmp_path, chiral, modes=[A.BindingMode("N_down", 1, 2.8, tilts=(0.0, 20.0))])
    rng = np.random.default_rng(1)
    seen = 0
    for _ in range(40):
        atoms, reasons, _ = sp.build(sp.random_genes(rng))
        assert "topology_changed" not in reasons and "stereo_inverted" not in reasons
        for idx in A._molecules(atoms):
            mol = atoms[idx]
            assert A.bond_list(mol) == sp.spec.topology
            assert A.same_chirality(sp.spec.chirality, A.labelled_chirality(mol, sp.spec.topology))
            seen += 1
    assert seen == 80


# ----------------------------------------------------------------------------- deduplication
def test_duplicates_repeated_translated_wrapped_rotated(space):
    base = (g(space, 0), g(space, 4, tilt=20.0))
    a = space.build(base)[0]
    assert space.same_structure(a, space.build(base)[0])                      # repeated
    assert space.same_structure(a, space.build(base[::-1])[0])                # molecule order
    assert space.canonical_key(base) == space.canonical_key(base[::-1])
    op = space.symops[4]                                                       # a surface translation
    moved = tuple(A.Gene(op.site_map[x.site_id], x.mode_id, x.conformer_id, x.azimuth_deg, x.tilt_deg) for x in base)
    assert moved != base and space.canonical_key(moved) == space.canonical_key(base)
    b = space.build(moved)[0]
    assert space.same_structure(a, b)                                         # translated
    w = b.copy()
    w.wrap()
    assert space.same_structure(a, w) and space.same_structure(w, a)          # wrapped
    rot = (g(space, 0, az=120.0), g(space, 4, tilt=20.0))                     # NH3 turned onto itself
    assert space.canonical_key(rot) == space.canonical_key(base)
    assert space.same_structure(a, space.build(rot)[0])


def test_distinct_configurations_are_kept_apart(space):
    a = space.build((g(space, 0), g(space, 1)))[0]                            # neighbours
    b = space.build((g(space, 0), g(space, 4)))[0]                            # diagonal
    c = space.build((g(space, 0), g(space, 1, "N_high")))[0]                  # other binding mode
    d = space.build((g(space, 0), g(space, 1, tilt=20.0)))[0]                 # other orientation
    for x in (b, c):
        assert not space.same_structure(a, x)
    # a 20 degree tilt moves NH3's H atoms by ~0.3 A: inside the default 0.5 A relaxed-structure tolerance (such
    # starts are told apart by the gene key below), outside a tighter one
    assert space.same_structure(a, d) and not space.same_structure(a, d, tol=0.2)
    keys = {space.canonical_key(k) for k in ((g(space, 0), g(space, 1)), (g(space, 0), g(space, 4)),
                                             (g(space, 0), g(space, 1, "N_high")), (g(space, 0), g(space, 1, tilt=20.0)))}
    assert len(keys) == 4
    off = A.SearchSpace(space.substrate, space.rec, space.sites, space.spec, 2, symmetry="none")
    assert len(off.symops) == 1
    moved = space.build((g(space, 3), g(space, 4)))[0]
    assert space.same_structure(a, moved) and not off.same_structure(a, moved)


def test_enantiomers_are_never_merged(tmp_path):
    m, anchor = chiral()
    mirror = m.copy()
    mirror.positions[:, 2] *= -1                                               # keeps the N-C axis direction roughly
    mode = [A.BindingMode("N_down", anchor, 2.8, tilts=(0.0,))]
    sp_r = make_space(tmp_path, lambda: (m, anchor), n=1, modes=mode)
    sp_s = make_space(tmp_path, lambda: (mirror, anchor), n=1, modes=mode)
    assert sp_r.spec.sha256_16() != sp_s.spec.sha256_16()
    for az in range(0, 360, 30):
        a = sp_r.build((g(sp_r, 0, az=0.0),))[0]
        b = sp_s.build((g(sp_s, 0, az=float(az)),))[0]
        assert not sp_r.same_structure(a, b)


def test_site_translations_report_how_well_the_substrate_maps(space):
    assert len(space.symops) == 9
    assert all(o.inorganic_residual_A < 1e-6 and o.organic_residual_A < 1e-6 for o in space.symops)   # ideal toy slab
    broken = space.substrate.copy()
    n_at = [i for i, s in enumerate(broken.get_chemical_symbols()) if s == "N"][0]
    broken.positions[n_at] += [0.0, 0.6, 0.0]                                 # one FA differs: symmetry is only approximate
    sp = A.SearchSpace(broken, space.rec, space.sites, space.spec, 2)
    assert len(sp.symops) == 9 and max(o.organic_residual_A for o in sp.symops) > 0.05


# ----------------------------------------------------------------------------- search loop, accounting
def test_budget_accounting_and_reproducibility(space):
    ev = A.SyntheticEvaluator(space)
    logs = {arm: A.run_search(space, ev, arm, 80, seed=7, n_init=10, pop_size=5) for arm in ("random", "evolutionary")}
    for log in logs.values():
        assert len(log["records"]) == 80 and sum(log["counts"].values()) == 80      # nothing leaves the budget
        assert log["evaluated"] == log["counts"].get("unique", 0) + log["counts"].get("duplicate_relaxed", 0)
        assert log["total_calls"] == sum(r["n_calls"] for r in log["records"]) == log["records"][-1]["cumulative_calls"]
        assert sum(log["rejection_reasons"].values()) >= log["counts"].get("rejected", 0)
        best = [t[2] for t in log["trace"] if t[2] is not None]
        assert best == sorted(best, reverse=True)                                    # running minimum
        for r in log["records"]:
            if r["status"].startswith("duplicate"):
                assert not r["ranked"] and "duplicate_of" in r
            if r["evaluated"]:
                assert r["final_state"] == "intact_adsorption" and r["initial_hash"] == r["final_hash"]
    ra, ev_ = logs["random"]["records"], logs["evolutionary"]["records"]
    assert [r.get("key") for r in ra[:10]] == [r.get("key") for r in ev_[:10]]      # same starting proposals
    assert any(r["operator"] in A.OPERATORS for r in ev_[10:])
    assert all(r["parents"] and all(p < r["attempt"] for p in r["parents"]) for r in ev_ if r["operator"] in A.OPERATORS)
    again = A.run_search(space, ev, "evolutionary", 80, seed=7, n_init=10, pop_size=5)
    assert [(r["status"], r.get("final_hash"), r["operator"]) for r in again["records"]] == \
           [(r["status"], r.get("final_hash"), r["operator"]) for r in ev_]
    other = A.run_search(space, ev, "evolutionary", 80, seed=8, n_init=10, pop_size=5)
    assert [r.get("key") for r in other["records"]] != [r.get("key") for r in ev_]


def test_relax_evaluator_fixes_far_field_and_counts_calls(space):
    from ase.calculators.calculator import Calculator, all_changes

    class Spring(Calculator):                       # toy: pulls molecule atoms 0.2 A outward; slab feels nothing
        implemented_properties = ["energy", "forces"]

        def __init__(self, target):
            super().__init__()
            self.target = target

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            d = atoms.positions - self.target
            m = atoms.arrays["mol_id"] > 0
            d[~m] = 0
            self.results = {"energy": float(0.5 * (d ** 2).sum()), "forces": -d}

    genes = (g(space, 0), g(space, 4))
    atoms, _, _ = space.build(genes)
    target = atoms.positions + 0.2 * space.n_out
    ev = A.AseRelaxEvaluator(Spring(target), fmax=0.01, steps=200, radius=4.0, mobile="local")
    face = A.AseRelaxEvaluator(Spring(target), radius=4.0, mobile="face", space=space)
    other, _, _ = space.build((g(space, 2), g(space, 7, "N_high")))
    sub_ids = lambda a, m: set(a.arrays["atom_id"][m & (a.arrays["mol_id"] == 0)].tolist())
    assert sub_ids(atoms, face.mobile_mask(atoms)) == sub_ids(other, face.mobile_mask(other)) != set()   # same fixed set
    assert sub_ids(atoms, ev.mobile_mask(atoms)) != sub_ids(other, ev.mobile_mask(other))               # "local" is not
    with pytest.raises(ValueError):
        A.AseRelaxEvaluator(Spring(target), mobile="face")
    mob = ev.mobile_mask(atoms)
    assert mob[atoms.arrays["mol_id"] > 0].all() and 0 < mob.sum() < len(atoms)
    fid = atoms.arrays["fragment_id"]
    assert all(len(set(mob[fid == f].tolist())) == 1 for f in np.unique(fid))       # lattice FA whole or not at all
    res = ev(atoms, genes)
    assert res.converged and res.n_calls >= res.info["steps"] > 0
    assert np.array_equal(res.atoms.positions[~mob], atoms.positions[~mob])         # far field untouched
    assert np.allclose(res.atoms.positions[atoms.arrays["mol_id"] > 0], target[atoms.arrays["mol_id"] > 0], atol=0.02)
    assert np.array_equal(res.atoms.arrays["atom_id"], atoms.arrays["atom_id"])
    assert np.array_equal(res.atoms.arrays["role"], atoms.arrays["role"])
    log = A.run_search(space, ev, "random", 4, seed=0, n_init=4)
    assert log["total_calls"] == sum(r["n_calls"] for r in log["records"]) > log["evaluated"]


def test_manifest_round_trip(space, tmp_path):
    ev = A.SyntheticEvaluator(space)
    logs = [A.run_search(space, ev, arm, 12, seed=0, n_init=6) for arm in ("random", "evolutionary")]
    man = A.write_manifest(tmp_path / "out", space, logs, {"note": "software check"}, write_structures="unique")
    disk = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert disk["schema"] == "adsearch_manifest/1" and len(disk["runs"]) == 2 == len(man["runs"])
    cov = disk["coverage_by_mode"]["N_down"]
    assert cov["molecules_per_nm2"] > 0 and cov["occupied_sites"] == 2 and cov["n_sites"] == 9
    from ase.io import read
    rec = next(r for r in disk["runs"][0]["records"] if r["status"] == "unique")
    back = read(tmp_path / "out" / rec["final_file"])
    for k in ("atom_id", "role", "mol_id"):
        assert k in back.arrays
    assert A.structure_hash(back) == rec["final_hash"]                                # hash survives the file
    assert {"genes", "operator", "parents", "seed", "validity", "key"} <= set(rec)


def test_acat_element_rule_misassigns_lattice_cations():
    pytest.importorskip("acat")
    from acat.settings import adsorbate_elements
    syms = toy_slab().get_chemical_symbols()
    assert {"C", "N", "H"} <= set(adsorbate_elements)
    assert sum(s in adsorbate_elements for s in syms) == 8                            # the lattice FA
