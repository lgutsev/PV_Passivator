import numpy as np
import pytest

from pvpassivator.surface import load_surface, find_sites
from pvpassivator.ledger import Species, UnbalancedReaction
from pvpassivator.oxidation import (oxygen_offsets, oxyanion_poses, oxidised_record,
                                    oxidation_reaction, IODINE_OXYANIONS)
from test_surface_poses_prep import toy_slab


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_offsets_bond_lengths_and_angles(n):
    d, a, _, _ = IODINE_OXYANIONS[n]
    off = oxygen_offsets(n, np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), phi_deg=17)
    assert np.allclose(np.linalg.norm(off, axis=1), d)
    assert np.all(off[:, 2] < 0)  if n < 4 else True       # O atoms opposite the lone pair
    if a is not None:
        u = off / d
        angs = [np.degrees(np.arccos(np.clip(u[i] @ u[j], -1, 1)))
                for i in range(n) for j in range(i + 1, n)]
        assert np.allclose(angs, a, atol=1e-6)


def _surface(tmp_path):
    at = toy_slab(nlayers=5)
    at.write(tmp_path / "POSCAR", format="vasp")
    return load_surface(tmp_path / "POSCAR", surface_id="toy")


def _top_inplane_iodide(atoms, rec):
    sites = [s for s in find_sites(atoms, rec, "top") if s.kind == "exposed_I"]
    h = rec.faces["top"]["edge_coordinate_A"]
    from pvpassivator.surface import unwrap_along
    z = unwrap_along(atoms, rec.normal_axis)
    return min(sites, key=lambda s: abs(z[s.atom_index] - h))


def test_iodate_poses_composition_charge_and_reaction(tmp_path):
    atoms, rec = _surface(tmp_path)
    site = _top_inplane_iodide(atoms, rec)
    poses = oxyanion_poses(atoms, rec, site, n_oxygen=3)
    assert poses, "no clash-free iodate orientation on the toy slab"
    for f, _ in poses:
        pb = [i for i, x in enumerate(f.get_chemical_symbols()) if x == "Pb"]
        assert min(f.get_distances(j, pb, mic=True).min() for j in range(len(f) - 3, len(f))) >= 2.05
    full, spec = poses[0]
    assert len(full) == len(atoms) + 3
    assert spec.label == "IO3-" and spec.oxidation_state == 5
    assert set(full.arrays["origin"][-3:]) == {1}
    # the halide keeps its identity, the O atoms join its fragment
    h = spec.halide_index
    assert full.arrays["atom_id"][h] == spec.halide_atom_id
    assert np.all(full.arrays["fragment_id"][-3:] == full.arrays["fragment_id"][h])
    r2 = oxidised_record(rec, full, spec)
    assert r2.formal_units.get("IO3-") == 1
    assert r2.formal_charge == rec.formal_charge
    assert r2.composition["O"] == 3
    slab = Species("slab", rec.composition)
    prod = Species("slab_IO3", r2.composition)
    rx = oxidation_reaction(slab, prod, 3)
    assert rx.atom_balance() == {}
    with pytest.raises(UnbalancedReaction):
        oxidation_reaction(slab, prod, 2)


def test_rejects_non_halide_site(tmp_path):
    atoms, rec = _surface(tmp_path)
    pb = [s for s in find_sites(atoms, rec, "top") if s.kind == "undercoordinated_Pb"][0]
    with pytest.raises(ValueError):
        oxyanion_poses(atoms, rec, pb)


def test_group_relaxed_poses_merges_duplicates_and_separates_motifs(tmp_path):
    from pvpassivator.oxidation import group_relaxed_poses, oxyanion_fingerprint
    atoms, rec = _surface(tmp_path)
    site = _top_inplane_iodide(atoms, rec)
    poses = oxyanion_poses(atoms, rec, site, n_oxygen=3)
    hid = poses[0][1].halide_atom_id
    a = poses[0][0]
    # same minimum: small noise, and the O atoms listed in a different order
    b = a.copy()
    b.positions += np.random.default_rng(0).normal(0, 0.01, b.positions.shape)
    b = b[list(range(len(b) - 3)) + [len(b) - 1, len(b) - 3, len(b) - 2]]
    # a different motif: a pose whose fingerprint differs
    fa = oxyanion_fingerprint(a, hid)
    other = next(f for f, _ in poses[1:]
                 if any(np.abs(fa[k] - oxyanion_fingerprint(f, hid)[k])[np.isfinite(fa[k])].max() > 0.3
                        for k in ("O-Pb", "I-Pb")))
    groups = group_relaxed_poses([other, a, b], hid, energies=[-1.0, -1.5, -1.505], labels=["c", "a", "b"])
    assert [g["representative"] for g in groups] == ["b", "c"]       # lowest energy first
    assert groups[0]["multiplicity"] == 2 and set(groups[0]["members"]) == {"a", "b"}
    assert groups[0]["energy_spread"] == pytest.approx(0.005)
    # energies only split geometric twins when energy_tol is given
    assert len(group_relaxed_poses([a, b], hid, energies=[0.0, 0.1])) == 1
    assert len(group_relaxed_poses([a, b], hid, energies=[0.0, 0.1], energy_tol=0.02)) == 2


def test_fingerprint_refuses_structures_without_pose_arrays(tmp_path):
    from pvpassivator.oxidation import oxyanion_fingerprint
    atoms, rec = _surface(tmp_path)
    full, spec = oxyanion_poses(atoms, rec, _top_inplane_iodide(atoms, rec), n_oxygen=3)[0]
    bare = full.copy()
    del bare.arrays["origin"]
    with pytest.raises(ValueError, match="origin"):
        oxyanion_fingerprint(bare, spec.halide_atom_id)


# ----------------------------------------------------------------------------- final-state classification
def _built(tmp_path, n=3, lp=None):
    atoms, rec = _surface(tmp_path)
    site = _top_inplane_iodide(atoms, rec)
    kw = {"lone_pairs": lp} if lp else {}
    full, spec = oxyanion_poses(atoms, rec, site, n_oxygen=n, **kw)[0]
    return full, rec, spec


def _normal(full, rec):
    v = full.cell.array[rec.normal_axis]
    return v / np.linalg.norm(v)


def test_classify_intact_iodate_reports_orientation_and_height(tmp_path):
    from pvpassivator.poses import classify_final_state
    full, rec, spec = _built(tmp_path)
    r = classify_final_state(full, full.copy(), rec)
    assert r["final_state_class"] == "intact_adsorption"
    (s,) = r["oxyanions"]
    assert s["oxyanion_class"].startswith("intact") and s["n_oxygen"] == 3
    assert all(x == pytest.approx(1.81, abs=1e-6) for x in s["x_o_A"])
    assert s["o_orientation_deg"] is not None and 0 <= s["o_orientation_deg"] <= 180
    assert abs(s["halide_height_A"]) < 1.0           # in-plane surface iodide


def test_classify_orientation_follows_the_oxygen_side(tmp_path):
    from pvpassivator.poses import classify_final_state
    atoms, rec = _surface(tmp_path)
    site = _top_inplane_iodide(atoms, rec)
    n = np.zeros(3); n[rec.normal_axis] = 1.0
    up = oxyanion_poses(atoms, rec, site, n_oxygen=1, lone_pairs=[("in", -n)], lifts=(0.0,))   # O to vacuum
    assert up
    s = classify_final_state(up[0][0], up[0][0].copy(), rec)["oxyanions"][0]
    assert s["o_orientation_deg"] < 10


def test_classify_o_transfer_cleavage_protonation_sinking_desorption(tmp_path):
    from pvpassivator.poses import classify_final_state
    full, rec, spec = _built(tmp_path)
    n = _normal(full, rec)
    h = spec.halide_index
    o = list(range(len(full) - 3, len(full)))
    syms = np.array(full.get_chemical_symbols())
    pb = [i for i in np.where(syms == "Pb")[0]]
    # O moved onto a Pb (1.9 A above it), away from the iodine
    f = full.copy()
    p = min(pb, key=lambda i: f.get_distance(i, o[0], mic=True))
    f.positions[o[0]] = f.positions[p] + 2.1 * n
    if f.get_distance(h, o[0], mic=True) <= 2.3:
        f.positions[o[0]] += 1.0 * n
    r = classify_final_state(full, f, rec)
    assert r["final_state_class"] in ("oxyanion_o_transfer_to_metal", "oxyanion_x_o_cleavage")
    assert r["oxyanions"][0]["cleaved_o"] == [0]
    # O pulled straight away into vacuum: cleavage, not transfer
    f = full.copy(); f.positions[o[1]] = f.positions[h] + 4.0 * n
    assert classify_final_state(full, f, rec)["final_state_class"] == "oxyanion_x_o_cleavage"
    # an FA hydrogen moved onto an O: protonation
    f = full.copy()
    hh = min(np.where(syms == "H")[0], key=lambda i: f.get_distance(i, o[2], mic=True))
    v = f.get_distance(o[2], hh, mic=True, vector=True)
    f.positions[hh] = f.positions[o[2]] + 0.98 * v / np.linalg.norm(v)
    r = classify_final_state(full, f, rec)
    assert r["final_state_class"] == "oxyanion_protonated" and r["oxyanions"][0]["protonated_o"] == [2]
    # whole oxyanion pushed 1.5 A into the slab: still intact, flagged below the metal plane
    f = full.copy(); f.positions[[h] + o] -= 1.5 * n
    s = classify_final_state(full, f, rec)["oxyanions"][0]
    assert "below_metal_plane" in s["flags"] and s["halide_height_A"] < 0
    # oxyanion lifted 8 A: desorbed
    f = full.copy(); f.positions[[h] + o] += 8.0 * n
    assert classify_final_state(full, f, rec)["final_state_class"] == "oxyanion_desorbed"


def test_classify_accepts_unsorted_vasp_readback(tmp_path):
    from ase.io import read, write
    from pvpassivator.poses import classify_final_state
    full, rec, spec = _built(tmp_path)
    write(tmp_path / "CONTCAR", full, format="vasp", direct=True, sort=False)
    back = read(tmp_path / "CONTCAR")
    assert "atom_id" not in back.arrays
    r = classify_final_state(full, back, rec)
    assert r["oxyanions"][0]["oxyanion_class"].startswith("intact")
    write(tmp_path / "CONTCAR_sorted", full, format="vasp", direct=True, sort=True)
    with pytest.raises(ValueError, match="species order"):
        classify_final_state(full, read(tmp_path / "CONTCAR_sorted"), rec)


def test_o_down_and_x_down_presets(tmp_path):
    from pvpassivator.oxidation import oxyanion_preset_poses, _bridge_direction, _nearest_pb
    atoms, rec = _surface(tmp_path)
    site = _top_inplane_iodide(atoms, rec)
    h = site.atom_index
    n = np.eye(3)[rec.normal_axis]
    x0 = atoms.positions[h].copy()
    # O-down IO3-: halide lifted by the I-O length, one O in the old halide site, that bond straight down
    poses = oxyanion_preset_poses(atoms, rec, site, 3, "O-down")
    assert [s.phi_deg for _, s in poses] == [0.0, 90.0]
    for f, spec in poses:
        d, _, _, _ = IODINE_OXYANIONS[3]
        assert np.allclose(f.positions[h], x0 + d * n)
        assert np.allclose(f.get_distances(h, [len(f) - 3, len(f) - 2, len(f) - 1], mic=True), d)
        assert min(np.linalg.norm(f.positions[-3:] - x0, axis=1)) < 1e-6
        assert spec.lone_pair == "O-down" and "/O-down/" in spec.pose_id
    # O-down IO2-, psi = 0: the O-I-O plane is perpendicular to the Pb-I-Pb bridge
    (f, spec), *_ = oxyanion_preset_poses(atoms, rec, site, 2, "O-down")
    u = f.get_distances(h, [len(f) - 2, len(f) - 1], mic=True, vector=True)
    plane_n = np.cross(u[0], u[1]) / np.linalg.norm(np.cross(u[0], u[1]))
    bridge = _bridge_direction(atoms, h, n, _nearest_pb(atoms, h))
    assert abs(plane_n @ bridge) > 0.999
    # X-down: halide stays, every O points to vacuum
    for f, spec in oxyanion_preset_poses(atoms, rec, site, 3, "X-down"):
        assert np.allclose(f.positions[h], x0)
        assert np.all(f.get_distances(h, [len(f) - 3, len(f) - 2, len(f) - 1], mic=True, vector=True) @ n > 0)
        assert "/X-down/" in spec.pose_id
    # X-down IO2- defaults to phi = 90: O-I-O plane perpendicular to the bridge, as for O-down
    (f, spec), = oxyanion_preset_poses(atoms, rec, site, 2, "X-down")
    assert spec.phi_deg == 90.0
    u = f.get_distances(h, [len(f) - 2, len(f) - 1], mic=True, vector=True)
    assert abs(np.cross(u[0], u[1]) @ bridge) / np.linalg.norm(np.cross(u[0], u[1])) > 0.999
    with pytest.raises(ValueError):
        oxyanion_preset_poses(atoms, rec, site, 3, "sideways")


def _toy_2x2(tmp_path):
    at = toy_slab(nlayers=5).repeat((1, 2, 2))
    at.write(tmp_path / "POSCAR", format="vasp")
    return load_surface(tmp_path / "POSCAR", surface_id="toy2x2")


def test_lead_iodate_unit_build_remove_and_reaction(tmp_path):
    from pvpassivator.oxidation import (shared_lead_pairs, lead_iodate_poses, lead_iodate_record,
                                        remove_lead_iodate, lead_iodate_extraction_reaction)
    atoms, rec = _toy_2x2(tmp_path)
    pairs = shared_lead_pairs(atoms, rec, "top")
    assert pairs and {round(p["angle_deg"]) for p in pairs} == {90, 180}
    cis = next(p for p in pairs if round(p["angle_deg"]) == 90)
    poses = lead_iodate_poses(atoms, rec, cis["pb"], cis["halides"], 3, "O-down")
    assert poses
    full, spec = poses[0]
    assert len(full) == len(atoms) + 6 and len(spec.unit_atom_ids()) == 9
    r2 = lead_iodate_record(rec, full, spec)
    assert r2.formal_units.get("IO3-") == 2 and r2.formal_charge == rec.formal_charge
    assert r2.composition["O"] == 6 and "unit_atom_ids" in r2.defects[-1]
    vac, r3 = remove_lead_iodate(full, r2, spec)
    assert len(vac) == len(full) - 9
    assert r3.composition["Pb"] == rec.composition["Pb"] - 1 and r3.composition["I"] == rec.composition["I"] - 2
    assert "O" not in r3.composition and r3.formal_charge == 0
    bulk = Species("Pb(IO3)2(bulk,Z=4)", {"Pb": 4, "I": 8, "O": 24}, kind="bulk")
    rx = lead_iodate_extraction_reaction(Species("slab_PbIO32", r2.composition),
                                         Species("slab_VPbI2", r3.composition), bulk)
    assert rx.products[1][0] == pytest.approx(0.25)
    with pytest.raises(Exception):          # energies are never defaulted
        rx.delta_e({"slab_PbIO32": 0.0, "slab_VPbI2": 0.0})
    with pytest.raises(ValueError):
        lead_iodate_extraction_reaction(Species("a", r2.composition), Species("b", r3.composition),
                                        Species("PbI2", {"Pb": 1, "I": 2}))
    # a relaxed structure whose iodate lost an O is not the declared unit: refuse to remove it
    broken = full.copy()
    broken.positions[-1] += 3.0 * np.eye(3)[rec.normal_axis]
    with pytest.raises(ValueError, match="not intact"):
        remove_lead_iodate(broken, r2, spec)


def test_lead_iodate_rejects_halide_not_on_that_pb(tmp_path):
    from pvpassivator.oxidation import shared_lead_pairs, lead_iodate_poses
    atoms, rec = _toy_2x2(tmp_path)
    pairs = shared_lead_pairs(atoms, rec, "top")
    other_pb = next(p["pb"] for p in pairs if p["pb"] != pairs[0]["pb"])
    x1, x2 = pairs[0]["halides"]
    if other_pb in [p["pb"] for p in pairs if x1 in p["halides"] and x2 in p["halides"]]:
        pytest.skip("toy geometry: both halides also bridge the other Pb")
    with pytest.raises(ValueError, match="not bonded"):
        lead_iodate_poses(atoms, rec, other_pb, (x1, x2))


def test_oxidation_bundle_writes_matched_jobs_with_triplet_o2(tmp_path):
    import json
    from dataclasses import replace
    from pvpassivator.vaspprep import oxidation_bundle, check_matched_settings, JobSpec
    atoms, rec = _surface(tmp_path)
    full, spec = oxyanion_poses(atoms, rec, _top_inplane_iodide(atoms, rec), n_oxygen=3)[0]
    plan = oxidation_bundle(tmp_path / "b", "io3_toy", atoms, rec, full, 3)
    root = tmp_path / "b" / "io3_toy"
    for role in ("bare_slab", "product", "ref_O2"):
        for step in ("relax", "static"):
            assert (root / role / step / "INCAR").exists()
    incar = (root / "ref_O2" / "static" / "INCAR").read_text()
    assert "ISPIN = 2" in incar and "NUPDOWN = 2" in incar
    assert plan["jobs"]["ref_O2"]["stoich"] == 1.5 and "1.5O2(g)" in plan["equation"].replace(" ", "")
    assert json.loads((root / "bundle.json").read_text())["o2_reference"]["box_A"] >= 15
    # a static job with another cutoff must be caught
    s = {k: JobSpec(**v["static"]) for k, v in plan["jobs"].items()}
    s["product"] = replace(s["product"], incar={**s["product"].incar, "ENCUT": 400})
    assert any("ENCUT" in p for p in check_matched_settings(s))
    with pytest.raises(ValueError):
        oxidation_bundle(tmp_path / "c", "small_box", atoms, rec, full, 3, o2_box=10.0)


def test_declared_o2_correction_is_applied_only_when_passed():
    from pvpassivator.ledger import ReferenceCorrection
    from pvpassivator.oxidation import O2_REFERENCE
    slab = Species("slab", {"Pb": 1, "I": 2})
    prod = Species("prod", {"Pb": 1, "I": 2, "O": 3})
    rx = oxidation_reaction(slab, prod, 3)
    E = {"slab": -10.0, "prod": -30.0, O2_REFERENCE.name: -10.0}
    plain = rx.delta_e(E)
    assert plain["delta_E_total_eV"] == pytest.approx(-5.0) and "corrections_applied" not in plain
    c = ReferenceCorrection("O2 overbinding fix", O2_REFERENCE.name, -0.5, "example fit, test only")
    cor = rx.delta_e(E, corrections=(c,))
    assert cor["delta_E_total_eV"] == pytest.approx(-5.0 + 1.5 * 0.5)
    assert cor["delta_E_uncorrected_eV"] == pytest.approx(-5.0) and cor["corrections_applied"][0]["source"]
    with pytest.raises(ValueError):
        ReferenceCorrection("no source", O2_REFERENCE.name, -0.5, "  ")
    with pytest.raises(ValueError):
        rx.delta_e(E, corrections=(ReferenceCorrection("wrong", "H2(g)", 0.1, "x"),))


def test_water_next_to_oxyanion(tmp_path):
    from pvpassivator.oxidation import oxyanion_water_poses, water_passivator
    atoms, rec = _toy_2x2(tmp_path)
    site = _top_inplane_iodide(atoms, rec)
    full, spec = oxyanion_preset_poses_first(atoms, rec, site)
    w = water_passivator()
    assert w.total_charge == 0 and w.multiplicity == 1 and w.atoms.get_chemical_formula() == "H2O"
    poses = oxyanion_water_poses(full, rec, spec.halide_atom_id, max_poses=40)
    assert poses
    oxy_o = np.where(full.arrays["origin"] == 1)[0]
    for f, sp in poses:
        wat = np.where(f.arrays["origin"] == 2)[0]
        assert len(wat) == 3 and len(f) == len(full) + 3
        assert np.all(f.arrays["origin"][oxy_o] == 1)          # oxyanion O keep their own tag
        if sp.site_kind == "oxyanion_O":
            dmin = min(f.get_distance(i, j, mic=True) for i in wat if f[i].symbol == "H" for j in oxy_o)
            assert dmin == pytest.approx(sp.height, abs=1e-6)


def oxyanion_preset_poses_first(atoms, rec, site):
    from pvpassivator.oxidation import oxyanion_preset_poses
    return oxyanion_preset_poses(atoms, rec, site, 3, "O-down")[0]


def test_cli_oxidize_writes_poses_and_refuses_overwrite(tmp_path):
    import json
    from pvpassivator.cli import main
    at = toy_slab(nlayers=5).repeat((1, 2, 2))
    at.write(tmp_path / "POSCAR", format="vasp")
    out = tmp_path / "ox"
    assert main(["oxidize", str(tmp_path / "POSCAR"), "--out", str(out), "--preset", "O-down", "--n", "3"]) == 0
    d = json.loads((out / "specs.json").read_text())
    assert d["halide_class"] == "in_plane" and d["poses"]
    assert all(p["formal_charge"] == 0 for p in d["poses"])
    assert (out / d["poses"][0]["folder"] / "start.extxyz").exists()
    assert main(["oxidize", str(tmp_path / "POSCAR"), "--out", str(out)]) == 2
    assert main(["oxidize", str(tmp_path / "POSCAR"), "--out", str(tmp_path / "x"), "--halide", "0"]) == 2


def test_lifted_out_is_relative_to_the_start_on_an_apical_site(tmp_path):
    """FAI-terminated face: an apical iodide sits ~3.2 A above its Pb, so an O-down oxyanion bound through one O
    is far above the Pb plane without being lifted out. Only a rise from its own start with no bond left counts."""
    from pvpassivator.oxidation import oxyanion_preset_poses, oxyanion_state
    from pvpassivator.surface import find_sites
    at = toy_slab(nlayers=4)                          # PbI2 | FAI | PbI2 | FAI -> FAI-terminated top
    at.write(tmp_path / "POSCAR", format="vasp")
    atoms, rec = load_surface(tmp_path / "POSCAR", surface_id="toyFAI")
    assert rec.faces["top"]["termination"] == "FAI"
    site = next(s for s in find_sites(atoms, rec, "top") if s.halide_class == "apical")
    full, spec = oxyanion_preset_poses(atoms, rec, site, 3, "O-down")[0]
    n = _normal(full, rec)
    h, o = spec.halide_index, list(range(len(full) - 3, len(full)))
    pb = min((i for i, s in enumerate(full.get_chemical_symbols()) if s == "Pb"),
             key=lambda i: full.get_distance(i, o[0], mic=True))
    # relaxed: the down O settles onto the Pb below (2.45 A); iodine stays ~4 A above the Pb plane
    bound = full.copy()
    bound.positions[[h] + o] -= (bound.get_distance(pb, o[0], mic=True) - 2.45) * n
    s = oxyanion_state(full, bound, rec, spec.halide_atom_id)
    assert s["oxyanion_class"] == "intact_bound" and s["halide_height_A"] > 3.5
    assert "lifted_out" not in s["flags"] and s["halide_rise_A"] < 0
    # risen 2.5 A from its start with no O-Pb or I-Pb bond left: lifted out
    up = full.copy()
    up.positions[[h] + o] += 2.5 * n
    s = oxyanion_state(full, up, rec, spec.halide_atom_id)
    assert "lifted_out" in s["flags"] and s["halide_rise_A"] == pytest.approx(2.5, abs=1e-6)
