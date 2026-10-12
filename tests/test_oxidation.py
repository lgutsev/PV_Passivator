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
