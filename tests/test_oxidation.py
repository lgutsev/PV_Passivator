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
