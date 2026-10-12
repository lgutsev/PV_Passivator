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
