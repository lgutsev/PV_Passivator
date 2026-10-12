import numpy as np
import pytest

from pvpassivator.surface import load_surface, find_sites, unwrap_along
from pvpassivator.oxidation import oxyanion_poses
from pvpassivator.render import display_cutout, _frame
from test_surface_poses_prep import toy_slab


def _iodate_slab(tmp_path):
    at = toy_slab(nlayers=5)
    at.write(tmp_path / "POSCAR", format="vasp")
    atoms, rec = load_surface(tmp_path / "POSCAR", surface_id="toy")
    z = unwrap_along(atoms, rec.normal_axis)
    site = min((s for s in find_sites(atoms, rec, "top") if s.kind == "exposed_I"),
               key=lambda s: abs(z[s.atom_index] - rec.faces["top"]["edge_coordinate_A"]))
    full, spec = oxyanion_poses(atoms, rec, site, n_oxygen=3)[0]
    return full, rec, spec


def test_frame_puts_outward_normal_up(tmp_path):
    full, rec, _ = _iodate_slab(tmp_path)
    assert rec.normal_axis == 0                         # toy slab normal is x
    R, A, B = _frame(full, rec.normal_axis, "top")
    assert np.allclose(R @ full.cell.array[0] / np.linalg.norm(full.cell.array[0]), [0, 0, 1])
    assert abs(A[2]) < 1e-9 and abs(B[2]) < 1e-9
    R2, _, _ = _frame(full, rec.normal_axis, "bottom")
    assert np.allclose(R2 @ full.cell.array[0] / np.linalg.norm(full.cell.array[0]), [0, 0, -1])


def test_cutout_shows_focus_once_with_its_oxyanion(tmp_path):
    full, rec, spec = _iodate_slab(tmp_path)
    c = display_cutout(full, spec.halide_index, normal_axis=rec.normal_axis, radius=7.0)
    assert c.focus is not None and np.allclose(c.positions[c.focus], 0)
    assert int((c.source_index == spec.halide_index).sum()) == 1     # periodic copies of the focus dropped
    assert int((c.symbols == "O").sum()) == 3
    o = np.flatnonzero(c.symbols == "O")
    io = [(i, j) for i, j in c.bonds if {i, j} & {c.focus} and {i, j} & set(o)]
    assert len(io) == 3                                               # I-O bonds of the iodate
    # every halide kept is bonded to a kept metal or is the focus
    pb = set(np.flatnonzero(c.symbols == "Pb"))
    for h in np.flatnonzero(c.symbols == "I"):
        if h == c.focus:
            continue
        assert any((h in b) and (set(b) & pb) for b in c.bonds)
    # metals bonded to an O donor lose their octahedron
    for m in c.no_octahedron:
        assert c.symbols[m] == "Pb" and any(m in b and c.symbols[b[0] if b[1] == m else b[1]] == "O" for b in c.bonds)


def test_render_needs_ovito_only_when_called(tmp_path):
    pytest.importorskip("ovito")
    from pvpassivator.render import render
    full, rec, spec = _iodate_slab(tmp_path)
    out = render(full, tmp_path / "x.png", spec.halide_index, normal_axis=rec.normal_axis, size=(200, 150),
                 radius=7.0)
    assert out.exists() and out.stat().st_size > 1000
