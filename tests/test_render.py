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
        if m in c.incomplete:
            continue
        assert c.symbols[m] == "Pb" and any(m in b and c.symbols[b[0] if b[1] == m else b[1]] == "O" for b in c.bonds)


def test_every_drawn_octahedron_is_complete(tmp_path):
    from ase.neighborlist import neighbor_list
    full, rec, spec = _iodate_slab(tmp_path)
    sym = np.array(full.get_chemical_symbols())
    i, j = neighbor_list("ij", full, 3.6)
    cn = {int(m): int(np.sum((i == m) & (sym[j] == "I"))) for m in np.flatnonzero(sym == "Pb")}
    for radius in (7.0, 10.0, 13.0):
        c = display_cutout(full, spec.halide_index, normal_axis=rec.normal_axis, radius=radius)
        assert set(c.incomplete) <= set(c.no_octahedron)
        for m in np.flatnonzero(c.symbols == "Pb"):
            if m in c.no_octahedron:
                continue
            shown = sum(1 for b in c.bonds if m in b and c.symbols[b[0] if b[1] == m else b[1]] == "I")
            assert shown == cn[int(c.source_index[m])]


def test_render_needs_ovito_only_when_called(tmp_path):
    pytest.importorskip("ovito")
    from pvpassivator.render import render
    full, rec, spec = _iodate_slab(tmp_path)
    out = render(full, tmp_path / "x.png", spec.halide_index, normal_axis=rec.normal_axis, size=(200, 150),
                 radius=7.0)
    assert out.exists() and out.stat().st_size > 1000


def test_web_export_matches_cutout(tmp_path):
    import json
    from pvpassivator.render import cutout_to_web
    full, rec, spec = _iodate_slab(tmp_path)
    c = display_cutout(full, spec.halide_index, normal_axis=rec.normal_axis, radius=9.0)
    o_atoms = [i for i in range(len(full)) if full.arrays["origin"][i] == 1]
    w = json.loads(json.dumps(cutout_to_web(c, highlight=o_atoms)))      # plain JSON round trip
    assert len(w["atoms"]) == len(c.positions) and w["bonds"] == [list(b) for b in c.bonds]
    assert [a["role"] for a in w["atoms"]].count("focus") == 1
    assert [a["role"] for a in w["atoms"]].count("highlight") == 3
    assert all(a["color"].startswith("#") and len(a["color"]) == 7 for a in w["atoms"])
    drawn = {o["metal"] for o in w["octahedra"]}
    assert drawn.isdisjoint(w["no_octahedron"])
    for o in w["octahedra"]:
        assert all(w["atoms"][h]["el"] == "I" for h in o["halides"])


def test_render_with_fragment_highlight(tmp_path):
    pytest.importorskip("ovito")
    from pvpassivator.render import render
    full, rec, spec = _iodate_slab(tmp_path)
    o_atoms = [i for i in range(len(full)) if full.arrays["origin"][i] == 1]
    out = render(full, tmp_path / "hl.png", spec.halide_index, normal_axis=rec.normal_axis, size=(200, 150),
                 radius=7.0, highlight=o_atoms)
    assert out.exists() and out.stat().st_size > 1000


def test_compose_panels(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pvpassivator.compose import compose_panels
    paths = []
    for k in range(3):
        p = tmp_path / f"p{k}.png"
        plt.imsave(p, np.full((90, 120, 3), 0.2 + 0.2 * k))
        paths.append(p)
    out = compose_panels(paths, tmp_path / "fig.png", ncols=2, titles=["A", "B", "C"], legend=["Pb", "I", "O"],
                         focus_label="I (oxidised)", scale_bar={"px_per_A": 10.0, "length_A": 5, "panels": [0]},
                         dpi=100)
    img = plt.imread(out)
    assert img.shape[1] > img.shape[0] > 0
    with pytest.raises(ValueError):
        compose_panels(paths, tmp_path / "bad.png", titles=["only one"])
