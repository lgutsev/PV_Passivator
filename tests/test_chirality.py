"""Geometry, symmetry and PBC checks for pvpassivator.chirality.

Real-structure tests use the experimental COD CIFs in examples/structures/mba2pbi4 (H atoms riding).
Fixtures named ``synthetic_*`` are constructed here and carry no physical meaning beyond their symmetry.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import make_supercell
from ase.io import read

from pvpassivator import chirality as ch

SDIR = Path(__file__).resolve().parents[1] / "examples" / "structures" / "mba2pbi4"
CIFS = {"R": "R-MBA2PbI4_COD7203475.cif", "S": "S-MBA2PbI4_COD7203476.cif", "rac": "rac-MBA2PbI4_COD2201646.cif"}
KEYS = ("tau_ax", "tau_eq", "tau_org")


@pytest.fixture(scope="module")
def cif():
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, f in CIFS.items():
            out[k] = read(SDIR / f)
    return out


def _desc(atoms, **kw):
    top = ch.build_topology(atoms, **kw)
    return top, ch.frame_descriptors(top, atoms)


# ------------------------------------------------------------------ topology and stereochemistry

def test_topology_and_stereocentres(cif):
    exp = {"R": (8, 0, 2), "S": (0, 8, 2), "rac": (2, 2, 1)}
    for k, a in cif.items():
        s = ch.build_topology(a).summary()
        assert (s["stereocentres"]["R"], s["stereocentres"]["S"], s["n_layers"]) == exp[k]
        assert s["n_molecules"] == 2 * s["n_metal"]           # A2PbI4 stoichiometry
        assert s["faces"]["+"] == s["faces"]["-"]


def test_stereo_descriptor_convention():
    # priorities 1,2,3 clockwise seen from +z with priority 4 along -z  ->  R
    c = np.zeros(3)
    p = [np.array([np.cos(t), np.sin(t), 0.33]) for t in np.radians([90, -30, 210])]
    assert ch.stereo_descriptor(c, p) == "R"
    assert ch.stereo_descriptor(c, [p[0], p[2], p[1]]) == "S"
    with pytest.raises(ch.TopologyError):
        ch.stereo_descriptor(c, [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, -1, 0])])


# ------------------------------------------------------------------ symmetry: reflection, rotation, conventions

def test_inversion_reverses_every_signed_descriptor(cif):
    top, d = _desc(cif["R"])
    inv = ch.invert(cif["R"])
    di = ch.frame_descriptors(top, inv)          # framework links are keyed by bridging halide: elementwise
    for k in ("tau_ax", "tau_eq"):
        np.testing.assert_allclose(di[k], -d[k], atol=1e-10)
    # organic links select periodic images by their reference separation, which inversion reverses,
    # so the inverted crystal gets its own topology (a trajectory never inverts)
    top_i, di2 = _desc(inv)
    for k in KEYS:
        np.testing.assert_allclose(np.sort(di2[k]), np.sort(-d[k]), atol=1e-10)
    assert set(ch.stereocentres(top, inv)) == {"S"}
    assert top_i.summary()["stereocentres"] == {"R": 0, "S": 8}


def test_mirror_plane_reverses_descriptors(cif):
    a = cif["R"].copy()
    top, d = _desc(a)
    m = a.copy()
    m.positions[:, 0] *= -1          # mirror x -> -x with the cell mirrored too (left-handed cell is fine for geometry)
    m.cell[:, 0] *= -1
    dm = ch.frame_descriptors(top, m)
    for k in KEYS:
        np.testing.assert_allclose(dm[k], -d[k], atol=1e-10)


def test_proper_rotation_invariance(cif):
    rng = np.random.default_rng(0)
    for key in ("R", "rac"):
        a = cif[key].copy()
        top, d = _desc(a)
        q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        b = a.copy()
        b.positions = a.positions @ q.T
        b.cell = a.cell.array @ q.T
        db = ch.frame_descriptors(top, b)
        for k in KEYS:
            np.testing.assert_allclose(db[k], d[k], atol=1e-10)


def test_normal_sign_convention_does_not_matter(cif):
    top, d = _desc(cif["R"])
    top2 = ch.build_topology(cif["R"], normal=-top.normal)
    d2 = ch.frame_descriptors(top2, cif["R"])
    for k in KEYS:
        np.testing.assert_allclose(d2[k], d[k], atol=1e-10)


def test_atom_order_does_not_matter(cif):
    a = cif["R"]
    _, d = _desc(a)
    perm = np.random.default_rng(1).permutation(len(a))
    _, dp = _desc(a[perm])
    for k in KEYS:
        np.testing.assert_allclose(np.sort(dp[k]), np.sort(d[k]), atol=1e-10)


# ------------------------------------------------------------------ periodic boundaries and cell size

def test_wrapping_and_lattice_shifts_do_not_matter(cif):
    a = cif["R"].copy()
    top, d = _desc(a)
    rng = np.random.default_rng(2)
    b = a.copy()
    b.positions += rng.integers(-2, 3, size=(len(a), 3)) @ a.cell.array    # every atom to a random image
    db = ch.frame_descriptors(top, b)
    c = a.copy()
    c.positions += np.array([3.1, -4.7, 11.3])                            # rigid origin shift, then wrap
    c.wrap()
    dc = ch.frame_descriptors(top, c)
    for k in KEYS:
        np.testing.assert_allclose(db[k], d[k], atol=1e-9)
        np.testing.assert_allclose(dc[k], d[k], atol=1e-9)


def test_supercell_gives_same_means(cif):
    for key in ("R", "rac"):
        a = cif[key]
        _, d = _desc(a)
        _, ds = _desc(make_supercell(a, np.diag([2, 2, 1])))
        for k in ("ax", "eq", "org"):
            assert ds[f"mean_{k}"] == pytest.approx(d[f"mean_{k}"], abs=1e-10)
            assert len(ds[f"tau_{k}"]) == 4 * len(d[f"tau_{k}"])


# ------------------------------------------------------------------ physical controls

def test_racemate_is_net_achiral_with_signed_pairs(cif):
    top, d = _desc(cif["rac"])
    for k in KEYS:
        assert d[k].mean() == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_allclose(np.sort(d[k]), -np.sort(d[k])[::-1], atol=1e-10)  # +/- partners
    assert np.max(np.abs(d["tau_ax"])) > 0.05      # locally twisted, net zero: domain cancellation, not achiral links


def test_R_and_S_crystals_have_opposite_signs(cif):
    _, dR = _desc(cif["R"])
    _, dS = _desc(cif["S"])
    for k in ("ax", "eq", "org"):
        assert np.sign(dR[f"mean_{k}"]) == -np.sign(dS[f"mean_{k}"]) != 0
    # separately refined crystals are NOT exact mirrors: the per-link twists agree to ~0.06, not 1e-10
    np.testing.assert_allclose(np.sort(dS["tau_org"]), np.sort(-dR["tau_org"]), atol=0.07)


def synthetic_layer(rot_deg=0.0, a=8.9, c=16.0):
    """SYNTHETIC: one PbI4 layer in a sqrt2 x sqrt2 cell with four CH3NH3+ (along +/-z, staggered H),
    optionally with an in-plane alternating octahedral rotation (achiral: keeps the layer mirror plane)."""
    pb = [(0, 0, 0), (a / 2, a / 2, 0)]
    br = [(a / 4, a / 4, 0), (3 * a / 4, a / 4, 0), (a / 4, 3 * a / 4, 0), (3 * a / 4, 3 * a / 4, 0)]
    phi = np.radians(rot_deg)
    br2 = []
    for x, y, z in br:
        r = np.array([x, y]) - np.round(np.array([x, y]) / a) * a     # vector from the nearest image of Pb(0,0)
        br2.append((x - phi * r[1], y + phi * r[0], z))
    ax = [(px, py, s * 3.2) for px, py, _ in pb for s in (1, -1)]
    syms, pos = ["Pb"] * 2 + ["I"] * 8, pb + br2 + ax
    for sx, sy in ((a / 2, 0), (0, a / 2)):
        for s in (1, -1):
            n, cc = np.array([sx, sy, s * 4.3]), np.array([sx, sy, s * 5.8])
            syms += ["N", "C"]
            pos += [tuple(n), tuple(cc)]
            for k in range(3):
                t = 2 * np.pi * k / 3
                syms += ["H", "H"]
                pos += [tuple(n + [0.95 * np.cos(t), 0.95 * np.sin(t), -s * 0.33]),
                        tuple(cc + [1.03 * np.cos(t + np.pi / 3), 1.03 * np.sin(t + np.pi / 3), s * 0.36])]
    return Atoms(syms, positions=pos, cell=[a, a, c], pbc=True)


@pytest.mark.parametrize("rot", [0.0, 8.0])
def test_synthetic_symmetric_layer_is_achiral(rot):
    at = synthetic_layer(rot)
    top, d = _desc(at)
    assert top.summary()["stereocentres"] == {"R": 0, "S": 0}
    for k in KEYS:
        np.testing.assert_allclose(d[k], 0.0, atol=1e-10)
    angles = ch.mxm_angles(top, at)
    assert np.all(angles <= 180.0 + 1e-9) and (rot > 0) == bool(np.any(angles < 179.0))


def test_synthetic_helix_and_zigzag_twists():
    """SYNTHETIC site vectors: a helix gives uniform sin(phi), its mirror -sin(phi); a planar zig-zag alternates."""
    def tau(v, e):
        return np.einsum("ij,ij->i", np.cross(v[:-1], v[1:]), e)
    phi = np.radians(20)
    v = np.array([[0, np.cos(k * phi), np.sin(k * phi)] for k in range(6)])
    e = np.tile([1.0, 0, 0], (5, 1))
    np.testing.assert_allclose(tau(v, e), np.sin(phi))
    vm, em = v * [-1, 1, 1], e * [-1, 1, 1]                       # mirror x -> -x
    np.testing.assert_allclose(tau(vm, em), -np.sin(phi))
    np.testing.assert_allclose(tau(v[::-1], -e), np.sin(phi))       # walking backwards: unchanged
    z = np.array([[0, np.cos(t), np.sin(t)] for t in np.radians([10, -10] * 3)])
    tz = tau(z, e)
    assert np.all(np.abs(tz) > 0.3) and tz.mean() == pytest.approx(0.0, abs=0.07)


# ------------------------------------------------------------------ trajectory pipeline on a synthetic trajectory

def test_synthetic_trajectory_pipeline(cif):
    """SYNTHETIC: R crystal + independent Gaussian noise (0.03 A) per frame, half the frames scattered over images."""
    a = cif["R"]
    top = ch.build_topology(a)
    rng = np.random.default_rng(3)
    frames, wrapped = [], []
    for t in range(12):
        f = a.copy()
        f.positions += rng.normal(scale=0.03, size=f.positions.shape)
        frames.append(f)
        g = f.copy()
        g.positions += rng.integers(-1, 2, size=(len(a), 3)) @ a.cell.array
        wrapped.append(g)
    r1 = ch.analyze_trajectory(top, frames)
    r2 = ch.analyze_trajectory(top, wrapped)
    for k in ("mean_ax", "mean_eq", "mean_org", "tau_org", "mxm_angle", "headgroup_tilt", "hbond"):
        np.testing.assert_allclose(np.asarray(r1[k], float), np.asarray(r2[k], float), atol=1e-9)
    assert r1["tau_org"].shape == (12, len(top.org_links))
    assert r1["nh_vectors"].shape == (12, 24, 3)
    assert all(set(ch.stereocentres(top, f)) == {"R"} for f in frames)
    acf = ch.rotational_acf(r1["nh_vectors"], 3)
    assert acf[0] == pytest.approx(1.0) and np.all(acf[1:] > 0.95)


# ------------------------------------------------------------------ hydrogen bonds and statistics

def test_hbond_lifetimes_continuous_vs_intermittent():
    h = np.tile([1, 1, 1, 0], 25)[:, None].astype(bool)
    r = ch.hbond_lifetimes(h, 4)
    assert r["occupancy"] == pytest.approx(0.75)
    assert r["C"][1] == pytest.approx(2 / 3, rel=0.03) and r["C"][4] == pytest.approx(1.0)
    assert r["S"][4] == 0.0 and r["S"][0] == pytest.approx(1.0)


def test_hbond_criteria_and_sensitivity(cif):
    top = ch.build_topology(cif["R"])
    tight = ch.hbond_matrix(top, cif["R"], ch.HBondCriteria(2.6, 150))[0].sum()
    loose = ch.hbond_matrix(top, cif["R"], ch.HBondCriteria(3.4, 110))[0].sum()
    assert loose >= tight
    sens = ch.hbond_sensitivity(top, [cif["R"]], (3.0, 3.4), (120.0,))
    assert sens[(3.4, 120.0)] >= sens[(3.0, 120.0)]


def test_block_average_and_autocorr_time():
    rng = np.random.default_rng(4)
    phi, n = 0.9, 2 ** 15
    x = np.empty(n)
    x[0] = 0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    tau = ch.integrated_autocorr_time(x)
    assert tau == pytest.approx((1 + phi) / (2 * (1 - phi)), rel=0.3)
    b = ch.block_average(x)
    assert b["se"] / b["naive_se"] == pytest.approx(np.sqrt(2 * tau), rel=0.35)


def test_cross_correlation_lag_and_chirality_census():
    rng = np.random.default_rng(5)
    x = rng.normal(size=2000)
    y = np.roll(x, 3) + 0.1 * rng.normal(size=2000)
    lags, r = ch.cross_correlation(x, y, 6)
    assert lags[np.argmax(r)] == 3
    assert ch.degree_of_chirality([0.2, 0.1, -0.1, 0.3]) == pytest.approx(0.5)
    assert ch.degree_of_chirality([0.2, -0.01], dead_band=0.05) == 1.0


def test_spatial_sign_correlation_detects_domains():
    at = Atoms("X8", positions=[[i * 2.0, 0, 0] for i in range(8)], cell=[16, 20, 20], pbc=True)
    vals = np.array([1, 1, 1, 1, -1, -1, -1, -1.0])   # two domains of four
    out = ch.spatial_sign_correlation(at.positions, vals, at, np.array([0, 3, 9]))
    near, far = out[(0.0, 3.0)][0], out[(3.0, 9.0)][0]
    assert near > far
