"""Checks of the optional ALCHEMI backend (pvpassivator.alchemi_backend).

Skipped unless nvalchemi-toolkit is importable; the model-based checks also need a MACE checkpoint in
``PVPASS_MACE_MODEL`` (they run on the CPU on a toy slab, about a minute each).  These establish that the SAME model
gives the same numbers through both interfaces and that identity, constraints and failure records survive batching.
They say nothing about whether the model is right for the chemistry.

    PVPASS_MACE_MODEL=/path/to/model  <alchemi env>/python -m pytest tests/test_alchemi_backend.py
"""
import os

import numpy as np
import pytest
from ase.build import molecule

pytest.importorskip("nvalchemi")
from pvpassivator import alchemi_backend as B  # noqa: E402
from test_surface_poses_prep import toy_slab  # noqa: E402

MODEL = os.environ.get("PVPASS_MACE_MODEL")
needs_model = pytest.mark.skipif(not MODEL or not os.path.exists(MODEL or ""), reason="set PVPASS_MACE_MODEL")


def make(h=2.8, tilt=0.0, reps=(1, 1, 1)):
    a = toy_slab(vac=24.0).repeat(reps)
    n = len(a)
    m = molecule("NH3")
    m.rotate(90, "y")
    m.rotate(tilt, "z")
    m.positions += [a.positions[:, 0].max() + h, 3.0, 3.0] - m.positions[0]
    a += m
    a.set_array("atom_id", np.arange(len(a)) + 7)
    a.set_array("role", np.r_[np.zeros(n, int), np.full(len(m), 2)])
    a.set_array("mol_id", np.r_[np.zeros(n, int), np.ones(len(m), int)])
    frozen = np.ones(len(a), bool)
    frozen[n:] = False
    return a, frozen


@pytest.fixture(scope="module")
def model():
    return B.load_mace(MODEL, "cpu", "float64")


@pytest.fixture(scope="module")
def native():
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=MODEL, device="cpu", default_dtype="float64")


def test_conversion_keeps_geometry_order_flags_and_ids():
    (a, fa), (b, fb) = make(2.6), make(3.0, 30.0, reps=(1, 1, 2))          # two different sizes in one batch
    b.pbc = (False, True, True)                                              # partial periodicity must survive
    batch, metas = B.to_batch([a, b], [fa, fb], ["cand-A", "cand-B"], device="cpu")
    assert batch.num_graphs == 2 and [m.candidate_id for m in metas] == ["cand-A", "cand-B"]
    pos = B._split_positions(batch, metas)
    for k, (src, fz) in enumerate(((a, fa), (b, fb))):
        assert np.array_equal(pos[k], src.positions)                         # float64 in, float64 out, same order
        assert np.array_equal(B._split(batch, metas, "atomic_numbers")[k].ravel(), src.numbers)
        assert np.array_equal(B._split(batch, metas, "frozen")[k].ravel(), fz)
        assert np.array_equal(metas[k].arrays["atom_id"], src.arrays["atom_id"])
        assert np.array_equal(metas[k].arrays["role"], src.arrays["role"])
    assert np.allclose(batch.cell.cpu().numpy()[0], a.cell.array) and np.allclose(batch.cell.cpu().numpy()[1], b.cell.array)
    assert batch.pbc.cpu().numpy().tolist() == [[True, True, True], [False, True, True]]
    assert batch.system_id.cpu().numpy().ravel().tolist() == [0, 1]
    B._check_identity(batch, metas)
    metas[0].numbers = metas[0].numbers[::-1].copy()                          # a scrambled order must be caught
    with pytest.raises(RuntimeError, match="atomic numbers or atom order"):
        B._check_identity(batch, metas)
    with pytest.raises(ValueError, match="frozen mask"):
        B.to_batch([a], [fa[:-1]], device="cpu")


@needs_model
def test_single_points_match_native_and_do_not_depend_on_the_batch(model, native):
    systems = [make(2.6)[0], make(3.1, 25.0)[0], make(2.8, 50.0, reps=(1, 1, 2))[0]]
    ref = []
    for a in systems:
        c = a.copy()
        c.calc = native
        ref.append((c.get_potential_energy(), c.get_forces()))
    one = B.single_points(model, systems, 1, device="cpu")
    for r, (e, f) in zip(one, ref):
        assert abs(r["energy"] - e) < 1e-9 and np.abs(r["forces"] - f).max() < 1e-9      # wrapper == native
    for variant in (B.single_points(model, systems, 3, device="cpu"),                     # one multi-system batch
                    B.single_points(model, systems[::-1], 3, device="cpu")[::-1],         # reordered
                    B.single_points(model, [systems[0], systems[0], systems[2], systems[1]], 4, device="cpu")[1:2]
                    + B.single_points(model, [systems[1], systems[0]], 2, device="cpu")[:1]
                    + B.single_points(model, [systems[2]], 1, device="cpu")):             # repeated / other companions
        assert len(variant) == 3
    multi = B.single_points(model, systems, 3, device="cpu")
    rev = B.single_points(model, systems[::-1], 3, device="cpu")[::-1]
    rep = B.single_points(model, [systems[0], systems[1], systems[0]], 3, device="cpu")
    for k in range(3):
        assert abs(multi[k]["energy"] - one[k]["energy"]) < 1e-9 and np.abs(multi[k]["forces"] - one[k]["forces"]).max() < 1e-9
        assert abs(rev[k]["energy"] - one[k]["energy"]) < 1e-9 and np.abs(rev[k]["forces"] - one[k]["forces"]).max() < 1e-9
    assert abs(rep[0]["energy"] - rep[2]["energy"]) < 1e-9 and abs(rep[0]["energy"] - one[0]["energy"]) < 1e-9


@needs_model
def test_partial_periodicity_matches_native(model, native):
    a = make(2.8)[0]                                   # 24 A of vacuum along x, far beyond the 6 A cutoff
    slab = a.copy()
    slab.pbc = (False, True, True)
    full = B.single_points(model, [a], 1, device="cpu")[0]
    part = B.single_points(model, [slab], 1, device="cpu")[0]
    c = slab.copy()
    c.calc = native
    assert abs(part["energy"] - c.get_potential_energy()) < 1e-9 and np.abs(part["forces"] - c.get_forces()).max() < 1e-9
    assert abs(part["energy"] - full["energy"]) < 1e-9          # the vacuum makes the two conventions equal here


@needs_model
def test_relaxation_freezes_records_and_isolates(model):
    (a, fa), (b, fb) = make(2.6), make(3.1, 25.0)
    recs, acct = B.relax(model, [a, b], [fa, fb], fmax=0.2, steps=60, ids=["A", "B"], device="cpu")
    assert [r.candidate_id for r in recs] == ["A", "B"] and acct["batches"] == 1
    for r, (src, fz) in zip(recs, ((a, fa), (b, fb))):
        assert r.status == "converged" and r.fmax_free <= 0.2 * 1.02 and r.frozen_max_move_any_step == 0.0
        assert np.array_equal(r.atoms.positions[fz], src.positions[fz])                  # frozen: bit-identical
        assert not np.array_equal(r.atoms.positions[~fz], src.positions[~fz])
        for key in ("atom_id", "role", "mol_id"):
            assert np.array_equal(r.atoms.arrays[key], src.arrays[key])
        assert r.atoms.arrays["model_forces"].shape == (len(src), 3) and r.atoms.info["model_energy"] == r.energy
        assert np.abs(r.atoms.arrays["model_forces"][fz]).max() > 0                       # raw forces, not the zeroed ones
    alone, _ = B.relax(model, [a], [fa], fmax=0.2, steps=60, ids=["A"], device="cpu")
    assert alone[0].steps_to_convergence == recs[0].steps_to_convergence                   # companions change nothing
    assert abs(alone[0].energy - recs[0].energy) < 1e-8
    assert np.abs(alone[0].atoms.positions - recs[0].atoms.positions).max() < 1e-8
    short, _ = B.relax(model, [b], [fb], fmax=1e-4, steps=3, ids=["B3"], device="cpu")
    assert short[0].status == "step_limit" and short[0].steps_to_convergence is None and short[0].atoms is not None
    bad = a.copy()
    bad.positions[-1] = bad.positions[-2] + 1e-9                                           # two atoms on top of each other
    mixed, _ = B.relax(model, [bad, b], [fa, fb], fmax=0.2, steps=60, ids=["bad", "B"], device="cpu")
    assert [r.candidate_id for r in mixed] == ["bad", "B"] and all(r.status in B.STATUSES for r in mixed)
    assert mixed[0].status != "converged"                                                 # the broken one is not passed off
