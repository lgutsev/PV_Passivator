import numpy as np
import pytest

from pvpassivator.benchmark import (BenchRun, same_composition_deltas, offset_fit, matched_adsorption,
                                    spearman, discover_runs)


def _run(rid, comp, e=None, problems=(), sp=()):
    return BenchRun(rid, "p", "", comp, sum(comp.values()), e, "converged_inferred", 10, 0.0,
                    list(problems), "CONTCAR", 0.0, list(sp))


MU = {"Pb": -3.0, "I": -1.5, "C": -5.0, "H": -2.0, "N": -4.0}


def _energy(comp, mu):
    return sum(mu[k] * v for k, v in comp.items())


def test_offset_fit_recovers_constant_shifts_exactly():
    rng = np.random.default_rng(1)
    runs, e_dft, e_ml = [], {}, {}
    for i in range(12):
        comp = {k: int(rng.integers(1, 9)) for k in MU}
        r = _run(f"r{i}", comp)
        runs.append(r)
        e_ml[r.run_id] = float(rng.normal(-500, 20))
        e_dft[r.run_id] = e_ml[r.run_id] + _energy(comp, MU)       # MLIP differs only by per-element offsets
    fit = offset_fit(runs, e_dft, e_ml)
    assert fit["in_sample"]["max_abs"] < 1e-9 and fit["loo"]["max_abs"] < 1e-8
    assert all(abs(fit["mu"][k] - MU[k]) < 1e-9 for k in MU)


def test_offset_fit_loo_exposes_a_single_bad_structure():
    rng = np.random.default_rng(2)
    runs, e_dft, e_ml = [], {}, {}
    for i in range(15):
        comp = {k: int(rng.integers(1, 9)) for k in MU}
        r = _run(f"r{i}", comp)
        runs.append(r)
        e_ml[r.run_id] = 0.0
        e_dft[r.run_id] = _energy(comp, MU) + (0.5 if i == 7 else 0.0)
    rows = {x["run"]: x for x in offset_fit(runs, e_dft, e_ml)["rows"]}
    worst = max(rows, key=lambda k: abs(rows[k]["loo"]))
    assert worst == "r7" and rows["r7"]["loo"] == pytest.approx(0.5, abs=1e-6)


def test_offset_fit_refuses_underdetermined():
    runs = [_run("a", {"Pb": 1, "I": 2}), _run("b", {"Pb": 2, "I": 4})]
    with pytest.raises(ValueError):
        offset_fit(runs, {"a": 0, "b": 0}, {"a": 0, "b": 0})


def test_same_composition_deltas_and_adsorption_matching():
    slab = _run("slab", {"Pb": 4, "I": 8})
    poseA = _run("A", {"Pb": 4, "I": 8, "C": 1, "H": 5, "N": 2})
    poseB = _run("B", {"Pb": 4, "I": 8, "C": 1, "H": 5, "N": 2})
    mol = _run("mol", {"C": 1, "H": 5, "N": 2})
    e_dft = {"slab": -10.0, "A": -15.0, "B": -14.6, "mol": -4.0}
    e_ml = {"slab": -20.0, "A": -25.1, "B": -24.6, "mol": -4.2}
    sc = same_composition_deltas([slab, poseA, poseB, mol], e_dft, e_ml)
    (row,) = sc["rows"]
    assert row["run"] == "B" and row["ref"] == "A"
    assert row["dE_dft"] == pytest.approx(0.4) and row["dE_mlip"] == pytest.approx(0.5)
    ads = matched_adsorption([poseA, poseB], [slab], [mol], e_dft, e_ml)
    got = {r["complex"]: (r["E_ads_dft"], r["E_ads_mlip"]) for r in ads["rows"]}
    assert got["A"] == pytest.approx((-1.0, -0.9)) and got["B"] == pytest.approx((-0.6, -0.4))
    assert ads["stats"]["mae"] == pytest.approx(0.15)


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2], [1, 2]) is None


def _vasp_run(d, nsw, steps, nelect=None, last_dE=-1e-4):
    d.mkdir()
    (d / "INCAR").write_text(f"NSW = {nsw}\nIBRION = 2\nISIF = 2\n" + (f"NELECT = {nelect}\n" if nelect else ""))
    pos = "x\n1.0\n10 0 0\n0 10 0\n0 0 10\nPb I\n1 2\nDirect\n0 0 0\n0.5 0 0\n0 0.5 0\n"
    (d / "POSCAR").write_text(pos)
    (d / "CONTCAR").write_text(pos)
    lines = [f"{i+1:4d} F= {-10 - i*0.01:.8E} E0= {-10 - i*0.01:.8E}  d E ={(-0.01 if i < steps-1 else last_dE):.6E}"
             for i in range(steps)]
    (d / "OSZICAR").write_text("\n".join(lines) + "\n")


def test_discover_runs_tiers(tmp_path):
    root = tmp_path / "set"
    root.mkdir()
    _vasp_run(root / "conv", nsw=99, steps=20)
    _vasp_run(root / "nsw_small", nsw=10, steps=10, last_dE=-0.004)
    _vasp_run(root / "nsw_big", nsw=10, steps=10, last_dE=-0.08)
    _vasp_run(root / "charged", nsw=99, steps=20, nelect=20)        # 14 + 2*7 = 28 -> q = +8
    (root / "X_old").mkdir()
    runs = {r.run_id.split("/")[-1]: r for r in discover_runs(root, "p")}
    assert set(runs) == {"conv", "nsw_small", "nsw_big", "charged"}
    assert runs["conv"].accepted and runs["conv"].single_point_ok
    assert not runs["nsw_small"].accepted and runs["nsw_small"].single_point_ok
    assert not runs["nsw_big"].single_point_ok and runs["nsw_big"].evaluable()
    assert runs["charged"].cell_charge == 8 and not runs["charged"].evaluable()
