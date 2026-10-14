import json

import numpy as np
import pytest

pytest.importorskip("rdkit")

from pvpassivator.discovery.chem import Eligibility, check, fingerprints, tanimoto  # noqa: E402
from pvpassivator.discovery.space import enumerate_spec, validate_spec, SCHEMA  # noqa: E402
from pvpassivator.discovery.objectives import ObjectiveSpec, Criterion, nondominated, hypervolume  # noqa: E402
from pvpassivator.discovery.store import EvaluationStore, Record, BudgetExceeded, SpecChanged  # noqa: E402
from pvpassivator.discovery.oracles import (SyntheticOracle, CachedDFTOracle, NotComputed, ApprovalRequired,  # noqa: E402
                                            Oracle)
from pvpassivator.discovery.optimize import GPParEGO, TanimotoGP  # noqa: E402
from pvpassivator.discovery.loop import run_pool, run_representation_loop  # noqa: E402
from pvpassivator.discovery.proposer import ReplayProposer, CallableProposer, feedback_prompt  # noqa: E402
from pvpassivator.discovery.score import DeviceResponse, ScoreModel, Descriptor, score  # noqa: E402


def spec2():
    return ObjectiveSpec("t", (Criterion("f1", "objective", "max"), Criterion("sa_score", "objective", "min"),
                               Criterion("g", "constraint", lower=-1.0)))


def grid_spec(subs, **kw):
    return {"schema": SCHEMA, "kind": "combinatorial", "id": kw.get("id", "g"),
            "scaffolds": [{"id": "pea", "smiles": "[NH3+]CCc1ccc([*:1])cc1"}], "substituents": {"1": subs}}


# --- Tier 0 -------------------------------------------------------------------------------------------------------

def test_known_passivator_cations_pass_and_match_reference_formulas():
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    expect = {"[NH3+]CCC(F)(F)F": "C3H7F3N+", "[NH3+]CC(F)(F)C(F)(F)F": "C3H5F5N+",
              "[NH3+]CCc1ccc(O)cc1": "C8H12NO+", "[NH3+]CCc1ccccc1": "C8H12N+"}
    for smi, formula in expect.items():
        c, why = check(smi)
        assert c is not None, why
        from rdkit import Chem
        assert CalcMolFormula(Chem.MolFromSmiles(c.smiles)) == formula


@pytest.mark.parametrize("smi,reason", [
    ("NCCc1ccccc1", "net charge"),                 # neutral amine: protonation state must be declared
    ("[NH3+]CC=O", "forbidden"),                   # aldehyde
    ("[NH3+]CCI", "forbidden"),                    # alkyl iodide
    ("C[N+](C)(C)C", "anchor"),                    # quaternary: no N-H anchor
    ("[NH3+]CCC(=O)[O-]", "net charge"),           # zwitterion, net 0
    ("[NH3+]CC[Si](C)(C)C", "elements"),
    ("[NH3+]CCc1ccccc1.[I-]", "net charge"),       # counter-ion must not be inside the ion record
    ("not a smiles", "unparseable"),
])
def test_tier0_rejections(smi, reason):
    c, why = check(smi)
    assert c is None and any(reason in w for w in why), why


def test_anion_rules_accept_sulfonates_only_with_anion_eligibility():
    assert check("[O-]S(=O)(=O)c1ccc2ccccc2c1", Eligibility.anion())[0] is not None
    assert check("[O-]S(=O)(=O)c1ccc2ccccc2c1")[0] is None


def test_tanimoto_identity_and_range():
    X = fingerprints(["[NH3+]CCc1ccccc1", "[NH3+]CCc1ccc(F)cc1", "[NH3+]CCCC"])
    T = tanimoto(X, X)
    assert np.allclose(np.diag(T), 1) and (T >= 0).all() and (T <= 1).all() and T[0, 1] > T[0, 2]


# --- representations ----------------------------------------------------------------------------------------------

def test_protected_fields_are_refused():
    d = grid_spec(["[*][H]"])
    d["objectives"] = {"anything": 1}
    d["eligibility"] = {"max_heavy": 99}
    errs = validate_spec(d)
    assert any("protected" in e for e in errs)
    cands, rep = enumerate_spec(d)
    assert not rep.ok and cands == []


def test_enumeration_attaches_dedups_and_filters():
    cands, rep = enumerate_spec(grid_spec(["[*][H]", "[*]F", "[*]F", "[*]C=O", "[*]C(F)(F)F"]))
    smis = {c.smiles for c in cands}
    assert "[NH3+]CCc1ccc(F)cc1" in smis and "[NH3+]CCc1ccccc1" in smis
    assert rep.n_enumerated == 5 and rep.n_duplicates == 1 and rep.n_valid == 3
    assert sum(v for k, v in rep.rejections.items() if "forbidden" in k) == 1


def test_spec_needs_mapped_attachment_points_and_size_cap():
    d = grid_spec(["[*]F"])
    d["scaffolds"][0]["smiles"] = "[NH3+]CCc1ccc([*])cc1"
    assert any("map number" in e for e in validate_spec(d))
    big = {"schema": SCHEMA, "kind": "combinatorial", "id": "big",
           "scaffolds": [{"id": "x", "smiles": "[*:1]C([*:2])([*:3])[NH3+]"}],
           "substituents": {k: [f"[*]{'C' * i}" for i in range(1, 21)] for k in "123"}}
    assert any("exceeds" in e for e in validate_spec(big))


# --- objectives ---------------------------------------------------------------------------------------------------

def test_pareto_and_hypervolume():
    Y = np.array([[0, 1], [1, 0], [1, 1], [0.5, 0.5]])
    assert nondominated(Y).tolist() == [True, True, False, True]
    # union of [0,2]x[1,2] (2.0), [0.5,2]x[0.5,1] (0.75), [1,2]x[0,0.5] (0.5)
    assert hypervolume(Y, np.array([2.0, 2.0])) == pytest.approx(3.25)
    Y3 = np.array([[0.0, 0.0, 0.0]])
    assert hypervolume(Y3, np.ones(3), n_mc=20000) == pytest.approx(1.0, abs=0.02)


def test_single_proxy_objective_is_refused(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"id": "x", "criteria": [{"name": "E_ads", "role": "objective", "direction": "min"}]}))
    with pytest.raises(ValueError, match="at least two"):
        ObjectiveSpec.load(p)


# --- store --------------------------------------------------------------------------------------------------------

def test_store_budget_dedup_resume_and_spec_lock(tmp_path):
    p = tmp_path / "s.jsonl"
    s = EvaluationStore(p, "h1", budget=2.0)
    s.add(Record("K1", "C", "o", "computed", {"a": 1}, 1.0))
    with pytest.raises(ValueError, match="duplicate"):
        s.add(Record("K1", "C", "o", "computed", {"a": 1}, 0.0))
    s.add(Record("K2", "C", "o", "computed", {"a": 2}, 1.0))
    with pytest.raises(BudgetExceeded):
        s.add(Record("K3", "C", "o", "computed", {"a": 3}, 0.5))
    s.add(Record("H1", "C", "o", "validated", {"a": 9}, 5.0, split="holdout"))   # holdout is not search budget
    with pytest.raises(ValueError, match="status"):
        s.add(Record("K4", "C", "o", "proven", {}, 0.0))
    r = EvaluationStore(p, "h1", budget=2.0)                                       # resume from file
    assert len(r.records) == 3 and r.spent == 2.0 and {x.key for x in r.search_view()} == {"K1", "K2"}
    with pytest.raises(SpecChanged):
        EvaluationStore(p, "h2", budget=2.0)


# --- oracles ------------------------------------------------------------------------------------------------------

def test_expensive_oracle_needs_approval_and_cached_oracle_never_invents(tmp_path):
    class Dft(Oracle):
        id = "dft"
        cost = 50.0
    with pytest.raises(ApprovalRequired):
        Dft()
    Dft(approved=True)
    c, _ = check("[NH3+]CCc1ccccc1")
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"profile": "r2scan_d3bj_v1", "results": {c.inchikey: {"E_ads": -1.0}}}))
    o = CachedDFTOracle(p)
    assert o.evaluate(c).status == "computed"
    with pytest.raises(NotComputed):
        o.evaluate(check("[NH3+]CCCC")[0])


def test_synthetic_oracle_reproducible_and_seed_dependent():
    c, _ = check("[NH3+]CCc1ccc(F)cc1")
    a, b, z = SyntheticOracle(1).evaluate(c).values, SyntheticOracle(1).evaluate(c).values, SyntheticOracle(2).evaluate(c).values
    assert a == b and a != z


# --- optimizer / loop ---------------------------------------------------------------------------------------------

def test_gp_interpolates_and_parego_returns_unique_indices():
    cands, _ = enumerate_spec(grid_spec(["[*][H]", "[*]F", "[*]Cl", "[*]Br", "[*]C", "[*]OC", "[*]O", "[*]C(F)(F)F",
                                         "[*]C#N", "[*]C(N)=O", "[*]SC", "[*]CC"]))
    X = fingerprints([c.smiles for c in cands[:10]])
    y = 5 * tanimoto(X, X[:1])[:, 0] + tanimoto(X, X[5:6])[:, 0]      # smooth in kernel space
    gp = TanimotoGP(X, y)
    m, s = gp.predict(X)
    far = fingerprints(["NC(=[NH2+])C1CCCCC1"])
    assert np.corrcoef(m, y)[0, 1] > 0.9 and s.mean() < gp.predict(far)[1][0]
    o = SyntheticOracle(0)
    spec = ObjectiveSpec("t", (Criterion("syn_trap_suppression", "objective", "max"),
                               Criterion("sa_score", "objective", "min"),
                               Criterion("syn_integrity", "constraint", lower=-5)))
    obs = [(c, o.evaluate(c).values) for c in cands[:6]]
    idx = GPParEGO(spec, n_init=6).suggest(cands[6:], obs, 4, np.random.default_rng(0))
    assert len(idx) == len(set(idx)) == 4 and all(0 <= i < len(cands) - 6 for i in idx)


def test_run_pool_respects_budget_and_pool():
    cands, _ = enumerate_spec(grid_spec(["[*][H]", "[*]F", "[*]Cl", "[*]Br", "[*]C"]))
    o = SyntheticOracle(0, cost=0.1)
    spec = ObjectiveSpec("t", (Criterion("syn_trap_suppression", "objective", "max"), Criterion("sa_score", "objective", "min")))
    res = run_pool("B", cands, GPParEGO(spec, n_init=2), o, spec, budget=0.3, batch=2, seed=0)
    assert len(res.evaluated) == 3 and res.store.spent == pytest.approx(0.3)
    res = run_pool("A", cands, GPParEGO(spec), o, spec, budget=10, batch=2, seed=0)
    assert len(res.evaluated) == 5 and res.unused_budget == pytest.approx(9.5)


def test_representation_loop_logs_bad_proposals_without_spending(tmp_path):
    good = grid_spec(["[*][H]", "[*]F", "[*]Cl", "[*]Br", "[*]C", "[*]OC"], id="good")
    bad = dict(good, id="bad", thresholds={"syn_binding": 0})
    texts = ["no json here", "```json\n" + json.dumps(bad) + "\n```", json.dumps(good)]
    it = iter(texts)
    prop = CallableProposer(lambda prompt: (next(it), {"tokens": len(prompt)}))
    o = SyntheticOracle(0, cost=0.1)
    spec = ObjectiveSpec("t", (Criterion("syn_trap_suppression", "objective", "max"), Criterion("sa_score", "objective", "min")))
    res = run_representation_loop(prop, GPParEGO(spec, n_init=2), o, spec, budget=0.6, batch=2, rounds=3, seed=0)
    assert [p.accepted for p in res.proposals] == [False, False, True]
    assert any("protected" in e for e in res.proposals[1].errors)
    assert res.proposals[0].usage["tokens"] > 0
    assert len(res.evaluated) == 6 and {c.source for c in res.evaluated} == {"good"}


def test_feedback_prompt_excludes_holdout_and_names_protected_fields():
    spec = ObjectiveSpec("t", (Criterion("f1", "objective", "max"), Criterion("sa_score", "objective", "min")))
    s = EvaluationStore(None, spec.hash, 10)
    s.add(Record("K1", "[NH3+]CCc1ccccc1", "o", "prediction", {"f1": 1.0, "sa_score": 2.0}, 1, source="r1"))
    s.add(Record("K2", "[NH3+]CCCCSECRET", "o", "validated", {"f1": 9.0, "sa_score": 1.0}, 1, split="holdout"))
    p = feedback_prompt(s, spec, [{"id": "r1", "n_valid": 3, "n_enumerated": 4}])
    assert "[NH3+]CCc1ccccc1" in p and "SECRET" not in p and "thresholds" in p


def test_replay_proposer_exhausts(tmp_path):
    f = tmp_path / "r.json"
    f.write_text("{}")
    r = ReplayProposer([f])
    assert r.propose("x")[0] == "{}" and r.propose("x")[0] == ""


# --- device-referenced score --------------------------------------------------------------------------------------

def toy_model():
    # TOY response for tests only (monotone PCE drop with interface trap density); not SCAPS output.
    resp = DeviceResponse((9.0, 10.0, 11.0, 12.0, 13.0, 14.0), (31.0, 30.5, 29.0, 25.0, 20.0, 16.0), "toy")
    return ScoreModel(resp, n0_cm2=1e13, n_res_cm2=1e10, mu_eV=-0.5, temperature_K=373)


def test_coverage_saturates_so_overbinding_is_not_rewarded():
    m = toy_model()
    th = m.coverage(np.array([-0.2, -0.5, -0.8, -1.5, -3.0]))
    assert th[1] == pytest.approx(0.5) and th[3] > 0.999 and th[4] - th[3] < 1e-6
    assert m.pce(-3.0, 0.9) == pytest.approx(m.pce(-1.5, 0.9), abs=1e-6)


def test_trap_removal_and_new_traps_drive_the_score():
    m = toy_model()
    assert m.pce(-1.5, 0.99) > m.pce(-1.5, 0.5) > m.pce(-1.5, 0.0)
    assert m.pce(-1.5, 0.99, n_new=5e12) < m.pce(-1.5, 0.99)
    r = score(m, Descriptor(-1.5, 0.0, "dft"), Descriptor(0.99, 0.0, "dft"))
    assert r.pce_p10 == pytest.approx(r.pce_p90) and r.delta_pce_mean > 0
    u = score(m, Descriptor(-0.5, 0.3, "mlip"), Descriptor(0.9, 0.1, "dft_on_mlip_geometry"))
    assert u.pce_p90 - u.pce_p10 > 1.0 and u.tiers["dg_bind"] == "mlip"


def test_score_refuses_to_extrapolate_beyond_scaps_table():
    m = toy_model()
    with pytest.raises(ValueError, match="outside the SCAPS table"):
        score(m, Descriptor(-2.0, 0.0), Descriptor(1.0, 0.0), Descriptor(1e15, 0.0))   # above the swept range
    lo = DeviceResponse((10.5, 14.0), (29.5, 16.0), "toy")
    with pytest.raises(ValueError, match="outside the SCAPS table"):
        score(ScoreModel(lo, 1e13, 1e10, -0.5), Descriptor(-2.0, 0.0), Descriptor(1.0, 0.0))   # below it
