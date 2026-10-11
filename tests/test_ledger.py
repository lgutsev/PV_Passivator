import pytest
from ase import Atoms

from pvpassivator.ledger import (Species, Reaction, ELECTRON, adsorption_reaction,
                                 UnbalancedReaction, MissingReference, identify_units)

# 335 FAI-rich slab recovered from archive: 45 FA, 36 Pb, 117 I
SLAB = Species("FAI_slab", {"C": 45, "H": 225, "N": 90, "Pb": 36, "I": 117},
               formal_units={"FA+": 45, "Pb2+": 36, "I-": 117})
PEAI = Species("PEAI", {"C": 8, "H": 12, "N": 1, "I": 1}, kind="salt")
FAI = Species("FAI", {"C": 1, "H": 5, "N": 2, "I": 1}, kind="salt")


def plus(a, b, name, sign=1):
    c = dict(a.composition)
    for e, n in b.composition.items():
        c[e] = c.get(e, 0) + sign * n
    return Species(name, {e: n for e, n in c.items() if n})


def test_slab_formal_neutral():
    assert SLAB.formal_charge() == 0
    assert SLAB.electronic_carriers() == 0


def test_neutral_salt_adsorption_balanced_and_normalised():
    cplx = plus(SLAB, PEAI, "slab+PEAI")
    r = adsorption_reaction(cplx, SLAB, [(1, PEAI)])
    assert r.check() == []
    out = r.delta_e({"slab+PEAI": -100.0, "FAI_slab": -90.0, "PEAI": -8.0})
    assert out["delta_E_per_unit_eV"] == pytest.approx(-2.0)


def test_two_face_normalisation_counts_all_adsorbates():
    cplx = plus(plus(SLAB, PEAI, "x"), PEAI, "slab+2PEAI")
    r = adsorption_reaction(cplx, SLAB, [(2, PEAI)])
    out = r.delta_e({"slab+2PEAI": -110.0, "FAI_slab": -90.0, "PEAI": -8.0})
    assert out["n_units"] == 2 and out["delta_E_per_unit_eV"] == pytest.approx(-2.0)


def test_counterion_loss_detected():
    pea_cation_only = Species("PEA", {"C": 8, "H": 12, "N": 1}, cell_charge=1)
    cplx = plus(SLAB, PEAI, "slab+PEAI")
    r = adsorption_reaction(cplx, SLAB, [(1, pea_cation_only)])
    probs = r.check()
    assert any("Atoms not conserved" in p and "I: +1" in p for p in probs)
    assert any("charge" in p for p in probs)
    with pytest.raises(UnbalancedReaction):
        r.delta_e({})


def test_isovalent_exchange_requires_fa_product():
    pea = Species("PEA+", {"C": 8, "H": 12, "N": 1}, cell_charge=1)
    fa = Species("FA+", {"C": 1, "H": 5, "N": 2}, cell_charge=1)
    exch = plus(plus(SLAB, fa, "t", -1), pea, "slab_PEA_for_FA")
    bad = Reaction([(1, SLAB), (1, pea)], [(1, exch)], energy_kind="exchange")
    assert bad.check()  # FA missing
    good = Reaction([(1, SLAB), (1, PEAI)], [(1, exch), (1, FAI)], energy_kind="exchange")
    assert good.check() == []


def test_vacancy_filling_needs_electron_reservoir():
    vac = Species("slab_VI+", {**SLAB.composition, "I": 116}, cell_charge=+1)
    cl_minus = Species("Cl-", {"Cl": 1}, cell_charge=-1)
    filled = Species("slab_ClI", {**SLAB.composition, "I": 116, "Cl": 1}, cell_charge=0)
    assert Reaction([(1, vac), (1, cl_minus)], [(1, filled)]).check() == []
    cl0 = Species("Cl", {"Cl": 1})
    r = Reaction([(1, vac), (1, cl0)], [(1, filled)])
    assert any("charge" in p for p in r.check())
    r2 = Reaction([(1, vac), (1, cl0), (1, ELECTRON)], [(1, filled)])
    assert r2.check() == []
    with pytest.raises(MissingReference):
        r2.delta_e({"slab_VI+": 0, "Cl": 0, "slab_ClI": 0})  # e- energy never assumed


def test_neutral_iodine_removal_is_not_VI_plus():
    vac_neutral = Species("slab_VI_neutral_cell", {**SLAB.composition, "I": 116}, cell_charge=0,
                          formal_units={"FA+": 45, "Pb2+": 36, "I-": 116})
    assert vac_neutral.formal_charge() == 1
    assert vac_neutral.electronic_carriers() == -1  # one excess electron vs formal picture


def test_cell_charge_from_nelect():
    # PbI2-rich 335: C45 H225 I153 N90 Pb54; standard PBE ZVAL -> 2682 electrons
    zval = {"C": 4, "H": 1, "I": 7, "N": 5, "Pb": 14}
    s = Species("x", {"C": 45, "H": 225, "I": 153, "N": 90, "Pb": 54})
    assert s.valence_electrons(zval) == 2682
    assert sum(zval[e] * n for e, n in s.composition.items()) - 2683 == -1


def test_identify_fa_units():
    fa = Atoms("CHNNHHHH", positions=[[0, 0, 0], [0, 0, 1.09], [1.15, 0, -0.6], [-1.15, 0, -0.6],
                                      [2.0, 0, -0.1], [1.2, 0, -1.6], [-2.0, 0, -0.1], [-1.2, 0, -1.6]],
               cell=[20, 20, 20], pbc=True)
    fa += Atoms("I", positions=[[8, 8, 8]])
    units, unknown = identify_units(fa)
    assert units == {"FA+": 1, "I-": 1} and unknown == []
