"""End-to-end on real archived OutPackLite data (skipped when archives are absent)."""
from pathlib import Path

import pytest

from pvpassivator.recovery import run_config, load_cases

CFG = Path(__file__).parents[1] / "examples" / "recovered_cases_pbe335.json"
cfg, _ = load_cases(CFG)
if not all(Path(p).exists() for p in cfg["archives"].values()):
    pytest.skip("research archives not available", allow_module_level=True)


@pytest.fixture(scope="module")
def results():
    return {r.case.case_id: r for r in run_config(CFG, Path(__file__).parents[1] / "data" / "cache")}


def test_salt_adsorption_balanced_from_poscars(results):
    for cid in ("PbI2rich_PEAI", "FAIrich_Schot_PEAAc", "FAIrich_Schot_PEATFA"):
        assert results[cid].balance_problems == []


def test_missing_bare_slab_energy_not_fabricated(results):
    r = results["PbI2rich_PEAI"]
    assert r.e_ads is None and "NOT REPRODUCIBLE" in r.verdict


def test_schottky_case_recomputed_from_raw(results):
    r = results["FAIrich_Schot_PEAAc"]
    assert r.energies["FAPI_333_x_FAI_Schot"] == pytest.approx(-2418.0321, abs=1e-4)
    assert r.e_ads == pytest.approx(-2586.5582 + 2418.0321 + 168.59662, abs=1e-4)
    assert "MISMATCH" in r.verdict and "NSW" in r.verdict
