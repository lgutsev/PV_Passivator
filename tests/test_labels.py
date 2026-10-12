import json

import pytest

from pvpassivator.labels import analyze

ZVAL = {"Pb": 14, "I": 7, "C": 4, "H": 1, "N": 5, "Cl": 7}


def _poscar(comp):
    syms = list(comp)
    return ("x\n1.0\n20 0 0\n0 20 0\n0 0 20\n" + " ".join(syms) + "\n" + " ".join(str(comp[s]) for s in syms)
            + "\nDirect\n" + "".join("0 0 0\n" for s in syms for _ in range(comp[s])))


def _outcar(comp, energy, relax=False, finished=True):
    syms = list(comp)
    lines = [f"   TITEL  = PAW_PBE {s} 01Jan2000" for s in syms]
    lines += [f"   POMASS =  1.0; ZVAL   =  {ZVAL[s]:.3f}    mass and valenz" for s in syms]
    lines += ["   ions per type =  " + "  ".join(str(comp[s]) for s in syms)]
    lines += [f"   NELECT =  {sum(ZVAL[s] * n for s, n in comp.items()):.4f}    total number of electrons"]
    lines += [f"  energy  without entropy=  {energy:.6f}  energy(sigma->0) =  {energy:.6f}"]
    if relax:
        lines += [" reached required accuracy - stopping structural energy minimisation"]
    if finished:
        lines += [" General timing and accounting informations for this job:"]
    return "\n".join(lines) + "\n"


def _job(root, job, comp, e_static, e_relax=None, done=True, relax_ok=True, kcheck=False):
    steps = {"static": e_static} if kcheck else {"relax": e_relax if e_relax is not None else e_static + 0.1,
                                                   "static": e_static}
    for step, e in steps.items():
        i = root / "inputs" / job / step
        o = root / "outputs" / job / step
        i.mkdir(parents=True); o.mkdir(parents=True)
        (i / "POSCAR").write_text(_poscar(comp))
        incar = "ISPIN = 1\nENCUT = 520\nMETAGGA = R2SCAN\n" + ("NSW = 300\nIBRION = 2\n" if step == "relax" else "NSW = 0\nIBRION = -1\n")
        (o / "INCAR").write_text(incar)
        (o / "KPOINTS").write_text("x\n0\nGamma\n1 1 1\n0 0 0\n" if not kcheck else "x\n0\nGamma\n1 2 2\n0 0 0\n")
        (o / "OUTCAR").write_text(_outcar(comp, e, relax=(step == "relax" and relax_ok)))
        if done:
            (o / "DONE").write_text("x\n")


def _package(tmp_path, k_complex=-1.04):
    root = tmp_path / "53_test"
    slab = {"Pb": 4, "I": 8, "C": 2, "H": 10, "N": 4}
    cx = {"Pb": 4, "I": 8, "C": 10, "H": 22, "N": 5, "Cl": 1}
    ref = {"C": 8, "H": 12, "N": 1, "Cl": 1}
    _job(root, "slab_X", slab, -100.0)
    _job(root, "complex_PEACl", cx, -150.0)
    _job(root, "ref_PEACl", ref, -48.8)
    _job(root, "kcheck_slab_X", slab, -101.03, kcheck=True)
    _job(root, "kcheck_complex_PEACl", cx, -150.0 + k_complex, kcheck=True)
    m = {"jobs": {"slab_X": {"role": "bare_slab"}, "complex_PEACl": {"role": "complex"},
                  "ref_PEACl": {"role": "reference"}, "kcheck_slab_X": {"role": "kpoint_check"},
                  "kcheck_complex_PEACl": {"role": "kpoint_check"}},
         "reactions": [{"name": "PEACl", "equation": "slab_X + ref_PEACl -> complex_PEACl"}],
         "kpoint_check": {"kpoints": [1, 2, 2], "jobs": ["slab_X", "complex_PEACl"]}}
    (root / "manifest.json").write_text(json.dumps(m))
    return root


def test_labels_eads_and_kpoint_check(tmp_path):
    r = analyze(_package(tmp_path))
    (row,) = r["reactions"]
    assert row["E_ads_eV"] == pytest.approx(-150.0 + 100.0 + 48.8)       # -1.2 eV, binding
    v = r["kpoint_check"]["verdicts"]["complex_PEACl"]
    assert v["k_error_eV"] == pytest.approx(-0.01) and v["pass"] and r["kpoint_check"]["pass"]
    assert not r["problems"] and r["settings_problem"] is None


def test_labels_kpoint_check_fails_beyond_tolerance(tmp_path):
    r = analyze(_package(tmp_path, k_complex=-1.10))
    assert r["kpoint_check"]["verdicts"]["complex_PEACl"]["pass"] is False and not r["kpoint_check"]["pass"]


def test_labels_never_use_unfinished_or_unconverged_steps(tmp_path):
    root = _package(tmp_path)
    (root / "outputs" / "ref_PEACl" / "static" / "DONE").unlink()
    r = analyze(root)
    assert r["reactions"][0]["E_ads_eV"] is None and r["reactions"][0]["missing"] == ["ref_PEACl"]
    assert "ref_PEACl/static" in r["problems"]
    root2 = tmp_path / "b"
    root2.mkdir()
    root2 = _package(root2)
    oc = root2 / "outputs" / "complex_PEACl" / "relax" / "OUTCAR"
    oc.write_text(oc.read_text().replace(" reached required accuracy - stopping structural energy minimisation\n", ""))
    r2 = analyze(root2)
    assert r2["reactions"][0]["E_ads_eV"] is None and "complex_PEACl/relax" in r2["problems"]


def test_cli_labels_exit_code(tmp_path, capsys):
    from pvpassivator.cli import main
    assert main(["labels", str(_package(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "PEACl" in out and "-1.200 eV" in out and "PASS" in out


def test_labels_flag_mismatched_settings(tmp_path):
    root = _package(tmp_path)
    inc = root / "outputs" / "complex_PEACl" / "static" / "INCAR"
    inc.write_text(inc.read_text().replace("ENCUT = 520", "ENCUT = 400"))
    assert "Hamiltonian" in analyze(root)["settings_problem"]
