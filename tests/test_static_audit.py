"""Software checks of the statics-only MLIP audit (prepare_static_audit.py, stage_static_audit.py) on toy fixtures."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read, write

SCRIPTS = Path(__file__).resolve().parents[1] / "examples" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_static_audit as PA  # noqa: E402
import stage_static_audit as SA  # noqa: E402
from test_adsearch import make_space  # noqa: E402

BASH = shutil.which("bash")
CFG = {**PA.DEFAULTS, "min_interatomic_A": 0.2}       # the toy slab has squashed lattice cations
CLUSTER = {"account": "acct", "partition": "workq", "ntasks_per_node": 64, "module": "vasp6/x",
           "potcar_dir": "/home/lgutsev/pot/potpaw_PBE",
           "stages": {"smoke": {"nodes": 2, "time": "01:00:00"},
                      "static_slab": {"nodes": 2, "time": "12:00:00", "max_parallel": 5},
                      "static_ref": {"nodes": 1, "time": "02:00:00"}},
           "cost_estimate_node_h": {"smoke": "1", "static_slab": "100-190", "static_ref": "1"}}


def make_pool(tmp_path, n=12):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sp = make_space(tmp_path)
    pool = tmp_path / "pool"
    (pool / "structures").mkdir(parents=True)
    rng = np.random.default_rng(0)

    def put(name, atoms, e):
        a = atoms.copy()
        a.set_array("model_forces", rng.normal(0, 0.3, (len(a), 3)))
        a.info["model_energy"] = e
        write(pool / "structures" / f"{name}.extxyz", a, format="extxyz")

    recs, seen = [], set()
    while len(recs) < n:
        genes = sp.random_genes(rng)
        atoms, reasons, _ = sp.build(genes)
        if reasons or sp.canonical_key(genes) in seen:
            continue
        seen.add(sp.canonical_key(genes))
        name = f"pool_{len(recs):02d}"
        e = -100.0 + 0.15 * len(recs)
        put(name, atoms, e)
        recs.append({"name": name, "group": "pool", "formula": atoms.get_chemical_formula(), "relaxed": True,
                     "converged": True, "substrate_ok": True, "final_state": "intact_adsorption",
                     "model_energy_eV": e, "file": f"structures/{name}.extxyz"})
    put("slab", sp.substrate, -90.0)
    mol = sp.spec.passivator.atoms.copy()
    mol.set_cell([25, 25, 25])
    mol.pbc = True
    mol.center()
    put("ref_molecule", mol, -5.0)
    (pool / "pool.json").write_text(json.dumps({
        "provenance": {"slab": {}, "molecule": {}, "code_commit": "test", "evaluator": {"kind": "toy"}},
        "structures": recs,
        "references": {"slab": {"file": "structures/slab.extxyz", "model_energy_eV": -90.0},
                       "molecule": {"file": "structures/ref_molecule.extxyz", "model_energy_eV": -5.0}}}))
    return pool


def test_prepare_writes_identical_geometry_statics_with_fixed_thresholds(tmp_path):
    pool = make_pool(tmp_path)
    man = PA.prepare(pool, tmp_path / "pkg", CFG, False)
    assert man["schema"] == "static_audit/1" and len(man["jobs"]) == 12 + 2
    assert man["thresholds"]["relative_energy_mae_eV"] == 0.10 and man["thresholds"]["force_rmse_near_eV_A"] == 0.05
    assert len(man["groups"]) == 1 and len(next(iter(man["groups"].values()))) == 12       # one composition group
    dip = set()
    for name, j in man["jobs"].items():
        d = tmp_path / "pkg" / name / "static"
        incar = dict(ln.split(" = ", 1) for ln in (d / "INCAR").read_text().splitlines())
        assert incar["NSW"] == "0" and incar["LREAL"] == ".FALSE." and incar["METAGGA"] == "R2SCAN"
        if j["role"] != "reference":
            dip.add(incar["DIPOL"])
            assert incar["IDIPOL"] == "1" and "selective" not in (d / "POSCAR").read_text().lower()
        # the POSCAR is the stored geometry, atom for atom (through the job's atom map)
        src = read(pool / j["source"])
        pos = read(d / "POSCAR", format="vasp").positions
        amap = json.loads((d / "job.json").read_text())["atom_map"]
        idx = {int(i): k for k, i in enumerate(src.arrays["atom_id"])} if "atom_id" in src.arrays else None
        perm = [idx[i] for i in amap] if idx else amap
        assert np.abs(src.positions[perm] - pos).max() < 1e-6
    assert len(dip) == 1                                                                   # one dipole centre
    with pytest.raises(FileExistsError):
        PA.prepare(pool, tmp_path / "pkg", CFG, False)


def test_prepare_refuses_a_structure_off_the_common_cell(tmp_path):
    pool = make_pool(tmp_path, n=2)
    f = pool / "structures" / "pool_01.extxyz"
    a = read(f)
    a.set_cell(a.cell.array * 1.01, scale_atoms=True)
    write(f, a, format="extxyz")
    with pytest.raises(ValueError, match="common cell"):
        PA.prepare(pool, tmp_path / "pkg", CFG, False)


def _returned(tmp_path, pool, e_err, f_err, seed=1):
    """A staged package plus a fake reader that plays the DFT: model numbers + chosen errors, in DFT atom order."""
    PA.prepare(pool, tmp_path / "pkg", CFG, False)
    (tmp_path / "c.json").write_text(json.dumps(CLUSTER))
    SA.main(["--package", str(tmp_path / "pkg"), "--cluster", str(tmp_path / "c.json"),
             "--out", str(tmp_path / "99_audit")])
    man = json.loads((tmp_path / "99_audit" / "manifest.json").read_text())
    rng = np.random.default_rng(seed)

    def reader(d: Path):
        name = d.parent.name
        j = man["jobs"][name]
        src = read(pool / j["source"])
        amap = json.loads((tmp_path / "99_audit" / "inputs" / name / "static" / "job.json").read_text())["atom_map"]
        idx = {int(i): k for k, i in enumerate(src.arrays["atom_id"])}
        perm = np.array([idx[i] for i in amap])
        return (j["model_energy_eV"] + 3.0 + rng.normal(0, e_err), src.positions[perm],
                src.arrays["model_forces"][perm] + rng.normal(0, f_err, (len(src), 3)))
    return tmp_path / "99_audit", reader


def test_analysis_passes_a_good_model_and_fails_a_bad_one(tmp_path):
    pool = make_pool(tmp_path)
    pkg, good = _returned(tmp_path, pool, e_err=0.01, f_err=0.01)
    rep = PA.analyze(pkg, pool, reader=good)
    grp = next(iter(rep["groups"].values()))
    assert rep["passes"] and grp["n"] == 12 and grp["spearman"] > 0.95 and grp["top3_overlap"] >= 2
    assert rep["forces_near"]["rmse_eV_A"] < 0.02 and rep["forces_near"]["n_atoms"] < rep["forces_far"]["n_atoms"]
    # a constant energy offset between model and DFT is not an error: only differences inside the group count
    assert grp["relative_energy_mae_eV"] < 0.05
    pkg_b, bad_e = _returned(tmp_path / "b", make_pool(tmp_path / "b"), e_err=0.5, f_err=0.01)
    assert not PA.analyze(pkg_b, tmp_path / "b" / "pool", reader=bad_e)["passes"]
    pkg_c, bad_f = _returned(tmp_path / "c", make_pool(tmp_path / "c"), e_err=0.01, f_err=0.2)
    r = PA.analyze(pkg_c, tmp_path / "c" / "pool", reader=bad_f)
    assert not r["passes"] and r["forces_near"]["rmse_eV_A"] > 0.05


def test_analysis_refuses_unfinished_steps_and_small_groups_skip_rank_test(tmp_path):
    pool = make_pool(tmp_path, n=4)
    pkg, reader = _returned(tmp_path, pool, 0.01, 0.01)
    rep = PA.analyze(pkg, pool)                         # real reader: nothing has run, so no DONE anywhere
    assert not rep["passes"] and len(rep["missing"]) == 4 and rep["structures"] == {}
    rep = PA.analyze(pkg, pool, reader=reader)
    grp = next(iter(rep["groups"].values()))
    assert grp["spearman"] is None and "not tested" in grp["rank_note"]


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_staged_package_layout_and_mock_run(tmp_path):
    pool = make_pool(tmp_path, n=3)
    pkg, _ = _returned(tmp_path, pool, 0.0, 0.0)
    assert (pkg / "lists" / "audit_slab.txt").read_text().split() == ["slab", "pool_00", "pool_01", "pool_02"]
    assert (pkg / "lists" / "audit_ref.txt").read_text().split() == ["ref_molecule"]
    st = (pkg / "run_audit_statics.slurm").read_text()
    assert "#SBATCH --array=0-3%4" in st and "--signal=B:USR1@" in st and 'run_step "$job" static static' in st
    assert "sleep $(( (SLURM_ARRAY_TASK_ID % 5) * 45 ))" in st
    for f in pkg.rglob("*"):
        if f.is_file() and f.name != ".keep":
            assert b"\r" not in f.read_bytes(), f
    assert "NELM = 4" in (pkg / "inputs" / "smoke" / "static" / "INCAR").read_text()
    assert json.loads((pkg / "package.json").read_text())["submitted"] is False
    for s in ("run_audit_smoke.slurm", "run_audit_statics.slurm", "run_audit_ref.slurm"):
        assert subprocess.run([BASH, "-n", str(pkg / s)]).returncode == 0
    for s in ("run_audit_statics.slurm", "run_audit_ref.slurm"):       # no sleeping in the mock run
        (pkg / s).write_text((pkg / s).read_text().replace("* 45 ))", "* 0 ))"), newline="\n")
    r = subprocess.run([BASH, str(SCRIPTS / "mock_run_package.sh"), str(pkg), "run_audit_smoke.slurm",
                        "run_audit_statics.slurm", "run_audit_ref.slurm"], capture_output=True, text=True,
                       env={**os.environ, "POLL_S": "1"})
    assert "FAIL" not in r.stdout and "DONE markers: 6" in r.stdout, r.stdout + r.stderr
