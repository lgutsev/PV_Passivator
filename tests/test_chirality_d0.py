"""Package generator for the chirality D0-D2 tranche (examples/scripts/prepare_chirality_d0.py)."""
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "scripts"))
import prepare_chirality_d0 as P  # noqa: E402

from pvpassivator import chirality as ch  # noqa: E402

CFG = {"title": "test", "kspacing": 0.25, "encut": 520, "encut_factor": 1.3,
       "parallel": {"default": {"NCORE": 16, "KPAR": 1}},
       "cluster": {"account": "acct", "partition": "workq", "ntasks_per_node": 64, "module": "vasp6/x",
                   "potcar_dir": "/pot", "stages": {k: {"nodes": 1, "time": "01:00:00"}
                                                    for k in ("smoke", "check", "soc", "relax", "static")}}}


@pytest.fixture()
def starts(tmp_path):
    d = tmp_path / "starts"
    d.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, f in P.CIFS.items():
            write(d / f"POSCAR_{k}", read(P.SDIR / f), format="vasp", direct=True)
    (d / "settle.json").write_text(json.dumps({"R": {"status": "pass"}, "rac": {"status": "pass"}}))
    return d


def _incar(p: Path) -> dict:
    out = {}
    for ln in p.read_text().splitlines()[1:]:
        k, v = ln.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def test_stage_layout_and_settings(starts, tmp_path):
    out = tmp_path / "pkg_chiral"
    P.stage(dict(CFG), starts, out)
    lists = {f.stem: f.read_text().split("\n")[:-1] for f in (out / "lists").glob("*.txt")}
    assert {k: len(v) for k, v in lists.items()} == {"smoke": 1, "check": 7, "soc": 2, "relax": 4, "static": 4}
    incars = list((out / "inputs").rglob("INCAR"))
    assert len(incars) == 18
    for p in incars:
        t = _incar(p)
        assert t["ISYM"] == "0" and t["METAGGA"] == "R2SCAN" and t["IVDW"] == "12"
        assert "LDIPOL" not in t                                      # bulk cells: no dipole correction
        soc = "LSORBIT" in t
        assert soc == ("_soc" in p.parent.parent.name)
        assert (p.parent / "EXE").read_text().strip() == ("vasp_ncl" if soc else "vasp_std")
        if soc:
            assert "ISPIN" not in t and t["MAGMOM"].endswith("*0.0")
        assert t["ENCUT"] == ("676" if p.parent.name == "relax" and t.get("ISIF") == "3"
                              else "600" if "e600" in str(p) else "520")
    # statics after relaxes read the relaxed CONTCAR of the same job
    assert all(r.split() == [r.split()[0]] * 2 for r in lists["static"])
    # POTCAR order follows POSCAR species blocks
    for d in (out / "inputs").rglob("POSCAR"):
        species = d.read_text().splitlines()[5].split()
        assert (d.parent / "POTCAR.names").read_text().split() == [{"Pb": "Pb_d"}.get(s, s) for s in species]
    # LF only everywhere
    assert not [f for f in out.rglob("*") if f.is_file() and b"\r" in f.read_bytes()]
    pkg = json.loads((out / "package.json").read_text())
    assert pkg["submitted"] is False


def test_mirror_job_is_exact_inversion(starts, tmp_path):
    out = tmp_path / "pkg_chiral"
    P.stage(dict(CFG), starts, out)
    r = read(out / "inputs" / "R_kref" / "static" / "POSCAR", format="vasp")
    m = read(out / "inputs" / "Rmirror_kref" / "static" / "POSCAR", format="vasp")
    tr, tm = ch.build_topology(r), ch.build_topology(m)
    assert tr.summary()["stereocentres"] == {"R": 8, "S": 0}
    assert tm.summary()["stereocentres"] == {"R": 0, "S": 8}
    dr, dm = ch.frame_descriptors(tr, r), ch.frame_descriptors(tm, m)
    for k in ("ax", "eq", "org"):
        assert dm[f"mean_{k}"] == pytest.approx(-dr[f"mean_{k}"], abs=1e-6)
    assert (r.cell.array == m.cell.array).all()


def test_refuses_failed_settle_and_nonempty_out(starts, tmp_path):
    (starts / "settle.json").write_text(json.dumps({"R": {"status": "fail"}, "rac": {"status": "pass"}}))
    with pytest.raises(SystemExit):
        P.stage(dict(CFG), starts, tmp_path / "a")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "x").write_text("x")
    with pytest.raises(SystemExit):
        P.stage(dict(CFG), starts, tmp_path / "b")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_slurm_scripts_parse(starts, tmp_path):
    out = tmp_path / "pkg_chiral"
    P.stage(dict(CFG), starts, out)
    for f in list(out.glob("*.slurm")) + [out / "lib_pvpass.sh"]:
        r = subprocess.run(["bash", "-n", f.name], cwd=out, capture_output=True, text=True)
        assert r.returncode == 0, (f.name, r.stderr)
