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


def test_signal_margin_below_walltime(starts, tmp_path):
    """Desk fix (package 105): USR1 must not fire at start-up of short jobs."""
    assert P.walltime_s("00:30:00") == 1800 and P.walltime_s("1-00:00:00") == 86400 and P.walltime_s("45:00") == 2700
    assert P.signal_margin({"time": "00:30:00"}) == 450
    assert P.signal_margin({"time": "72:00:00"}) == 1800
    with pytest.raises(SystemExit):
        P.signal_margin({"time": "00:30:00", "signal_margin_s": 1800})
    out = tmp_path / "pkg_chiral"
    P.stage(dict(CFG), starts, out)        # every stage is 1 h in CFG
    for f in out.glob("*.slurm"):
        line = next(ln for ln in f.read_text().splitlines() if "--signal" in ln)
        assert line == "#SBATCH --signal=B:USR1@900", (f.name, line)


def test_package_json_has_per_file_sha256(starts, tmp_path):
    import hashlib
    out = tmp_path / "pkg_chiral"
    P.stage(dict(CFG), starts, out)
    files = json.loads((out / "package.json").read_text())["files_sha256"]
    on_disk = {f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file() and f.name != "package.json"}
    assert set(files) == on_disk and len(files) > 90
    for rel, h in files.items():
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == h


def test_diag_mode_lreal_encut_statics(starts, tmp_path):
    out = tmp_path / "pkg_diag"
    P.stage(dict(CFG), starts, out, mode="diag")
    assert (out / "lists" / "diag.txt").read_text().split() == ["R_e520_lrealF", "R_e600_lrealF", "R_e700_lrealF", "R_e700"]
    assert sorted(f.name for f in out.glob("*.slurm")) == ["run_diag.slurm"]
    got = {}
    for job in ("R_e520_lrealF", "R_e600_lrealF", "R_e700_lrealF", "R_e700"):
        t = _incar(out / "inputs" / job / "static" / "INCAR")
        got[job] = (t["ENCUT"], t["LREAL"], t["ISYM"], t["IBRION"])
        k = (out / "inputs" / job / "static" / "KPOINTS").read_text().splitlines()[3]
        assert k == "3 3 1"                                   # same mesh as package 105 R_kref
    assert got == {"R_e520_lrealF": ("520", ".FALSE.", "0", "-1"), "R_e600_lrealF": ("600", ".FALSE.", "0", "-1"),
                   "R_e700_lrealF": ("700", ".FALSE.", "0", "-1"), "R_e700": ("700", "Auto", "0", "-1")}
    # identical start geometry to the D0 package's R_kref
    d0 = tmp_path / "pkg_d0"
    P.stage(dict(CFG), starts, d0)
    assert (out / "inputs/R_e700/static/POSCAR").read_bytes() == (d0 / "inputs/R_kref/static/POSCAR").read_bytes()
