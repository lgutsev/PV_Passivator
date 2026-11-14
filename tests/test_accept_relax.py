"""Owner acceptance of an unfinished relax: accept_relax.sh (thresholds, refusals, record), run_step (skip the
accepted relax, start the static from it) and pvpass labels (use it, but report it as owner-accepted)."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pvpassivator.labels import analyze

SCRIPTS = Path(__file__).resolve().parents[1] / "examples" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_label_set as S  # noqa: E402
import test_labels as L  # noqa: E402

BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="needs bash")


def forces_block(fmax):
    return (" POSITION                                       TOTAL-FORCE (eV/Angst)\n"
            " -----------------------------------------------------------------------------------\n"
            f"      0.00000      0.00000      0.00000         {fmax:.5f}      0.00000      0.00000\n"
            "      2.00000      0.00000      0.00000         0.01000      0.00000      0.00000\n"
            " -----------------------------------------------------------------------------------\n")


def oszicar(energies, start=1):
    return "".join(f"RMM:  12   -0.1E+04   -0.1E-06\n {start + i} F= {e:.8E} E0= {e:.8E}  d E =-0.1E-02\n"
                   for i, e in enumerate(energies))


def relax_package(tmp_path, last_fmax, e_tail_meV, done=False, contcar_differs=False):
    root = tmp_path / "pkg"
    (root / "inputs" / "cx" / "relax").mkdir(parents=True)
    (root / "inputs" / "cx" / "relax" / "INCAR").write_text("IBRION = 1\nEDIFFG = -0.03\n")
    out, work = root / "outputs" / "cx" / "relax", root / "runs" / "cx" / "relax"
    out.mkdir(parents=True); work.mkdir(parents=True)
    (out / "OUTCAR").write_text(forces_block(0.5) + forces_block(last_fmax))   # only the LAST block counts
    seg1 = [-7000.0 - 0.05 * i for i in range(8)]                               # earlier segment
    cur = [seg1[-1] - e_tail_meV / 1000 * i / 10 for i in range(1, 13)]        # 12 flat steps: 20 in all; last 10 span e_tail
    (out / "OSZICAR.seg1").write_text(oszicar(seg1))
    (out / "OSZICAR").write_text(oszicar(cur))
    (out / "CONTCAR").write_text("geometry\n")
    (work / "CONTCAR").write_text("geometry\n" if not contcar_differs else "newer\n")
    if done:
        (out / "DONE").write_text("x\n")
    (root / "accept_relax.sh").write_bytes(S.ACCEPT.encode())
    return root


def accept(root, fmax_max="0.035", de_max="5"):
    return subprocess.run([BASH, "accept_relax.sh", "cx", fmax_max, de_max, "owner 2026-10-30: plateau"],
                          cwd=root, capture_output=True, text=True, timeout=60)


@needs_bash
def test_accept_within_thresholds_writes_record(tmp_path):
    root = relax_package(tmp_path, last_fmax=0.033, e_tail_meV=4.0)
    r = accept(root)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = (root / "outputs" / "cx" / "relax" / "ACCEPTED").read_text()
    assert "last_max_force_eV_per_A: 0.0330" in rec and "ionic_steps_all_segments: 20" in rec
    assert "note: owner 2026-10-30: plateau" in rec and "contcar_sha256: " in rec and "incar_ediffg: -0.03" in rec
    assert not (root / "outputs" / "cx" / "relax" / "DONE").exists()


@needs_bash
@pytest.mark.parametrize("kw,why", [({"last_fmax": 0.050, "e_tail_meV": 4.0}, "force"),
                                    ({"last_fmax": 0.033, "e_tail_meV": 12.0}, "energy"),
                                    ({"last_fmax": 0.033, "e_tail_meV": 4.0, "done": True}, "done"),
                                    ({"last_fmax": 0.033, "e_tail_meV": 4.0, "contcar_differs": True}, "contcar")])
def test_accept_refuses(tmp_path, kw, why):
    root = relax_package(tmp_path, **kw)
    r = accept(root)
    assert r.returncode != 0, why
    assert not (root / "outputs" / "cx" / "relax" / "ACCEPTED").exists(), why


DRIVER = r'''set -eo pipefail
source ./lib_pvpass.sh
set -u
module() { :; }
srun() { "$@"; }
POLL_S=0.2 WRAPPER_TRIES=2 WRAPPER_WAIT_S=0
run_step cx relax relax
run_step cx static static cx
'''
FAKE_VASP = '#!/bin/bash\ncp POSCAR used_geometry\necho " EDIFF is reached" > OUTCAR\nprintf "<modeling>\\n</modeling>\\n" > vasprun.xml\n'


@needs_bash
def test_run_step_skips_accepted_relax_and_static_uses_its_contcar(tmp_path):
    potdir = tmp_path / "pot" / "Pb_d"; potdir.mkdir(parents=True)
    (potdir / "POTCAR").write_text("   TITEL  = PAW_PBE Pb_d 06Sep2000\n")
    (tmp_path / "lib_pvpass.sh").write_bytes(S.LIB.replace("__MODULE__", "m").replace("__POTDIR__", (tmp_path / "pot").as_posix()).encode())
    (tmp_path / "bin").mkdir(); (tmp_path / "bin" / "vasp_std").write_bytes(FAKE_VASP.encode())
    for step in ("relax", "static"):
        i = tmp_path / "inputs" / "cx" / step; i.mkdir(parents=True)
        for f, t in {"INCAR": "x\n", "KPOINTS": "k\n", "EXE": "vasp_std\n", "POTCAR.names": "Pb_d\n", "POSCAR": "input\n"}.items():
            (i / f).write_bytes(t.encode())
    o = tmp_path / "outputs" / "cx" / "relax"; o.mkdir(parents=True)
    (o / "ACCEPTED").write_text("accepted: owner\n"); (o / "CONTCAR").write_text("accepted-geometry\n")
    (tmp_path / "driver.sh").write_bytes(DRIVER.encode())
    r = subprocess.run([BASH, "-c", 'chmod +x bin/vasp_std; export PATH="$PWD/bin:$PATH"; bash driver.sh'],
                       cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "accepted by the owner" in r.stdout and not (tmp_path / "runs" / "cx" / "relax").exists()
    assert (tmp_path / "runs" / "cx" / "static" / "used_geometry").read_text() == "accepted-geometry\n"
    assert (tmp_path / "outputs" / "cx" / "static" / "DONE").exists()


def test_labels_use_accepted_relax_but_report_it(tmp_path):
    root = L._package(tmp_path)
    relax = root / "outputs" / "complex_PEACl" / "relax"
    (relax / "DONE").unlink()
    comp = {"Pb": 4, "I": 8, "C": 10, "H": 22, "N": 5, "Cl": 1}
    (relax / "OUTCAR").write_text(L._outcar(comp, -149.9, relax=False, finished=False))   # not converged, stopped
    assert analyze(root)["reactions"][0]["E_ads_eV"] is None                         # without acceptance: unusable
    (relax / "ACCEPTED").write_text("accepted: owner\nlast_max_force_eV_per_A: 0.0330\ndE_last10_meV: -4.0\nnote: plateau\n")
    r = analyze(root)
    (row,) = r["reactions"]
    assert row["E_ads_eV"] == pytest.approx(-1.2) and row["owner_accepted_relaxes"] == ["complex_PEACl"]
    assert r["accepted_relaxes"]["complex_PEACl"]["last_max_force_eV_per_A"] == "0.0330"


SD_CONTCAR = ("cx\n1.0\n10 0 0\n0 10 0\n0 0 10\nPb I\n1 1\nSelective dynamics\nDirect\n"
              "0.0 0.0 0.0 T T T\n0.2 0.0 0.0 F F F\n")


@needs_bash
@pytest.mark.parametrize("frozen_force,ok", [(0.30, True), (0.01, True)])
def test_accept_ignores_frozen_atom_forces(tmp_path, frozen_force, ok):
    """VASP prints real forces on frozen atoms; the limit applies to free atoms (selective-dynamics T) only."""
    root = relax_package(tmp_path, last_fmax=0.033, e_tail_meV=4.0)
    out = root / "outputs" / "cx" / "relax"
    block = (" POSITION                                       TOTAL-FORCE (eV/Angst)\n"
             " -----------------------------------------------------------------------------------\n"
             "      0.00000      0.00000      0.00000         0.03300      0.00000      0.00000\n"
             f"      2.00000      0.00000      0.00000         {frozen_force:.5f}      0.00000      0.00000\n"
             " -----------------------------------------------------------------------------------\n")
    (out / "OUTCAR").write_text(block)
    for p in (out / "CONTCAR", root / "runs" / "cx" / "relax" / "CONTCAR"):
        p.write_text(SD_CONTCAR)
    r = accept(root)
    assert (r.returncode == 0) == ok, r.stdout + r.stderr
    rec = (out / "ACCEPTED").read_text()
    assert "last_max_force_free: 0.0330" in rec and "free_atoms: 1" in rec
    assert f"last_max_force_all: {max(frozen_force, 0.033):.4f}" in rec


@needs_bash
def test_accept_refuses_free_atom_over_limit_with_selective_dynamics(tmp_path):
    root = relax_package(tmp_path, last_fmax=0.050, e_tail_meV=4.0)   # first atom (free) carries 0.050
    out = root / "outputs" / "cx" / "relax"
    for p in (out / "CONTCAR", root / "runs" / "cx" / "relax" / "CONTCAR"):
        p.write_text(SD_CONTCAR)
    r = accept(root)
    assert r.returncode == 1 and not (out / "ACCEPTED").exists(), r.stdout + r.stderr
