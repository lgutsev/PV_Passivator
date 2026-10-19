"""lib_pvpass.sh setup_vasp: an incomplete module wrapper is a transient failure (exit 75 + TRANSIENT), not a crash."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "examples" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import stage_label_set as S  # noqa: E402

BASH = shutil.which("bash")

DRIVER = r'''set -eo pipefail
source ./lib_pvpass.sh
set -u
module() { :; }
srun() { "$@"; }
WRAPPER_TRIES=2 WRAPPER_WAIT_S=0 POLL_S=0.2
run_step j1 static static
'''

OK_VASP = '#!/bin/bash\necho " EDIFF is reached" > OUTCAR\nprintf "<modeling>\\n</modeling>\\n" > vasprun.xml\n'


def _package(tmp_path: Path, vasp_text: str) -> Path:
    potdir = tmp_path / "pot" / "Pb_d"
    potdir.mkdir(parents=True)
    (potdir / "POTCAR").write_text("   TITEL  = PAW_PBE Pb_d 06Sep2000\n")
    lib = S.LIB.replace("__MODULE__", "vasp6/test").replace("__POTDIR__", (tmp_path / "pot").as_posix())
    (tmp_path / "lib_pvpass.sh").write_bytes(lib.encode())
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "vasp_std").write_bytes(vasp_text.encode())
    inp = tmp_path / "inputs" / "j1" / "static"
    inp.mkdir(parents=True)
    for f, t in {"INCAR": "NSW = 0\n", "KPOINTS": "k\n0\nGamma\n1 1 1\n", "EXE": "vasp_std\n",
                 "POTCAR.names": "Pb_d\n",
                 "POSCAR": "x\n1.0\n5 0 0\n0 5 0\n0 0 5\nPb\n1\nDirect\n0 0 0\n"}.items():
        (inp / f).write_bytes(t.encode())
    (tmp_path / "driver.sh").write_bytes(DRIVER.encode())
    return tmp_path / "outputs" / "j1" / "static"


def _run(tmp_path: Path):
    return subprocess.run([BASH, "-c", 'chmod +x bin/vasp_std; export PATH="$PWD/bin:$PATH"; bash driver.sh'],
                          cwd=tmp_path, capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_incomplete_wrapper_is_transient(tmp_path):
    out = _package(tmp_path, "")                     # empty file: the wrapper never becomes a script
    r = _run(tmp_path)
    assert r.returncode == 75, r.stdout + r.stderr
    assert "transient, resubmit" in r.stderr
    assert (out / "TRANSIENT").exists() and not (out / "DONE").exists()


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_ready_wrapper_clears_transient(tmp_path):
    out = _package(tmp_path, OK_VASP)
    out.mkdir(parents=True)
    (out / "TRANSIENT").write_text("earlier attempt\n")
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "DONE").exists() and not (out / "TRANSIENT").exists()


def test_guard_waits_ten_minutes_by_default():
    assert 'WRAPPER_TRIES:-40' in S.LIB and 'WRAPPER_WAIT_S:-15' in S.LIB
