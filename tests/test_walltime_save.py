"""lib_pvpass.sh run_step under a walltime signal: outputs/ must hold the run even though VASP never finished."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "examples" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import stage_label_set as S  # noqa: E402

BASH = shutil.which("bash")

FAKE_VASP = r'''#!/bin/bash
# Fake relax: one ionic step every 0.3 s, never converges; honours STOPCAR (LSTOP) like VASP.
cp POSCAR CONTCAR
for i in $(seq 1 200); do
  echo "  energy  without entropy= -$i.0  energy(sigma->0) = -$i.0" >> OUTCAR
  echo "   $i F= -.$i E0= -.$i" >> OSZICAR
  echo "step $i" > CONTCAR.step
  if [ -f STOPCAR ]; then echo "LSTOP: stopping after step $i" >> OUTCAR; cat POSCAR > CONTCAR; exit 0; fi
  sleep 0.3
done
'''

DRIVER = r'''set -eo pipefail
source ./lib_pvpass.sh
set -u
module() { :; }
srun() { "$@"; }
POLL_S=0.2 SAVE_EVERY=3
( sleep 1.5; kill -USR1 $$ ) &
run_step j1 relax relax
'''


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_walltime_signal_saves_outputs(tmp_path):
    potdir = tmp_path / "pot" / "Pb_d"
    potdir.mkdir(parents=True)
    (potdir / "POTCAR").write_text("   TITEL  = PAW_PBE Pb_d 06Sep2000\n")
    lib = S.LIB.replace("__MODULE__", "vasp6/test").replace("__POTDIR__", (tmp_path / "pot").as_posix())
    (tmp_path / "lib_pvpass.sh").write_bytes(lib.encode())
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "vasp_std").write_bytes(FAKE_VASP.encode())
    inp = tmp_path / "inputs" / "j1" / "relax"
    inp.mkdir(parents=True)
    for f, t in {"INCAR": "NSW = 200\n", "KPOINTS": "k\n0\nGamma\n1 1 1\n", "EXE": "vasp_std\n",
                 "POTCAR.names": "Pb_d\n",
                 "POSCAR": "x\n1.0\n5 0 0\n0 5 0\n0 0 5\nPb\n1\nDirect\n0 0 0\n"}.items():
        (inp / f).write_bytes(t.encode())
    (tmp_path / "driver.sh").write_bytes(DRIVER.encode())
    r = subprocess.run([BASH, "-c", 'chmod +x bin/vasp_std; export PATH="$PWD/bin:$PATH"; bash driver.sh'], cwd=tmp_path, capture_output=True, text=True,
                       timeout=60)
    out = tmp_path / "outputs" / "j1" / "relax"
    # stopped by the signal, not converged -> exit 1 with the resubmit message; nothing marked DONE
    assert r.returncode == 1, r.stdout + r.stderr
    assert "did not reach EDIFFG" in r.stderr
    assert not (out / "DONE").exists()
    # but the run is in outputs/: OUTCAR with the LSTOP stop, CONTCAR, the walltime marker
    assert (out / "WALLTIME_STOP").exists()
    assert "LSTOP: stopping" in (out / "OUTCAR").read_text()
    assert (out / "CONTCAR").exists() and (out / "POTCAR.used").read_text().split() == ["Pb_d"]
    assert not (tmp_path / "runs" / "j1" / "relax" / "STOPCAR").exists()


def test_generated_scripts_request_the_signal():
    assert "#SBATCH --signal=B:USR1@1800" in S.SLURM
