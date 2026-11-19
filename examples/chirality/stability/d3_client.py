"""D3(BJ, r2SCAN) client for ASE's socket calculator; runs in /project/lgutsev/env/d3_torchdftd (torch-dftd).
    python d3_client.py STRUCTURE PORT CUTOFF_BOHR CNTHR_BOHR [stress]
(the optional 5th argument 'stress' returns the D3 virial, needed for NPT; without it the virial is zero, as in 137/140)
Parameters as VASP IVDW=12 with the r2SCAN set (s8 0.7898, a1 0.4948, a2 5.7308; torch-dftd xc="r2scan"),
float64, GPU when available. Same pattern as the D3 clients of packages 97/119/125 (there: zero damping, PBE).
"""
import sys

import torch
from ase import units
from ase.calculators.socketio import SocketClient
from ase.io import read
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

atoms = read(sys.argv[1], index=0)
atoms.set_constraint()
atoms.calc = TorchDFTD3Calculator(damping="bj", xc="r2scan", cutoff=float(sys.argv[3]) * units.Bohr,
                                  cnthr=float(sys.argv[4]) * units.Bohr, dtype=torch.float64,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
SocketClient(host="localhost", port=int(sys.argv[2])).run(atoms, use_stress=len(sys.argv) > 5 and sys.argv[5] == "stress")
