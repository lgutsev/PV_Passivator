"""D3(0) dispersion helper for MLIP label targets (VASP ``IVDW = 11``).

Training target for the PBE-D3(0) MLIP: E_DFT - E_D3(0), F_DFT - F_D3(0), sigma_DFT - sigma_D3(0). The model learns
short-range PBE; the *same* D3(0) backend, with the same cutoffs, is added back at run time. Never add D3 twice:
frames carry ``energy_target`` and :func:`subtract_d3` refuses frames that were already subtracted.

Backends (both give energy, forces and stress in ASE units and sign convention):

* :class:`SDFTD3Executable`: the ``s-dftd3`` command-line program (as used for the desk's L01 check). Its JSON
  ``virial`` divided by the cell volume equals the ASE stress (checked by finite differences, 2026-11-07).
  Its cutoffs are the program defaults; they cannot be set from the command line.
* :class:`TorchDFTD3`: ``torch-dftd`` (``damping='zero'``, ``xc='pbe'``, no three-body term). Its default cutoffs
  (95 bohr, CN 40 bohr) are close to the VASP defaults (``VDW_RADIUS`` 50 A, ``VDW_CNRADIUS`` 20 A).

Neither backend is imported at module import time; the module itself needs only numpy and ASE.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase.io import write
from ase.units import Bohr, Hartree

# Grimme D3 zero damping for PBE, as used by VASP IVDW = 11 (no three-body term in VASP's IVDW = 11).
D3ZERO_PBE = {'s6': 1.0, 's8': 0.722, 'rs6': 1.217, 'rs8': 1.0, 'alp': 14.0, 's9': 0.0}
# VASP INCAR tags for IVDW = 11 and their PBE defaults; an explicit tag must equal the default to match D3ZERO_PBE.
VASP_IVDW11_DEFAULTS = {'VDW_S6': 1.0, 'VDW_SR': 1.217, 'VDW_S8': 0.722, 'VDW_RADIUS': 50.0, 'VDW_CNRADIUS': 20.0}
TARGET_FULL = 'dft_free_energy'
TARGET_MINUS_D3 = 'dft_free_energy_minus_d3zero'


def check_incar_d3zero(incar: dict) -> list[str]:
    """Reasons the INCAR is not PBE + D3(0) with default parameters; [] means it matches."""
    problems = []
    if str(incar.get('IVDW', '')).strip() != '11':
        problems.append(f"IVDW is {incar.get('IVDW')!r}, not 11 (D3 zero damping)")
    if 'METAGGA' in incar or str(incar.get('LHFCALC', 'F')).strip('.').upper() in ('T', 'TRUE'):
        problems.append('not a PBE run (METAGGA or LHFCALC set)')
    gga = str(incar.get('GGA', 'PE')).strip().upper()
    if gga != 'PE':
        problems.append(f'GGA = {gga}, not PE')
    for key, default in VASP_IVDW11_DEFAULTS.items():
        if key in incar and not np.isclose(float(incar[key]), default, rtol=0, atol=1e-6):
            problems.append(f'{key} = {incar[key]} differs from the IVDW = 11 PBE default {default}')
    return problems


class SDFTD3Executable:
    """D3(0)/PBE from the ``s-dftd3`` program (``run --zero pbe --json --grad``)."""

    def __init__(self, exe: str | os.PathLike | None = None):
        exe = exe or os.environ.get('PVPASS_SDFTD3') or shutil.which('s-dftd3') or shutil.which('s-dftd3.exe')
        if not exe or not Path(exe).is_file():
            raise FileNotFoundError('s-dftd3 executable not found (pass a path or set PVPASS_SDFTD3)')
        self.exe = str(exe)

    def describe(self) -> dict:
        out = subprocess.run([self.exe, '--version'], capture_output=True, text=True, check=True)
        return {'backend': 's-dftd3', 'exe': self.exe, 'version': out.stdout.strip(), 'damping': 'zero',
                'xc': 'pbe', 'cutoffs': 'program defaults'}

    def compute(self, atoms):
        env = dict(os.environ, PATH=str(Path(self.exe).parent) + os.pathsep + os.environ.get('PATH', ''))
        with tempfile.TemporaryDirectory() as tmp:
            periodic = bool(atoms.pbc.any())
            name = 'POSCAR' if periodic else 'in.xyz'
            if periodic:
                write(Path(tmp) / name, atoms, format='vasp', direct=True)
            else:
                write(Path(tmp) / name, atoms, format='xyz')
            subprocess.run([self.exe, 'run', '--zero', 'pbe', '--json', 'd3.json', '--grad', 'g.txt', '--noedisp', name],
                           cwd=tmp, env=env, capture_output=True, text=True, check=True)
            res = json.loads((Path(tmp) / 'd3.json').read_text())
        params = res.get('damping parameters', {})
        for key, value in D3ZERO_PBE.items():
            if key in params and not np.isclose(params[key], value, rtol=0, atol=1e-8):
                raise ValueError(f's-dftd3 used {key} = {params[key]}, expected {value}')
        energy = float(res['energy']) * Hartree
        forces = -np.asarray(res['gradient'], float).reshape(-1, 3) * Hartree / Bohr
        stress = None
        if periodic and 'virial' in res:
            stress = np.asarray(res['virial'], float).reshape(3, 3) * Hartree / atoms.get_volume()
        return energy, forces, stress


class TorchDFTD3:
    """D3(0)/PBE from torch-dftd, float64, no three-body term."""

    def __init__(self, device: str = 'cpu', cutoff: float | None = None, cnthr: float | None = None):
        import torch
        from torch_dftd.dftd3_xc_params import get_dftd3_default_params
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        p = get_dftd3_default_params('zero', 'pbe')
        want = {'s6': D3ZERO_PBE['s6'], 'rs6': D3ZERO_PBE['rs6'], 's18': D3ZERO_PBE['s8'], 'rs18': D3ZERO_PBE['rs8'],
                'alp': D3ZERO_PBE['alp']}
        if any(abs(p[k] - v) > 1e-8 for k, v in want.items()):
            raise ValueError(f'torch-dftd zero/pbe parameters {p} differ from VASP IVDW = 11 {want}')
        kw = {}
        if cutoff is not None:
            kw['cutoff'] = cutoff
        if cnthr is not None:
            kw['cnthr'] = cnthr
        self.calc = TorchDFTD3Calculator(damping='zero', xc='pbe', abc=False, device=device, dtype=torch.float64, **kw)
        self.settings = {'device': device, **kw}

    def describe(self) -> dict:
        import torch_dftd
        return {'backend': 'torch-dftd', 'version': getattr(torch_dftd, '__version__', '?'), 'damping': 'zero',
                'xc': 'pbe', 'abc': False, **self.settings}

    def compute(self, atoms):
        a = atoms.copy()
        a.calc = self.calc
        energy, forces = float(a.get_potential_energy()), a.get_forces()
        stress = a.get_stress(voigt=False) if a.pbc.all() else None
        return energy, forces, stress


def _voigt(stress3):
    s = np.asarray(stress3, float)
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])


def subtract_d3(frames, backend, incar: dict | None = None):
    """Turn full PBE-D3(0) labels into E - D3, F - D3, sigma - D3 targets, in place.

    Keeps the originals as DFT_energy / DFT_forces / DFT_stress and the dispersion part as D3_*. REF_stress (Voigt,
    ASE sign) is changed only where present; a frame with REF_stress needs a backend that returns stress.
    """
    if incar is not None:
        problems = check_incar_d3zero(incar)
        if problems:
            raise ValueError('labels are not PBE + D3(0): ' + '; '.join(problems))
    description = backend.describe()
    for at in frames:
        if at.info.get('energy_target') == TARGET_MINUS_D3 or 'D3_energy' in at.info:
            raise ValueError('D3 already subtracted from this frame; refusing to subtract twice')
        energy, forces, stress = backend.compute(at)
        if forces.shape != at.arrays['REF_forces'].shape or not np.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError('D3 backend returned malformed results')
        at.info['DFT_energy'] = float(at.info['REF_energy'])
        at.set_array('DFT_forces', at.arrays['REF_forces'].copy())
        at.info['D3_energy'] = energy
        at.set_array('D3_forces', np.asarray(forces, float))
        at.info['REF_energy'] = at.info['DFT_energy'] - energy
        at.arrays['REF_forces'] = at.arrays['DFT_forces'] - forces
        if 'REF_stress' in at.info:
            if stress is None:
                raise ValueError('frame has a stress label but the D3 backend gave no stress')
            at.info['DFT_stress'] = np.asarray(at.info['REF_stress'], float).copy()
            at.info['D3_stress'] = _voigt(stress)
            at.info['REF_stress'] = at.info['DFT_stress'] - at.info['D3_stress']
        at.info['energy_target'] = TARGET_MINUS_D3
        at.info['d3_backend'] = description['backend']
    return description


def add_d3(frames, backend):
    """Inverse of :func:`subtract_d3` with a fresh D3 evaluation (round-trip check, or rebuilding full targets)."""
    for at in frames:
        if at.info.get('energy_target') != TARGET_MINUS_D3:
            raise ValueError('frame target is not E - D3(0); refusing to add D3')
        energy, forces, stress = backend.compute(at)
        at.info['REF_energy'] = float(at.info['REF_energy']) + energy
        at.arrays['REF_forces'] = at.arrays['REF_forces'] + forces
        if 'REF_stress' in at.info:
            at.info['REF_stress'] = np.asarray(at.info['REF_stress'], float) + _voigt(stress)
        for key in ('D3_energy', 'D3_stress', 'DFT_energy', 'DFT_stress', 'd3_backend'):
            at.info.pop(key, None)
        for key in ('D3_forces', 'DFT_forces'):
            if key in at.arrays:
                del at.arrays[key]
        at.info['energy_target'] = TARGET_FULL
