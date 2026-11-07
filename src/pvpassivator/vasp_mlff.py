"""DFT steps from VASP on-the-fly MLFF runs (``ML_LMODE``/``ML_MODE = train``) as label frames.

Only configurations that VASP computed with DFT are kept, and only when two independent records agree:

* the OUTCAR ionic step: electronic iterations ran, "aborting loop because EDIFF is reached" was printed, fewer than
  NELM iterations, a force block and a ``free  energy   TOTEN`` line without the ``ML`` marker;
* an ``ML_AB`` training-structure entry with the same atoms, cell, positions, forces, energy and (if present) stress.

Labels are taken from the OUTCAR (free energy TOTEN, forces, ``in kB`` stress converted with ASE's own
``convert_vasp_outcar_stress``: ASE sign, eV/A^3, Voigt xx yy zz yz xz xy). ML_AB values serve as the cross-check.

The ML-step markers (``(ML)`` in the force header, ``ML`` before TOTEN / "energy without entropy") and the ML_AB
layout follow VASP 6.3-6.4 output as described in the VASP wiki; they were written against synthetic fixtures, not a
real ML_AB file (none exists on the desk laptop, 2026-11-07). Anything that does not match is rejected with a reason,
never guessed, so the first real run must be checked with ``mlff_dft_frames`` before export.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.vasp_parsers.vasp_outcar_parsers import convert_vasp_outcar_stress

from .vasp_io import parse_incar

_SEP_STAR = re.compile(r'^\s*\*{10,}\s*$')
_SEP_EQ = re.compile(r'^\s*={10,}\s*$')
_SEP_DASH = re.compile(r'^\s*-{10,}\s*$')


def _is_sep(line):
    return bool(_SEP_STAR.match(line) or _SEP_EQ.match(line) or _SEP_DASH.match(line))


def parse_ml_ab(path):
    """Training structures of an ML_AB file: list of dicts (symbols, cell, positions, energy, forces, stress_kB)."""
    lines = Path(path).read_text(errors='replace').splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip().startswith('Configuration num.')]
    configs = []
    for n, start in enumerate(starts):
        block = lines[start:starts[n + 1] if n + 1 < len(starts) else len(lines)]
        sections, i = {}, 1
        while i < len(block):
            if (_SEP_EQ.match(block[i]) or _SEP_STAR.match(block[i])) and i + 2 < len(block) and _SEP_DASH.match(block[i + 2]):
                name, j, body = block[i + 1].strip(), i + 3, []
                while j < len(block) and not (_SEP_EQ.match(block[j]) or _SEP_STAR.match(block[j])):
                    body.append(block[j])
                    j += 1
                sections[name] = body
                i = j
            else:
                i += 1
        try:
            symbols = []
            for row in sections['Atom types and atom numbers']:
                if row.strip():
                    el, count = row.split()
                    symbols += [el] * int(count)
            nums = lambda key: np.array([[float(x) for x in r.split()] for r in sections[key] if r.strip()])
            cell = nums('Primitive lattice vectors (ang.)')
            pos = nums('Atomic positions (ang.)')
            forces = nums('Forces (eV ang.^-1)')
            energy = float(next(r for r in sections['Total energy (eV)'] if r.strip()).split()[0])
            stress = None
            if 'Stress (kbar)' in sections:
                vals = [r for r in sections['Stress (kbar)'] if r.strip() and not _is_sep(r)
                        and not r.strip().startswith(('XX', 'XY'))]
                diag, off = [float(x) for x in vals[0].split()], [float(x) for x in vals[1].split()]
                stress = diag + off               # XX YY ZZ XY YZ ZX, kbar, as written by VASP
            if cell.shape != (3, 3) or pos.shape != (len(symbols), 3) or forces.shape != pos.shape:
                raise ValueError('malformed ML_AB configuration')
        except (KeyError, ValueError, StopIteration, IndexError) as exc:
            configs.append({'index': n, 'error': f'{type(exc).__name__}: {exc}'})
            continue
        configs.append({'index': n, 'symbols': symbols, 'cell': cell, 'positions': pos, 'energy': energy,
                        'forces': forces, 'stress_kB': stress})
    return configs


def outcar_steps(path, natoms):
    """Ionic steps of an MLFF-run OUTCAR, each tagged DFT or ML.

    Returns dicts with keys: index, method ('dft' | 'ml' | 'unclear'), n_scf, scf_converged, cell, positions,
    forces, free_energy, energy_no_entropy, stress_kB (XX YY ZZ XY YZ ZX) or None.
    """
    steps, cell, cur = [], None, None

    def new():
        return {'n_scf': 0, 'scf_converged': False, 'force_header_ml': None, 'energy_ml': None, 'cell': None,
                'positions': None, 'forces': None, 'free_energy': None, 'energy_no_entropy': None, 'stress_kB': None}
    cur = new()
    with open(path, errors='replace') as fh:
        it = iter(fh)
        for line in it:
            if 'Iteration' in line and '(' in line and '---' in line:
                cur['n_scf'] += 1
            elif 'aborting loop because EDIFF is reached' in line:
                cur['scf_converged'] = True
            elif 'direct lattice vectors' in line:
                rows = [next(it) for _ in range(3)]
                cell = np.array([[float(x) for x in r.split()[:3]] for r in rows])
            elif line.strip().startswith('in kB') or line.strip().startswith('ML in kB'):
                try:
                    cur['stress_kB'] = [float(x) for x in line.split()[-6:]]
                except ValueError:
                    cur['stress_kB'] = 'unreadable'
            elif 'POSITION' in line and 'TOTAL-FORCE' in line:
                cur['force_header_ml'] = '(ML)' in line or line.strip().startswith('ML')
                next(it)
                rows = []
                for r in it:
                    if r.strip().startswith('---'):
                        break
                    rows.append([float(x) for x in r.split()[:6]])
                rows = np.array(rows)
                if rows.shape == (natoms, 6):
                    cur['positions'], cur['forces'] = rows[:, :3], rows[:, 3:]
                cur['cell'] = None if cell is None else cell.copy()
            elif 'free  energy' in line and 'TOTEN' in line:
                # double spaces: the ionic-step line, not the per-SCF-iteration "free energy    TOTEN"
                cur['energy_ml'] = 'ML' in line.split('TOTEN')[0]
                cur['free_energy'] = float(line.split('=')[1].split()[0])
            elif 'energy  without entropy' in line and cur['free_energy'] is not None:
                # printed after TOTEN; closes the ionic step
                cur['energy_no_entropy'] = float(re.findall(r'[-+]?\d+\.\d+(?:[eE][-+]?\d+)?', line)[0])
                cur['index'] = len(steps)
                ml_marks = {cur['force_header_ml'], cur['energy_ml']}
                if ml_marks == {False} and cur['n_scf'] > 0:
                    cur['method'] = 'dft'
                elif ml_marks == {True} and cur['n_scf'] == 0:
                    cur['method'] = 'ml'
                else:
                    cur['method'] = 'unclear'
                steps.append(cur)
                cur = new()
    return steps


def _same_geometry(cell_a, pos_a, cell_b, pos_b, tol):
    if not np.allclose(cell_a, cell_b, atol=tol, rtol=0):
        return False
    frac = np.linalg.solve(np.asarray(cell_a).T, (np.asarray(pos_a) - np.asarray(pos_b)).T).T
    frac -= np.round(frac)
    return bool(np.abs(frac @ np.asarray(cell_a)).max() <= tol)


def mlff_dft_frames(folder, expected_symbols, outcar='OUTCAR', ml_ab='ML_AB', nelm=None,
                    pos_tol=1e-4, energy_tol=1e-4, force_tol=1e-4, stress_tol=1e-2):
    """Cross-checked DFT steps of one VASP on-the-fly MLFF run.

    Returns (frames, rejected, summary). frames is a list of (OUTCAR step index, Atoms) with REF_energy (free energy),
    REF_forces and, where the step printed it, REF_stress (ASE Voigt, eV/A^3).
    """
    folder = Path(folder)
    incar = parse_incar(folder / 'INCAR') if (folder / 'INCAR').exists() else {}
    mode = str(incar.get('ML_LMODE', incar.get('ML_MODE', ''))).strip().lower()
    if str(incar.get('ML_LMLFF', 'F')).strip('.').upper() not in ('T', 'TRUE') or mode != 'train':
        raise ValueError('not an on-the-fly training run (need ML_LMLFF = .TRUE. and ML_LMODE/ML_MODE = train)')
    nelm = int(float(incar.get('NELM', 60))) if nelm is None else nelm
    steps = outcar_steps(folder / outcar, len(expected_symbols))
    configs = parse_ml_ab(folder / ml_ab)
    frames, rejected, used = [], [], set()
    summary = {'outcar_steps': len(steps), 'dft_steps': 0, 'ml_steps': 0, 'unclear_steps': 0,
               'ml_ab_configs': len(configs), 'ml_ab_unmatched': 0, 'dft_steps_not_in_ml_ab': 0,
               'ml_ab_energy_matches': {}}
    for c in configs:
        if 'error' in c:
            rejected.append({'ml_ab_config': c['index'], 'reason': 'unreadable ML_AB entry: ' + c['error']})
    for s in steps:
        summary[s['method'] + '_steps'] += 1
        if s['method'] == 'ml':
            continue
        tag = {'ionic_step': s['index']}
        if s['method'] == 'unclear':
            rejected.append({**tag, 'reason': 'step not clearly DFT or ML (markers disagree)'})
            continue
        if not s['scf_converged'] or s['n_scf'] >= nelm:
            rejected.append({**tag, 'reason': 'SCF did not reach EDIFF or NELM reached'})
            continue
        if s['positions'] is None or s['cell'] is None:
            rejected.append({**tag, 'reason': 'incomplete force block or cell'})
            continue
        if not (np.isfinite(s['forces']).all() and np.isfinite(s['free_energy'])):
            rejected.append({**tag, 'reason': 'nonfinite labels'})
            continue
        match = None
        for c in configs:
            if 'error' in c or c['index'] in used or c['symbols'] != list(expected_symbols):
                continue
            if _same_geometry(c['cell'], c['positions'], s['cell'], s['positions'], pos_tol):
                match = c
                break
        if match is None:
            summary['dft_steps_not_in_ml_ab'] += 1
            rejected.append({**tag, 'reason': 'DFT step has no matching ML_AB entry (not cross-checked)'})
            continue
        problems = []
        if not np.allclose(match['forces'], s['forces'], atol=force_tol, rtol=0):
            problems.append('forces differ between ML_AB and OUTCAR')
        which = [k for k in ('free_energy', 'energy_no_entropy')
                 if s[k] is not None and abs(match['energy'] - s[k]) <= energy_tol]
        if not which:
            problems.append('energy differs between ML_AB and OUTCAR')
        stress = None
        if isinstance(s['stress_kB'], list):
            if match['stress_kB'] is not None and not np.allclose(match['stress_kB'], s['stress_kB'], atol=stress_tol, rtol=0):
                problems.append('stress differs between ML_AB and OUTCAR (value or sign convention)')
            stress = convert_vasp_outcar_stress(s['stress_kB'])
        elif s['stress_kB'] == 'unreadable':
            problems.append('unreadable stress line')
        if problems:
            rejected.append({**tag, 'ml_ab_config': match['index'], 'reason': '; '.join(problems)})
            continue
        used.add(match['index'])
        summary['ml_ab_energy_matches'][which[0]] = summary['ml_ab_energy_matches'].get(which[0], 0) + 1
        at = Atoms(list(expected_symbols), positions=s['positions'], cell=s['cell'], pbc=True)
        at.info['REF_energy'] = float(s['free_energy'])
        at.set_array('REF_forces', np.asarray(s['forces'], float))
        if stress is not None:
            at.info['REF_stress'] = np.asarray(stress, float)
        at.info['mlff_step'] = s['index']
        at.info['ml_ab_config'] = match['index']
        frames.append((s['index'], at))
    summary['ml_ab_unmatched'] = sum(1 for c in configs if 'error' not in c and c['index'] not in used)
    return frames, rejected, summary
