"""Exporter extensions for the Pb/Sn grain-boundary MLIP (branch mlip-pbsn-gb): stress, families, element set,
VASP on-the-fly MLFF DFT steps, D3(0) subtraction. Synthetic fixtures only; no VASP run."""
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from ase.units import GPa

from pvpassivator.dispersion import (D3ZERO_PBE, SDFTD3Executable, TARGET_MINUS_D3, add_d3, check_incar_d3zero,
                                     subtract_d3)
from pvpassivator.training_data import (PBSN_GB_ELEMENTS, PBSN_GB_ELEMENTS_F, export_dataset, prepare_finetune,
                                        vasp_kbar_to_ase_stress, xml_frames)
from pvpassivator.vasp_mlff import mlff_dft_frames, outcar_steps, parse_ml_ab

ZVAL = {'Cs': 9.0, 'Pb': 14.0, 'Sn': 14.0, 'I': 7.0, 'F': 7.0, 'He': 2.0}


# ----------------------------------------------------------------------------------------------- fixtures
def vxml(symbols, cell, frames, stress=None, nelect=None, encut=520):
    """vasprun.xml text; frames = list of (fractional positions, energy, forces)."""
    calcs = []
    for k, (frac, energy, forces) in enumerate(frames):
        scf = ''.join(f'<scstep><energy><i name="e_fr_energy">{e}</i><i name="e_0_energy">{e}</i></energy></scstep>'
                      for e in (energy - 1e-6, energy))
        st = ''
        if stress is not None:
            st = '<varray name="stress">' + ''.join('<v>' + ' '.join(map(str, r)) + '</v>' for r in stress[k]) + '</varray>'
        calcs.append(f'<calculation>{scf}<energy><i name="e_fr_energy">{energy}</i></energy><structure><crystal>'
                     '<varray name="basis">' + ''.join('<v>' + ' '.join(map(str, r)) + '</v>' for r in cell) + '</varray>'
                     '</crystal><varray name="positions">' + ''.join('<v>' + ' '.join(map(str, r)) + '</v>' for r in frac)
                     + '</varray></structure><varray name="forces">' + ''.join('<v>' + ' '.join(map(str, r)) + '</v>' for r in forces)
                     + '</varray>' + st + '</calculation>')
    ne = sum(ZVAL[s] for s in symbols) if nelect is None else nelect
    return (f'<modeling><incar><i name="ENCUT">{encut}</i><i name="ISPIN">1</i></incar>'
            f'<parameters><i name="NELECT">{ne}</i><i name="NELM">60</i><i name="EDIFF">0.0001</i></parameters>'
            '<atominfo><array name="atoms"><set>' + ''.join(f'<rc><c>{s}</c></rc>' for s in symbols) + '</set></array></atominfo>'
            + ''.join(calcs) + '</modeling>')


def species_blocks(symbols):
    order = list(dict.fromkeys(symbols))
    return order, [symbols.count(s) for s in order]


def outcar_header(symbols, nelect=None):
    order, counts = species_blocks(symbols)
    ne = sum(ZVAL[s] for s in symbols) if nelect is None else nelect
    lines = [f'   TITEL  = PAW_PBE {s} 01Jan2000' for s in order]
    lines += [f'   POMASS =  1.000; ZVAL   =  {ZVAL[s]:.3f}    mass and valenz' for s in order]
    lines += ['   ions per type =  ' + ' '.join(map(str, counts)), f'   NELECT =  {ne:.4f}    total number of electrons']
    return '\n'.join(lines) + '\n'


def poscar(symbols, cell, frac):
    order, counts = species_blocks(symbols)
    return ('x\n1.0\n' + ''.join(' '.join(map(str, r)) + '\n' for r in cell) + ' '.join(order) + '\n'
            + ' '.join(map(str, counts)) + '\nDirect\n' + ''.join(' '.join(map(str, r)) + '\n' for r in frac))


def job(root, name, symbols, cell, frames, *, kpoints='Gamma\n1 1 1', stress=None, incar_extra='', step='static',
        nelect=None, encut=520):
    inp, out = root / 'inputs' / name / step, root / 'outputs' / name / step
    inp.mkdir(parents=True)
    out.mkdir(parents=True)
    (inp / 'POSCAR').write_text(poscar(symbols, cell, frames[0][0]))
    (out / 'INCAR').write_text(f'ENCUT={encut}\nISPIN=1\nNSW=0\nIBRION=-1\n{incar_extra}')
    (out / 'KPOINTS').write_text(f'k\n0\n{kpoints}\n0 0 0\n')
    (out / 'DONE').write_text('done')
    (out / 'OUTCAR').write_text(outcar_header(symbols, nelect) +
                                ' energy  without entropy= -1.0 energy(sigma->0) = -1.0\n General timing and accounting\n')
    with gzip.open(out / 'vasprun.xml.gz', 'wt') as fh:
        fh.write(vxml(symbols, cell, frames, stress, nelect, encut))


def bulk_frames(symbols, n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    base = rng.random((len(symbols), 3))
    return [((base + shift + 0.01 * k) % 1.0, -10.0 - 0.1 * k - shift, rng.normal(0, 0.1, (len(symbols), 3)))
            for k in range(n)]


CSPBI3 = ['Cs', 'Pb', 'I', 'I', 'I']
CSSNI3 = ['Cs', 'Sn', 'I', 'I', 'I']


def family_package(root, **over):
    """Two families with different cells and k-meshes, no reactions; F1 carries stress."""
    root.mkdir()
    cellA, cellB = np.eye(3) * 6.3, np.diag([6.3, 6.3, 25.0])
    stressA = [np.diag([5.0, 6.0, 7.0]) + np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]) for _ in range(2)]
    job(root, 'bulk_pb', CSPBI3, cellA, bulk_frames(CSPBI3, 2, 1), kpoints='Gamma\n4 4 4', stress=stressA, incar_extra='ISIF=2\n')
    job(root, 'bulk_sn', CSSNI3, cellA, bulk_frames(CSSNI3, 2, 2), kpoints='Gamma\n4 4 4', stress=stressA, incar_extra='ISIF=2\n')
    job(root, 'slab', CSSNI3, cellB, bulk_frames(CSSNI3, 3, 3), kpoints=over.get('slab_k', 'Gamma\n4 4 1'))
    jobs = {'bulk_pb': {'role': 'bulk', 'family': 'F1', 'stress': True, 'steps': ['static']},
            'bulk_sn': {'role': 'bulk', 'family': 'F1', 'stress': True, 'steps': ['static']},
            'slab': {'role': 'surface', 'family': 'F2', 'steps': ['static']}}
    jobs.update(over.get('jobs', {}))
    manifest = {'cell_policy': 'per_job', 'jobs': jobs,
                'bundles': over.get('bundles', [['bulk_pb'], ['bulk_sn'], ['slab']])}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


# ----------------------------------------------------------------------------------------------- stress
def test_vasp_stress_sign_units_and_voigt_order():
    kb = np.array([[1.0, 4.0, 5.0], [4.0, 2.0, 6.0], [5.0, 6.0, 3.0]])
    s = vasp_kbar_to_ase_stress(kb)
    # positive kB (VASP: cell under compression, wants to expand) is a negative ASE stress; 1 kB = 0.1 GPa
    assert np.allclose(s, -0.1 * GPa * np.array([1, 2, 3, 6, 5, 4]))


def test_xml_stress_export_and_missing_stress_rejected(tmp_path):
    cell = np.eye(3) * 6.3
    frames = bulk_frames(CSPBI3, 2, 0)
    kb = [np.diag([10.0, 20.0, 30.0]), np.diag([1.0, 2.0, 3.0])]
    p = tmp_path / 'vasprun.xml'
    p.write_text(vxml(CSPBI3, cell, frames, kb))
    ok, bad = xml_frames(p, CSPBI3, 60, 1e-4, want_stress=True)
    assert len(ok) == 2 and not bad
    assert np.allclose(ok[0][1].info['REF_stress'], -0.1 * GPa * np.array([10, 20, 30, 0, 0, 0]))
    p.write_text(vxml(CSPBI3, cell, frames))
    ok, bad = xml_frames(p, CSPBI3, 60, 1e-4, want_stress=True)
    assert not ok and all('stress' in r['reason'] for r in bad)
    assert len(xml_frames(p, CSPBI3, 60, 1e-4)[0]) == 2       # stress is optional unless requested


ARCHIVED = Path(r'D:\Research_Storage\CsPbI2Br_2\CsPbI2Br_Surface_Calcs\PBE\CsI\CsPbI2Br_333_SurfCsI_6\vasprun.xml')


@pytest.mark.skipif(not ARCHIVED.exists(), reason='archived PBE-D3(0) ISIF=4 vasprun not on this machine')
def test_stress_matches_ase_reader_on_archived_vasp_run():
    symbols = read(ARCHIVED, index=0).get_chemical_symbols()
    frames, _ = xml_frames(ARCHIVED, symbols, 60, 1e-4, want_stress=True)
    ref = read(ARCHIVED, index=':')
    assert frames
    for i, at in frames[:5]:
        assert np.allclose(at.info['REF_stress'], ref[i].get_stress(voigt=True), atol=1e-10)


# ----------------------------------------------------------------------------------------------- families
def test_multi_family_cells_kmeshes_tags_and_stress(tmp_path):
    p = family_package(tmp_path / 'p')
    r = export_dataset([p], tmp_path / 'ds', elements=PBSN_GB_ELEMENTS)
    assert r['files']['train']['frames'] == 7 and not r['rejected']
    assert r['families']['F1']['stress_frames'] == 4 and r['families']['F2']['stress_frames'] == 0
    assert r['families']['F1']['kpoints'] != r['families']['F2']['kpoints']
    assert 'No stress labels' not in r['limitations']
    ats = read(tmp_path / 'ds/train.extxyz', index=':')
    assert {a.info['family'] for a in ats} == {'F1', 'F2'}
    assert {round(a.cell[2, 2], 1) for a in ats} == {6.3, 25.0}          # two cell shapes in one dataset
    slab = [a for a in ats if a.info['family'] == 'F2']
    assert all('REF_stress' not in a.info and a.info['config_stress_weight'] == 0.0 for a in slab)
    bulk = [a for a in ats if a.info['family'] == 'F1']
    assert all(np.asarray(a.info['REF_stress']).shape == (6,) for a in bulk)
    # bundle split policy kept: one bundle to valid, the rest stay train
    prefix = next(iter(r['jobs'])).split('/')[0]
    r2 = export_dataset([p], tmp_path / 'ds2', {'job_splits': {prefix + '/slab': 'valid'}})
    assert r2['ready_for_pilot'] and r2['files']['valid']['frames'] == 3


def test_kmesh_must_agree_within_a_family(tmp_path):
    p = family_package(tmp_path / 'p', jobs={'slab': {'role': 'surface', 'family': 'F1', 'steps': ['static']}})
    with pytest.raises(ValueError, match=r'\[F1\] KPOINTS'):
        export_dataset([p], tmp_path / 'ds')


def test_bundle_cannot_cross_splits(tmp_path):
    p = family_package(tmp_path / 'p', bundles=[['bulk_pb', 'bulk_sn'], ['slab']])
    from pvpassivator.training_data import sha256
    prefix = sha256(p / 'manifest.json')[:16]
    with pytest.raises(ValueError, match='crosses data splits'):
        export_dataset([p], tmp_path / 'ds', {'job_splits': {prefix + '/bulk_pb': 'train', prefix + '/bulk_sn': 'test'}})


def test_fixed_cell_run_must_keep_its_cell(tmp_path):
    p = family_package(tmp_path / 'p')
    gz = p / 'outputs/slab/static/vasprun.xml.gz'
    with gzip.open(gz, 'wt') as fh:
        fh.write(vxml(CSSNI3, np.diag([6.3, 6.3, 26.0]), bulk_frames(CSSNI3, 3, 3)))
    r = export_dataset([p], tmp_path / 'ds')
    assert any('fixed-cell run changed its cell' in x['reason'] for x in r['rejected'])
    assert r['families'].get('F2') is None


def test_element_set_and_neutrality_checks(tmp_path):
    root = tmp_path / 'p'
    root.mkdir()
    cell = np.eye(3) * 6.3
    csPbF = ['Cs', 'Pb', 'I', 'I', 'F']
    job(root, 'withF', csPbF, cell, bulk_frames(csPbF, 1, 5))
    job(root, 'charged', CSPBI3, cell, bulk_frames(CSPBI3, 1, 6, 0.3), nelect=sum(ZVAL[s] for s in CSPBI3) + 1)
    job(root, 'odd', ['Cs', 'Pb', 'I', 'I', 'I', 'I'], cell, bulk_frames(['Cs', 'Pb', 'I', 'I', 'I', 'I'], 1, 7))
    jobs = {n: {'role': 'bulk', 'family': 'F1', 'steps': ['static']} for n in ('withF', 'charged', 'odd')}
    (root / 'manifest.json').write_text(json.dumps({'cell_policy': 'per_job', 'jobs': jobs,
                                                    'bundles': [['withF'], ['charged'], ['odd']]}))
    r = export_dataset([root], tmp_path / 'a', elements=PBSN_GB_ELEMENTS)
    reasons = ' | '.join(x['reason'] for x in r['rejected'])
    assert "outside the allowed set: ['F']" in reasons
    assert 'neutral electron count unverified' in reasons          # NELECT = ZVAL sum + 1
    assert 'odd electron count' in reasons                          # CsPbI4: 9+14+28 = 51 electrons
    assert r['files']['train']['frames'] == 0
    r = export_dataset([root], tmp_path / 'b', elements=PBSN_GB_ELEMENTS_F)
    assert r['files']['train']['frames'] == 1
    assert 'F' in PBSN_GB_ELEMENTS_F and 'F' not in PBSN_GB_ELEMENTS and {'Sn', 'Cs', 'Cl', 'S'} <= PBSN_GB_ELEMENTS


def test_atom_order_change_rejected(tmp_path):
    p = family_package(tmp_path / 'p')
    gz = p / 'outputs/bulk_pb/static/vasprun.xml.gz'
    with gzip.open(gz, 'wt') as fh:
        fh.write(vxml(['Pb', 'Cs', 'I', 'I', 'I'], np.eye(3) * 6.3, bulk_frames(CSPBI3, 2, 1), [np.eye(3)] * 2))
    r = export_dataset([p], tmp_path / 'ds')
    assert any('atom order' in x['reason'] for x in r['rejected'])


# ----------------------------------------------------------------------------------------------- fine-tune config
def test_prepare_finetune_stress_multihead_and_elements(tmp_path):
    p = family_package(tmp_path / 'p')
    from pvpassivator.training_data import sha256
    prefix = sha256(p / 'manifest.json')[:16]
    export_dataset([p], tmp_path / 'ds', {'job_splits': {prefix + '/bulk_sn': 'valid'}})
    model = tmp_path / 'foundation.model'
    model.write_bytes(b'checkpoint')
    prov = prepare_finetune(tmp_path / 'ds', model, tmp_path / 'run', '0.3.16', stress_weight=1.0)
    cfg = json.loads((tmp_path / 'run/mace_config.json').read_text())
    assert cfg['loss'] == 'stress' and cfg['stress_key'] == 'REF_stress' and cfg['stress_weight'] == 1.0
    assert cfg['multiheads_finetuning'] is False and prov['train_frames_with_stress'] == 2
    with pytest.raises(ValueError, match='replay file'):
        prepare_finetune(tmp_path / 'ds', model, tmp_path / 'run2', '0.3.16', multihead=True)
    replay = tmp_path / 'replay.extxyz'
    replay.write_text('')
    prepare_finetune(tmp_path / 'ds', model, tmp_path / 'run3', '0.3.16', multihead=True, replay_file=replay)
    cfg = json.loads((tmp_path / 'run3/mace_config.json').read_text())
    assert cfg['multiheads_finetuning'] is True and cfg['pt_train_file'].endswith('replay.extxyz')
    assert 'loss' not in cfg                                         # stress off unless asked
    with pytest.raises(ValueError, match="lacks required elements: \\['Cl'\\]"):
        prepare_finetune(tmp_path / 'ds', model, tmp_path / 'run4', '0.3.16', required_elements=['Cs', 'Cl'])


def test_prepare_finetune_refuses_stress_weight_without_stress(tmp_path):
    p = family_package(tmp_path / 'p', jobs={'bulk_pb': {'role': 'bulk', 'family': 'F1', 'steps': ['static']},
                                            'bulk_sn': {'role': 'bulk', 'family': 'F1', 'steps': ['static']}})
    from pvpassivator.training_data import sha256
    prefix = sha256(p / 'manifest.json')[:16]
    export_dataset([p], tmp_path / 'ds', {'job_splits': {prefix + '/slab': 'valid'}})
    model = tmp_path / 'm.model'
    model.write_bytes(b'x')
    with pytest.raises(ValueError, match='no training frame carries REF_stress'):
        prepare_finetune(tmp_path / 'ds', model, tmp_path / 'run', '0.3.16', stress_weight=0.5)


# ----------------------------------------------------------------------------------------------- VASP MLFF
def mlab_text(configs):
    star, eq, dash = '*' * 50, '=' * 50, '-' * 50
    out = [' 1.0 Version', star, '     The number of configurations', dash, f'        {len(configs)}', star]
    for n, c in enumerate(configs, 1):
        order, counts = species_blocks(c['symbols'])
        out += [star, f'     Configuration num.{n:7d}', eq, '     System name', dash, '     test', eq,
                '     The number of atom types', dash, f'       {len(order)}', eq, '     The number of atoms', dash,
                f'       {len(c["symbols"])}', star, '     Atom types and atom numbers', dash]
        out += [f'     {s} {k}' for s, k in zip(order, counts)]
        out += [eq, '     CTIFOR', dash, '   2.0E-003', eq, '     Primitive lattice vectors (ang.)', dash]
        out += ['  ' + '  '.join(f'{x:.10f}' for x in r) for r in c['cell']]
        out += [eq, '     Atomic positions (ang.)', dash]
        out += ['  ' + '  '.join(f'{x:.10f}' for x in r) for r in c['positions']]
        out += [eq, '     Total energy (eV)', dash, f'  {c["energy"]:.10f}', eq, '     Forces (eV ang.^-1)', dash]
        out += ['  ' + '  '.join(f'{x:.10f}' for x in r) for r in c['forces']]
        if c.get('stress') is not None:
            s = c['stress']
            out += [eq, '     Stress (kbar)', dash, '     XX YY ZZ', dash, '  ' + '  '.join(f'{x:.6f}' for x in s[:3]),
                    dash, '     XY YZ ZX', dash, '  ' + '  '.join(f'{x:.6f}' for x in s[3:])]
    out.append(star)
    return '\n'.join(out) + '\n'


def outcar_step(cell, pos, forces, energy, *, method='dft', n_scf=8, converged=True, stress=None):
    ml = method == 'ml'
    lines = []
    if not ml:
        for k in range(1, n_scf + 1):
            lines.append(f'----------------------------------------- Iteration    1({k:4d})  ---------------------------------------')
            lines.append(f'  free energy    TOTEN  =     {energy + 1.0 / k:.8f} eV')
        lines.append('  aborting loop because EDIFF is reached' if converged else '  aborting loop EDIFF was not reached (unconverged)')
    if stress is not None:
        lines.append(('  ML in kB  ' if ml else '  in kB     ') + ' '.join(f'{x:11.5f}' for x in stress))
    lines.append('      direct lattice vectors                 reciprocal lattice vectors')
    lines += ['    ' + ' '.join(f'{x:12.9f}' for x in r) + '     0.0 0.0 0.0' for r in cell]
    lines.append(' POSITION                                       TOTAL-FORCE (eV/Angst)' + (' (ML)' if ml else ''))
    lines.append(' -----------------------------------------------------------------------------------')
    lines += [' ' + ' '.join(f'{x:12.5f}' for x in p) + ' ' + ' '.join(f'{x:13.6f}' for x in f) for p, f in zip(pos, forces)]
    lines.append(' -----------------------------------------------------------------------------------')
    lines.append(('  ML ' if ml else '  ') + f'free  energy{" ML" if ml else ""}   TOTEN  =     {energy:.8f} eV')
    lines.append(('  ML ' if ml else '  ') + f'energy  without entropy=     {energy + 0.001:.8f}  energy(sigma->0) =     {energy + 0.0005:.8f}')
    return '\n'.join(lines) + '\n'


def mlff_run(folder, symbols=CSPBI3, incar='ML_LMLFF = .TRUE.\nML_LMODE = train\nIBRION = 0\nISIF = 2\nNSW = 4\nISPIN = 1\nENCUT = 520\n',
             ml_ab_energy='free', ab_force_shift=0.0, ab_stress_sign=1.0):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(11)
    cell = np.eye(3) * 6.3
    steps = []
    for k in range(5):
        pos = (rng.random((len(symbols), 3)) * 6.3)
        steps.append(dict(pos=pos, forces=rng.normal(0, 0.2, (len(symbols), 3)), energy=-20.0 - 0.05 * k,
                          stress=[3.0 + k, 4.0, 5.0, 0.5, -0.25, 0.125]))
    plan = [dict(method='dft'),                      # 0: kept
            dict(method='ml'),                       # 1: ML step, skipped
            dict(method='dft', n_scf=60),            # 2: NELM reached
            dict(method='dft'),                      # 3: converged DFT, absent from ML_AB
            dict(method='dft', converged=False)]     # 4: SCF not converged
    text = outcar_header(symbols)
    for s, opt in zip(steps, plan):
        text += outcar_step(cell, s['pos'], s['forces'], s['energy'], stress=s['stress'], **opt)
    text += ' General timing and accounting\n'
    (folder / 'OUTCAR').write_text(text)
    (folder / 'INCAR').write_text(incar)
    (folder / 'KPOINTS').write_text('k\n0\nGamma\n1 1 1\n0 0 0\n')
    pick = lambda s: s['energy'] if ml_ab_energy == 'free' else s['energy'] + 0.001
    configs = [dict(symbols=symbols, cell=cell, positions=steps[i]['pos'] + (6.3 if i == 0 else 0.0),  # wrapped image
                    forces=steps[i]['forces'] + ab_force_shift, energy=pick(steps[i]),
                    stress=list(ab_stress_sign * np.array(steps[i]['stress']))) for i in (0, 2, 4)]
    (folder / 'ML_AB').write_text(mlab_text(configs))
    return steps


def test_ml_ab_and_outcar_parsers(tmp_path):
    steps = mlff_run(tmp_path / 'run')
    cfg = parse_ml_ab(tmp_path / 'run/ML_AB')
    assert len(cfg) == 3 and cfg[0]['symbols'] == CSPBI3 and np.allclose(cfg[1]['forces'], steps[2]['forces'])
    assert cfg[0]['stress_kB'] == pytest.approx(steps[0]['stress'])
    oc = outcar_steps(tmp_path / 'run/OUTCAR', 5)
    assert [s['method'] for s in oc] == ['dft', 'ml', 'dft', 'dft', 'dft']
    assert oc[2]['n_scf'] == 60 and not oc[4]['scf_converged']


def test_mlff_dft_steps_cross_checked(tmp_path):
    steps = mlff_run(tmp_path / 'run')
    frames, rejected, summary = mlff_dft_frames(tmp_path / 'run', CSPBI3)
    assert [i for i, _ in frames] == [0]
    at = frames[0][1]
    assert at.info['REF_energy'] == pytest.approx(steps[0]['energy'])
    assert np.allclose(at.arrays['REF_forces'], steps[0]['forces'], atol=1e-6)
    s = steps[0]['stress']   # OUTCAR order XX YY ZZ XY YZ ZX -> ASE Voigt xx yy zz yz xz xy, sign flipped
    assert np.allclose(at.info['REF_stress'], -0.1 * GPa * np.array([s[0], s[1], s[2], s[4], s[5], s[3]]))
    reasons = {r['ionic_step']: r['reason'] for r in rejected if 'ionic_step' in r}
    assert 'NELM' in reasons[2] and 'no matching ML_AB' in reasons[3] and 'EDIFF' in reasons[4]
    assert summary['ml_steps'] == 1 and summary['dft_steps'] == 4 and summary['ml_ab_energy_matches'] == {'free_energy': 1}


def test_mlff_mismatches_and_non_training_runs_rejected(tmp_path):
    mlff_run(tmp_path / 'f', ab_force_shift=0.01)
    frames, rejected, _ = mlff_dft_frames(tmp_path / 'f', CSPBI3)
    assert not frames and any('forces differ' in r['reason'] for r in rejected)
    mlff_run(tmp_path / 's', ab_stress_sign=-1.0)
    frames, rejected, _ = mlff_dft_frames(tmp_path / 's', CSPBI3)
    assert not frames and any('sign convention' in r['reason'] for r in rejected)
    mlff_run(tmp_path / 'e', ml_ab_energy='no_entropy')
    frames, _, summary = mlff_dft_frames(tmp_path / 'e', CSPBI3)
    assert len(frames) == 1 and summary['ml_ab_energy_matches'] == {'energy_no_entropy': 1}
    mlff_run(tmp_path / 'p', incar='ML_LMLFF = .TRUE.\nML_LMODE = run\nIBRION = 0\n')
    with pytest.raises(ValueError, match='on-the-fly training run'):
        mlff_dft_frames(tmp_path / 'p', CSPBI3)


def test_mlff_job_through_export_dataset(tmp_path):
    root = tmp_path / 'pkg'
    run = root / 'outputs/md300/md'
    steps = mlff_run(run)
    (run / 'DONE').write_text('done')
    inp = root / 'inputs/md300/md'
    inp.mkdir(parents=True)
    cell = np.eye(3) * 6.3
    (inp / 'POSCAR').write_text(poscar(CSPBI3, cell, np.linalg.solve(cell.T, steps[0]['pos'].T).T % 1.0))
    manifest = {'cell_policy': 'per_job', 'bundles': [['md300']],
                'jobs': {'md300': {'role': 'gb_md', 'family': 'F3b', 'kind': 'vasp_mlff', 'steps': ['md'], 'stress': True}}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    r = export_dataset([root], tmp_path / 'ds', elements=PBSN_GB_ELEMENTS)
    assert r['files']['train']['frames'] == 1 and r['families']['F3b']['stress_frames'] == 1
    assert next(iter(r['jobs'].values()))['mlff']['md']['dft_steps'] == 4


# ----------------------------------------------------------------------------------------------- D3(0)
class PairD3:
    """Deterministic stand-in backend: E = -sum_i<j c/r^6 (minimum image), with analytic forces and stress."""
    def describe(self):
        return {'backend': 'test-pair'}

    def compute(self, atoms):
        c, pos, cell = 5.0, atoms.positions, atoms.cell.array
        e, f, w = 0.0, np.zeros_like(pos), np.zeros((3, 3))
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                d = pos[j] - pos[i]
                d -= np.round(np.linalg.solve(cell.T, d)) @ cell
                r = np.linalg.norm(d)
                e += -c / r ** 6
                g = 6 * c / r ** 8 * d          # dE/d(d)
                f[i] += g
                f[j] -= g
                w += np.outer(d, g)
        return e, f, w / atoms.get_volume()


def labelled(seed, stress=True):
    rng = np.random.default_rng(seed)
    at = Atoms(CSPBI3, scaled_positions=rng.random((5, 3)), cell=np.eye(3) * 6.3, pbc=True)
    at.info['REF_energy'] = -12.3
    at.set_array('REF_forces', rng.normal(0, 0.3, (5, 3)))
    if stress:
        at.info['REF_stress'] = rng.normal(0, 1e-3, 6)
    return at


def test_d3_subtract_add_round_trip_and_double_subtraction():
    frames = [labelled(1), labelled(2, stress=False)]
    orig = [(a.info['REF_energy'], a.arrays['REF_forces'].copy(), a.info.get('REF_stress')) for a in frames]
    subtract_d3(frames, PairD3(), incar={'IVDW': '11', 'ENCUT': '520'})
    e, f, s = PairD3().compute(frames[0])
    assert frames[0].info['REF_energy'] == pytest.approx(orig[0][0] - e)
    assert np.allclose(frames[0].arrays['REF_forces'], orig[0][1] - f)
    assert frames[0].info['energy_target'] == TARGET_MINUS_D3 and 'REF_stress' not in frames[1].info
    with pytest.raises(ValueError, match='twice'):
        subtract_d3(frames, PairD3())
    add_d3(frames, PairD3())
    for a, (e0, f0, s0) in zip(frames, orig):
        assert a.info['REF_energy'] == pytest.approx(e0, abs=1e-12)
        assert np.allclose(a.arrays['REF_forces'], f0, atol=1e-12)
        if s0 is not None:
            assert np.allclose(a.info['REF_stress'], s0, atol=1e-14)
        assert 'D3_energy' not in a.info and a.info['energy_target'] == 'dft_free_energy'


def test_d3_incar_check():
    assert check_incar_d3zero({'IVDW': '11'}) == []
    assert check_incar_d3zero({'IVDW': '11', 'VDW_S8': '0.722', 'GGA': 'PE'}) == []
    assert any('IVDW' in p for p in check_incar_d3zero({'IVDW': '12'}))
    assert any('VDW_SR' in p for p in check_incar_d3zero({'IVDW': '11', 'VDW_SR': '1.1'}))
    assert any('METAGGA' in p for p in check_incar_d3zero({'IVDW': '11', 'METAGGA': 'R2SCAN'}))
    with pytest.raises(ValueError, match='not PBE'):
        subtract_d3([labelled(3)], PairD3(), incar={'IVDW': '12'})
    assert D3ZERO_PBE['s9'] == 0.0 and D3ZERO_PBE['rs6'] == 1.217 and D3ZERO_PBE['s8'] == 0.722


def _sdftd3():
    for exe in (os.environ.get('PVPASS_SDFTD3'), r'C:\Users\lguts\micromamba\envs\xtb\Library\bin\s-dftd3.exe'):
        if exe and Path(exe).is_file():
            return SDFTD3Executable(exe)
    return None


@pytest.mark.skipif(_sdftd3() is None, reason='s-dftd3 executable not available')
def test_sdftd3_backend_round_trip_and_finite_differences():
    b = _sdftd3()
    at = labelled(4)
    e, f, s = b.compute(at)
    h = 1e-4
    for i, k in ((0, 0), (2, 1)):
        num = []
        for d in (h, -h):
            x = at.copy()
            x.positions[i, k] += d
            num.append(b.compute(x)[0])
        assert -(num[0] - num[1]) / (2 * h) == pytest.approx(f[i, k], abs=1e-5)
    num = []
    for d in (h, -h):
        x = at.copy()
        x.set_cell(at.cell.array @ (np.eye(3) + np.diag([d, 0, 0])), scale_atoms=True)
        num.append(b.compute(x)[0])
    assert (num[0] - num[1]) / (2 * h) / at.get_volume() == pytest.approx(s[0, 0], abs=1e-6)
    frames = [at]
    e0, f0, s0 = at.info['REF_energy'], at.arrays['REF_forces'].copy(), at.info['REF_stress'].copy()
    subtract_d3(frames, b)
    add_d3(frames, b)
    assert at.info['REF_energy'] == pytest.approx(e0, abs=1e-10)
    assert np.allclose(at.arrays['REF_forces'], f0, atol=1e-10) and np.allclose(at.info['REF_stress'], s0, atol=1e-12)


@pytest.mark.skipif(_sdftd3() is None, reason='s-dftd3 executable not available')
def test_subtract_d3_script_writes_hashed_minus_d3_dataset(tmp_path):
    import subprocess
    import sys
    p = family_package(tmp_path / 'p')
    export_dataset([p], tmp_path / 'ds')
    script = Path(__file__).resolve().parents[1] / 'examples/scripts/subtract_d3.py'
    incar = tmp_path / 'INCAR'
    incar.write_text('IVDW = 11\nENCUT = 520\n')
    cmd = [sys.executable, str(script), str(tmp_path / 'ds'), '--out', str(tmp_path / 'ds_md3'), '--backend', 's-dftd3',
           '--exe', _sdftd3().exe, '--incar', str(incar)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    rep = json.loads((tmp_path / 'ds_md3/dataset.json').read_text())
    assert rep['energy_target'] == TARGET_MINUS_D3 and rep['d3_backend']['backend'] == 's-dftd3'
    full, minus = read(tmp_path / 'ds/train.extxyz', index=':'), read(tmp_path / 'ds_md3/train.extxyz', index=':')
    for a, b in zip(full, minus):
        assert b.info['REF_energy'] == pytest.approx(a.info['REF_energy'] - b.info['D3_energy'])
        assert b.info['energy_target'] == TARGET_MINUS_D3
    again = subprocess.run(cmd[:3] + ['--out', str(tmp_path / 'twice')] + cmd[5:], capture_output=True, text=True)
    assert again.returncode == 0   # the source dataset is still the full target; the guard is per frame
    twice = subprocess.run([sys.executable, str(script), str(tmp_path / 'ds_md3'), '--out', str(tmp_path / 'x'),
                            '--backend', 's-dftd3', '--exe', _sdftd3().exe], capture_output=True, text=True)
    assert twice.returncode != 0 and 'twice' in (twice.stderr + twice.stdout)
