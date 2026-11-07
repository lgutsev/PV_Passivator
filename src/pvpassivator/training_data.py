"""Neutral VASP energy/force data export and guarded MACE pilot preparation.

Uses the force-consistent VASP free energy, not sigma->0 adsorption energies.
No model download, training, or cluster submission occurs during preparation.

Multi-family extension (branch mlip-pbsn-gb, 2026-11-07), all opt-in through the manifest so that existing label-set
packages export exactly as before:

* ``family`` (package or job level) tags every frame; k-meshes must agree within a family, not across families.
* ``cell_policy: "per_job"`` replaces the common slab cell check: fixed-cell runs must keep their input cell.
* ``bundles``: lists of job names that form one indivisible split unit without a reaction equation
  (MD trajectories, relaxation paths, reference groups).
* job ``stress: true``: the frames carry REF_stress (only for cells without vacuum).
* job ``kind: "vasp_mlff"``: DFT steps of an on-the-fly MLFF run, cross-checked between ML_AB and OUTCAR.
* ``elements``: an allowed element set (e.g. PBSN_GB_ELEMENTS); other jobs are rejected.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.units import GPa

from .vasp_io import parse_run, compatible, HAMILTONIAN_KEYS

# Element set of the Pb/Sn grain-boundary MLIP (UNIROMA_P2 plan 2026-11-07, section 2.1); F only with 4F-PEAI.
PBSN_GB_ELEMENTS = frozenset({'H', 'C', 'N', 'O', 'S', 'Cl', 'I', 'Cs', 'Sn', 'Pb'})
PBSN_GB_ELEMENTS_F = PBSN_GB_ELEMENTS | {'F'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_key(at):
    # Atom-order invariant, including periodic wrapping; no rounded force labels.
    rows = np.column_stack((at.numbers, np.round(at.get_scaled_positions(wrap=True), 7)))
    rows = rows[np.lexsort(tuple(rows[:, i] for i in reversed(range(4))))]
    return hashlib.sha256(np.round(at.cell.array, 7).tobytes() + rows.tobytes()).hexdigest()


def vasp_kbar_to_ase_stress(kbar_3x3):
    """vasprun.xml <varray name="stress"> (kB, VASP sign) -> ASE Voigt stress (eV/A^3, xx yy zz yz xz xy).

    Same conversion as ase.io's vasprun reader: stress = -0.1 GPa * kB (checked against ase.io.read on an archived
    ISIF = 4 vasprun, 2026-11-07).
    """
    s = -0.1 * GPa * np.asarray(kbar_3x3, float)
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])


def xml_frames(path, expected_symbols, nelm, ediff, expected_nelect=None, expected_incar=None, want_stress=False):
    """Read complete XML (including gzip); reject individual unconverged SCFs.

    Truncated XML raises instead of silently accepting its prefix. Restart segments
    are intentionally usable only if complete, with their own OUTCAR metadata.
    With want_stress, every frame also needs the XML stress tensor (REF_stress, ASE Voigt, eV/A^3).
    """
    opener = gzip.open if str(path).endswith('.gz') else open
    frames, rejected, symbols, parameters_ok = [], [], None, expected_nelect is None
    def normalized(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            text = str(value).strip().strip('.').upper()
            return {'T': 'TRUE', 'F': 'FALSE'}.get(text, text)
    with opener(path, 'rb') as fh:
        for _, node in ET.iterparse(fh, events=('end',)):
            if node.tag == 'incar' and expected_incar:
                actual = {i.get('name'): i.text.strip() for i in node.findall('.//i') if i.text}
                for key in HAMILTONIAN_KEYS:
                    if key in expected_incar and (key not in actual or normalized(expected_incar[key]) != normalized(actual[key])):
                        raise ValueError(f'XML/OUTCAR-side INCAR mismatch: {key}')
            elif node.tag == 'parameters' and expected_nelect is not None:
                actual = {i.get('name'): i.text.strip() for i in node.findall('.//i') if i.text}
                if not np.isclose(float(actual.get('NELECT', 'nan')), expected_nelect, rtol=0, atol=1e-6):
                    raise ValueError('XML electron count differs from neutral OUTCAR')
                if int(float(actual.get('NELM', -1))) != nelm or not np.isclose(float(actual.get('EDIFF', 'nan')), ediff, rtol=1e-6, atol=0):
                    raise ValueError('XML convergence settings differ from INCAR')
                parameters_ok = True
            elif node.tag == 'atominfo':
                symbols = [rc.find('c').text.strip() for rc in node.findall("array[@name='atoms']/set/rc")]
                if symbols != expected_symbols:
                    raise ValueError('XML atom order/composition differs from input POSCAR')
                node.clear()
            elif node.tag == 'calculation':
                if not parameters_ok:
                    raise ValueError('XML electron-count/convergence metadata missing')
                index = len(frames) + len(rejected)
                scf = node.findall('scstep')
                reason = None
                if len(scf) < 2 or len(scf) >= nelm:
                    reason = 'SCF convergence unproven or NELM reached'
                else:
                    for name in ('e_fr_energy', 'e_0_energy'):
                        vals = [s.find(f"energy/i[@name='{name}']") for s in scf[-2:]]
                        if any(v is None for v in vals) or abs(float(vals[-1].text) - float(vals[0].text)) > ediff:
                            reason = 'last SCF energy changes exceed EDIFF or are missing'
                try:
                    def array(xpath):
                        return np.array([[float(x) for x in v.text.split()] for v in node.findall(xpath)])
                    cell = array("structure/crystal/varray[@name='basis']/v")
                    pos = array("structure/varray[@name='positions']/v")
                    forces = array("varray[@name='forces']/v")
                    energy = float(node.find("energy/i[@name='e_fr_energy']").text)
                    if symbols is None or cell.shape != (3, 3) or pos.shape != (len(symbols), 3) or forces.shape != pos.shape:
                        raise ValueError('missing/malformed geometry or forces')
                    if not all(np.isfinite(x).all() for x in (cell, pos, forces, energy)) or abs(np.linalg.det(cell)) < 1e-8:
                        raise ValueError('nonfinite labels or singular cell')
                    stress = None
                    if want_stress:
                        stress = array("varray[@name='stress']/v")
                        if stress.shape != (3, 3) or not np.isfinite(stress).all():
                            raise ValueError('stress label requested but missing/malformed in XML')
                    if reason is None:
                        at = Atoms(symbols, scaled_positions=pos, cell=cell, pbc=True)
                        at.info['REF_energy'] = energy
                        at.set_array('REF_forces', forces)
                        if stress is not None:
                            at.info['REF_stress'] = vasp_kbar_to_ase_stress(stress)
                        frames.append((index, at))
                except (ValueError, AttributeError, TypeError) as exc:
                    reason = str(exc)
                if reason:
                    rejected.append({'frame': index, 'reason': reason})
                node.clear()
    return frames, rejected


def _fixed_cell(incar):
    """True when VASP keeps the cell fixed (ISIF < 3; MD default ISIF = 0, otherwise 2)."""
    ibrion = int(float(incar.get('IBRION', -1)))
    isif = int(float(incar.get('ISIF', 0 if ibrion == 0 else 2)))
    return isif < 3


def export_dataset(packages, out, split_plan=None, stride=1, elements=None):
    """Export finished neutral reaction jobs; split connected reference bundles.

    split_plan is {"job_splits": {"<manifest SHA16>/<job>": "train|valid|test"}}.
    Omitted components go to train. Connected jobs may never cross partitions.
    elements, if given, is the allowed element set; jobs containing other elements are rejected.
    """
    if stride < 1:
        raise ValueError('stride must be positive')
    allowed = None if elements is None else set(elements)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory must be new or empty')
    plan = split_plan or {'job_splits': {}}
    assignments = plan.get('job_splits', {})
    records, rejected, frames, runs, parent = {}, [], [], [], {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(b)] = find(a)
    for package in packages:
        root = Path(package).resolve()
        mpath = root / 'manifest.json'
        manifest = json.loads(mpath.read_text())
        prefix = sha256(mpath)[:16]
        per_job_cells = manifest.get('cell_policy') == 'per_job'
        if not per_job_cells and 'common_cell_A' not in manifest:
            raise ValueError('manifest needs common_cell_A or cell_policy "per_job"')
        selected = set()
        for reaction in manifest.get('reactions', []):
            sides = reaction['equation'].split(' -> ')
            if len(sides) != 2:
                raise ValueError('reaction must use an explicit lhs -> rhs equation')
            inventories = []
            for side in sides:
                inventory = Counter()
                for job in side.split(' + '):
                    at = read(root / 'inputs' / job.strip() / 'static' / 'POSCAR', format='vasp')
                    inventory.update(at.get_chemical_symbols())
                inventories.append(inventory)
            if inventories[0] != inventories[1]:
                raise ValueError('reaction atom inventories do not balance')
            names = [n.strip() for side in reaction['equation'].split(' -> ') for n in side.split(' + ')]
            selected.update(names)
            for n in names:
                find(prefix + '/' + n)
                union(prefix + '/' + names[0], prefix + '/' + n)
        for bundle in manifest.get('bundles', []):
            names = [str(n).strip() for n in bundle]
            if not names:
                raise ValueError('empty bundle in manifest')
            selected.update(names)
            for n in names:
                find(prefix + '/' + n)
                union(prefix + '/' + names[0], prefix + '/' + n)
        for name in sorted(selected):
            if name not in manifest['jobs']:
                raise ValueError(f'unknown reaction job {name}')
            key = prefix + '/' + name
            job = manifest['jobs'][name]
            family = str(job.get('family', manifest.get('family', 'default')))
            kind = job.get('kind', 'vasp')
            if kind not in ('vasp', 'vasp_mlff'):
                raise ValueError(f'{name}: unknown job kind {kind!r}')
            want_stress = bool(job.get('stress', False))
            records[key] = {'package': str(root), 'job': name, 'family': family, 'kind': kind, 'stress': want_stress}
            for step in job.get('steps', ('relax', 'static')):
                folder = root / 'outputs' / name / step
                # an owner-accepted relax (accept_relax.sh writes ACCEPTED, never DONE) supplies frames like a finished one
                accepted = step == 'relax' and not (folder / 'DONE').exists() and (folder / 'ACCEPTED').exists()
                if not (folder / 'DONE').exists() and not accepted:
                    rejected.append({'job': key, 'step': step, 'reason': 'missing DONE'})
                    continue
                if kind == 'vasp_mlff':
                    sources = [(folder / 'ML_AB', 'OUTCAR')]
                else:
                    xmls = sorted(folder.glob('vasprun.xml.seg*.gz'))
                    final = folder / 'vasprun.xml.gz'
                    if not final.exists():
                        final = folder / 'vasprun.xml'
                    xmls += [final]
                    sources = [(xml, 'OUTCAR' + s if (s := xml.name.removeprefix('vasprun.xml').removesuffix('.gz')).startswith('.seg') else 'OUTCAR')
                               for xml in xmls]
                for source, oc in sources:
                    run = parse_run(folder, outcar=oc)
                    bad = run.acceptance(require_relax=(kind == 'vasp' and step == 'relax' and oc == 'OUTCAR' and not accepted))
                    if accepted:
                        bad = [b for b in bad if not b.startswith('run did not finish')]
                    if run.cell_charge is None or abs(run.cell_charge) > 1e-6:
                        bad.append('neutral electron count unverified')
                    if int(float(run.incar.get('ISPIN', 1))) != 1 or str(run.incar.get('LSORBIT', '')).strip('.').upper() in ('TRUE', 'T'):
                        bad.append('pilot supports non-SOC, ISPIN=1 neutral states only')
                    if not source.exists():
                        bad.append('ML_AB missing' if kind == 'vasp_mlff' else 'XML trajectory missing')
                    if bad:
                        rejected.append({'job': key, 'step': step, 'source': str(source), 'reason': '; '.join(bad)})
                        continue
                    try:
                        start = read(root / 'inputs' / name / step / 'POSCAR', format='vasp')
                        symbols = start.get_chemical_symbols()
                        if dict(zip(*np.unique(symbols, return_counts=True))) != dict(zip(run.species, run.ions_per_type)):
                            raise ValueError('OUTCAR composition differs from input')
                        if allowed is not None and not set(symbols) <= allowed:
                            raise ValueError(f'elements outside the allowed set: {sorted(set(symbols) - allowed)}')
                        nelm, ediff = int(float(run.incar.get('NELM', 60))), float(run.incar.get('EDIFF', 1e-4))
                        if kind == 'vasp_mlff':
                            from .vasp_mlff import mlff_dft_frames
                            extracted, skipped, mlff_summary = mlff_dft_frames(folder, symbols, outcar=oc, nelm=nelm)
                            records[key].setdefault('mlff', {})[step] = mlff_summary
                            if want_stress:
                                for i, at in extracted:
                                    if 'REF_stress' not in at.info:
                                        raise ValueError('stress label requested but missing in an MLFF DFT step')
                            else:
                                for i, at in extracted:
                                    at.info.pop('REF_stress', None)
                        else:
                            extracted, skipped = xml_frames(source, symbols, nelm, ediff, run.nelect, run.incar,
                                                            want_stress=want_stress)
                        rejected += [dict(job=key, step=step, source=str(source), **r) for r in skipped]
                        selected_frames = extracted[::stride]
                        for i, at in selected_frames:
                            if per_job_cells:
                                if _fixed_cell(run.incar) and not np.allclose(at.cell.array, start.cell.array, atol=1e-5, rtol=0):
                                    raise ValueError('fixed-cell run changed its cell')
                            elif job.get('role') in ('bare_slab', 'complex') and not np.allclose(at.cell.array, manifest['common_cell_A'], atol=1e-5, rtol=0):
                                raise ValueError('frame cell differs from manifest common slab cell')
                        source_hash = sha256(source)
                        for i, at in selected_frames:
                            at.info.update(job_id=key, step=step, source_sha256=source_hash, frame_index=i,
                                           energy_kind='vasp_free_energy', energy_target='dft_free_energy',
                                           cell_charge=0, config_type=job.get('role', 'unspecified'), family=family,
                                           kpoints=run.kpoints or '', relax_owner_accepted=bool(accepted))
                            if 'REF_stress' not in at.info:
                                at.info['config_stress_weight'] = 0.0
                            frames.append(at)
                        if extracted:
                            runs.append((family, run))
                    except (ValueError, ET.ParseError, OSError) as exc:
                        rejected.append({'job': key, 'step': step, 'source': str(source), 'reason': str(exc)})
    unknown = set(assignments) - set(records)
    if unknown:
        raise ValueError(f'split plan contains unknown jobs: {sorted(unknown)}')
    if runs:
        # One Hamiltonian everywhere; one k-mesh within each family (cells of different size may differ across families).
        mismatches = compatible([r for _, r in runs], allow_kpoint_difference=True)
        for fam in sorted({f for f, _ in runs}):
            mismatches += [f'[{fam}] ' + m for m in compatible([r for f, r in runs if f == fam], allow_kpoint_difference=False)
                           if m.startswith('KPOINTS')]
        if mismatches:
            raise ValueError('incompatible training Hamiltonians: ' + '; '.join(mismatches))
    groups = {}
    for key in records:
        groups.setdefault(find(key), []).append(key)
    splits = {}
    for component, keys in groups.items():
        choices = {assignments[k] for k in keys if k in assignments}
        if not choices:
            choices = {'train'}
        if len(choices) != 1 or not choices <= {'train', 'valid', 'test'}:
            raise ValueError('connected reaction/reference bundle crosses data splits')
        for key in keys:
            splits[key] = next(iter(choices))
    partition = {s: [] for s in ('train', 'valid', 'test')}
    seen, deduplicated = {}, []
    for at in frames:
        split = splits[at.info['job_id']]
        digest = geometry_key(at)
        if digest in seen:
            previous_split, previous = seen[digest]
            if split != previous_split:
                raise ValueError('duplicate geometry crosses data splits (shared reference or copied trajectory)')
            # The final static is the authoritative recalculation of the relaxed
            # geometry; do not reject ordinary changes from tighter SCF settings.
            if at.info['job_id'] == previous.info['job_id'] and at.info['step'] == 'static' and previous.info['step'] == 'relax':
                deduplicated.append({'geometry': digest, 'superseded_source': previous.info['source_sha256'],
                                     'kept_source': at.info['source_sha256'], 'reason': 'final static supersedes same-job relaxation endpoint'})
                previous.info.update(at.info)
                previous.set_array('REF_forces', at.arrays['REF_forces'].copy())
                continue
            if not np.isclose(at.info['REF_energy'], previous.info['REF_energy'], atol=1e-5, rtol=0) or not np.allclose(at.arrays['REF_forces'], previous.arrays['REF_forces'], atol=1e-5, rtol=0):
                raise ValueError('duplicate geometry has conflicting energy/force labels')
            continue
        seen[digest] = (split, at)
        at.info['bundle_id'] = find(at.info['job_id'])
        at.info['split'] = split
        partition[split].append(at)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, ats in partition.items():
        path = out / (split + '.extxyz')
        if ats:
            write(path, ats, format='extxyz')
        else:
            path.write_text('')
        files[split] = {'file': path.name, 'sha256': sha256(path), 'frames': len(ats)}
    families = {}
    for split, ats in partition.items():
        for at in ats:
            f = families.setdefault(at.info['family'], {'frames': {}, 'atoms_min': len(at), 'atoms_max': len(at),
                                                         'stress_frames': 0, 'kpoints': []})
            f['frames'][split] = f['frames'].get(split, 0) + 1
            f['atoms_min'], f['atoms_max'] = min(f['atoms_min'], len(at)), max(f['atoms_max'], len(at))
            f['stress_frames'] += int('REF_stress' in at.info)
            if at.info['kpoints'] not in f['kpoints']:
                f['kpoints'].append(at.info['kpoints'])
    n_stress = sum(f['stress_frames'] for f in families.values())
    limitations = ['No independent validation when all references form one connected bundle',
                   'Training data include dispersion in the DFT target; do not add D3 again to the fitted potential '
                   '(or subtract D3(0) first with examples/scripts/subtract_d3.py and add it back at run time)',
                   'Geometry hashes detect rounded exact copies, not all near duplicates; bundle grouping controls path leakage']
    if not n_stress:
        limitations.insert(0, 'No stress labels')
    result = {'schema': 'pvpass_training/2', 'energy_kind': 'vasp_free_energy', 'energy_target': 'dft_free_energy',
              'force_units': 'eV/Angstrom', 'stress_units': 'eV/Angstrom^3, ASE sign, Voigt xx yy zz yz xz xy',
              'allowed_elements': sorted(allowed) if allowed is not None else None,
              'files': files, 'families': families, 'jobs': records, 'job_splits': splits, 'components': groups,
              'rejected': rejected, 'deduplicated': deduplicated,
              'ready_for_pilot': bool(partition['train'] and partition['valid']),
              'limitations': limitations}
    (out / 'dataset.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def prepare_finetune(dataset, checkpoint, out, expected_mace_version, stress_weight=0.0, multihead=False,
                     replay_file=None, required_elements=None):
    """Prepare an explicit-checkpoint, no-auto-download MACE pilot configuration.

    stress_weight > 0 trains on REF_stress (MACE loss "stress"); frames without REF_stress carry weight 0.
    multihead=True writes a multihead replay fine-tune; it needs an explicit local replay_file, because MACE would
    otherwise download its default replay set. required_elements: every element must occur in the training split.
    """
    dataset, checkpoint, out = Path(dataset).resolve(), Path(checkpoint).resolve(), Path(out)
    report = json.loads((dataset / 'dataset.json').read_text())
    if not report['ready_for_pilot']:
        raise ValueError('need nonempty train and independent validation bundles; random frame splitting is forbidden')
    if not checkpoint.is_file():
        raise ValueError('provide a local foundation checkpoint; no automatic download')
    if not expected_mace_version or not expected_mace_version.strip():
        raise ValueError('pin the MACE version used on the training machine')
    if stress_weight < 0:
        raise ValueError('stress_weight must be non-negative')
    if multihead and (replay_file is None or not Path(replay_file).is_file()):
        raise ValueError('multihead fine-tuning needs an explicit local replay file; no automatic download')
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory must be new or empty')
    for split in ('train', 'valid', 'test'):
        rec = report['files'][split]
        if sha256(dataset / rec['file']) != rec['sha256']:
            raise ValueError(f'{split} data changed since export')
    train_atoms = read(dataset / 'train.extxyz', index=':')
    train_elements = set().union(*(set(at.numbers) for at in train_atoms))
    targets = {at.info.get('energy_target', 'dft_free_energy') for at in train_atoms}
    for split in ('valid', 'test'):
        if report['files'][split]['frames']:
            for at in read(dataset / report['files'][split]['file'], index=':'):
                if not set(at.numbers) <= train_elements:
                    raise ValueError(f'{split} contains elements absent from the training partition')
                targets.add(at.info.get('energy_target', 'dft_free_energy'))
    if len(targets) != 1:
        raise ValueError(f'mixed energy targets across frames: {sorted(targets)}')
    if required_elements is not None:
        from ase.data import atomic_numbers
        missing = sorted(e for e in required_elements if atomic_numbers[e] not in train_elements)
        if missing:
            raise ValueError(f'training partition lacks required elements: {missing}')
    n_stress = sum('REF_stress' in at.info for at in train_atoms)
    if stress_weight > 0 and not n_stress:
        raise ValueError('stress_weight > 0 but no training frame carries REF_stress')
    # Conservative initial settings; a pilot, not tuned production parameters.
    config = {'name': 'pvpass_neutral_pilot', 'foundation_model': str(checkpoint), 'multiheads_finetuning': bool(multihead),
              'train_file': str(dataset / 'train.extxyz'), 'valid_file': str(dataset / 'valid.extxyz'),
              'energy_key': 'REF_energy', 'forces_key': 'REF_forces', 'E0s': 'estimated',
              'energy_weight': 10.0, 'forces_weight': 10.0, 'lr': 0.0001, 'batch_size': 1,
              'max_num_epochs': 20, 'seed': 17, 'device': 'cuda', 'default_dtype': 'float64',
              'ema': True, 'ema_decay': 0.99, 'amsgrad': True, 'clip_grad': 1.0,
              'model_dir': 'models', 'checkpoints_dir': 'checkpoints', 'results_dir': 'results', 'log_dir': 'logs'}
    if stress_weight > 0:
        config.update({'loss': 'stress', 'stress_key': 'REF_stress', 'stress_weight': float(stress_weight),
                       'compute_stress': True})
    if multihead:
        config.update({'pt_train_file': str(Path(replay_file).resolve())})
    if report['files']['test']['frames']:
        config['test_file'] = str(dataset / 'test.extxyz')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'mace_config.json').write_text(json.dumps(config, indent=2) + '\n')
    provenance = {'mace_version': expected_mace_version, 'checkpoint': str(checkpoint),
                  'checkpoint_sha256': sha256(checkpoint), 'dataset': str(dataset),
                  'dataset_report_sha256': sha256(dataset / 'dataset.json'), 'files': report['files'],
                  'energy_target': next(iter(targets)), 'stress_weight': float(stress_weight),
                  'train_frames_with_stress': int(n_stress), 'multihead': bool(multihead),
                  'replay_sha256': sha256(replay_file) if multihead else None,
                  'config_sha256': sha256(out / 'mace_config.json')}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance
