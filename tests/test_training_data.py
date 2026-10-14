import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from pvpassivator.training_data import export_dataset, geometry_key, prepare_finetune, sha256, xml_frames


def xml(positions=(0.0, 0.1), scf_bad=False, charge=False, count=1):
    calculations = []
    for x in positions:
        scf = ''.join(f'<scstep><energy><i name="e_fr_energy">{e}</i><i name="e_0_energy">{e}</i></energy></scstep>'
                      for e in (-1.000001, -1.0 if not scf_bad else -0.1))
        calculations.append(f'<calculation>{scf}<energy><i name="e_fr_energy">{-1+x}</i></energy>'
                            '<structure><crystal><varray name="basis"><v>10 0 0</v><v>0 10 0</v><v>0 0 10</v></varray></crystal>'
                            '<varray name="positions">'+''.join(f'<v>{x} {0.2*k} 0</v>' for k in range(count))+'</varray></structure>'
                            '<varray name="forces">'+'<v>0.2 0.3 0.4</v>'*count+'</varray></calculation>')
    return ('<modeling><incar><i name="ENCUT">520</i><i name="ISPIN">1</i></incar>'
            f'<parameters><i name="NELECT">{3 if charge else 2*count}</i><i name="NELM">60</i><i name="EDIFF">0.0001</i></parameters>'
            '<atominfo><array name="atoms"><set>'+'<rc><c>He</c></rc>'*count+'</set></array></atominfo>'
            + ''.join(calculations) + '</modeling>')


def package(root, offset=0.0):
    root.mkdir()
    jobs = {'slab': {'role': 'bare_slab'}, 'ref': {'role': 'reference'}, 'complex': {'role': 'complex'}}
    manifest = {'jobs': jobs, 'common_cell_A': np.eye(3).tolist(),
                'reactions': [{'equation': 'slab + ref -> complex'}]}
    manifest['common_cell_A'] = (np.eye(3)*10).tolist()
    # Unique package identity while retaining an independent configuration bundle.
    manifest['source'] = offset
    (root/'manifest.json').write_text(json.dumps(manifest))
    for j, job in enumerate(jobs):
        count=2 if job=='complex' else 1
        for step in ('relax', 'static'):
            inp = root/'inputs'/job/step
            out = root/'outputs'/job/step
            inp.mkdir(parents=True); out.mkdir(parents=True)
            (inp/'POSCAR').write_text(f'He\n1\n10 0 0\n0 10 0\n0 0 10\nHe\n{count}\nDirect\n'+'0 0 0\n'*count)
            (out/'INCAR').write_text('ENCUT=520\nISPIN=1\nNSW='+('20' if step=='relax' else '0')+'\nIBRION=2\n')
            (out/'KPOINTS').write_text('k\n0\nGamma\n1 1 1\n0 0 0\n')
            (out/'DONE').write_text('done')
            (out/'OUTCAR').write_text(f'TITEL = PAW_PBE He 01Jan2000\n POMASS = 4.0; ZVAL = 2.0\n ions per type = {count}\n NELECT = {2*count}.0\n energy  without entropy= -1.0 energy(sigma->0) = -1.0\n reached required accuracy\n General timing and accounting\n')
            with gzip.open(out/'vasprun.xml.gz', 'wt') as f:
                f.write(xml((offset+j*0.03, offset+j*0.03+0.01) if step=='relax' else (offset+j*0.03+0.01,),count=count))
    return root


def test_xml_force_energy_pair_and_scf_rejection(tmp_path):
    p=tmp_path/'vasprun.xml'; p.write_text(xml())
    frames, rejected = xml_frames(p, ['He'], 60, 1e-4)
    assert not rejected and len(frames)==2
    assert frames[1][1].info['REF_energy']==pytest.approx(-0.9)
    assert frames[1][1].positions[0,0]==pytest.approx(1.0)
    assert np.allclose(frames[1][1].arrays['REF_forces'], [[0.2,0.3,0.4]])
    p.write_text(xml(scf_bad=True))
    assert len(xml_frames(p,['He'],60,1e-4)[1])==2


def test_truncated_xml_and_metadata_rejected(tmp_path):
    p=tmp_path/'x.xml'; p.write_text(xml()[:-12])
    import xml.etree.ElementTree as ET
    with pytest.raises(ET.ParseError): xml_frames(p,['He'],60,1e-4)
    p.write_text(xml(charge=True))
    with pytest.raises(ValueError,match='electron count'): xml_frames(p,['He'],60,1e-4,2)
    p.write_text(xml())
    with pytest.raises(ValueError,match='atom order'): xml_frames(p,['H'],60,1e-4)
    with pytest.raises(ValueError,match='ENCUT'): xml_frames(p,['He'],60,1e-4,2,{'ENCUT':'400'})


def test_dataset_roundtrip_dedup_and_whole_bundle_split(tmp_path):
    p=package(tmp_path/'p')
    result=export_dataset([p], tmp_path/'dataset')
    assert result['files']['train']['frames']==6
    assert not result['ready_for_pilot']
    ats=read(tmp_path/'dataset/train.extxyz',index=':')
    assert all('REF_forces' in a.arrays and 'REF_energy' in a.info for a in ats)
    prefix=sha256(p/'manifest.json')[:16]
    with pytest.raises(ValueError,match='crosses data splits'):
        export_dataset([p],tmp_path/'bad',{'job_splits':{prefix+'/slab':'train',prefix+'/complex':'valid'}})
    with pytest.raises(ValueError,match='independent validation'):
        prepare_finetune(tmp_path/'dataset',tmp_path/'no_model',tmp_path/'run','0.3.13')


def test_independent_packages_prepare_and_hash_guards(tmp_path):
    p=package(tmp_path/'p',0); q=package(tmp_path/'q',0.4)
    prefix=sha256(q/'manifest.json')[:16]
    result=export_dataset([p,q],tmp_path/'dataset',{'job_splits':{prefix+'/slab':'valid'}})
    assert result['ready_for_pilot']
    model=tmp_path/'foundation.model'; model.write_bytes(b'test checkpoint')
    prepare_finetune(tmp_path/'dataset',model,tmp_path/'run','0.3.13')
    config=json.loads((tmp_path/'run/mace_config.json').read_text())
    assert config['valid_file'].endswith('valid.extxyz') and 'valid_fraction' not in config
    assert config['energy_key']=='REF_energy'
    with (tmp_path/'dataset/train.extxyz').open('a') as f: f.write('\n')
    with pytest.raises(ValueError,match='changed'):
        prepare_finetune(tmp_path/'dataset',model,tmp_path/'run2','0.3.13')


def test_copied_geometry_cannot_cross_splits(tmp_path):
    p=package(tmp_path/'p'); q=package(tmp_path/'q')
    m=json.loads((q/'manifest.json').read_text()); m['identity']='copy'
    (q/'manifest.json').write_text(json.dumps(m))
    prefix=sha256(q/'manifest.json')[:16]
    with pytest.raises(ValueError,match='duplicate geometry crosses'):
        export_dataset([p,q],tmp_path/'dataset',{'job_splits':{prefix+'/slab':'valid'}})


def test_charged_unfinished_missing_force_records_excluded(tmp_path):
    p=package(tmp_path/'p')
    for step in ('relax','static'):
        oc=p/'outputs/slab'/step/'OUTCAR'
        oc.write_text(oc.read_text().replace('NELECT = 2.0','NELECT = 1.0'))
        (p/'outputs/ref'/step/'DONE').unlink()
        gz=p/'outputs/complex'/step/'vasprun.xml.gz'
        with gzip.open(gz,'wt') as f: f.write(xml().replace('<varray name="forces"><v>0.2 0.3 0.4</v></varray>',''))
    result=export_dataset([p],tmp_path/'dataset')
    assert result['files']['train']['frames']==0 and result['rejected']


def test_staging_preserves_restart_xml():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'examples/scripts/stage_label_set.py'
    spec=importlib.util.spec_from_file_location('stage',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert 'vasprun.xml.seg$seg.gz' in mod.LIB


def test_restarted_xml_frames_are_included(tmp_path):
    p=package(tmp_path/'p')
    folder=p/'outputs/slab/relax'
    (folder/'OUTCAR.seg1').write_text((folder/'OUTCAR').read_text())
    with gzip.open(folder/'vasprun.xml.seg1.gz','wt') as fh: fh.write(xml((0.2,)))
    result=export_dataset([p],tmp_path/'dataset')
    assert result['files']['train']['frames']==7


def test_incompatible_profiles_stop_export(tmp_path):
    p=package(tmp_path/'p')
    for step in ('relax','static'):
        inc=p/'outputs/ref'/step/'INCAR'
        inc.write_text(inc.read_text().replace('520','400'))
        with gzip.open(p/'outputs/ref'/step/'vasprun.xml.gz','wt') as fh:
            fh.write(xml((0.03,0.04)).replace('520','400'))
    with pytest.raises(ValueError,match='incompatible training Hamiltonians'):
        export_dataset([p],tmp_path/'dataset')


def test_late_bad_cell_does_not_export_partial_source(tmp_path):
    p=package(tmp_path/'p')
    for step in ('relax','static'):
        body=xml((0.01,0.02)).replace('<v>10 0 0</v>', '<v>11 0 0</v>', 1)
        with gzip.open(p/'outputs/slab'/step/'vasprun.xml.gz','wt') as fh: fh.write(body)
    result=export_dataset([p],tmp_path/'dataset')
    assert result['files']['train']['frames']==4


def test_prediction_scoring_checks_identity_and_geometry(tmp_path):
    import importlib.util
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    script=Path(__file__).resolve().parents[1]/'examples/scripts/evaluate_mlip_predictions.py'
    spec=importlib.util.spec_from_file_location('evaluate',script)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    p=package(tmp_path/'p');export_dataset([p],tmp_path/'dataset')
    ref=tmp_path/'dataset/train.extxyz'
    atoms=read(ref,index=':')
    for at in atoms:
        at.calc=SinglePointCalculator(at,energy=at.info['REF_energy']+0.5,forces=at.arrays['REF_forces']+0.1)
    pred=tmp_path/'pred.extxyz';write(pred,atoms,format='extxyz')
    result=mod.score(ref,pred)
    assert result['energy_total_eV']['mae']==pytest.approx(0.5)
    assert result['force_components_eV_A']['rmse']==pytest.approx(0.1)
    atoms[0].positions[0,0]+=0.1
    atoms[0].calc=SinglePointCalculator(atoms[0],energy=0,forces=np.zeros((len(atoms[0]),3)))
    write(pred,atoms,format='extxyz')
    with pytest.raises(ValueError,match='geometry'):mod.score(ref,pred)


def test_unbalanced_reaction_rejected(tmp_path):
    p=package(tmp_path/'p')
    m=json.loads((p/'manifest.json').read_text());m['reactions'][0]['equation']='slab -> complex'
    (p/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='do not balance'):export_dataset([p],tmp_path/'dataset')
