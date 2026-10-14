"""Score precomputed energy/force predictions against a held-out export.

Predicted extxyz must preserve source_sha256, frame_index, job_id, and step,
and contain predicted energy/forces as ASE calculator results. No fitting or
model execution occurs here. Run separately for foundation and fine-tuned models.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from ase.io import read


def identity(at):
    return tuple(at.info[k] for k in ('job_id','step','source_sha256','frame_index'))


def score(reference, predictions):
    ref=read(reference,index=':'); pred=read(predictions,index=':')
    index={identity(a):a for a in pred}
    if len(index)!=len(pred) or len(ref)!=len(pred) or set(index)!={identity(a) for a in ref}:
        raise ValueError('predictions must contain exactly one matching record for every reference frame')
    rows=[]; energy=[]; forces=[]
    for at in ref:
        p=index[identity(at)]
        if not np.array_equal(p.numbers,at.numbers) or not np.allclose(p.positions,at.positions,atol=1e-7,rtol=0) or not np.allclose(p.cell,at.cell,atol=1e-7,rtol=0):
            raise ValueError('prediction geometry or atom order changed')
        e=p.get_potential_energy()-float(at.info['REF_energy'])
        f=p.get_forces(apply_constraint=False)-at.arrays['REF_forces']
        if not np.isfinite(e) or not np.isfinite(f).all():
            raise ValueError('nonfinite prediction')
        energy.append(e); forces.extend(f.ravel())
        rows.append({'job_id':at.info['job_id'],'bundle_id':at.info['bundle_id'],
                     'energy_error_eV':float(e),'energy_error_eV_per_atom':float(e/len(at)),
                     'force_component_rmse_eV_A':float(np.sqrt(np.mean(f*f)))})
    def stats(v):
        v=np.asarray(v)
        return {'n':len(v),'mae':float(np.abs(v).mean()),'rmse':float(np.sqrt((v*v).mean()))}
    if not rows:
        raise ValueError('empty held-out set')
    return {'energy_total_eV':stats(energy),'force_components_eV_A':stats(forces),
            'n_bundles':len({r['bundle_id'] for r in rows}),'rows':rows,
            'limitations':['No adsorption or ranking metric inferred from per-configuration errors',
                           'No test-set offset fitting; foundation predictions need the declared matching dispersion convention']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference');p.add_argument('predictions');p.add_argument('--out',required=True)
    a=p.parse_args()
    try:
        result=score(a.reference,a.predictions)
        dest=Path(a.out)
        if dest.exists(): raise ValueError('refusing to overwrite an existing evaluation')
        dest.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    except (ValueError,KeyError,OSError) as exc: p.exit(2,f'{exc}\n')
