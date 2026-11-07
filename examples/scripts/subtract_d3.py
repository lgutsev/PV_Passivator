"""Write a copy of an exported dataset with the D3(0) part removed from every label (E - D3, F - D3, sigma - D3).

    python examples/scripts/subtract_d3.py DATASET --out DATASET_minus_d3 --backend s-dftd3 [--exe PATH]
    python examples/scripts/subtract_d3.py DATASET --out DATASET_minus_d3 --backend torch-dftd [--device cuda]

The source labels must be PBE + D3(0) (VASP IVDW = 11 with default parameters; pass --incar to check one INCAR).
Add the SAME backend back at run time. Never add D3 to a model trained on the full DFT target.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from ase.io import read, write
from pvpassivator.dispersion import SDFTD3Executable, TorchDFTD3, subtract_d3, TARGET_MINUS_D3
from pvpassivator.training_data import sha256
from pvpassivator.vasp_io import parse_incar

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('dataset')
    p.add_argument('--out', required=True)
    p.add_argument('--backend', choices=['s-dftd3', 'torch-dftd'], required=True)
    p.add_argument('--exe')
    p.add_argument('--device', default='cpu')
    p.add_argument('--incar', help='INCAR of the label runs, checked for IVDW = 11 defaults')
    a = p.parse_args()
    src, out = Path(a.dataset), Path(a.out)
    try:
        if out.exists() and any(out.iterdir()):
            raise ValueError('output directory must be new or empty')
        report = json.loads((src / 'dataset.json').read_text())
        for split, rec in report['files'].items():
            if sha256(src / rec['file']) != rec['sha256']:
                raise ValueError(f'{split} data changed since export')
        backend = SDFTD3Executable(a.exe) if a.backend == 's-dftd3' else TorchDFTD3(device=a.device)
        incar = parse_incar(a.incar) if a.incar else None
        out.mkdir(parents=True, exist_ok=True)
        description = None
        for split, rec in report['files'].items():
            path = out / rec['file']
            if rec['frames']:
                frames = read(src / rec['file'], index=':')
                description = subtract_d3(frames, backend, incar)
                write(path, frames, format='extxyz')
            else:
                path.write_text('')
            rec.update(sha256=sha256(path), source_sha256=sha256(src / rec['file']))
        report.update(energy_target=TARGET_MINUS_D3, d3_backend=description or backend.describe(),
                      derived_from={'dataset': str(src.resolve()), 'dataset_json_sha256': sha256(src / 'dataset.json')})
        report['limitations'] = [l for l in report['limitations'] if not l.startswith('Training data include dispersion')]
        report['limitations'].append('Target is DFT minus D3(0); add the same D3(0) backend back at run time, once')
        (out / 'dataset.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'files': report['files'], 'd3_backend': report['d3_backend']}, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f'{exc}\n')
