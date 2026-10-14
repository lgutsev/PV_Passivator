"""Verify prepared data/checkpoint/environment; launch only with --execute."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from pvpassivator.training_data import sha256


def preflight(folder):
    folder = Path(folder).resolve()
    prov = json.loads((folder / 'provenance.json').read_text())
    config = json.loads((folder / 'mace_config.json').read_text())
    if sha256(folder / 'mace_config.json') != prov['config_sha256']:
        raise ValueError('configuration changed after preparation')
    if sha256(prov['checkpoint']) != prov['checkpoint_sha256']:
        raise ValueError('foundation checkpoint changed')
    if sha256(Path(prov['dataset']) / 'dataset.json') != prov['dataset_report_sha256']:
        raise ValueError('dataset report changed')
    for split, rec in prov['files'].items():
        if sha256(Path(prov['dataset']) / rec['file']) != rec['sha256']:
            raise ValueError(f'{split} dataset changed')
    version = importlib.metadata.version('mace-torch')
    if version != prov['mace_version']:
        raise ValueError(f'MACE version {version} differs from pinned {prov["mace_version"]}')
    import torch
    if not torch.cuda.is_available():
        raise ValueError('pilot configured for CUDA; run on the selected GPU environment')
    return folder, config


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prepared_run')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    try:
        folder, config = preflight(a.prepared_run)
        cmd = [sys.executable, '-m', 'mace.cli.run_train']
        for k, v in config.items():
            if v is True:
                cmd.append('--' + k)
            elif v is False:
                cmd.append('--' + k + '=False')
            else:
                cmd.append('--' + k + '=' + str(v))
        print(json.dumps({'status': 'preflight passed', 'command': cmd, 'execute': a.execute}, indent=2))
        if a.execute:
            sys.exit(subprocess.run(cmd, cwd=folder).returncode)
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        p.exit(2, f'{exc}\n')
