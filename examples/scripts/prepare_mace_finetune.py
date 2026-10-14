"""Prepare a pinned-checkpoint MACE pilot; does not run training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from pvpassivator.training_data import prepare_finetune

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mace-version', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    try:
        print(json.dumps(prepare_finetune(a.dataset, a.checkpoint, a.out, a.mace_version), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f'{exc}\n')
