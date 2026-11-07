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
    p.add_argument('--stress-weight', type=float, default=0.0,
                   help='> 0 trains on REF_stress (MACE loss "stress"); frames without stress get weight 0')
    p.add_argument('--multihead', action='store_true', help='multihead replay fine-tune (needs --replay-file)')
    p.add_argument('--replay-file', help='local replay extxyz for the pretrained head (no automatic download)')
    p.add_argument('--required-elements', help='comma list that must all occur in the training split')
    a = p.parse_args()
    req = [e.strip() for e in a.required_elements.split(',')] if a.required_elements else None
    try:
        print(json.dumps(prepare_finetune(a.dataset, a.checkpoint, a.out, a.mace_version, stress_weight=a.stress_weight,
                                          multihead=a.multihead, replay_file=a.replay_file, required_elements=req),
                         indent=2))
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f'{exc}\n')
