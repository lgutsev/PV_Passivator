"""Export returned label packages without launching any calculations."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from pvpassivator.training_data import export_dataset

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('packages', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--split-plan')
    p.add_argument('--stride', type=int, default=1)
    args = p.parse_args()
    plan = json.loads(Path(args.split_plan).read_text()) if args.split_plan else None
    try:
        result = export_dataset(args.packages, args.out, plan, args.stride)
        print(json.dumps({k: result[k] for k in ('files', 'ready_for_pilot')}, indent=2))
        print(f"{len(result['rejected'])} rejected frames/steps; see dataset.json")
        sys.exit(0 if result['files']['train']['frames'] else 1)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f'{exc}\n')
