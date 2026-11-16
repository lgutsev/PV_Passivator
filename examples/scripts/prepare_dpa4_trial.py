"""Stage a cluster trial: fine-tune a pretrained DPA-4 on the audit frames with the same folds as the MACE rounds.

    python examples/scripts/prepare_dpa4_trial.py --cv data/cache/adsearch/ft110/cv_v3
           --dataset data/cache/adsearch/ft110/dataset_v3 --out STAGING/<name> [--variant Neo] [--steps 4000]

Nothing is submitted and nothing is downloaded here.  The package carries:

  data/fold{k}/{train,valid,test}/<formula>/   DeePMD npy systems, one per composition, written from the fold's
  data/final/{train,valid}/<formula>/          extxyz files (cv_audit_finetune.py), so the splits are byte-for-byte
                                               the ones the MACE folds used.  Energy = DFT - D3 (REF_energy_unaligned),
                                               forces = DFT - D3 forces.  type_map is the full periodic table, as the
                                               released DPA-4 checkpoints require.  names.txt and mol_id.npy ride along
                                               for scoring.
  fetch_pretrained.sh      run once on a login node (needs internet): the checkpoint and its released config.
  make_input.py            builds input.json from the RELEASED config: model section untouched, data/lr/steps replaced.
  eval_dpa4.py             predictions (energy, forces) for every system of a fold, as npz.
  run_dpa4_smoke.slurm     environment report, checkpoint load, zero-shot predictions on all frames, 100 fine-tune
                           steps to time one step.  Everything that is unknown about the cluster module is tested here.
  run_dpa4_folds.slurm     array 0-5: five folds and the all-frames model.

Scoring happens locally on the returned predictions, with the same quantities as the MACE rounds (scoring script not
written yet; it is needed only once predictions exist).

Local checks (Windows, deepmd 3.2.0, CPU): make_input.py runs on a released-style config, and a fine-tune with a
stand-in descriptor trains on the exported data.  A local DPA-4 (SeZM) run needs 'vesin' or 'nvalchemiops' for its
neighbour list, and loading any checkpoint into DeepPot hung on this desktop, so eval_dpa4.py and the DPA-4 path are
tested only by the cluster smoke job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols
from ase.io import read

TYPE_MAP = chemical_symbols[1:119]          # H .. Og, the released checkpoints' type_map
HF = "https://huggingface.co/deepmodelingcommunity/DPA4-OMat24/resolve/main"
VERSION = "v20260805"


def lf(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def write_systems(frames, root: Path) -> dict:
    """One DeePMD system per composition; returns {formula: [job names]}."""
    groups: dict[str, list] = {}
    for a in frames:
        groups.setdefault(a.get_chemical_formula(), []).append(a)
    out = {}
    for formula, ats in sorted(groups.items()):
        z0 = ats[0].numbers
        for a in ats:
            if not (a.numbers == z0).all():
                raise ValueError(f"{formula}: atom order differs between frames ({a.info['job']})")
        d = root / formula
        (d / "set.000").mkdir(parents=True, exist_ok=True)
        lf(d / "type.raw", "".join(f"{int(z) - 1}\n" for z in z0))
        lf(d / "type_map.raw", "".join(f"{s}\n" for s in TYPE_MAP))
        n = len(ats)
        np.save(d / "set.000" / "coord.npy", np.array([a.positions.reshape(-1) for a in ats], dtype=np.float64))
        np.save(d / "set.000" / "box.npy", np.array([a.cell.array.reshape(-1) for a in ats], dtype=np.float64))
        np.save(d / "set.000" / "energy.npy", np.array([a.info["REF_energy_unaligned"] for a in ats], dtype=np.float64))
        np.save(d / "set.000" / "force.npy", np.array([a.arrays["REF_forces"].reshape(-1) for a in ats], dtype=np.float64))
        np.save(d / "mol_id.npy", np.array([a.arrays["mol_id"] for a in ats], dtype=np.int32))
        lf(d / "names.txt", "".join(f"{a.info['job']}\t{a.info['family']}\t{a.info['group']}\n" for a in ats))
        out[formula] = [a.info["job"] for a in ats]
        assert np.load(d / "set.000" / "coord.npy").shape == (n, 3 * len(z0))
    return out


MAKE_INPUT = r'''"""input.json for one task, from the RELEASED training config of the pretrained checkpoint.

    python make_input.py RELEASED.json DATA_DIR OUT.json [--steps N] [--start-lr X] [--stop-lr Y]

The model section is copied unchanged (the release notes require it, full type_map included).  Replaced: data,
learning-rate values, number of steps, checkpoint/print cadence.  Removed: the pretraining statistics file, EMA,
epoch-based length, logging extras.  Every change is printed.
"""
import json, sys
from pathlib import Path
a = sys.argv[1:]
opt = {a[i]: a[i + 1] for i in range(3, len(a), 2)}
rel, data, out = json.loads(Path(a[0]).read_text()), Path(a[1]).resolve(), Path(a[2])
steps, lr0, lr1 = int(opt.get("--steps", __STEPS__)), float(opt.get("--start-lr", 1e-4)), float(opt.get("--stop-lr", 1e-6))
cfg = {"model": rel["model"]}
for k in ("optimizer", "loss", "learning_rate"):
    if k in rel:
        cfg[k] = json.loads(json.dumps(rel[k]))
lr = cfg.setdefault("learning_rate", {"type": "exp"})
print("learning_rate: start", lr.get("start_lr"), "->", lr0, "; stop", lr.get("stop_lr"), "->", lr1)
lr["start_lr"], lr["stop_lr"] = lr0, lr1
if lr.get("type") == "exp":
    lr["decay_steps"] = max(1, steps // 100)
loss = cfg.setdefault("loss", {"type": "ener"})
for k in ("start_pref_v", "limit_pref_v"):
    if loss.get(k):
        print("loss:", k, loss[k], "-> 0 (no virials in this data)")
        loss[k] = 0
systems = lambda s: sorted(str(p) for p in (data / s).iterdir() if (p / "type.raw").exists())
old = rel.get("training", {})
tr = {"training_data": {"systems": systems("train"), "batch_size": 1},
      "validation_data": {"systems": systems("valid"), "batch_size": 1, "numb_btch": 4},
      "numb_steps": steps, "seed": int(old.get("seed", 42)), "disp_file": "lcurve.out", "disp_freq": 50,
      "save_freq": max(100, steps // 8), "save_ckpt": "model.ckpt", "max_ckpt_keep": 3}
if "gradient_max_norm" in old:
    tr["gradient_max_norm"] = old["gradient_max_norm"]
print("training: dropped", sorted(set(old) - set(tr) - {"training_data", "validation_data"}))
print("training systems:", [Path(s).name for s in tr["training_data"]["systems"]], "valid:", [Path(s).name for s in tr["validation_data"]["systems"]])
cfg["training"] = tr
out.write_text(json.dumps(cfg, indent=1) + "\n")
'''

EVAL = r'''"""Predictions of a DeePMD model for every system under the given folders.

    python eval_dpa4.py MODEL OUT_DIR SPLIT_DIR [SPLIT_DIR ...]

Writes OUT_DIR/<split>/<formula>.npz with energy (n,), force (n, N, 3) and the frame names, plus summary.json with
plain per-system errors.  Uses the native neighbour list when the installed deepmd accepts the option.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from deepmd.infer import DeepPot
model, out = sys.argv[1], Path(sys.argv[2])
try:
    dp = DeepPot(model, nlist_backend="native")
    how = "native nlist"
except TypeError:
    dp = DeepPot(model)
    how = "default nlist"
tmap = dp.get_type_map()
summ = {"model": model, "neighbour_list": how, "systems": {}}
for sd in sys.argv[3:]:
    sd = Path(sd)
    for s in sorted(p for p in sd.iterdir() if (p / "type.raw").exists()):
        own = (s / "type_map.raw").read_text().split()
        types = np.array([tmap.index(own[int(t)]) for t in (s / "type.raw").read_text().split()])
        coord, box = np.load(s / "set.000" / "coord.npy"), np.load(s / "set.000" / "box.npy")
        e0, f0 = np.load(s / "set.000" / "energy.npy"), np.load(s / "set.000" / "force.npy")
        E, F, t0 = [], [], time.time()
        for i in range(len(coord)):                     # one frame at a time: 597 atoms, modest memory
            e, f, _ = dp.eval(coord[i:i + 1], box[i:i + 1], types)
            E.append(float(np.ravel(e)[0])), F.append(np.asarray(f).reshape(-1, 3))
        E, F = np.array(E), np.array(F)
        (out / sd.name).mkdir(parents=True, exist_ok=True)
        names = [ln.split("\t")[0] for ln in (s / "names.txt").read_text().splitlines()]
        np.savez_compressed(out / sd.name / f"{s.name}.npz", energy=E, force=F, names=np.array(names))
        d = F - f0.reshape(F.shape)
        summ["systems"][f"{sd.name}/{s.name}"] = {
            "n_frames": len(E), "n_atoms": int(len(types)), "seconds_per_frame": round((time.time() - t0) / len(E), 3),
            "energy_mean_error_eV": float((E - e0).mean()), "energy_rel_mae_eV": float(np.abs((E - e0) - (E - e0).mean()).mean()),
            "force_rmse_eV_A": float(np.sqrt((d ** 2).mean())), "finite": bool(np.isfinite(E).all() and np.isfinite(F).all())}
        print(sd.name, s.name, summ["systems"][f"{sd.name}/{s.name}"], flush=True)
(out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
'''

FETCH = r'''#!/bin/bash
# Run ONCE on a login node (needs internet), from the package directory, before any sbatch.
# Downloads the pretrained DPA-4 checkpoint and its released training config (licence: CC-BY-NC-4.0, per the model card).
# One master copy in the LONI foundation-model store; pretrained/ holds symlinks to it, so no model binary sits in a
# work folder (desk rule, package 138). A file already in the store is reused, not downloaded again.
set -euo pipefail
V=${DPA4_VARIANT:-__VARIANT__}
B=__HF__
M=${DPA4_STORE:-/ddnB/project/ramu/lgutsev/MLIP_PROJECT_STORAGE/MLIP_Foundational_Models/DPA4-OMat24}
PKG=$PWD
mkdir -p "$M" pretrained
cd "$M"
for f in DPA4-$V-OMat24-__VERSION__.pt DPA4-$V-OMat24-__VERSION__.json; do
  if [ ! -s "$f" ]; then
    curl -fL --retry 3 -o "$f.part" "$B/$f"
    mv "$f.part" "$f"
  fi
done
grep -q '"model"' DPA4-$V-OMat24-__VERSION__.json || { echo "released config has no \"model\" section" >&2; exit 3; }
echo "config has a model section"
sha256sum DPA4-$V-OMat24-__VERSION__.pt DPA4-$V-OMat24-__VERSION__.json > "SHA256SUMS.$V"
cd "$PKG/pretrained"
for f in DPA4-$V-OMat24-__VERSION__.pt DPA4-$V-OMat24-__VERSION__.json; do ln -sfn "$M/$f" "$f"; done
cp "$M/SHA256SUMS.$V" SHA256SUMS
sha256sum -c SHA256SUMS
cat SHA256SUMS
ls -lL
'''

HEAD = r'''#!/bin/bash
#SBATCH --job-name=__JOB__
#SBATCH --account=__ACCOUNT__
#SBATCH --partition=__PARTITION__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=__TIME__
__ARRAY__#SBATCH --output=logs/%x_%A_%a.out
# __PACKAGE__: __PURPOSE__
# Generated by PV_Passivator examples/scripts/prepare_dpa4_trial.py; nothing was submitted by the generator.
# Runtime as the SiN-TiN desk's DPA-4 runs: an Apptainer deepmd module with /ddnB, /project, /work bound.
set -eo pipefail
module purge
module load "${DPA4_MODULE:-__MODULE__}"
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}/ddnB,/project,/work"
export SINGULARITY_BIND="$APPTAINER_BIND"
for v in PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID \
         DP_INTRA_OP_PARALLELISM_THREADS=8 DP_INTER_OP_PARALLELISM_THREADS=1; do
  export "$v" "APPTAINERENV_$v" "SINGULARITYENV_$v"
done
set -u
cd "${SLURM_SUBMIT_DIR:?submit from the package directory}"
V=${DPA4_VARIANT:-__VARIANT__}
CKPT=$PWD/pretrained/DPA4-$V-OMat24-__VERSION__.pt
REL=$PWD/pretrained/DPA4-$V-OMat24-__VERSION__.json
[ -s "$CKPT" ] && [ -s "$REL" ] || { echo "run fetch_pretrained.sh on a login node first" >&2; exit 2; }
( cd pretrained && sha256sum -c --quiet SHA256SUMS )
sha256sum -c --quiet SHA256SUMS
latest() { ls model.ckpt-*.pt 2>/dev/null | sed -n 's/^model\.ckpt-\([0-9]*\)\.pt$/\1 &/p' | sort -n | tail -n 1 | cut -d' ' -f2; }
'''

SMOKE = HEAD + r'''W=outputs/smoke
[ ! -e "$W/DONE" ] || { echo "smoke already done"; exit 0; }
mkdir -p "$W" && cd "$W"
{ echo "host $(hostname)  $(date -u +%FT%TZ)  job ${SLURM_JOB_ID:-?}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  python -c "import torch, deepmd; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'deepmd', deepmd.__version__)"; } | tee env.txt
# 1. does this deepmd load the released checkpoint at all?  zero-shot predictions on all 40 frames (also the baseline)
python ../../eval_dpa4.py "$CKPT" zeroshot ../../data/final/train ../../data/final/valid > zeroshot.log 2>&1 \
  && echo "zero-shot evaluation OK" | tee -a env.txt || { echo "zero-shot evaluation FAILED (see zeroshot.log)" | tee -a env.txt; tail -n 30 zeroshot.log; }
# 2. does a fine-tune start, and how long is a step on a 597-atom frame?
python ../../make_input.py "$REL" ../../data/fold0 input.json --steps 100 | tee make_input.log
set +e
/usr/bin/time -v dp --pt train input.json --finetune "$CKPT" > train.log 2>&1; rc=$?
set -e
echo "fine-tune 100 steps exit $rc" | tee -a env.txt
grep -i "average training time\|wall time\|error\|Traceback" train.log | tail -n 12 | tee timing.txt || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a env.txt
C=$(latest); [ -z "$C" ] || { python ../../eval_dpa4.py "$C" after100 ../../data/fold0/test > after100.log 2>&1 && echo "evaluation of a fine-tuned checkpoint OK" | tee -a env.txt || echo "evaluation of a fine-tuned checkpoint FAILED" | tee -a env.txt; rm -f model.ckpt-*.pt model.ckpt.pt; }
[ "$rc" -eq 0 ] && date -u +%FT%TZ > DONE || exit 5
'''

FOLDS = HEAD + r'''T=${SLURM_ARRAY_TASK_ID:?use --array}
case "$T" in 0|1|2|3|4) NAME=fold$T ;; 5) NAME=final ;; *) echo "task must be 0-5" >&2; exit 2 ;; esac
W=outputs/$NAME
[ ! -e "$W/DONE" ] || { echo "$NAME already done"; exit 0; }
mkdir -p "$W" && cd "$W"
echo "host $(hostname)  $(date -u +%FT%TZ)  task $T ($NAME)  job ${SLURM_ARRAY_JOB_ID:-?}" | tee -a env.txt
python -c "import torch, deepmd; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'deepmd', deepmd.__version__)" | tee -a env.txt
if [ ! -e TRAINED ]; then
  R=$(latest)
  if [ -n "$R" ] && [ -s input.json ]; then
    python ../../make_input.py "$REL" ../../data/$NAME input.regen.json --steps "${DPA4_STEPS:-__STEPS__}" > /dev/null
    cmp -s input.json input.regen.json || { echo "input.json differs from make_input.py output; not resuming" >&2; exit 4; }
    rm -f input.regen.json
    [ ! -e train.log ] || mv train.log "train_$(date -u +%Y%m%dT%H%M%SZ).log"
    echo "resume from $R" | tee -a env.txt
    dp --pt train input.json --restart "$R" > train.log 2>&1
  else
    python ../../make_input.py "$REL" ../../data/$NAME input.json --steps "${DPA4_STEPS:-__STEPS__}" | tee make_input.log
    echo "fine-tune from $(basename "$CKPT")" | tee -a env.txt
    dp --pt train input.json --finetune "$CKPT" > train.log 2>&1
  fi
  date -u +%FT%TZ > TRAINED
fi
C=$(latest); [ -n "$C" ] || { echo "no checkpoint" >&2; exit 3; }
SPLITS="../../data/$NAME/train ../../data/$NAME/valid"; [ "$NAME" = final ] || SPLITS="$SPLITS ../../data/$NAME/test"
python ../../eval_dpa4.py "$C" pred $SPLITS > eval.log 2>&1
cp "$C" model_final.pt; sha256sum model_final.pt | tee -a env.txt
ls model.ckpt-*.pt | grep -v "^$C$" | xargs -r rm -f
grep -i "average training time" train.log | tail -n 1 | tee -a env.txt || true
date -u +%FT%TZ > DONE
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", required=True), ap.add_argument("--dataset", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="Neo"), ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--account", default="loni_perovsk27"), ap.add_argument("--partition", default="gpu2")
    ap.add_argument("--module", default="deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    cv, ds, out = Path(a.cv), Path(a.dataset), Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise SystemExit(f"{out} is not empty")
    pkg = out.name
    layout = {}
    folds = sorted(p for p in cv.iterdir() if p.name.startswith("fold") and (p / "data").is_dir())
    for f in folds:
        for sp in ("train", "valid", "test"):
            layout[f"{f.name}/{sp}"] = write_systems(read(f / "data" / f"{sp}.extxyz", index=":"), out / "data" / f.name / sp)
    for sp in ("train", "valid"):
        layout[f"final/{sp}"] = write_systems(read(ds / f"{sp}.extxyz", index=":"), out / "data" / "final" / sp)
    if (ds / "test.extxyz").exists() and (ds / "test.extxyz").stat().st_size:
        raise SystemExit("the all-frames dataset has a test split; expected train + valid only")

    def fill(t, **kw):
        base = {"VARIANT": a.variant, "VERSION": VERSION, "HF": HF, "STEPS": str(a.steps), "ACCOUNT": a.account,
                "PARTITION": a.partition, "MODULE": a.module, "PACKAGE": pkg, **kw}
        for k, v in base.items():
            t = t.replace(f"__{k}__", v)
        return t
    lf(out / "make_input.py", fill(MAKE_INPUT))
    lf(out / "eval_dpa4.py", EVAL)
    lf(out / "fetch_pretrained.sh", fill(FETCH))
    lf(out / "run_dpa4_smoke.slurm", fill(SMOKE, JOB="dpa4-smoke", TIME="01:00:00", ARRAY="",
                                           PURPOSE="load the released checkpoint, zero-shot predictions, time 100 fine-tune steps"
                                           ).replace("%x_%A_%a.out", "%x_%j.out"))
    lf(out / "run_dpa4_folds.slurm", fill(FOLDS, JOB="dpa4-folds", TIME="08:00:00", ARRAY="#SBATCH --array=0-5%3\n",
                                           PURPOSE="fine-tune five folds (0-4) and the all-frames model (5), then predict"))
    lf(out / "export.list", "outputs/*/env.txt\noutputs/*/*.log\noutputs/*/timing.txt\noutputs/*/input.json\noutputs/*/lcurve.out\n"
                            "outputs/*/pred/*\noutputs/*/zeroshot/*\noutputs/*/after100/*\noutputs/*/model_final.pt\noutputs/*/DONE\nlogs/*\npretrained/SHA256SUMS\n")
    lf(out / "logs" / ".keep", "")
    lf(out / ".gitattributes", "# keep data byte-identical to SHA256SUMS\n*.npy binary\n*.raw -text\n*.txt -text\n")
    n_frames = {k: sum(len(v) for v in d.values()) for k, d in layout.items()}
    lf(out / "README.md", f"""# {pkg}

Trial: fine-tune the pretrained DPA4-{a.variant}-OMat24-{VERSION} on 40 r2SCAN single points of melamine on a FAPbI3
slab (597 atoms at most), with the same five family-held-out folds as the MACE fine-tune it is compared with.
Prepared by PV_Passivator (`examples/scripts/prepare_dpa4_trial.py`). Nothing was submitted or downloaded.

Target: DFT energy and forces with D3 subtracted (the MACE rounds use the same target; D3 is added back at use).
The pretrained model is PBE-level (OMat24); the fine-tune has to move it to r2SCAN on these frames.

## Commands, in order (from this directory)

1. `bash fetch_pretrained.sh` on a login node (internet): about 5 MB from Hugging Face, writes `pretrained/SHA256SUMS`.
2. `sbatch run_dpa4_smoke.slurm` (1 GPU, <= 1 h). Read `outputs/smoke/env.txt` and `timing.txt` BEFORE step 3:
   it says whether this deepmd build loads the checkpoint, whether a fine-tune starts, and what one step costs.
3. `sbatch run_dpa4_folds.slurm` (array 0-5%3, 1 GPU each, <= 8 h each; {a.steps} steps, batch 1).

Every task writes `outputs/<name>/DONE` only at the end; finished tasks are skipped and an interrupted one resumes
from its last checkpoint. Override with `DPA4_VARIANT`, `DPA4_STEPS`, `DPA4_MODULE` (export before sbatch).

## Known unknowns, all tested by the smoke job
- whether module `{a.module}` (a beta) reads a checkpoint and config released for deepmd 3.2.0;
- whether keys of the released config (optimizer, compile, mixed precision) are accepted by it;
- how the fine-tune sets the per-element energy reference from 4 compositions of 5 elements (relative energies
  within one composition do not depend on it; absolute ones do);
- step time and memory for a 597-atom frame.

Frames per split: {json.dumps(n_frames)}
""")
    sha = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(out.rglob("*")) if p.is_file() and p.name not in ("package.json", "SHA256SUMS")}
    lf(out / "SHA256SUMS", "".join(f"{h}  {f}\n" for f, h in sha.items() if not f.startswith("logs/") and f != "README.md"))
    lf(out / "package.json", json.dumps({"package": pkg, "generator": "PV_Passivator prepare_dpa4_trial.py", "submitted": False,
                                          "pretrained": {"name": f"DPA4-{a.variant}-OMat24-{VERSION}", "source": HF,
                                                         "licence": "CC-BY-NC-4.0 (model card)", "downloaded": False},
                                          "steps": a.steps, "layout": layout, "files_sha256": sha}, indent=1) + "\n")
    print(f"staged {pkg}: {len(folds)} folds + final, {len(sha)} files -> {out}")
    print(json.dumps(n_frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
