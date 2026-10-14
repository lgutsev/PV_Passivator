# Neutral passivation MLIP training pilot

Implemented 13 October 2026. Preparation does not download models, launch training, submit DFT, or modify existing staged cluster packages.

## What is ready

- `examples/scripts/export_training_data.py`: reads returned label-set packages and exports train/valid/test extended XYZ plus a provenance/rejection report.
- `examples/scripts/prepare_mace_finetune.py`: prepares a first MACE fine-tuning configuration from an explicit local checkpoint and pinned MACE version.
- `examples/scripts/run_mace_pilot.py`: checks data/config/checkpoint hashes, package version and CUDA; defaults to preflight only. `--execute` starts training explicitly.
- `examples/scripts/evaluate_mlip_predictions.py`: scores aligned precomputed predictions on energies and raw force components. Use separately for the foundation and fine-tuned model.
- `stage_label_set.py`: future generated packages preserve compressed XML when a relaxation restarts, alongside the existing OUTCAR segments.

## Data and acceptance

Inputs: package `manifest.json`, `inputs/<job>/<step>/POSCAR`, and returned `outputs/<job>/<step>/{DONE,INCAR,KPOINTS,OUTCAR,vasprun.xml.gz}`. Both complete relaxations and final static calculations can supply labels. Archived OutPackLite records cannot supply missing force arrays.

The pilot supports verified neutral, ISPIN=1, non-SOC states. Run acceptance and XML/OUTCAR-side settings checks precede extraction. Each ionic frame needs finite cell/positions/forces/free energy; two final SCF energy changes below EDIFF and fewer iterations than NELM. Incomplete XML and unverified SCF convergence are rejected, with reasons in `dataset.json`. Complete preserved restart XML segments are read with their corresponding OUTCAR segment; partial/truncated segments are rejected. Older already-overwritten segments cannot be recovered by this change.

Training uses VASP `e_fr_energy` and the raw XML force arrays, including forces on constrained atoms. This preserves a force-consistent target at the specified smearing. It does not mix sigma->0 adsorption energies with free-energy training labels. No stress training is enabled. Inputs from different Hamiltonians or k-meshes are not pooled. Complex/slab cells must match the manifest common cell. The new potential targets the full DFT energy including dispersion; do not add D3 a second time to its predictions. Baseline foundation predictions require matching dispersion and energy-reference treatment before comparison.

Frames retain job, source checksum, step, frame index, energy convention, split and bundle provenance. Exact rounded geometry copies are deduplicated within a split and forbidden across splits. Conflicting labels for identical geometry stop export. No atom constraints are imposed on training forces. Near-duplicate detection beyond exact rounded copies is not claimed.

## Split policy and the current campaign

Reaction jobs and all their shared references are connected into indivisible bundles. All frames and conformers in a bundle stay together. By default all bundles are exported as training data; no random validation fraction is generated. Fine-tuning preparation requires nonempty training and validation partitions.

The current nine-salt package shares one slab reference, so it forms ONE connected bundle. It can supply training data and a pipeline feasibility pilot, but cannot independently supply both training and validation under this policy. An additional independently referenced surface/reaction bundle is needed. Merely copying the common slab into another package does not solve this; the duplicate-geometry check rejects it. Broader scaffold/surface generalization needs additional independent groups and an untouched test partition. No fixed frame count is promised to establish model quality.

Example export on the machine containing returned results:

```text
python examples/scripts/export_training_data.py RETURNED_PACKAGE --out data/training/neutral_v1
```

The report lists job identifiers as `<manifest_sha16>/<job>`. For multiple independent packages, a split plan can assign one job in each component; the assignment propagates to the complete component:

```json
{"job_splits": {"ACTUAL_MANIFEST_SHA16/slab_NAME": "valid"}}
```

```text
python examples/scripts/export_training_data.py TRAIN_PACKAGE VALID_PACKAGE --split-plan split_plan.json --stride 5 --out data/training/neutral_v2
python examples/scripts/prepare_mace_finetune.py --dataset data/training/neutral_v2 --checkpoint LOCAL_FOUNDATION.model --mace-version ACTUAL_INSTALLED_VERSION --out data/training/pilot_v1
python examples/scripts/run_mace_pilot.py data/training/pilot_v1
python examples/scripts/run_mace_pilot.py data/training/pilot_v1 --execute
```

Export and preparation require new/empty output directories. Prepare on the training machine so absolute checkpoint/data paths resolve there. Foundation-model selection remains a comparison decision: the archive relative-energy and adsorption diagnostics favor different models, and the adsorption bundles are not trusted labels. Require Pb/I/organic element coverage in the chosen checkpoint. No model was downloaded or selected automatically.

## First experiment and success criteria

The generated configuration is a narrow naive-fine-tuning baseline, 20 epochs, batch size 1, learning rate 1e-4, force/energy weights 10, one seed and training-only estimated atomic reference energies. These are initial settings, not optimized values. Hashes pin the input data, configuration and checkpoint; the launcher checks the explicitly supplied MACE package version. Full GPU execution and checkpoint compatibility remain unverified until run in the chosen environment.

1. Confirm parser/export readiness on returned revision-2 results; inspect rejection counts and energy/force ranges.
2. Qualify an independent validation bundle; preserve a test bundle where possible.
3. Evaluate the unmodified foundation model with the same target convention on that held-out data. Preserve record identity fields in prediction XYZs.
4. Run the fine-tuning pilot and score exactly the same held-out configurations. Report total-energy and per-atom errors separately from force-component errors. Fit reference offsets only on training data.
5. Use complete, qualified free-energy reference bundles to evaluate reaction errors and rankings; the per-frame scorer alone does not establish adsorption accuracy. Retain the existing sigma->0 DFT adsorption report as a separately labeled target.
6. Before generating MLIP trajectories, test physical stability and energy conservation, then measure sampling/force failures. Collect displaced and thermal configurations where needed; relaxation paths alone do not prove thermal coverage.

Official MACE fine-tuning interface consulted 13 October 2026: https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html . Narrow-system naive fine-tuning is the initial baseline; replay is a separate future experiment if preservation outside this domain is required.

## Current limitation

Infrastructure is ready for returning force-bearing labels; no trained passivation checkpoint or held-out improvement is claimed. Existing staged package 53 is unchanged. Regenerate/restage a new reviewed package to use the XML preservation change; do not silently patch a package already submitted.
