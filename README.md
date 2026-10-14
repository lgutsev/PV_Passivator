# pvpassivator

Charge-balanced preparation and DFT-backed analysis for passivated FAPbI3 surfaces. A validated DFT-free predictor is a later stage. See `docs/WORKFLOW_AUDIT.md` for the current status.

## Status

| Component | State |
|---|---|
| Atom/charge ledger, balanced reactions (`ledger.py`) | implemented and unit-tested |
| VASP OUTCAR/OSZICAR parsing, settings fingerprints (`vasp_io.py`) | implemented; OSZICAR path tested on archives |
| Archive reproduction (`recovery.py`, `pvpass reproduce`) | implemented; exposes spreadsheet/archive mismatches |
| Charge displacement Δn / λ(z) / CD(z) (`density.py`) | implemented, synthetic fixture only; no archived CHGCARs |
| NAMD frame export (`namd_adapter.py`, `pvpass namd-export`) | implemented and tested on synthetic trajectories |
| Surface iodide oxyanions IO⁻…IO₄⁻ (`oxidation.py`) | implemented and unit-tested: idealised end states in scanned orientations, balanced `slab + n/2 O2 -> slab(IO_n-)` ledger entry; O2 is a reservoir only, no O2–surface paths; relaxed poses grouped by O–Pb bonding + first-shell fingerprint (`group_relaxed_poses`); O-down/X-down presets; Pb(IO₃)₂ surface unit with PbI₂-vacancy removal and a declared bulk reference; r2SCAN job bundle with a triplet O₂ reference (`vaspprep.oxidation_bundle`) and optional declared O₂ corrections (`ledger.ReferenceCorrection`); water next to an oxyanion; `pvpass oxidize` CLI |
| Structure renders in the house style (`render.py`, `compose.py`, `docs/FIGURE_STYLE.md`) | implemented; cutout geometry unit-tested (complete octahedra), fragment highlight, multi-panel composer, JSON web export; OVITO render smoke-tested (optional extra `[render]`) |
| MLIP benchmark vs archived DFT (`benchmark.py`, `examples/scripts/run_mlip_singlepoints.py`, `analyze_mlip_benchmark.py`) | implemented: energy-only single points on DFT final structures, compared within one settings group (same k-mesh) via same-composition differences, per-element-offset LOO residuals and composition-matched E_ads |
| Consistent DFT label sets (`examples/scripts/prepare_label_set.py`, `stage_label_set.py`, `labels.py`, `pvpass labels`) | implemented: one cell/k-mesh/profile per reaction family, cluster package with tested scripts, results reader with balance, settings and k-point checks |
| UI | not started (pose generation is in `poses.py`, r2SCAN job prep in `vaspprep.py`) |
| Neutral MLIP training infrastructure (`training_data.py`, export/configuration/preflight/evaluation scripts) | implemented and tested on synthetic fixtures; awaits qualified energy/force outputs and an independent validation bundle; see `docs/TRAINING_PILOT.md` |

## Usage

```
pip install -e .[dev]
pytest
pvpass reproduce examples/recovered_cases_pbe335.json
pvpass oxidize CONTCAR --out iodate_starts --n 3 --preset O-down
pvpass labels <finished label-set package>      # E_ads per reaction + k-point check; refuses unfinished steps
```

## First MLIP fine-tuning pilot

See [the training protocol](docs/TRAINING_PILOT.md) for the energy/force exporter, complete-reference-bundle split policy, local-checkpoint MACE configuration and guarded launcher. Preparation does not download models or launch training. The current nine-salt campaign shares one slab reference and therefore forms one indivisible training bundle; independent validation data are still needed. Future generated cluster packages preserve XML relaxation segments across restarts.
