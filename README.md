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
| Surface iodide oxyanions IO⁻…IO₄⁻ (`oxidation.py`) | implemented and unit-tested: idealised end states in scanned orientations, balanced `slab + n/2 O2 -> slab(IO_n-)` ledger entry; O2 is a reservoir only, no O2–surface paths; relaxed poses grouped by O–Pb bonding + first-shell fingerprint (`group_relaxed_poses`) |
| Structure renders in the house style (`render.py`, `docs/FIGURE_STYLE.md`) | implemented; cutout geometry unit-tested, OVITO render smoke-tested (optional extra `[render]`) |
| MLIP benchmark, UI | not started (pose generation is in `poses.py`, r2SCAN job prep in `vaspprep.py`) |

## Usage

```
pip install -e .[dev]
pytest
pvpass reproduce examples/recovered_cases_pbe335.json
```
