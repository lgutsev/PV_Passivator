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
| Pose generation, VASP job prep, MLIP benchmark, UI | not started |

## Usage

```
pip install -e .[dev]
pytest
pvpass reproduce examples/recovered_cases_pbe335.json
```
