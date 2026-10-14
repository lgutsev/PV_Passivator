# Neutral MLIP training infrastructure handoff — 13 October 2026

Added a guarded VASP energy/force exporter, explicit reference-bundle splits, pinned-checkpoint MACE configuration/preflight launcher, held-out prediction scorer and usage protocol in `docs/TRAINING_PILOT.md`.

The exporter reads complete XML (plain or gzip) plus OUTCAR metadata, checks charge, atom ordering, Hamiltonian/k-mesh compatibility, common slab cells and per-frame SCF convergence, then writes free-energy/raw-force labels. Reaction atom inventories must balance. Incomplete and missing inputs are recorded, not invented. Exact duplicate geometries may not cross splits; final statics supersede matching same-job relaxation endpoints. Complete restart XML segments can be recovered using their segment OUTCAR. The staging generator now preserves those XML segments on future restarts; already staged packages were not patched.

First experiment is a narrow neutral, ISPIN=1, non-SOC MACE fine-tune using a local checkpoint and explicit package version. No model download, GPU training, or cluster submission occurred. The runtime configuration cannot be prepared without a nonempty independent validation split. See the protocol for data target, dispersion treatment and scope limits.

Important new readiness finding: the nine-salt package shares one slab reference, connecting all reaction cases into a single bundle. Under complete-reference-bundle isolation, this is training data, not a train/validation split. Obtain a genuinely independent bundle; duplicating the same bare slab fails the cross-split geometry check.

The current returned smoke/bulk results do not contain completed passivation reaction jobs. A real-package dry-run therefore produces zero frames and a missing-output report. This is expected, and means the exporter is ready to ingest the next qualified outputs rather than that training has started.

No changes to existing physics modules or calculation settings. Changes to the cluster library are solely XML preservation. Public repository documentation contains no private grant text.

Validation: full repository suite passed, **81 tests**, including 12 new data integrity/round-trip tests. Checks cover complete XML energy/force pairing, per-frame SCF rejection, truncated XML, electron-count/atom-order mismatch, connected-bundle leakage, copied geometry across splits, conflicting profiles, restart segments, atom balance, hash guards and prediction alignment. The exporter was also run on the actual desk package: zero frames, 38 missing-output entries, pilot readiness false. MACE/GPU execution was not performed or claimed.
