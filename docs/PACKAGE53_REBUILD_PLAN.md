# Package 53: rebuilding seven complexes on the relaxed slab, and the MLIP pre-relax (plan, 2026-10-20)

**Status: plan only.** Nothing is built or staged. Background: `PACKAGE53_RELAX_DIAGNOSIS.md`.

## Where things stand

- **Hold** (owner, 2026-10-20). The desk gave these commands to the owner:
  - `scancel 1116000_5` (PFPACl, pending);
  - `sbatch --array=0,6,9 --dependency=afterany:1116000:1116001 run_relax_slabs.slurm`.

  Only `slab_FAI_Schot`, TFPACl and TFPAI get further rounds. PEAAc, PEATFA, PEAI, PEACl, THCl and PFPAI finish their running segments and stop; their frames come back in the next export.
- **Foundation MLIP test** (`maceh` env, RTX 5070, MACE-matpes-r2scan-omat-ft + torch-dftd D3(BJ, r2SCAN)). Run on the latest DFT frame of each job:

  | Job | MACE Fmax vs DFT Fmax (eV/Å) | Force RMSE vs DFT (eV/Å) | Corr. | cos(F_ML, F_DFT), \|F_DFT\| > 0.05 | MACE FIRE relax from the DFT frame |
  |---|---|---:|---:|---:|---|
  | slab_FAI_Schot (504) | 0.449 vs 0.115 | 0.074 | 0.36 | 0.66 | 600 steps, −2.94 eV, mean move 0.29 Å |
  | complex_PEAAc (532) | 0.595 vs 0.134 | 0.085 | 0.23 | 0.60 | 565 steps, −2.24 eV, mean move 0.22 Å |

  Near the r2SCAN minimum the foundation model's force error is larger than the residual DFT forces, and its own minimum is somewhere else. **Do not pre-relax the slabs with the foundation model.** It is still useful for removing clashes after a transplant, as a short, local and constrained relax. A single point takes 4–7 s and a full FIRE relax 20–50 min on the laptop GPU.

## Plan

**A0. Fine-tuned pre-relaxer.** This is optional, but probably the biggest win.
- Package 53 now holds about 1,400–1,500 r2SCAN relax frames: 9 exported jobs × 2 segments (PFPACl's first segment is still on LONI), same cell, k-mesh and profile, neutral, ISPIN 1. `export_training_data.py` reads exactly these.
- Fine-tune MACE-matpes-r2scan on them **as a pre-relax tool, not as a label model**. Hold out the last 30 frames of two jobs (for example TFPAI and THCl) and score them.
- Acceptance: force RMSE ≤ 0.03 eV/Å and cos ≥ 0.9 on held-out frames with \|F\| > 0.05; the relaxed geometry within 0.1 Å of the DFT trajectory's latest frame.
- The training-pilot rule (independent bundles before any label claim) still applies to label use. A pre-relaxer only shortens the path to a DFT minimum, and DFT makes the final call.
- Compute: an 8 GB laptop GPU is marginal for 530-atom fine-tuning. A LONI GPU package through the desk is the realistic route, at a few GPU-hours.

**A1. Converge the bare slab in DFT.** Task 0 keeps running. If A0 passes before the slab is DONE, the owner can restart task 0 from a fine-tuned pre-relaxed geometry. That is a new segment, so DFT still finishes it.

**A2. Transplant the seven salts onto the relaxed slab.**
- For each of PEAAc, PEATFA, PEAI, PEACl, PFPACl, THCl and PFPAI: align the archived complex onto the relaxed slab using the best rigid translation of the Pb sublattice (the method in the diagnosis), then copy the salt atoms across.
- Every complex is slab (504) + salt, so the Schottky vacancies are the same.
- Check that the cation N sits at the V_FA site and the anion at the V_I site of the relaxed slab.
- Acceptance: minimum distance ≥ 1.0 Å, no H–lattice contact shorter than in the archived complex, and a spectator mismatch to the slab of 0 by construction.
- A new repo tool is needed: `examples/scripts/rebuild_on_slab.py`, with a unit test, writing a JSON report per complex.

**A3. Local settle.** A short foundation-MACE + D3 FIRE relax, or the A0 model if it passes.
- Only atoms within 8 Å of the salt move; everything else stays fixed at the relaxed-slab positions.
- This removes transplant strain without touching the spectators.

**A4. DFT relax of the rebuilt complexes: freeze the far field (revised 2026-10-31).**
- Each rebuilt complex relaxes with selective dynamics. Atoms more than R from the salt are frozen at the converged bare slab's positions; everything within R relaxes. Start with R = 10 Å.
- **Why the free relax is no longer recommended:** the spectator check (61cfd91) on the latest CONTCARs shows that even TFPACl and TFPAI, which started consistent with the slab (0.055 and 0.047 Å), drifted to 0.36 and 0.28 Å.
  - The drift is mostly FA reorientation: N moved 0.44–0.61 Å, C about 0.25 Å.
  - The Pb–I framework moved as well: Pb 0.11–0.26 Å, I 0.20–0.31 Å.
  - It barely decays with distance: TFPACl 0.38 / 0.36 / 0.32 Å at 8 / 10 / 12 Å.
  - So the far region has many near-degenerate minima, and two independent relaxations need not end in the same one. A free relax would need luck for every complex.
- **Freezing makes the far field identical by construction,** so it cancels in E_ads. It also cuts the degrees of freedom: roughly 170 of about 300 non-H atoms lie beyond 10 Å from the salt in this cell.
- **The frozen reference must be a converged bare slab.** It has itself moved 0.31–0.55 Å since its start. A1 comes first.
- **Convergence test** on two complexes (one TFPA, one PEA), relaxed with R = 8 and R = 10 Å: accept R when E_ads changes by less than about 0.02 eV.
- **Code needed:** a `freeze_beyond_A` option in `prepare_label_set.py`, which writes the selective-dynamics flags from each complex's distance to its salt (the same adsorbate identification as `spectator.py`).

**A5. Acceptance check for E_ads (built: `pvpass labels`, commit 61cfd91).**
- With A4, the far-field atoms match by construction, so the frozen-region check becomes a consistency test of the inputs.
- **Add a boundary-shell check:** the mean displacement of the free atoms between R − 2 Å and R from the salt, relative to the bare slab. If it is large, the response has not decayed at R; increase R.
- The current all-free complexes (the nine on rev 5.1) will be flagged by the existing check. Their statics remain training frames, not labels.

**A6. Packaging.** Revision 5 of 53, or a new number; the desk decides.
- Seven complex inputs change (new structure hashes), and `kcheck_complex_PEAAc` follows the rebuilt PEAAc.
- The revision-4 outputs of the seven move to `outputs_rev4/` and stay training frames; nothing is deleted.
- Generate with commit a8442ac or later, with `signal_margin_s: 3600` for `relax_slab`.

## Cost [P]

- Rebuilt complexes start near a minimum, so expect 30–80 ionic steps at about 0.5 h each: 30–80 node-h per complex, about 210–560 node-h for seven, plus about 20 for statics.
- Compare that with the current path: about 1,440 node-h per round, with no end in sight for the mismatched seven.
- A0 is a few GPU-hours.
- The bare slab's remaining rounds are needed in every plan.

## Effect on label set v3

- v3a derives from the PEACl and PEAAc complexes, so it waits for their rebuilt and relaxed versions.
- v3b needs the same discipline on the PbI₂ slab: relax the bare PbI₂ slab first, then build all six molecules on that relaxed geometry.

## Owner decisions

1. A0: fine-tune a pre-relaxer, and where (LONI GPU package or laptop)?
2. A4: freeze the far field at the converged bare slab (now recommended, R = 10 Å with an 8/10 Å convergence test), or keep the free relax?
3. Start A2 on the *current* slab geometry now to save calendar time? Not recommended: the slab is still dropping about 10 meV per step, so the spectator states would diverge again.
4. Revision 5 or a new package number: the desk decides once staged.

## A0 status (2026-10-27): draft package ready, not staged

- **Dataset:** `export_prerelax_frames.py` (tested) wrote `data/prerelax_a0/dataset_v2` from the 2026-10-26 outputs. It holds 2,485 frames: train 1,668, valid 295 (trajectory tails), test 522 (all of TFPAI and THCl). The target is DFT minus D3(BJ, r2SCAN). torch-dftd's parameters equal the INCAR's.
- **Draft package:** `data/prerelax_a0/NN_mace_pvpass_prerelax_a0`, git-ignored, made by `make_prerelax_package.py`: gpu2, mace_env, 60 epochs, foundation E0s, forces weight 100, hash-checked data and checkpoint.
- **Local smoke run (maceh env, RTX 5070):** MACE 0.3.16 accepts the config. One epoch on 6 frames moved held-out TFPAI force RMSE from 0.062 (foundation) to 0.039 eV/Å.
- **Before staging:**
  - commit the exporter and generator, so `package.json` names a commit that contains them;
  - confirm the checkpoint exists at the LONI path;
  - agree with the desk how the 205 MB `data/` reaches LONI (not through git).
