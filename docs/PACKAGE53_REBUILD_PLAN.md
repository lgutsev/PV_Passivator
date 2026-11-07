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

**A2 status (2026-11-07): tool built, dry run on the converged slab.** `rebuild_on_slab.py` + `tests/test_rebuild_on_slab.py`.
- **The bare slab is converged:** `outputs/slab_FAI_Schot/relax` is DONE (29 Oct; "reached required accuracy", EDIFFG −0.03). So A1 is done.
- **The archived complexes do not put the salt into the slab's vacancies.** Matching whole lattice units (Pb, I, FA molecules), not atoms:
  - In 7 of 9 the cation sits on an A-site that the slab fills with FA, and that FA sits on the slab's V_FA (3–7 Å away).
  - In 5 of 9 (PEAAc, PEATFA, PEACl, PFPACl, TFPACl) the anion sits on a surface I site, and that iodide fills V_I (4–9 Å away).
  - Only THCl and PFPAI have the cation in V_FA as built.
  - The composition is still slab + salt, so E_ads is defined. But the cation-at-V_FA / anion-at-V_I check in the bullets above does not hold as written.
  - **Owner decision (2026-11-07): keep the archived sites** (decision 5). The check that applies instead is the tool's: each site stays in its local Pb frame.
- **What the tool does with this:** it moves such an exchanged unit with the salt (it is "relocated") when either the unit's complex position or the slab site it vacated lies within 8 Å of the salt. Every other atom sits exactly at its slab position.
- **Salt identification fixed in `spectator.align`:** the salt is now the fragments whose formula never occurs in the slab. Per-element matching had swapped salt H/C with lattice FA atoms. This also corrects the adsorbate set used by `pvpass labels` (A5) and `freeze_beyond_A`.
- **Dry-run result** (slab CONTCAR sha 972f6436d50775d5; complex CONTCARs; output `data/cache/rebuild_a2_dryrun`, not committed):

  | Complex | Status | Relocated | Salt–slab min (Å), archived → rebuilt | Salt H–Pb/I min (Å), archived → rebuilt |
  |---|---|---|---|---|
  | PEAAc | pass | FA, I | 1.79 → 2.02 | 2.82 → 2.81 |
  | PEACl | pass | FA, I | 2.34 → 2.44 | 2.51 → 2.50 |
  | PEATFA | needs_settle | FA, I | 1.89 → 2.34 | 2.83 → 2.74 |
  | PEAI | needs_settle | FA | 2.54 → 2.45 | 2.54 → 2.45 |
  | PFPACl | needs_settle | FA, I | 2.23 → 2.43 | 2.50 → 2.43 |
  | TFPACl | needs_settle | FA, I | 2.25 → 2.15 | 2.49 → 2.15 |
  | TFPAI | needs_settle | FA | 2.43 → 2.13 | 2.43 → 2.13 |
  | THCl | needs_settle | – | 2.33 → 2.35 | 2.51 → 2.35 |
  | PFPAI | needs_settle | – | 2.54 → 2.40 | 2.54 → 2.40 |

  - **needs_settle:** a salt H–lattice contact is shorter than in the archived complex. There is no clash; the closest is 2.13 Å. The A3 local settle is meant to remove this.
  - **Hard checks passed in all nine:** salt–slab ≥ 1.0 Å; the cation N, the anion and every relocated unit stay within 0.23 Å of their local Pb frame; no site exchange outside 8 Å; far-field mismatch to the slab exactly 0.
  - **The local frame matters:** it differs from the global alignment by 0.12–1.12 Å.

**A3. Local settle.** A short foundation-MACE + D3 FIRE relax, or the A0 model if it passes.
- Only atoms within 8 Å of the salt move; everything else stays fixed at the relaxed-slab positions.
- This removes transplant strain without touching the spectators.

**A3 status (2026-11-07): tool built, run on the nine rebuilt complexes.** `local_settle.py` + `tests/test_local_settle.py`.
- **Setup:** `maceh` env, RTX 5070, foundation MACE-matpes-r2scan + D3(BJ, r2SCAN) [torch-dftd], FIRE.
- **What moves:** atoms within 8 Å of the salt, plus the relocated units; molecules move whole. That is 170–192 mobile atoms; the rest are fixed and verified unchanged.
- **Stop:** fmax ≤ 0.1 eV/Å or 150 steps. Every case converged in 20–53 steps, 1–2.5 min each; the run took 14 min.
- **Output:** `data/cache/settle_a3`, not committed. Its input is the A2 output above.

| Complex | Status | Steps | fmax (eV/Å), start → end | Mean / max move (Å) | Salt H–Pb/I min (Å), archived → settled | Largest site shift (Å) |
|---|---|---:|---|---|---|---:|
| PEAAc | pass | 51 | 0.73 → 0.10 | 0.05 / 0.30 | 2.82 → 2.78 | 0.21 |
| PEAI | pass | 27 | 0.84 → 0.07 | 0.05 / 0.30 | 2.54 → 2.50 | 0.21 |
| PFPACl | pass | 27 | 0.70 → 0.10 | 0.05 / 0.28 | 2.50 → 2.53 | 0.21 |
| PFPAI | pass | 20 | 0.96 → 0.09 | 0.03 / 0.25 | 2.54 → 2.55 | 0.19 |
| TFPAI | pass | 27 | 1.02 → 0.09 | 0.05 / 0.25 | 2.43 → 2.40 | 0.16 |
| PEACl | review | 27 | 0.89 → 0.08 | 0.05 / 0.29 | 2.51 → 2.45 | 0.18 |
| THCl | review | 38 | 1.36 → 0.09 | 0.05 / 0.40 | 2.51 → 2.44 | 0.22 |
| TFPACl | review | 48 | 0.98 → 0.10 | 0.05 / 0.25 | 2.49 → 2.40 | 0.20 |
| PEATFA | fail | 53 | 0.89 → 0.10 | 0.06 / 0.22 | 2.83 → 2.66 | 0.305 (TFA O) |

- **The settle did its job.** The shortest salt H–Pb/I contact rose from 2.13 to 2.40 Å, and every salt–slab contact is ≥ 1.92 Å. Moves are small: 0.03–0.06 Å on average, 0.4 Å at most.
- **review** (PEACl, THCl, TFPACl): a salt H–lattice contact is still 0.06–0.09 Å shorter than archived, beyond the 0.05 Å tolerance. All are ≥ 2.40 Å, normal N–H···I distances.
- **PEATFA fail is marginal.** The TFA anchor O shifted 0.305 Å against its archived Pb frame (limit 0.30). It keeps all 4 of its Pb neighbours, so it has not left its site.
- **For the owner:** accept the review cases and PEATFA as they are, or rerun them with a tighter radius or fmax. The thresholds were not loosened after the fact.

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
5. **Decided (owner, 2026-11-07): keep the archived sites.** That means the salt on an exchanged A-site and, for five complexes, on a surface I site, which is the tool's default. The alternative, placing the salt into the slab's own V_FA / V_I, would have been a new adsorption geometry rather than a transplant.

## A0 status (2026-10-27): draft package ready, not staged

- **Dataset:** `export_prerelax_frames.py` (tested) wrote `data/prerelax_a0/dataset_v2` from the 2026-10-26 outputs. It holds 2,485 frames: train 1,668, valid 295 (trajectory tails), test 522 (all of TFPAI and THCl). The target is DFT minus D3(BJ, r2SCAN). torch-dftd's parameters equal the INCAR's.
- **Draft package:** `data/prerelax_a0/NN_mace_pvpass_prerelax_a0`, git-ignored, made by `make_prerelax_package.py`: gpu2, mace_env, 60 epochs, foundation E0s, forces weight 100, hash-checked data and checkpoint.
- **Local smoke run (maceh env, RTX 5070):** MACE 0.3.16 accepts the config. One epoch on 6 frames moved held-out TFPAI force RMSE from 0.062 (foundation) to 0.039 eV/Å.
- **Before staging:**
  - commit the exporter and generator, so `package.json` names a commit that contains them;
  - confirm the checkpoint exists at the LONI path;
  - agree with the desk how the 205 MB `data/` reaches LONI (not through git).
