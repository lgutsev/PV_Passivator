# Finite-temperature chirality transfer: feasibility, first checkpoint (8 Nov 2026)

Tags: **[V]** verified on the publication page, **[R]** read from our files, **[M]** measured or computed here,
**[I]** inferred, **[P]** proposed, not done.

## 0. Reviewer summary

| | What | Where |
|---|---|---|
| **Run** [M] | Static descriptors, topology and stereochemistry checks on three experimental COD crystal structures: (R)-, (S)- and rac-MBA₂PbI₄. | `examples/scripts/chirality_bulk_benchmark.py` → `docs/CHIRALITY_MBA2PBI4_MANIFEST.json` |
| **Run** [M] | Exact S mirror built by inverting R, then compared with the separately refined S crystal. | same manifest, `S_mirror` |
| **Run** [M] | Analysis module tests: 20 new, 167/167 in the full suite. Test inputs are the three real CIFs plus labelled synthetic fixtures. | `src/pvpassivator/chirality.py`, `tests/test_chirality.py` |
| **Run** [M] | Five-call MACE throughput timing on a laptop GPU, plus an R/S energy check with the r2SCAN foundation model. No MD was run. | numbers in §8 and `examples/chirality/plan_matrices.json` |
| **Planned only** [P] | DFT validation matrix, MLIP acceptance criteria, MD matrix, restraint intervention, cost estimates. **Nothing is staged or submitted, and no package number is assigned.** | `examples/chirality/plan_matrices.json`, §8 |
| **Not done** | No DFT, no MLIP training and no MD trajectory. The restraint code is not written. No hydration work. | — |

**Recommendation: GO for a small DFT validation tranche (approval needed). NO-GO for production MD.** The
reasons are in §9.

**Update 2026-11-08: the owner approved the D0–D2 tranche ("ok approve"). It is staged, not submitted.**
- Package `LONI_staging/chiral_d0_vasp_mba2pbi4`, hand-off `LONI_staging/HANDOFF_TO_DISPATCH_chiral_d0.md`, no
  package number yet.
- Generator: `examples/scripts/prepare_chirality_d0.py` (tests: `tests/test_chirality_d0.py`).
- Starts:
  - R and rac with only the riding H relaxed, using foundation MACE + D3(BJ). fmax went from 7.4 / 9.0 to
    ≤ 0.05 eV/Å on H.
  - Heavy atoms are bit-for-bit at the X-ray positions, so every descriptor is unchanged. X–H bonds are
    1.03–1.09 Å.
  - The exact mirror of R is included.
- 18 runs, in order: smoke 1, check 7, soc 2, relax 4, static 4. `ISYM = 0` throughout.
- Cost: the staged package totals about 80–360 node-h. That is above the 30–120 node-h approved, because of
  `ISYM = 0` and four relaxations. Only smoke + check + soc (about 17–55 node-h) is requested now. The relax and
  static arrays wait for measured timings and a re-confirmation by the owner if the projection exceeds 120 node-h.
- Checks: local mock run 18/18 DONE, `bash -n` clean, no CR bytes.
- **Desk intake (2026-11-08):** taken in as **package 105** (`105_vasp_pvpass_mba2pbi4_chirality_d0`, desk commit
  `2e1d165`), renamed only.
  - The desk changed one thing: the smoke job's USR1 margin, from 1800 s to 300 s on a 30-min job.
  - The generator now uses min(1800 s, walltime/4) and refuses larger margins. It also writes per-file
    `files_sha256` into `package.json`.
  - Smoke, check and soc are released to the owner for submission. Relax and static are held until the owner
    re-confirms after the check timings.
- **Released runs finished (desk report, 2026-11-08):** smoke 3:16; check 10:03 / 8:16 / 19:17 / 13:03 / 10:02 /
  5:01 / 11:03 (R_kref, R_kdown, R_kup, R_kref_e600, Rmirror_kref, rac_kref, rac_kup; 1 node each); soc 1:16:22 (R)
  and 28:08 (rac) on 2 nodes. That is about 0.2 node-h for check and 3.5 node-h for soc, against 17–55 estimated.
  - Energies, forces and LOOP+ timings await the export. `examples/scripts/analyze_chirality_d0.py` (tests:
    `tests/test_analyze_chirality_d0.py`) applies the D0 thresholds and recommends the mesh.
  - **Projected relax + static cost** [I]: a converged R static takes ~10 min (~16 min at ENCUT 676), and an ionic
    step is about half a static. 40–120 steps per relax gives about 14–50 node-h for the four relaxes plus < 1 node-h
    for the statics. That is below the 120 node-h re-confirmation line and inside the approved 30–120.
- **D0 results [M] (export of package 105, `analyze_chirality_d0.py`):**

  | Check | Result | Verdict |
  |---|---|---|
  | k-mesh, R | 3×3×1 vs 4×4×1: max force change 1.1 meV/Å (2×2×1 is off by 16.9) | pass |
  | k-mesh, rac | 3×3×2 vs 4×4×3: 0.8 meV/Å | pass |
  | R − rac energy | −96.8 meV/f.u. at the reference meshes, −96.4 at the dense ones (change 0.4) | pass |
  | Mirror (D2) | R vs exact inversion: ΔE = 0.0 meV/cell, forces opposite to 0.3 meV/Å | pass (a settings check, not physics) |
  | SOC (D0c) | Max force change: Pb 12.8, I 9.3, others ≤ 3.8 meV/Å; −18.2 meV/atom in energy | SOC off is acceptable for structure (< 20) |
  | **ENCUT** | **520 vs 600 eV: max 25.9 meV/Å, systematic on I (RMS 22.5), C 9.5 RMS, H/N/Pb ≤ 5** | **fail** |
  | Start forces | DFT fmax on the H-settled X-ray cells: 1.46 (R) and 1.93 (rac) eV/Å | relaxation needed |

  - The owner's condition for releasing relax/static ("Yes, if it passes") is **not met**. Both arrays stay held,
    and the receipt is recorded in the desk ledger.
  - I has the softest POTCAR (ENMAX 176 eV), so the I error points to `LREAL = Auto` or r2SCAN grid sensitivity
    rather than basis incompleteness [I].
  - The framework descriptors are built from iodide positions, so this must be resolved before relaxing.
  - **Proposed** [P], not staged: four R statics at about 1–1.5 node-h: `LREAL = .FALSE.` at 520 / 600 / 700 eV,
    and `LREAL = Auto` at 700 eV. If the error is LREAL, relax at 520 with `LREAL = .FALSE.`; otherwise take the
    lowest ENCUT whose forces are within 5 meV/Å of the next step up. Either way the relax inputs are regenerated.
    The settings then differ from the package 53 label profile, which matters if these frames are later mixed
    into the same training set.
  - **Staged 2026-11-08** (owner: "Yes, stage it"), not submitted: `LONI_staging/chiral_d0e_vasp_mba2pbi4_lreal`
    with `HANDOFF_TO_DISPATCH_chiral_d0e.md`, no package number. Generator mode
    `prepare_chirality_d0.py stage --mode diag`.
    - Four R statics on 105's exact start and mesh: `LREAL = .FALSE.` at 520 / 600 / 700, and `LREAL = Auto` at 700.
    - Checks: mock run 4/4 DONE, `bash -n` clean.
    - Sent with the owner's "Yes, send it". Taken in as **package 108** (`108_vasp_pvpass_mba2pbi4_lreal_diag`, desk
      commit `48df5cd`; all 27 hashes match) and released to the owner for submission.
    - The export will be read with `analyze_chirality_d0.py <105 dir> --diag <108 dir>`. That compares all six R
      statics and returns the cause (LREAL, basis or unresolved) and the setting for the regenerated relax/static.
  - **D0e results [M] (package 108 against 105's baselines; max per-atom force difference, meV/Å):**

    | Pair | max | I | Pb | C | H | N |
    |---|---|---|---|---|---|---|
    | `LREAL = .FALSE.` 520 vs 600 | 5.1 | 1.1 | 0.3 | 5.1 | 4.8 | 1.6 |
    | `LREAL = .FALSE.` 600 vs 700 | 10.0 | 0.7 | 0.4 | 7.8 | 5.9 | 10.0 |
    | `LREAL = .FALSE.` 520 vs 700 | 11.2 | 1.7 | 0.6 | 10.3 | 10.5 | 11.2 |
    | Auto vs `.FALSE.`, 520 | 11.5 | 4.4 | 4.6 | 11.5 | 0.4 | 0.7 |
    | Auto vs `.FALSE.`, 600 | 28.0 | 28.0 | 2.1 | 3.1 | 0.2 | 0.5 |
    | Auto vs `.FALSE.`, 700 | 8.3 | 8.3 | 2.5 | 1.6 | 0.9 | 2.0 |

    - **The iodide error is `LREAL = Auto`.** With `.FALSE.`, the framework atoms (I, Pb) are converged at
      520 eV to ≤ 1.7 meV/Å.
    - The organic atoms still change by 5–11 meV/Å with ENCUT, and the change grows from 520→600 to 600→700. That
      is not a converging basis; r2SCAN grid noise is the likely cause [I].
    - By the pre-set 5 meV/Å rule the analyzer returns "unresolved". 105 relax/static stay held (receipt: desk ledger
      `2e10f6b`). The owner's decision is pending: accept 520 + `LREAL = .FALSE.` with a ~12 meV/Å tolerance on C/H/N,
      or run a further grid check (e.g. `ADDGRID`, 800 eV).
    - **Decision 2026-11-09** (owner: "Yes, regenerate"): 520 eV + `LREAL = .FALSE.`. The accepted residual on C/H/N
      is ~12 meV/Å, recorded as the D1 tolerance.
    - Revision 1 of 105's relax/static is staged, not submitted: `LONI_staging/chiral_d1_vasp_mba2pbi4_relax_rev1`,
      hand-off `HANDOFF_TO_DISPATCH_105_REV1.md`, `prepare_chirality_d0.py stage --mode d1`.
    - Only `LREAL` differs from 105. The starts are byte-identical to the 105 runs. Mock run 8/8 DONE.
    - Estimated 19–67 node-h.
    - **Relaxations done** (LONI job 1171640, about 25 node-h; read with `analyze_chirality_d1.py`; desk ledger
      `b53431d`). All four are intact: stereocentres kept, Pb six-coordinate, NH₃⁺ kept, final fmax < 0.01 eV/Å.

      | Job | Volume vs X-ray | Net τ_ax / τ_eq / τ_org (X-ray → relaxed) | Pb–I–Pb (°) |
      |---|---|---|---|
      | R fixed cell | 0 | 0.00016 / 0.0065 / −0.029 → 0.00017 / 0.0054 / −0.018 | 149.3, 155.8 |
      | R free cell | −3.3 % (a 8.79, b 9.12, c 28.45 Å; still orthorhombic) | → 0.00027 / 0.0056 / −0.022 | 148.0, 155.5 |
      | rac fixed cell | 0 | 0 / 0 / 0 → 0 / 0 / 0 (per-link τ_ax ±0.100) | 154.1 |
      | rac free cell | −4.8 % (β 100.1° → 99.6°) | 0 / 0 / 0 (per-link τ_ax ±0.095) | 152.3 |

      - Every net descriptor keeps its X-ray sign. τ_eq shrinks by ~15 % and τ_org by 25–40 %; τ_ax stays at the
        2×10⁻⁴ noise level.
      - The inversion of relaxed R gives exactly the negated values, so the exact-mirror S for later work is the
        inversion of relaxed R.
      - The volume loss mixes thermal expansion (DFT is at 0 K; the X-ray data were taken at 173 / 293 K) with
        r2SCAN+D3 overbinding. These runs cannot separate the two.
      - Fixed-cell relax energies (520 eV, `LREAL = .FALSE.`): rac is 52 meV/f.u. lower than R.
      - 520 eV statics (job 1173509, received at desk 302a6bf):

        | Cell | E(R) − E(rac) | D3 part |
        |---|---|---|
        | fixed (X-ray) cells | +51.1 meV/f.u. | −19.0 meV/f.u. (D3 favours R) |
        | free (relaxed) cells | **+75.0 meV/f.u.** | +16.3 meV/f.u. |

        - Cell relaxation lowers rac by 33.9 meV/f.u. but R by only 10.0.
        - The racemic crystal is the lower-energy one at 0 K. Its sign agrees with the existence of the racemic
          compound. The size is a static 0 K number (no ZPE or vibrational free energy).
        - **The MLIP R−rac target is +75.0 meV/f.u. on the free-cell minima** (+51.1 on the fixed cells).
      - These relaxed cells are the DFT reference minima for the MLIP acceptance tests in §8.
- **Zero-shot test of the foundation model [M] (2026-11-10, local, no labels written):**
  - Script `examples/scripts/mlip_vs_dft_chirality.py`, run in the `maceh` env. Model: MACE-matpes-r2scan-omat-ft +
    D3(BJ, r2scan), float64.
  - Data: every 4th frame of the DFT fixed-cell relax trajectories (24 R frames, 27 rac frames), then an MLIP
    fixed-cell relax from each DFT minimum.

  | Item | R | rac | Criterion | Verdict |
  |---|---|---|---|---|
  | Force RMSE C / N | 0.180 / 0.182 | 0.181 / 0.150 eV/Å | ≤ 0.05, none > 0.08 | fail |
  | Force RMSE I_axial / I_bridging | 0.153 / 0.027 | 0.145 / 0.028 | | fail (axial) |
  | Force RMSE H on N / H on C / Pb | 0.073 / 0.067 / 0.013 | 0.073 / 0.060 / 0.000 (Pb on inversion centres) | | fail (H) |
  | MLIP fmax at the DFT minimum | 0.55 eV/Å | 0.51 eV/Å | — | — |
  | MLIP relax away from the DFT minimum | 0.12 Å RMS, 0.27 max | 0.07 Å RMS, 0.17 max | — | — |
  | Net τ_eq (DFT → MLIP) | 0.0054 → 0.0022 (−58 %) | 0 → 0 | same sign, within 25 % | sign ok, magnitude fail |
  | Net τ_org | −0.018 → −0.029 (+60 %) | 0 → 0 | same sign, within 25 % | sign ok, magnitude fail |
  | Max per-link difference (org) | 0.042 | 0.020 | ≤ 0.02 | fail (R) |
  | R − rac energy | 35.3 vs DFT 51.1 meV/f.u. | | within 10 meV/f.u. | fail |
  | Stereocentres | kept | kept | — | ok |

  - **Verdict: the foundation model is not usable for chirality dynamics.** This confirms the NO-GO for MD in §9.
  - The largest errors sit exactly at the transfer interface: axial I (H-bond acceptors), N, C and N–H. The
    bridging I and Pb framework is described well (≤ 0.03 eV/Å).
  - Fine-tuning needs labels that sample the cation–axial-iodide coupling: rotor scans (D3–D5), distortions (D6),
    finite-T frames (D8), and a held-out D9.
  - Measured cost changes the estimate: a 188-atom R static at 520 eV with `LREAL = .FALSE.` takes ~12 min on one
    node, so the ~330 single points of D3–D9 are ≈ 70–100 node-h rather than the 100–330 inferred earlier.
- **Label tranche A staged 2026-11-10** (owner: "Yes, stage it"), not submitted: `LONI_staging/chiral_labels_a_vasp_mba2pbi4`,
  hand-off `HANDOFF_TO_DISPATCH_chiral_labels_a.md`, generator `examples/scripts/prepare_chirality_labels.py`
  (tests: `tests/test_chirality_labels.py`).
  - 62 single points on the 105 rev1 fixed-cell minima, same profile (520 eV, `LREAL = .FALSE.`, `ISYM = 0`):

    | Block | Points |
    |---|---|
    | D3 NH₃⁺ rotor | 26 |
    | D4 ring torsion | 9 (11 rigid-rotation points dropped for contacts < 1.5 Å; the kept ones cover ±20° libration around 0° and 180°, plus R at 80°) |
    | D5 methyl rotor | 7 |
    | D6 Pb/I displacements | 12 (a substitute for symmetry-adapted tilt patterns) |
    | D7 in-plane strain | 8 |

  - About 12–20 node-h. Mock run 62/62 DONE.
  - Not in this tranche: D8 finite-T frames and the D9 held-out set. They need local pre-relaxer MD frames: about
    6 runs × 2–3 h on the laptop GPU, which needs the owner's OK under the no-long-GPU-jobs rule.
  - Sent 2026-11-10 ("Yes, send it"). The desk took it in as package 113 (LONI job 1174343, array 0-61%20).
- **D8/D9 frame generation with the foundation model failed its integrity check (2026-11-10).** The owner said
  "Yes, run it" for the local MD. Generator: `examples/scripts/run_chirality_premd.py`, 12 runs, about 3.7 GPU-h.
  Setup:
  - Langevin NVT on the 105 rev1 fixed-cell minima.
  - Every saved frame is re-checked: six-coordinate Pb within 3.8 Å, intact NH₃⁺, unchanged stereocentres.

  | System | 173 K | 293 K | 358 K |
  |---|---|---|---|
  | R, D8 (30 wanted) | 4, broken (Pb 5-coordinate) | 4, broken | 0, broken (an N–H lost in the first frame) |
  | rac, D8 (30 wanted) | **30 ok** | 2, broken | 2, broken |
  | R, D9 (10 wanted) | 1, broken | 0, broken | 0, broken |
  | rac, D9 (10 wanted) | **10 ok** | **10 ok** | **10 ok** |

  - Every break is the same pattern: the longest Pb–I bond stretches from 3.33 Å (DFT minimum) to 3.6–3.75 Å
    within 0.5–2 ps, until one crosses 3.8 Å. Even the rac frames that pass sit at 3.4–3.7 Å.
  - Mean kinetic temperatures run 7–13 % above the thermostat target.
  - Reading: this is the foundation model's soft axial-iodide response, which matches its axial-I force RMSE
    of 0.15 eV/Å. It is not a physical result.
  - The frames are biased: labelling them would spend DFT on model-softened geometries and would barely
    cover R. **Not staged.**
  - The usual fix is one active-learning round: fine-tune on tranche A (113) plus the D1 relax trajectories,
    then regenerate D8/D9 with the fine-tuned model and the same checks. The 70 intact rac frames are kept in
    `data/cache/chirality_premd` and are not labels.
- **Fine-tune first (owner, 2026-11-11).** The dataset and fine-tune configuration are prepared with the existing
  training-pilot tools. Nothing has been trained yet: training waits for the package 113 export.
  - Exporter: `examples/scripts/export_chirality_training.py` (tests: `tests/test_export_chirality_training.py`).
    - It reuses the run-acceptance, per-frame SCF, Hamiltonian-fingerprint and geometry-hash checks of
      `pvpassivator.training_data`.
    - It writes the same `pvpass_training/1` layout, so `prepare_mace_finetune.py` and `run_mace_pilot.py` run
      unchanged.
  - Sources:
    - 113 label single points.
    - 105 rev1 **fixed-cell** relax trajectories, every 4th frame plus the last, all in train.
    - The four 105 rev1 statics at 520 eV, also in train. The fixed-cell statics supersede their relax
      endpoints; the free-cell statics add the relaxed cells.
    - Excluded with a recorded reason: the free-cell relaxes (ENCUT 676 for Pulay stress, so a different
      Hamiltonian) and the D0 checks (`LREAL = Auto`).
    - R and rac differ only in k-mesh, at one k-spacing (converged in D0).
  - Target: D3(BJ, r2SCAN) is subtracted from the DFT labels, and D3 is added back at inference.
    - This matches the foundation model, which was trained without dispersion.
    - torch-dftd reproduces the VASP `Edisp` to ≤ 0.07 meV per cell on the 105 relax endpoints
      (R −8.25201 vs −8.25207 eV; rac −4.08800 vs −4.08804 eV).
  - Splits (`examples/chirality/finetune_splits.json`), fixed before any label returned. Whole scans or seeds
    only:

    | Split | Content | Points from 113 |
    |---|---|---|
    | test | rac NH₃⁺ scan (part of the plan's D9) | 13 |
    | valid | CH₃ scan, D6 seed 3, ±1 % strain | 14 |
    | train | everything else, plus the 105 relax frames | 35 |

    Later D9 MD frames go to test.
  - Configuration (`examples/chirality/finetune_overrides.json`, via a new `--overrides` option of
    `prepare_mace_finetune.py`):
    - E0s from the foundation model: two crystals of one composition cannot identify per-element references.
    - Force weight 100, energy weight 1: the failure is in forces.
    - Up to 60 epochs, patience 15, batch size 2, float64.
  - Checked on 2026-11-11 with the 105 relax frames only (51 frames: 24 R, 27 rac; nothing rejected):
    - Export works. D3 on the CPU takes ~7 s per frame, so use `--device cuda`.
    - The prepare and preflight steps pass unchanged.
    - A throwaway 2-epoch run trained on R and validated on rac, with MACE 0.3.16 and `E0s=foundation`:
      - it runs;
      - rac force RMSE fell from 123 to 32 meV/Å;
      - the energy RMSE stays at ~54 meV/atom, a near-constant reference offset that the energy term removes
        slowly.
    - These are near-minimum relax frames, so this is a pipeline check, not evidence of accuracy.
    - Measured cost: ~3.9 min per epoch for 24 frames of 188 atoms in float64 on the laptop GPU. The full set
      (~85 train frames, half of them 94-atom rac) is ≈ 8–10 min per epoch, so 60 epochs ≈ 8–10 GPU-h.
      float32 would be faster but is unmeasured. This is a long GPU job and needs the owner's OK when 113
      returns.
  - Pass criteria are the ones already fixed in `plan_matrices.json`, scored with D3 added back:
    - per-environment force RMSE ≤ 0.05 eV/Å (none above 0.08) on the held-out test;
    - relaxed net τ with the same signs and within 25 % of DFT;
    - rotor profiles within max(20 meV, 15 %);
    - R−rac within 10 meV per formula unit.

    A model that passes on the 113 test split only regenerates D8/D9. The production gate still needs the full D9.
- **Package 113 returned 2026-11-12, 62/62 DONE** (job 1174343; task 3 rerun as 1176163). Receipt recorded at desk
  d1b82bf. Every point converged in ≤ 21 SCF steps.
  - Dataset `data/training/chirality_v1` (git-ignored): train 85, valid 13, test 12 frames; nothing rejected.
    - The 0° scan points are the DFT minima, already in train through the 105 statics. Two of them sat in
      held-out splits (`rac_D3_NH3_000` in test, `R_D5_CH3_000` in valid) and were removed under a rule
      written into `finetune_splits.json` before any model was trained.
    - Their labels agree with the 105 statics to 0.0 meV and 0.0 meV/Å: the two packages give identical
      labels for the same geometry.
  - **Foundation baseline** (`examples/scripts/score_chirality_mlip.py`; output
    `data/cache/chirality_scores/foundation.json`). This is the bar the fine-tune must clear:

    | Criterion | Foundation MACE-matpes-r2scan | Required | |
    |---|---|---|---|
    | Test forces (rac NH₃⁺ scan), RMSE: C / N / I_axial | 0.181 / 0.149 / 0.141 eV/Å | ≤ 0.05 | fail |
    | Test forces, RMSE: H on N / H on C | 0.075 / 0.060 eV/Å | ≤ 0.05 | fail |
    | Test forces, RMSE: Pb / I_bridging | 0.007 / 0.029 eV/Å | ≤ 0.05 | pass |
    | NH₃⁺ rotor barrier, R / rac | 231 vs DFT 223 / 235 vs 237 meV, profile RMSE 3.7 / 4.2 meV | max(20 meV, 15 %); ≤ 10 meV | pass |
    | CH₃ rotor barrier (R) | 93 vs 162 meV, RMSE 24.7 | same | fail |
    | Ring torsion (R / rac, rigid) | 2259 vs 3106 / 262 vs 371 meV | same | fail |
    | R − rac, fixed / free cells | 35.3 vs 51.1 / 40.1 vs 75.0 meV/f.u. | within 10 | fail |

    - The foundation model already reproduces the NH₃⁺ rotor energy profile, but not the forces along it.
      The force error sits on the cation's heavy atoms and the axial iodides, not on Pb or the bridging I.
    - The R D4 80° point is a near-clash of a rigidly rotated ring (3.1 eV). The barrier criterion was
      written for relaxed barriers, so the rigid ring scans are a force and energy check, not a barrier check.
    - Three D6 frames (σ ≥ 0.10 Å) break the 3.8 Å six-coordination test. They are scored with plain element
      masks and listed in the score notes.
  - Next: the fine-tune itself (`prepare_mace_finetune.py --overrides examples/chirality/finetune_overrides.json`,
    then `run_mace_pilot.py --execute`), ≈ 8–10 GPU-h. It needs the owner's OK. Since 2026-11-12 the laptop's CPU
    and GPU are shared with other agents (CPU cap 4 threads; one D3 export already hit a GPU out-of-memory).
  - The owner chose LONI over the laptop for the fine-tune (2026-11-12). The package is staged, not sent:
    `LONI_staging/chiral_ft1_mace_mba2pbi4`, with hand-off `HANDOFF_TO_DISPATCH_chiral_ft1.md`.
    - Generator: `examples/scripts/make_chirality_finetune_package.py`.
    - Job: one gpu2 GPU, ≤ 8 h, estimate 1–3 GPU-h. The 2.6 MB dataset is inside the package.
    - Scoring against the criteria (`score_chirality_mlip.py`) runs locally after the export.
- **Fine-tune ft1 back (2026-11-12): LONI package 121, job 1178376.**
  - Run: one A100 80 GB, 60/60 epochs in 8.4 min, mace 0.3.15 / torch 2.5.0 (the YAML was checked with 0.3.16).
  - Model: `D:\MLIP_Custom_Models\MACE\mba2pbi4_chirality_ft1.model`, sha256 `7aa5d279…`, verified. It has a
    provenance file. Add D3(BJ, r2SCAN) at inference.
  - Training curve: validation force RMSE 28 → 8.8 meV/Å, still falling slightly at epoch 60; no early stop.
  - Scores: `data/cache/chirality_scores/ft1.json` and `ft1_relax.json`, against the same criteria as the
    foundation baseline above.

    | Criterion | Foundation | ft1 | Required | ft1 |
    |---|---|---|---|---|
    | Test forces (rac NH₃⁺ scan), worst environment RMSE | 0.181 eV/Å (C) | 0.008 eV/Å (H on N); largest atom 0.070 | ≤ 0.05; none > 0.08 | **pass** |
    | Valid forces, worst RMSE / largest atom | 0.181 / 0.58 | 0.017 (Pb) / 0.198 (C) | — | (not gated) |
    | NH₃⁺ rotor, R (train) / rac (test) | 231 / 235 vs 223 / 237 meV | 229 / 246 vs 223 / 237; RMSE 2.7 / 4.2 | max(20 meV, 15 %); ≤ 10 | **pass** |
    | CH₃ rotor (valid, never trained) | 93 vs 162 meV | 85 vs 162, RMSE 28.8 | same | fail |
    | Ring torsion, R / rac (train, rigid) | 2259 / 262 vs 3106 / 371 | 2681 / 313 vs 3106 / 371; RMSE 161 / 23 | same | fail |
    | R − rac, fixed cells | 35.3 vs 51.1 meV/f.u. | 15.4 vs 51.1 | within 10 | fail |
    | R − rac, free cells | 40.1 vs 75.0 | 65.5 vs 75.0 | within 10 | **pass** |
    | Relax from the DFT minima (+D3) | 0.12 Å RMS; τ_eq −58 %, τ_org +60 % | < 0.0003 Å; net τ equal to 1e-4; per-link ≤ 1e-4 | same sign, ≤ 25 %, links ≤ 0.02 | pass (but these minima are training data) |

  - Reading:
    - Forces are fixed everywhere the data reach. The C, N and axial-I errors drop by ~20×, and the held-out rac
      NH₃⁺ scan passes. So the cation–axial-iodide coupling that failed zero-shot is now learned.
    - **Energies are the weak point.** The energy RMSE of 0.5–0.7 meV/atom is ≈ 25–35 meV per formula unit
      (47 atoms) and ≈ 100 meV per 188-atom cell. That is the size of the remaining R − rac (fixed cell) and
      ring-profile misses.
      - Cause: the loss weights forces 100 : energy 1.
      - Fix: a second-stage energy weighting (MACE `swa`, energy weight ~1000 for the last epochs) in the next
        fine-tune.
    - The CH₃ rotor is a genuine extrapolation failure. The whole scan sits in valid, so no methyl rotation was
      ever trained, and the fine-tune left it as soft as the foundation model.
  - Gate (rule written before training: "a model that passes on the 113 test split only regenerates D8/D9"): the
    test split passes, both forces and the rac NH₃⁺ profile. **ft1 qualifies only as the D8/D9 configuration
    generator.** It is not a production model. Its trajectories are not results, and every D8/D9 frame is
    relabelled by DFT.
- **D8/D9 with ft1 (2026-11-12/13, owner: "Yes, run it"): the same integrity breaks as the foundation model.**
  - 12 runs in ~6 h on the laptop GPU (output `data/cache/chirality_premd_ft1`, git-ignored).
  - Intact: R 173 K D8 (30 frames), rac 173 K D8 (30), and rac D9 at all three temperatures (3 × 10).
  - The other 7 stopped within 0–2 frames, on the same Pb atoms as the foundation runs (169/170 in R, 89 in rac),
    each because a sixth Pb–I passed 3.8 Å.
  - The integration is ruled out. NVE from the R minimum (0.5 ps, initialized at 293 K) conserves energy to
    −0.01 meV/atom/ps in all four setups, against a 1 meV/atom/ps stop criterion:
    - float32 + D3 at 40 bohr;
    - float64 + D3 at 40 bohr;
    - float64 + full D3;
    - float64 without D3.

    So neither float32 nor the shortened D3 cutoff heats the system.
  - **The 3.8 Å coordination cutoff is the problem, not the dynamics.**
    - Across all 171 MD frames of both models, the sixth Pb–I is ≤ 3.79 Å and the seventh ≥ 5.79 Å.
    - In the DFT-labelled D6 distortions the sixth reaches 4.10 Å and the seventh stays ≥ 5.43 Å.
    - So no frame ever lost an iodide. The 3.8 Å value is the topology builder's default and sits inside
      ordinary thermal Pb–I stretching. Rough estimate: 0.12 eV for +0.45 Å at k ≈ 1 eV/Å², about 5 kT at
      293 K over ~24 bonds per cell.
  - The plan's stop condition reads "any Pb loses or gains an I (coordination != 6)". It stays as written; only
    its distance definition changes.
  - Proposal: count Pb–I bonds out to 4.3 Å, the largest cutoff both cells accept (half-cell-width guard). Rerun
    only the broken runs with `run_chirality_premd.py --mx-cut 4.3 --redo-broken`, about 3–4 GPU-h; the old
    files are kept as `*.broken_mx3.8`.
  - Unexplained: the rac D9 runs average +13–15 % above the thermostat target in both models (same seed); R runs
    are within ±5 %. NVE shows no heating, so this is noted but not acted on.
  - Rerun with `--mx-cut 4.3 --redo-broken` (owner: "Yes, rerun them", 2026-11-13; CPU capped at 2 threads, idle
    priority, while the owner used the laptop). 10 of 12 runs are now complete.
    - Both R 358 K runs stopped again, D8 at frame 15 and D9 at frame 4, this time because an ammonium N–H passed
      the topology builder's covalent tolerance (1.15 × (r_N + r_H) ≈ 1.17 Å) between saved frames.
    - The saved frames show N–H ≤ 1.15 Å and H…I ≥ 2.17 Å throughout, far from the plan's proton-transfer
      criterion (H…I < 1.9 Å).
    - Most likely the same kind of tolerance artifact as the Pb–I one, but unproven: the breaking frame isn't
      saved. Not pursued, to keep the laptop free.
  - **Label tranche B staged** (not sent): `LONI_staging/chiral_labels_b_vasp_mba2pbi4_md`, with hand-off
    `HANDOFF_TO_DISPATCH_chiral_labels_b.md`.
    - Generator: `examples/scripts/prepare_chirality_md_labels.py` (test `tests/test_chirality_md_labels.py`).
    - 219 single points: D8 R 75 / rac 90, D9 R 24 / rac 30. R 358 K contributes 15 + 4.
    - Same profile as 113. About 30–45 node-h.
    - Every frame was re-checked at staging, and none was refused.
- **Package 126 back (2026-11-14/16): 219/219 DONE** (job 1180657; tasks 24/45/47 exited 75 and were rerun as
  1182128). All points converged in ≤ 21 SCF steps. Receipt recorded at the desk.
  - Dataset `data/training/chirality_v2`: train 250 (adds D8 R 75 / rac 90), valid 13, test 66 (rac NH₃⁺ scan plus
    D9 R 24 / rac 30). Nothing rejected; the same 7 duplicates as v1 were handled the same way.
  - **ft1 fails on the finite-temperature held-out frames** (`data/cache/chirality_scores/ft1_on_v2.json`): test
    force RMSE C 0.113, N 0.064, H on C 0.043 eV/Å, worst atoms 0.65 / 0.28 / 0.54 eV/Å.
    - ft1 was trained on near-minimum geometries only. Its 0.008 eV/Å on the rac NH₃⁺ scan does not carry over
      to thermal configurations.
    - So ft1 was usable as a configuration generator, whose frames are all relabelled by DFT, but not as a
      model.
  - The scorer now builds its environment topology at Pb–I 4.3 Å, so thermal frames keep the axial and
    bridging iodides separate.
  - **ft2 staged** (not sent): `LONI_staging/chiral_ft2_mace_mba2pbi4`, hand-off `HANDOFF_TO_DISPATCH_chiral_ft2.md`.
    - Dataset v2.
    - Starts from the foundation model, like ft1.
    - Stage one: forces 100 / energy 1. Stage two from epoch 70: energy 1000 / forces 100.
    - 100 epochs, one gpu2 GPU, ≤ 1 GPU-h estimated.
    - Settings: `examples/chirality/finetune_overrides_ft2.json`.
- **ft2 back (2026-11-16): LONI package 133, job 1185342.**
  - Run: one A100, mace 0.3.15.
  - Model: `D:\MLIP_Custom_Models\MACE\mba2pbi4_chirality_ft2.model`, sha256 `c051e23a…`, verified. This is the
    stage-two model, from epoch 95.
  - Stage two from epoch 70 lowered the validation energy RMSE from 0.44 to 0.12 meV/atom, with forces held at
    4.5 meV/Å.
  - Scores: `data/cache/chirality_scores/ft2_on_v2.json` and `ft2_relax.json`, on dataset v2.
  - The held-out test is the rac NH₃⁺ scan plus the 54 D9 frames from independent MD seeds.

    | Criterion | ft1 | ft2 | Required | ft2 |
    |---|---|---|---|---|
    | Test forces, worst environment RMSE | 0.113 eV/Å (C) | 0.027 (Pb); C 0.026, N 0.025 | ≤ 0.05 | **pass** |
    | Test forces, largest single atom | 0.65 (C) | 0.16 (C), 0.14 (Pb), 0.13 (N), 0.12 (H on C) | "none > 0.08" | see below |
    | NH₃⁺ rotor, R / rac | RMSE 2.7 / 4.2 | 228 / 243 vs 223 / 237 meV; RMSE 2.0 / 2.4 | max(20 meV, 15 %); ≤ 10 | **pass** |
    | CH₃ rotor (valid, never trained) | 85 vs 162 meV | 158 vs 162; RMSE 1.9 | same | **pass** |
    | Ring torsion, rac | 313 vs 371; RMSE 23 | 363 vs 371; RMSE 3.8 | same | **pass** |
    | Ring torsion, R (rigid, includes a 3.1 eV clash point) | 2681 vs 3106; RMSE 161 | 2877 vs 3106; RMSE 91 | same | fail (barrier within 15 %, profile RMSE not) |
    | R − rac, fixed cells | 15.4 vs 51.1 | 41.2 vs 51.1 meV/f.u. | within 10 | **pass** (9.9) |
    | R − rac, free cells | 65.5 vs 75.0 | 75.1 vs 75.0 | within 10 | **pass** |
    | Relax from the DFT minima | exact | ≤ 0.0001 Å; net τ equal | same sign, ≤ 25 % | pass (training minima) |

  - **The clause "none > 0.08" is ambiguous.**
    - Read as a per-environment RMSE cap, which is how `mlip_vs_dft_chirality.py` reads it, ft2 passes.
    - Read as a per-atom maximum, which is how `score_chirality_mlip.py` reads it, ft2 fails: the largest errors
      on thermal frames are 0.12–0.16 eV/Å.
    - The text was written before any model existed and does not settle the question. It is the owner's call,
      recorded here, not decided by the analysis.
  - Reading:
    - The D8 thermal frames fixed what ft1 lacked. Held-out thermal force errors drop about 4×.
    - The never-trained CH₃ rotor is now reproduced to 2 meV RMSE.
    - Both R − rac energies pass.
    - What remains is per-atom error tails on thermal frames, and the rigid R ring clash point. Those 3.1 eV
      geometries are far outside any thermal sampling.
  - Gate status:
    - Under the per-environment reading, ft2 passes every criterion that bears on MD sampling at 173–358 K.
    - The full gate also needs the MD-stability items of the plan (NVE drift, stop conditions over long runs),
      which have not been run with ft2.
    - Production MD (~14 ns) remains NO-GO until the owner decides the "none > 0.08" reading and those stability
      checks pass.
  - **Owner decision (2026-11-16): "none > 0.08" is read per environment.** Recorded as
    `forces_heldout_reading` in `plan_matrices.json`; the criterion text itself is unchanged.
    `score_chirality_mlip.py` now gates on the environment RMSE and reports per-atom maxima without gating.
  - Under this reading **ft2 passes the held-out force criterion** (worst environment 0.027 eV/Å), the NH₃⁺, CH₃
    and rac ring profiles, and both R − rac energies.
  - Still open:
    - the rigid R ring profile (its 3.1 eV clash point);
    - the MD-stability items: NVE drift ≤ 1 meV/atom/ps and the stop conditions, over production-length runs in
      the production cell.
  - Production MD stays NO-GO until the stability check passes.
- **MD-stability check staged (2026-11-16, owner: "Yes, prepare it"), not sent:** `LONI_staging/chiral_stab1_mace_mba2pbi4`,
  hand-off `HANDOFF_TO_DISPATCH_chiral_stab1.md`.
  - Runner: `examples/chirality/stability/run_stability.py`. Generator: `make_chirality_stability_package.py`.
  - Units: R and rac in the production cell (2×2×1; 752 / 376 atoms) at 358 K.
  - Per unit: 20 ps NVT, then 10 ps NVE, with ft2 + D3(BJ, r2SCAN). On LONI the D3 runs as a socket client in the
    d3_torchdftd env.
  - Every 100 fs the plan's stop conditions are checked:
    - stereocentres;
    - Pb six-coordination at 4.3 Å;
    - NH₃⁺ identity, with organic bonds at 1.3 × covalent radii so a stretched N–H is not counted as broken;
    - proton transfer (ammonium H…I < 1.9 Å);
    - runaway force.
  - PASS also needs |NVE drift| ≤ 1 meV/atom/ps.
  - Ran end to end locally on the CPU with a tiny protocol. About 3–8 GPU-h on LONI.
- **MD-stability check back: LONI 137 (2026-11-17, A100 80GB).** The array was submitted twice: 1185807 and
  1186690. The desk reported only 1186690.
  - **rac_358K: PASS.**
    - Task 1185807_1 ran the full 30 ps with no stop condition. NVE drift was 0.0004 meV/atom/ps, and NVT ⟨T⟩ was
      366.7 K.
    - 1186690_1 then restarted from the step-55 800 frame while the first run was still writing. That left 21
      duplicate frames and 210 duplicate thermo rows in the same files.
    - The two tails are clean and diverge only by chaos (≤ 0.23 Å; E_tot within 1 meV).
    - Drift from either deduplicated copy is 0.0003 meV/atom/ps.
    - The largest 6th Pb–I distance is 4.27 Å, and the 7th iodide stays ≥ 5.37 Å.
  - **R_358K: STOP under the protocol as written.**
    - Task 1185807_0 stopped at step 41 000 (20.50 ps, the first 0.5 ps of NVE): "Pb 357 has 5 halides within
      4.3 Å".
    - The 6th Pb–I distance was **4.309 Å**, 9 mÅ past the cut. The 7th iodide was at 5.87 Å, so this is a stretch,
      not a lost iodide.
    - Earlier in the run, Pb 169 reached 4.299 Å at 14.4 ps. Eleven saved frames have a 6th Pb–I distance above
      4.1 Å. The 4.3 Å cut was set from earlier runs, whose longest stretch was 4.10 Å, so the margin is smaller
      than assumed.
    - 1186690_0 then resumed from that frame and ran to 30 ps. Its "PASS" (drift 0.0005 meV/atom/ps) is not the
      protocol's verdict, because the runner did not treat an earlier STOP as final.
    - No other stop condition fired anywhere in either unit: stereocentres, molecule/NH₃⁺ identity, ammonium
      H…I ≥ 1.9 Å, and forces ≤ 8.7 eV/Å.
  - Runner fix: `run_stability.py` now exits without running when `DONE` already holds a verdict, so a
    resubmission cannot run past a stop. Test: `test_runner_does_not_rerun_a_unit_with_a_verdict`.
  - **Gate status:** the stability item is not passed. R tripped the Pb coordination stop by 9 mÅ. Whether a
    6th-neighbour stretch with a ≥ 5.3 Å 7th iodide should count as loss of six-coordination is the owner's call.
    If it should not, the rule has to be redefined in advance and R rerun. The post-stop continuation cannot count
    as that rerun. Production MD stays NO-GO.
  - **Owner decision (2026-11-17): "No, redefine and rerun."** A stretched 6th Pb–I is not a loss of
    six-coordination.
    - New rule, set before the rerun: Pb is six-coordinate when exactly 6 iodides are within **4.6 Å**. The rule is
      recorded as `pb_six_coordination_reading` in `plan_matrices.json`.
    - Margins: 4.6 Å is 0.29 Å above the longest 6th Pb–I seen in 137 (4.31 Å) and 0.69 Å below the shortest 7th
      (5.29 Å). Every saved frame of both 137 runs passes it.
    - Test: a 6th Pb–I stretched to 4.4 Å passes, and one pulled to 4.8 Å stops.
    - `result.json` now records the margin (`pb_i_shell_A`).
    - rac stands as passed: it passed the stricter 4.3 Å rule.
  - **R rerun staged, not sent:** `LONI_staging/chiral_stab2_mace_mba2pbi4`, hand-off
    `HANDOFF_TO_DISPATCH_chiral_stab2.md`.
    - R_358K only, fresh run, same protocol and model as 137 apart from the cut.
    - About 14–15 GPU-h on an A100.
- **R rerun back: LONI 140 (2026-11-19, job 1189314, single submission, A100 80GB, 14 h 06 min). R_358K: PASS.**
  - Ran the full 30 ps (60 000 steps) with no stop condition under the 4.6 Å Pb rule.
  - NVE drift 0.0001 meV/atom/ps (1000 frames); NVT ⟨T⟩ 361.2 K; largest force 8.2 eV/Å.
  - Pb–I shell over the 300 saved frames: longest 6th 4.309 Å, shortest 7th 5.283 Å. The same seed reproduces the
    137 stretch at 20.5 ps, which now sits 0.29 Å inside the cut.
  - Independent recheck here of every saved frame: no stop condition; all 32 stereocentres stay R.
  - Caveat: this is one seed and the same starting velocities as 137, a stability test of the model, not an
    ensemble or a physical result.
  - **Gate status:** ft2 passes the MD-stability item for both R and rac at 358 K. Only the rigid R ring profile
    (the 3.1 eV clash point) remains open. Production MD stays NO-GO until the owner rules on that item.
  - **Owner decision (2026-11-19): "Accept, outside sampling."** The rigid R ring miss is driven by its 3.1 eV clash
    point, far above anything sampled at 173–358 K; it no longer blocks production MD. Recorded as
    `rigid_r_ring_reading` in `plan_matrices.json`.
  - **Gate status: ft2 + D3(BJ, r2SCAN) clears every gate item** (held-out forces per environment, rotor profiles
    with the ring item accepted, R − rac energies, MD stability for R and rac at 358 K).
  - Production MD is no longer blocked by the gate, but it is a new GPU campaign: nothing is staged or submitted
    without the owner's go-ahead. Its trajectories remain model predictions to be checked against DFT, not validated
    physics.

## 1. Hypothesis under test

> Spacer rigidity and persistent asymmetric organic–inorganic hydrogen bonding control how much handedness is
> transferred to a halide-perovskite framework at finite temperature. Interfacial water may disrupt this transfer
> before gross structural degradation becomes visible.

This is our proposed question, not a quotation; the LinkedIn post that prompted it does not state it. The dry
bulk mechanism is the primary test. Hydration is a conditional later extension.

## 2. Repository audit [R]

| Item | State |
|---|---|
| Repository | `lgutsev/PV_Passivator` (**public**). Work is on branch `chirality-feasibility` in worktree `PV_Passivator_chirality`, cut from `main` at `f39feef`. |
| Instructions | No `AGENTS.md` or `CLAUDE.md` in the repo. Followed the `docs/*.md` conventions, the tag legend and the "plan only, nothing staged" status lines. |
| Active work on `main` | Package 53 rebuild, revision 6: relax array running on LONI (job 1169113, `docs/PACKAGE53_REBUILD_PLAN.md`). Label set v3: draft plan. Discovery prototype. This branch touches none of them. |
| Structures | FAPbI₃ slab and passivant complexes only (package 53). Bulk CIFs in `D:\MLIP_Work_Folder\reference_structures` have no MBA or other chiral spacer. **Before this branch the repo had no chiral structures.** |
| Datasets | `data/prerelax_a0/dataset_v2`: r2SCAN+D3(BJ) relaxation frames of the package 53 complexes (FAI-Schottky slab with PEA⁺, TH⁺, TFPA⁺, PFPA⁺ salts). `label_grade: false`, pre-relaxer only. `data/labels_r2scan_v1–v3`: adsorption label sets. `data/bulk_refs_scanrvv10`: SCAN-rVV10, HYDRA profile, **must not be mixed**. None of these contain finite-temperature frames, rotor scans or chiral spacers. |
| Checkpoints | `NN_mace_pvpass_prerelax_a0`: pre-relaxer fine-tune package. Foundation models in `D:\MLIP_Foundational_Models`: MACE-matpes-r2scan-omat-ft, MACE-MPA-0, MACE-OFF23, GRACE, DPA-2/3, UMA. MACE-matpes-r2scan gives 0.07–0.085 eV/Å force RMSE near minima of FAPbI₃ slab complexes, so it is a pre-relaxer only. |
| Calculators | VASP r2SCAN+D3(BJ) on LONI, routed through the desk. MACE in the `maceh` venv on a local RTX 5070 laptop GPU. |
| Reusable code | `namd_adapter.py` (frame selection and continuity checks), `training_data.py` and the export/preflight scripts (energy/force export, split policy), `benchmark.py`, `vaspprep.py` (r2SCAN job prep). The new module reads ASE frames and does not add a second training framework. |

## 3. Prior art and novelty

| Work | Shown | Not shown |
|---|---|---|
| Pols, Brocks, Calero, Tao, *JPCL* **15**, 8057 (2024), doi:10.1021/acs.jpclett.4c01629 [V, main text only; SI returned 403] | (S)-, (R)- and rac-MBA₂PbI₄ bulk; on-the-fly MLFF trained on SCAN energies, forces and stresses of (S); NpT MD from 50 to 400 K. Vector-chirality descriptors for the cations, the in-plane framework and the out-of-plane framework, plus an H-bond asymmetry Δr_HB. Framework chirality falls faster than cation chirality: at 400 K χ_in-plane is 0.32 and χ_out-of-plane 0.05, while χ_cation stays at 0.97. NH₃⁺ reorientation from ~200 K; τ = 6.0 ps (chiral) vs 17.5 ps (rac) at 400 K. Static descriptor table for five more chiral spacers. | MD for any spacer other than MBA. No rigidity comparison, no intervention on headgroup motion, no water or surfaces. Main text gives no validation metrics, cell sizes, run lengths or dispersion treatment; these are in SI Notes 3 and 6, which we could not read. |
| Ramesh *et al.*, arXiv:2507.10036 (v3, 24 Aug 2026; no journal reference on the abstract page) [V, abstract only; PDF over the fetch size limit] | Experiment: circularly polarised transient absorption on (R)-, (S)- and rac-MBA₂PbI₄. A ~5.7 meV mode is present only in the chiral crystal; it is assigned to coupled organic libration and Pb–I bending, with transient CD whose sign reverses between R and S. Supported by ab-initio calculations. | Methods not read. No spacer series, no hydration, no finite-temperature structural statistics that we could verify. |
| Jana *et al.*, *Nat. Commun.* **11**, 4699 (2020); Son *et al.*, *Nat. Commun.* **14**, 3124 (2023); Apergi *et al.*, *JPCL* **14**, 11565 (2023) [as cited by Pols; not opened] | Static structural-chirality transfer; H-bond-isomer control of transfer; tight-binding CD. | — |

**What is not new:** temperature-dependent loss of framework chirality in bulk MBA₂PbI₄ (Pols 2024). Reproducing
it here is a **validation benchmark only**.

**Defensible new contribution** [P], in order of cost:
1. A controlled spacer comparison at matched halide, layer thickness, charge and anchoring group, asking whether
   framework chirality persistence follows rigidity or H-bond persistence. Confounding packing differences must
   be reported.
2. A headgroup-rotation restraint intervention with a control restraint. Nobody has reported this. It changes
   the Hamiltonian, so it is a mechanistic probe, not a material prediction.
3. Local-domain statistics: per-link signed distributions, spatial sign correlations, and racemate versus
   enantiopure. Published work reports global χ only.
4. Hydrated interfaces, conditional on 1–3.

**Experiments that could tell the predictions apart:**
- Temperature-dependent CD/ECD and X-ray (Pb–I–Pb angles, Δβ) on spacer pairs.
- Low-frequency Raman/THz of the ~5.7 meV coupled mode versus temperature: this mode should soften or
  broaden where we predict loss of coupling.
- Quasi-elastic neutron scattering or ²H NMR for NH₃⁺ rotation rates.
- In-situ humidity CD versus XRD, to show chirality loss before lattice degradation.

## 4. Capability and gap table

| Need | Have | Gap |
|---|---|---|
| Species H, C, N, Pb, I | [R] all present in r2SCAN slab labels | none in a 2D bulk or chiral context |
| Molecular topology | [R] PEA⁺, a structural isomer of MBA⁺ with an achiral CH₂CH₂ linker, bound to a 3D FAPbI₃ surface | no α-methyl stereocentre; no spacer layers of a stoichiometric 2D phase |
| Phases | [R] cubic-like FAPbI₃ slabs | no n = 1 RP bulk (P2₁2₁2₁ or P2₁/a) |
| Surface environments | [R] FAI-Schottky slab | not needed for the dry bulk pilot; needed for hydration |
| H-bonding N–H···I | [R] in relaxation frames near minima | no H-bond breaking or rearrangement sampling |
| Rotational barriers | none | NH₃⁺, ring-torsion and methyl-rotor scans (D3–D5) |
| Force/stress labels | [R] forces on slab relaxations; no bulk stresses | bulk stresses needed for NPT (D7) |
| Uncertainty calibration | none for this chemistry | committee or other uncertainty calibrated on D9 |
| Electronic properties | [R] slab dipoles only (`PET_ELECTRONIC_FEASIBILITY.md`) | DFT+SOC band splitting and CD: a separate stage, outside MLIP scope |
| Analysis code | [M] `chirality.py` (this branch) | tested on static crystals and synthetic fixtures only; no real trajectory yet |

**Which system can the current machinery represent?** It is built around surface-bound passivants on a 3D FAPbI₃
slab. A stoichiometric 2D spacer crystal is a different material. Chirality transfer in (MBA)₂PbI₄ says nothing
directly about a chiral passivant on FAPbI₃. The bulk benchmark is still useful to the project for three reasons:
- it tests the analysis and the r2SCAN MLIP route on a well-characterised crystal with known R/S/rac controls;
- its spacer chemistry is next to PEA⁺, which the project already labels;
- it gives the descriptor, H-bond and rotor validation recipe needed before any chiral-passivant claim on a 3D
  surface.

## 5. Smallest physical system and structure manifest [M]

Full machine-readable record: `docs/CHIRALITY_MBA2PBI4_MANIFEST.json`. CIFs are in
`examples/structures/mba2pbi4/` (COD data are public domain).

| | R | S (crystal) | rac | S (exact mirror) |
|---|---|---|---|---|
| Source | COD 7203475 | COD 7203476 | COD 2201646 | inversion of R, r → −r, cell unchanged |
| Citation | Billing & Lemmerer, *CrystEngComm* 2006, doi:10.1039/b606987h | same paper | Billing, *Acta Cryst. E* 58, m669 (2002), doi:10.1107/S1600536802019025 | — |
| Space group, T | P2₁2₁2₁, 173 K | P2₁2₁2₁, 173 K | P2₁/a (P2₁/c setting), 293 K | — |
| Cell (Å) | 8.868, 9.247, 28.729 | 8.853, 9.231, 28.650 | 8.794, 9.391, 14.643, β = 100.09° | as R |
| Flack | 0.095(16) | 0.057(13) | centrosymmetric | — |
| Z, atoms | 4, 188 (2 layers) | 4, 188 | 2, 94 (1 layer) | 188 |
| Stereocentres found | 8 R / 0 S | 0 R / 8 S | 2 R / 2 S | 0 R / 8 S |
| Pb–I (Å), Pb–I–Pb (°) | 3.171–3.346; 150.8, 156.4 | 3.166–3.344; 150.6, 156.5 | 3.190–3.323; 155.4 | as R |
| σ² cis-angle (deg²) | 18.3 | 18.9 | 14.6 | 18.3 |

Checks that passed for every structure:
- Stoichiometry (C₈H₁₂N)₂PbI₄ per formula unit, no disorder.
- Every N carries 3 H (NH₃⁺), so 2(+1) + PbI₄²⁻ = 0.
- Six-coordinate Pb with 2 axial and 4 bridging I.
- Corner-sharing n = 1 layers.
- Each cation has one stereocentre: the N-bearing α-carbon (bonded to N, H, CH₃ and phenyl).

COD names the cation "(R)-β-phenethylammonium". The connectivity shows it is 1-phenylethylammonium, i.e. MBA⁺.

H atoms are riding (constrained) positions in all three CIFs. The foundation MLIP gives fmax = 7.2 eV/Å on the
X-ray R cell. **None of these geometries is a relaxed or DFT structure.**

The racemate is centrosymmetric, but its per-link twists are **not** zero (τ_ax = ±0.104). Each layer
contains locally twisted links of both signs that cancel exactly. A racemic composition is therefore an
achiral-in-the-mean reference, not a structure free of local handedness.

**Exact mirror vs separately refined crystal.** The S crystal matches the inverted R after a translation:
- RMS (max) deviations: Pb 0.005 (0.007) Å, I 0.023 (0.037) Å, N 0.058 (0.060) Å, C 0.036 (0.100) Å.
- c differs by 0.08 Å.

That is small, but the net organic twist of the S crystal is +0.0145 against +0.0294 for the exact mirror.
MD and DFT must therefore use the **exact inversion** of whatever R structure is relaxed. Comparing the two
crystal refinements would mix refinement noise into an R/S difference.

**Second spacer** [P, not verified]: (S)-1-methylheptylammonium lead iodide ((S)-1-Me-HA₂PbI₄). It is listed in
Pols Table 1 and has the same halide, layer thickness n = 1, charge and NH₃⁺ anchor, with a flexible alkyl tail in
place of the rigid aryl. Expected confounds: different packing (alkyl interdigitation instead of aryl contacts)
and possibly a different space group. Its CIF provenance was not checked: the COD query failed on 8 Nov.
Do not start a molecular screen.

## 6. Descriptor definitions (implemented in `pvpassivator/chirality.py`)

All handedness descriptors are **link twists**, τ_ij = (v_i × v_j)·ê_ij, where ê_ij is the minimum-image unit
vector between neighbouring sites i and j.

| Descriptor | Sites and links | v |
|---|---|---|
| τ_ax (out-of-plane framework) | Pb–Pb links through a bridging I | axial trans axis X_ax(+) − X_ax(−), oriented along +n |
| τ_eq (in-plane framework) | same links | the equatorial trans axis that does not contain the bridging I, oriented along n × ê |
| τ_org (organic packing) | cations on the same face of the same layer, in-plane N–N < 6.5 Å, every periodic image | N − centroid(C) |

Properties, each enforced by a test:

| Property | Test |
|---|---|
| Invariant to atom order and to i ↔ j | `test_atom_order_does_not_matter` |
| Invariant to the sign of the layer normal | `test_normal_sign_convention_does_not_matter` |
| Invariant to proper rotation of cell and atoms | `test_proper_rotation_invariance` |
| Invariant to wrapping, per-atom lattice shifts and origin shifts | `test_wrapping_and_lattice_shifts_do_not_matter` |
| Same means in a 2×2×1 supercell | `test_supercell_gives_same_means` |
| Exact sign reversal under inversion and under a mirror | `test_inversion_*`, `test_mirror_plane_*` |
| Racemate: net zero, with signed ± partners | `test_racemate_*` |
| R and S crystals have opposite signs | `test_R_and_S_*` |
| Synthetic symmetric layer (untilted, and with an achiral in-plane rotation) gives zero | `test_synthetic_symmetric_layer_is_achiral` |
| Synthetic helix gives sin φ; a planar zig-zag alternates with mean ~0 | `test_synthetic_helix_and_zigzag_twists` |

Why link twists and not projections on the layer normal: a projection perpendicular to a chain direction
reverses when the walking direction reverses. It is polar, not chiral. Pols *et al.* project some of their
descriptors on n̂. Their exact vector sets are in SI Note 2, which we have not read, so **our numbers are not
comparable to theirs** until that is reconciled.

**Static values** [M] (sine of the twist, dimensionless):

| | per-link τ_ax | per-link τ_eq | per-link τ_org | net (cell mean) ax / eq / org |
|---|---|---|---|---|
| R | −0.013, +0.014 | +0.002, +0.011 | −0.271, +0.212 | +0.0002 / +0.0065 / −0.0294 |
| S crystal | −0.015, +0.014 | −0.011, −0.002 | −0.210, +0.239 | −0.0002 / −0.0063 / +0.0145 |
| rac | ±0.104 | ±0.006 | ±0.209 | 0 / 0 / 0 |

Reading:
- The net values are small differences between alternating per-link twists. The ordering
  |org| ≫ |eq| ≫ |ax| matches the qualitative ordering in Pols Table 1 (cation > in-plane > out-of-plane).
- The net τ_ax of the chiral crystal (2×10⁻⁴) is at the level of refinement noise. Out-of-plane framework
  chirality is not resolved in the static X-ray structure.
- Finite-temperature work must therefore track **signed per-link distributions and their block-averaged means**,
  not only a global χ.

Other observables implemented:
- Pb–I–Pb angles; octahedral Δd and σ².
- C→N headgroup tilt; N–H vectors with P₁/P₂ rotational ACFs (Pols Eq. 4 form).
- N–H···I occupancy, continuous S(t) and intermittent C(t) H-bond lifetimes; per-cation r_HB (two shortest H···I).
- Face asymmetry of r_HB. This is polar, not chiral, and is reported only for comparison.
- Block averaging (Flyvbjerg–Petersen); Sokal τ_int; cross-correlation with lag (supporting evidence only);
  sign census χ (our operational definition); spatial sign correlation for domains.

**H-bond definition sensitivity** [M]: on the static R crystal, H-bonds per N–H are:

| r_max (H···I) | H-bonds per N–H |
|---|---|
| 2.8 Å | 0.17 |
| 3.0 Å | 0.67 |
| 3.2 Å | 1.0 |
| 3.4 Å | 0.83–1.33 (depends on the angle cut-off) |

The riding H positions make this worse. Occupancies and lifetimes must be reported over the criteria grid
(`hbond_sensitivity`), never at a single cut-off.

## 7. Missing inputs

1. Pols *et al.* SI (Notes 2, 3, 6, 7): descriptor vector sets, χ formula, MLFF validation, MD lengths and cells,
   SCAN/dispersion/SOC settings. Needed to reconcile the descriptors and to avoid repeating their benchmark blindly.
2. Ramesh *et al.* full text (methods).
3. Phase-transition and decomposition data for the chiral phases between 173 and 358 K. Only the racemate's
   melting point, 166.5 °C, is verified.
4. Any DFT label of a 2D chiral spacer crystal at the project profile: none exist.
5. A verified CIF for the second spacer.
6. Restraint implementation (ASE constraint on NH₃⁺ and methyl dihedrals) and a measured MD throughput on the
   target HPC hardware.

## 8. Prepared plans (not submitted)

`examples/chirality/plan_matrices.json` holds the full matrices. In summary:

**DFT** (r2SCAN+D3(BJ), project INCAR family, bulk cells without a dipole correction, SOC off with a check):
- D0: convergence, a timing single point, and an SOC on/off check.
- D1: R and rac relaxations at fixed and free cell.
- D2: mirror single point.
- D3–D5: NH₃⁺, ring and methyl rotor scans.
- D6: tilt/twist distortions.
- D7: strain.
- D8: 180 finite-T frames.
- D9: a 60-frame held-out test set, split by trajectory.
- Cost estimate [I]: about 150–500 node-h. **First tranche for approval: D0–D2, about 30–120 node-h.**

**MLIP acceptance**, fixed now, before any production run:
- Held-out force RMSE ≤ 0.05 eV/Å per environment (none > 0.08).
- Every net τ of the relaxed MLIP structure has the DFT sign and lies within 25 % of the DFT value; per-link τ
  within 0.02.
- Rotor barriers within max(20 meV, 15 %).
- R vs rac energy within 10 meV/f.u.
- Stress within 0.3 GPa.
- The R/S mirror equality is automatic for an O(3)-invariant model: reported, but not counted as evidence.

**MD design:**
- Conditions: R, exact-mirror S and rac at 173, 293 and 358 K, 3 seeds each.
- Cells: 2×2×1 (752 / 376 atoms); the decisive comparison repeated in 4×4×1.
- Protocol: 25 ps NPT, then 250 ps NVT at the mean cell, then NVE segments for correlation functions.
- Controls: matched-cell runs; NH₃⁺ restraint at three strengths plus a methyl-rotor control restraint.
- Stop conditions are listed in the plan file.
- About 14 ns in total.
- Measured [M]: 0.38 s per step for 752 atoms in float32 on the laptop GPU, so ≈ 58 h per run and ~3500–4000
  laptop-GPU-h in total. Benchmark the HPC throughput before any request.

## 9. Risks and go/no-go

| Risk | Consequence |
|---|---|
| No validated MLIP for this chemistry. Foundation model error near minima (0.07–0.085 eV/Å) is comparable to the forces that set 1–3° twists. | Any MD now would be a stable-looking trajectory with no physical standing. |
| Net framework handedness is tiny in the static structure (τ_eq ≈ 0.006, τ_ax ≈ 2×10⁻⁴). | The temperature trend may sit below model error. That outcome would be **inconclusive**, not a falsification. |
| Bulk benchmark ≠ passivant on 3D FAPbI₃. | Success transfers to the analysis and validation recipe, not to passivation claims. |
| H-bond statistics depend on the definition. | Report over the criteria grid. |
| Benchmark overlaps Pols 2024. | Novelty rests on the spacer comparison and the intervention, both of which need the validated model first. |

**Decision.**
- **GO:** the analysis module and the R / exact-S / rac structure set are ready, and the DFT tranche D0–D2 is
  ready to stage once the owner approves and the desk assigns a package number.
- **NO-GO:** MD production, fine-tuning, the spacer comparison and hydration. They wait until D1 minima exist
  and a fine-tuned model passes the criteria in §8 on the held-out set.
- If the fine-tuned model fails, the next step is the smallest extra-label proposal, not a positive feasibility
  claim.
