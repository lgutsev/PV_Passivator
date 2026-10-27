# Package 53: why the slab relaxes are slow, and a consistency problem (2026-10-20)

This uses export 20261020_1107 (revision 4), reading `OUTCAR.seg1` and `OUTCAR` of the slab-type relaxes.
- The slabs carry no selective dynamics, so the max force is taken over all atoms.
- The target is EDIFFG = −0.02 eV/Å, with IBRION = 2 (CG) and the default POTIM.
- Each ionic step takes about 0.5 h on 2 nodes.

## 1. The structures are still relaxing, not stuck

| Job | Steps so far | Fmax now (eV/Å) | Atoms > 0.02 | E drop, last 10 / 50 steps (meV) | E above final, from step 0 (eV) |
|---|---:|---:|---:|---:|---:|
| slab_FAI_Schot | 164 | 0.115 | 382 | −109 / −629 | 2.13 |
| complex_PEAAc | 174 | 0.134 | 188 | −52 / −225 | 1.69 |
| complex_TFPACl | 147 | 0.396 | 394 | −144 / −619 | 1.84 |
| complex_THCl | 170 | 0.249 | 237 | −39 / −153 | 1.34 |

- The energy falls steadily, by about 4–15 meV per step. The structures are not oscillating near a minimum, so a looser EDIFFG would not rescue the energies.
- In the bare slab, the largest forces sit on FA C/H throughout the slab, up to about 10 Å below the surface. The mean Pb–I–Pb tilt grew from 4.4° to 5.9°, and the mean displacement is 0.39 Å.
- The whole slab is reorganising: FA reorientation and octahedral tilting in the remapped cell. A pseudo-cubic α-FAPbI₃ slab at 0 K can keep finding lower tilt/FA patterns, which CG explores slowly.
- In the complexes the adsorbate dominates the largest forces, but hundreds of FA and I atoms are still above 0.02 eV/Å.

## 2. Most complexes do not share the bare slab's spectator state

Method:
- Align each complex onto the bare slab with the best rigid translation of the Pb sublattice.
- Assign atoms one-to-one per element (Hungarian).
- Compare atoms more than 10 Å from the adsorbate.

| Complex | Pb/I > 10 Å: mean mismatch at start → now (Å) | FA N/C > 10 Å: start → now (Å) |
|---|---|---|
| TFPACl | 0.04 → 0.22 | 0.06 → 0.30 |
| TFPAI | 0.04 → 0.15 | 0.06 → 0.23 |
| PEAI | 0.28 → 0.33 | 0.33 → 0.37 |
| PEACl | 0.36 → 0.36 | 0.45 → 0.43 |
| THCl | 0.54 → 0.47 | 0.64 → 0.56 |
| PFPAI | 0.61 → 0.52 | 0.78 → 0.66 |
| PEAAc | 0.61 → 0.77 | 0.84 → 1.21 |
| PEATFA | 0.65 → 0.78 | 0.88 → 1.22 |

What this shows:
- Only the two TFPA complexes were built on the same slab geometry. The other seven come from separate archived runs, with a different tilt/FA pattern far from the adsorbate.
- Relaxation does not bring them together; PEAAc and PEATFA are drifting further apart.
- For those seven, E_ads = E(complex) − E(slab) − E(salt) includes the energy difference between two spectator states. Given the 1–2 eV that the bare slab alone has dropped during relaxation, that offset could be large, and it would remain even at full convergence. **As built, these seven E_ads would not be reliable labels.** The relaxed geometries and their forces are still valid training frames.

## 3. Options for the owner

- **A. Rebuild the complexes on one slab, the recommended fix for the labels.**
  - Converge the bare slab first.
  - Transplant each salt, with its local contact geometry from the archived complex (and the V_FA/V_I fill for the Schottky cases), onto that relaxed slab, so the spectators are identical by construction. This is how TFPACl and TFPAI were already built.
  - Each complex relax then starts close to a minimum, with only a local rearrangement left.
- **B. Faster bare-slab convergence.**
  - Pre-relax with an MLIP, then let r2SCAN finish.
  - Or switch the optimizer for the remaining far-from-minimum stage: FIRE (IBRION = 3, POTIM = 0, IOPT = 7 if the VTST build has it) or a damped MD quench, then CG/RMM-DIIS near the end.
  - Freezing the bottom Pb layers with selective dynamics would cut the degrees of freedom. v2 relaxes every atom, so freezing breaks that convention.
- **C. Hold further rounds of the current array.**
  - Keep running segments, so nothing is lost, but don't queue more 72 h rounds for the seven mismatched complexes until A is decided.
  - Each round is about 10 × 2 nodes × 72 h ≈ 1,440 node-h, and the forces are still 6–20× the target after 150–170 steps.
  - The bare slab and the two TFPA complexes stay useful under every option.

The same lesson applies to label set v3. Its v3a cases already derive from package-53 complexes, and its v3b complexes are to be built on one PbI₂ slab, so both should be built on the *relaxed* bare slab.

Scripts used (session scratchpad, not committed): `relax_diag.py` (per-step energy and forces), `spectator2.py` (rigid alignment and spectator mismatch).
