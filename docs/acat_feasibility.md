# Adsorption-arrangement search with ACAT: feasibility, pilot and benchmark specification (2026-11-09)

Branch `acat-adsorption-search` (worktree `Test_Code/PV_Passivator_acat`, from `main` 5297f43, not pushed).

**Status in one paragraph.** A project-native generator for fixed-composition arrangements of N passivant molecules on one slab face is implemented and tested (`src/pvpassivator/adsearch.py`, 17 software checks). ACAT 2.0.3 was installed and probed on the real slab: **ACAT's adsorbate bookkeeping and GA operators are a no-go for this surface**; its site finder works only on the bare Pb sublattice with hand-given layers and a NumPy shim, and its graph comparator runs but is not needed. The pilot system is chosen and a valid candidate set is exported. **No search-performance result and no physical ranking exists**: the only scores run so far are a made-up software-check score and a handful of relaxations with an unvalidated foundation MLIP (plumbing and throughput). The next step that needs a decision is a DFT validation tranche (§8).

Tags: **[V]** verified here by running it, **[U]** present but untested for this use, **[M]** missing input.

## 1. Sources and versions inspected

- ACAT documentation (index, `ga.html`, `notes.html`, `settings.html`, `build.html`, `installation.html`), read 2026-11-09; header "ACAT 2.0.3". The PyPI page and the GitLab README could not be fetched.
- Installed and run: **ACAT 2.0.3, ASE 3.29.0, ase-ga 1.0.3, autoadsorbate 0.2.5, NetworkX 3.6.1, NumPy 2.4.4, Python 3.14.4, Windows 11**, in a throw-away venv (not a project dependency). An environment with the same ACAT/ase-ga versions already exists at `D:\MLIP_Work_Folder\envs\acat_fe16` (another project's).
- Version findings the documentation does not state **[V]**:
  - `acat.ga.adsorbate_operators` fails to import on ASE 3.29 until `ase-ga` is installed, because `ase.ga` moved out of ASE.
  - `CustomSurface` site finding imports `autoadsorbate`, an undeclared dependency.
  - ACAT 2.0.3 calls `numpy.in1d`, which NumPy 2.4 removed; the probe aliases it to `numpy.isin` and records that it did.
  - `CustomSurface(n_layers=None)` (automatic layer detection) did not return on this slab in any representation; the layer count has to be given.
  - `asap3` (listed as required) is not installed; nothing probed here needed it. It is needed for the surrogate-metal/EMT and CNA paths, which are metal-only anyway. It was not built on Windows.
- Han et al., npj Comput. Mater. 9, 139 (2023) and Yu et al., Adv. Mater. Interfaces (2025) were not re-read for this checkpoint; nothing below depends on a number from them.
- `PV_Passivator_Chirality_Worker_Assignment.md` was not found (repo, worktrees, Downloads). The chirality branch's `chirality.py` (CIP descriptors, `invert`) is unmerged; see §5 for what is shared.

## 2. Repository audit

`main` at 5297f43, clean, 14 commits ahead of `origin/main`; other worktrees: `chirality-feasibility`, `mlip-pbsn-gb`. No `AGENTS.md` or `CLAUDE.md` in the repository. Current priority on `main`: package 53 revision 6 (frozen-far-field relaxes of nine salt complexes on the FAI Schottky slab, running on the cluster).

| Topic | State | Evidence |
|---|---|---|
| Slabs | **[V]** α-FAPbI₃ (100), 3×3 in-plane, normal along **a/x**. PbI₂-terminated 567 atoms (45 FA⁺, 54 Pb²⁺, 153 I⁻, formal charge 0), both faces PbI₂, 9 five-coordinate Pb per face, 44.5 Å vacuum, 32.5 Å thick. FAI-terminated 513 atoms; FAI Schottky 504 atoms | `surface.load_surface`, `WORKFLOW_AUDIT.md` §2 |
| Defects, cell charge | **[V]** Schottky removal with formal-charge ledger (`poses.remove_fragments`); charged cells exist only in the legacy PBE archive (`_NELECT` folders). The pilot uses none | `ledger.py`, tests |
| Passivant identity, protonation, charge | **[V]** `poses.Passivator` requires declared charge, multiplicity and per-fragment formal charges; counterions are never dropped | `poses.load_passivator` |
| Topology, stereochemistry at construction | **[V] new**: bond list and labelled chirality of every conformer and every built molecule (`adsearch.bond_list`, `labelled_chirality`). **[U]** CIP R/S labels: only on the unmerged chirality branch | §5, tests |
| Binding atoms, anchoring modes | **[V] new**: documented binding-mode library per molecule (`adsearch.BindingMode`), including a bidentate form with a second-anchor distance check. **[M]** no bidentate mode has been exercised on a real molecule | `examples/adsearch/melamine2_pbi2.json` |
| Site finding | **[V]** `surface.find_sites`: under-coordinated Pb, exposed halide (in-plane/apical/subsurface), FA | existing tests |
| Random / sequential initialization | **[V]** `poses.generate_poses`: one molecule, systematic site × anchor × height × tilt × azimuth, shuffled, covalent-radius clash filter. **No multi-molecule placement and no structure deduplication existed before this branch** | `poses.py` |
| Atom identity through relaxation | **[V]** `atom_id` / `fragment_id` / `origin` arrays; `classify_final_state` compares fragments by id (intact / proton transfer / reactive rearrangement / desorbed) | `poses.py` |
| DFT labels, target profile r2SCAN+D3(BJ) | **[M] for any multi-molecule arrangement.** Package 53 (nine salts, FAI Schottky slab, one molecule each) is still relaxing. The PbI₂-face Lewis-base set (label set v3b) is a draft, not built. Legacy PBE+D3(zero) energies exist for single and double melamine but are ISIF=4, mostly unconverged (NSW reached), and a different functional | `PACKAGE53_REBUILD_PLAN.md`, `data/labels_r2scan_v3/PLAN.md` |
| Checkpoints | **[V]** foundation `MACE-matpes-r2scan-omat-ft` + torch-dftd D3(BJ, r2SCAN), `maceh` env (MACE 0.3.16, RTX 5070 8 GB). Near the r2SCAN minimum its force error exceeds the residual DFT forces (RMSE 0.074–0.085 eV/Å, cos 0.60–0.66 on the salt complexes). **[M]** fine-tuned model: A0 pre-relaxer is a draft package, not trained. **[M]** any model validated on a PbI₂-face molecular interface, or on melamine | rebuild plan, "Foundation MLIP test" and A0 |
| ASE calculator interface | **[V]** `local_settle.make_calc` (MACE + D3 `SumCalculator`); reused unchanged | `examples/scripts/local_settle.py` |
| Uncertainty calibration | **[M]** no committee, no calibration data | – |
| Provenance conventions | **[V]** sha256 of inputs, settings fingerprints, JSON manifests, git-ignored `data/cache`; followed here | `vaspprep.py`, `labels.py` |

## 3. What ACAT does with this surface (`examples/scripts/acat_surface_probe.py`)

Input: the neutral r2SCAN PbI₂ slab (567 atoms, sha `51a2d7560eeb9fcd`) rotated so the normal is z (ACAT assumes z), and the legacy two-melamine complex `MelMel_C`. Each ACAT path runs in its own process with a time limit; the full report is `data/cache/acat_probe/pbi2_melmel.json` (git-ignored, regenerate with the script).

| ACAT path | Result **[V]** |
|---|---|
| Element rule (`acat.settings.adsorbate_elements` = H, C, N, O, F, S) | On the **bare** slab ACAT counts 207 catalyst atoms (Pb, I) and **360 adsorbate atoms**: all 45 C, 90 N and 225 H of lattice formamidinium. The project's roles say 567 substrate, 0 adsorbate |
| `CustomSurface` + `SlabAdsorptionSites`, Pb sublattice only (54 atoms), `n_layers=6` given by hand | Works in 0.04 s, after one shim (ACAT calls `numpy.in1d`, removed in NumPy 2.4). 36 sites: 9 on-top, 18 bridge, 9 four-fold. All 9 project-native under-coordinated Pb sites have an ACAT on-top site. The bridge and four-fold sites sit where the surface iodides are |
| Same, Pb + I (207 atoms) | With `n_layers` given: stops with "The total number of planes is not a multiple of n_layers". ACAT requires every layer to hold as many atoms as the surface layer; PbI₂ and FAI/I planes alternate. With automatic layer detection: no result within 120 s (and none within 15 min in an earlier run, at 2 GB) |
| Same, all 567 atoms | No result within 120 s |
| `SlabAdsorbateCoverage`, bare slab, default; and complex with explicit `adsorbate_indices` | No result within 120 s in either: both first build the site list above. The explicit-index argument therefore could not be shown to repair the bookkeeping |
| GA operators (`MoveAdsorbate`, …) | Constructible with `catalyst_indices`, but adsorbates are **names** resolved by `acat.settings.adsorbate_molecule` / `ase.build.molecule`: "C3H6N6" and "melamine" return nothing. An arbitrary `ase.Atoms` molecule, a binding-mode library or a conformer cannot be passed. Heights are per site type for small adsorbates. Not run on the slab, because they need the coverage object above |
| `WLGraphComparator` | Runs on the 597-atom complex (2.7 s per comparison) and merges a translated, wrapped copy with the original. It was not shown to separate binding modes or enantiomers; a graph comparator cannot see chirality |


**ACAT-specific verdict: no-go for adsorbate bookkeeping and the stock GA operators; site generation works only on the bare Pb sublattice, where it adds nothing to `find_sites`; the comparator is optional.** Making ACAT work would mean overriding its element rule in every module, supplying a hand-made site list in its internal format and re-implementing the operators for intact molecules, which is the adapter below without the benefit. The assignment's fallback applies: keep the explicit project-native site list. The evolutionary arm is therefore "ACAT-style" (same operator families: move, rotate, replace mode, cut-and-splice of whole adsorbates) but contains no ACAT code. If a like-for-like ACAT run is wanted later, the only realistic entry point is a small-adsorbate problem on a Pb/I-only representation, which is a different scientific question.

## 4. Pilot system

**PbI₂-terminated α-FAPbI₃ (100) 3×3 slab + 2 melamine (C₃H₆N₆) on one face.** Config: `examples/adsearch/melamine2_pbi2.json`.

- **Why this one.** It is the only family in the project with existing same-composition multi-molecule arrangements: three two-melamine structures on the PbI₂ face (`MelMel`, `_B`, `_C`) and two on the FAI face, plus single-melamine structures, all legacy PBE. The slab is the label-set-v3b slab and exists in the target profile. Melamine is neutral, closed-shell, rigid and achiral: no counterion, no protonation choice, no conformer problem in the first pilot. The cell is 19.4 × 18.9 Å, so two molecules have inequivalent arrangements (adjacent, diagonal, further) with image distances ≥ 12 Å for adjacent pairs.
- **What it costs.** Model coverage is worse than for the "best-supported" pair. The best-supported pair in the repository is FAI Schottky slab + PEA salts (package 53), but that slab has one vacancy pair, so a second salt would change the slab and the defect population, not just the arrangement. PEAI on the PbI₂ face is the alternative if the owner prefers salt chemistry: the generator handles ion pairs (counterion role), but it needs a conformer library and the proton-transfer question, so it is the second system, not the first.
- **Not substituted:** the chiral 2D spacer work (MBA₂PbI₄ bulk) is a different question and is not used here.
- **Stage 1, single molecule** (9 sites × 3 modes × orientations, construction only): ring-N end-on 270/324 valid (54 slab clashes), amine-N end-on 324/324, flat 36/36. Anchor distances at construction 2.6–2.7 Å; legacy relaxed N–Pb is 2.58–2.69 Å.
- **Stage 2, fixed composition:** N = 2, neutral cell, same 567 slab atoms in every candidate. Coverage: **2 molecules per 365.3 Å² = 0.55 nm⁻²**; occupied sites 2 of 9 under the convention "one site per anchoring atom declared by the binding mode" (sites merely shadowed by a molecule are not counted; a bidentate mode counts 2). Distinct candidates on the 30° grid: 76 states per molecule, 23,104 arrangements after removing surface translations and molecule order.
- A second count (N = 3 or 4) is deliberately not set up.

## 5. Candidate representation, operators, filters (implemented)

- **Roles.** Per-atom `role` (substrate, defect, passivant, counterion, solvent) and `mol_id`, set once and never rewritten; `atom_id` as in `poses`. Operators act on genes (site, mode, conformer, azimuth, tilt); the substrate is copied verbatim into every candidate, so no operator can move, splice or delete a lattice atom. `substrate_integrity` checks ids, elements, roles and count after construction (bit-identical positions) and after relaxation (no heavy lattice atom more than 2.5 Å above either face).
- **Operators:** move an intact molecule to a free site; rotate about the normal / change tilt within the mode's bounded list; switch binding mode or conformer; exchange the sites of two molecules; recombine complete molecules from two parents with the count fixed.
- **Filters, each counted by reason:** site conflict; anchor–site distance window (and second anchor for bidentate modes); passivant–substrate contacts below the sum of covalent radii (declared anchor contact exempt); molecule–molecule and molecule–own-image contacts below 1.3 × that sum, under PBC; buried (below the outer inorganic layer); too close to the opposite face's image; bonding changed; labelled centre inverted.
- **Stereochemistry.** A conformer whose bonding differs, or whose labelled chirality sign flips at any non-planar centre, is refused at library construction ("declare it as a separate molecule"). This is a construction check on atom labels, stricter than CIP. Shared with the chirality work: the atom-id convention and the idea of never comparing separately built enantiomers. Not shared yet: CIP labels, because `chirality.py` is unmerged.
- **Reactions during relaxation** are not repaired: `classify_final_state` labels proton transfer, rearrangement and desorption; only `intact_adsorption` structures that converged enter the ranking, the others stay in the log with their label.
- **Deduplication.** Before evaluation: canonical gene key (smallest over surface translations and molecule order, azimuth folded by the mode's period). After relaxation: two structures are the same if one surface translation and one pairing of molecules puts every passivant atom on a same-element atom within 0.5 Å (minimum image). Matching is by position, not label, so a molecule turned onto itself is a duplicate; mirror images and different binding modes are not.
- **Surface symmetry used:** the 9 in-plane translations of the site lattice. On the r2SCAN slab they map the whole substrate, FA included, to within 0.001 Å (the archive relax kept the 1×1 periodicity), so here they are true symmetries. On a slab with disordered FA the code reports the residuals and the merge becomes approximate (`symmetry: "none"` turns it off). Point-group operations are not used.

## 6. Local checks run

- `pytest`: **164 passed, 1 skipped** (the skip is the ACAT-dependent check, which passes in the ACAT venv). 17 new checks in `tests/test_adsearch.py`, all on a synthetic toy slab and toy molecules: substrate and roles survive construction and every operator; damage is detected (missing atom, relabelled role, extracted lattice atom); roles from a reference slab keep lattice FA as substrate and find a counterion; every rejection reason; clashes through the periodic boundary and against the molecule's own image; other stereoisomer / broken bond refused; duplicates found when repeated, translated, wrapped, reordered and self-rotated; distinct site separation, mode and enantiomer kept apart; budget accounting; identical starting proposals across arms; same seed → same log; relax evaluator freezes the far field and counts calls; manifest hash survives a file round trip.
- **Site/role demonstration on the real slab** (`adsearch_pilot.py demo`): labelled bare slab with 9 site markers, single-molecule table, 12 valid two-melamine candidates from 14 attempts (1 molecule clash, 2 slab clashes), each with genes, hash and bit-identical substrate. Output `data/cache/adsearch/demo_melamine2` (git-ignored; regenerate with the command in §9).
- **Loop and accounting on the real slab with the made-up score** (200 attempts × 3 seeds × 2 arms): every attempt accounted for. Random arm: 157–169 unique, 24–41 rejected, 2–7 duplicates. Evolutionary arm: 115–129 unique, 13–15 rejected, 56–72 duplicate proposals. Generator cost ≈ 0.06 s per build; the whole loop 0.3–0.6 s per attempt. **This is a software check. It is not evidence about which arm searches better**, and the duplicate rate of the evolutionary arm on a real energy surface is unknown (one cause, a rotate operator that could redraw the same orientation, was removed afterwards).

- **Single-molecule sanity stage with the foundation model** (MACE-matpes-r2scan-omat-ft + D3(BJ), FIRE to 0.1 eV/Å, molecule + atoms within 8 Å mobile; RTX 5070). One melamine on the central Pb site, one start per binding mode:

  | Start mode | Steps | Wall (s) | Final state | Closest N–Pb (Å) | Closest H···I (Å) | Largest move, molecule / slab (Å) |
  |---|---:|---:|---|---:|---:|---|
  | ring N end-on | 28 | 330 | intact, slab intact | 2.56 | 2.56 | 0.11 / 0.13 |
  | amine N end-on | 59 | 644 | intact, slab intact | 3.06 | 3.09 | 0.28 / 0.27 |
  | flat | 36 | 394 | intact, slab intact | 3.59 | 3.16 | 0.16 / 0.13 |

  All three converge, keep their mode, keep every atom id and role, and leave the lattice in place. The ring-N contact (2.56 Å) matches the legacy relaxed complexes (2.58–2.69 Å), and the flat start keeps the 3.6–3.7 Å separation of the flat molecule in legacy `MelMel`. **The model energies are not reported as a ranking**: the model is unvalidated here, and these runs used the molecule-centred mobile set, so the three minima are not under identical constraints. That weakness is why the ranking default is now the face-wide mobile set (`mobile: "face"`: the same 135 slab atoms + the molecules for every candidate, 165 of 597 atoms).
- **Throughput measured:** 11–18 s per energy/force call for 582–597 atoms on the 8 GB laptop GPU (memory 7.9 GB, at the limit; the slower figure was measured while other jobs shared the machine), 29–60 calls per relaxation, so **5.5–11 min per candidate**.
- **Two-molecule plumbing run, same model:** two ring-N end-on melamines on adjacent Pb sites converged in 35 steps (652 s, 137 mobile atoms): both N–Pb contacts kept (2.58 Å), state intact, slab intact, largest moves 0.25 Å (molecule) and 0.32 Å (slab). Two further random two-molecule candidates were still running when this note was written; their output lands in `data/cache/adsearch/mace_sanity/mace_sanity.json`.

## 7. Equal-budget benchmark specification (not run)

| Item | Setting |
|---|---|
| Problem | 2 melamine, PbI₂ slab bottom face, config above |
| Arms | (1) random placement; (2) evolutionary (operators of §5, rank-weighted parents from the 10 best unique ranked structures) |
| Budget | 200 **attempts** per arm per seed; rejected and duplicate proposals consume it |
| Seeds | 3 per arm; for one seed both arms start from the same 20 proposals (same generator stream) |
| Relaxation | identical for both arms: FIRE, fmax 0.1 eV/Å, ≤ 150 steps; mobile = molecules + every slab atom within 8 Å of the face's outer layer (the same 135 slab atoms for every candidate), lattice molecules whole; the rest fixed at the bare-slab positions |
| Ranked | converged, substrate intact, `intact_adsorption`; everything else logged with its label |
| Recorded per attempt | arm, seed, operator, parents, genes, validity details, status, energy/force calls, wall time, hashes |
| Reported | best energy vs cumulative attempts and vs cumulative calls; unique ranked structures within 0.10 eV of the best and their motifs; invalid and duplicate fractions; spread over seeds; total calls and wall time |
| Claim rule | no search-performance claim unless the calculator has passed §8 on this system, and never from one seed |

**Cost of running it with the foundation model on the laptop GPU:** about 115–170 evaluated candidates per run in the software check, so roughly 700–1,000 relaxations for 2 arms × 3 seeds, at 5.5–11 min each: **65–180 GPU-hours** (more with the larger face-wide mobile set). That is a substantial GPU campaign under the assignment's rule, so it is specified, not launched. The budget should be re-set from the throughput of whichever model passes §8.

## 8. MLIP validation and the DFT proposal (nothing staged, nothing submitted)

**Why DFT comes before the benchmark.** No model here has seen a molecule on the PbI₂ face, and the foundation model is already known to mis-place the r2SCAN minimum of the salt complexes. An equal-budget comparison on its energy surface would measure the search on the wrong landscape.

**Energy resolution needed, fixed now.** The question is the ordering of arrangements at fixed composition. The three legacy two-melamine structures span 0.92 eV (PBE, unconverged, so only an order of magnitude). Target: resolve differences of **0.10 eV per cell** (0.05 eV per molecule).

**Acceptance thresholds, fixed before any test result exists:**

| Quantity | Threshold |
|---|---|
| Relative energy among same-composition structures, family-separated hold-out | MAE ≤ 0.10 eV and worst case ≤ 0.20 eV |
| Rank agreement on ≥ 10 held-out structures spanning ≥ 0.5 eV | Spearman ρ ≥ 0.8; at least 2 of the DFT top 3 in the model's top 3 |
| Forces on passivant atoms and on substrate atoms within 4 Å of them (reported separately from the far field) | RMSE ≤ 0.05 eV/Å, cos ≥ 0.9 for \|F\| > 0.05 eV/Å |
| Binding-mode stability | DFT relax from the model minimum keeps the mode; anchor N–Pb changes by ≤ 0.15 Å |

"Family" = all frames descended from one starting arrangement; neighbouring relaxation frames never count as independent. Structures used for fitting stay out of the final audit set. Committee agreement, if a committee is trained, flags extrapolation and is not treated as accuracy.

**Reference protocol** (the project's `r2scan_d3bj_v1` profile): VASP, METAGGA = R2SCAN, IVDW = 12 with s8 = 0.7898, a1 = 0.4948, a2 = 5.7308, ENCUT 520, ISPIN 1, neutral cell, Γ only, the v2 common cell (70.0 × 19.4203 × 18.9076 Å), dipole correction along the normal with one pinned centre, no cell relaxation. One deliberate difference from the label profile: **`LREAL = .FALSE.`** instead of `Auto`, because the audit compares forces and package 108 showed that `Auto` shifts forces by up to about 12 meV/Å at 520 eV on these elements. Energies of this set must therefore not be pooled with label sets run with `Auto`.

**Tranche V1, statics only: prepared, not submitted** (owner: "Prepare V1", 2026-11-09). 22 single points at exactly the geometries where the model's energy and forces are stored, so the comparison is atom by atom and needs no DFT relaxation:

| Group | n | What |
|---|---|---|
| single | 3 | one melamine in each binding mode, model-relaxed |
| legacy | 3 | the three archived two-melamine arrangements transplanted to this slab, model-relaxed |
| pool | 11 | random two-melamine arrangements (seed 0, 13 attempts, 2 clash rejections), model-relaxed; **not** selected by model energy |
| asbuilt | 3 | unrelaxed starting geometries of pool_00–02 (forces up to 11 eV/Å: a far-from-minimum test) |
| reference | 2 | bare slab (same cell), isolated molecule |

Changes from the plan first proposed here, and why. (1) The pool is unselected instead of "5 lowest model minima + alternatives + controls": choosing by the model's own ranking would bias the audit toward where the model is self-consistent, and no search has been run at a budget that could produce credible minima. (2) The slab's face region (135 atoms) was relaxed once with the model before any molecule was placed (44 steps, −0.42 eV, largest move 0.15 Å); otherwise every candidate relaxation spends most of its steps re-relaxing the same slab. The slab is therefore *not* a DFT-relaxed slab in this cell, and V1 yields model-audit data, **not** adsorption-energy labels. The DFT-relaxed slab remains a prerequisite for V2 and for any label use. (3) Cost: with `LREAL = .FALSE.` the estimate is **about 100–190 node-h** (not 60–100): 597 atoms, NELECT 2778, 2 nodes each, array of 21 with 5 at a time, plus a 1 node-h timing job that must be read before the array is sized.

Model results on the pool (foundation MACE-matpes-r2scan-omat-ft + D3(BJ), unvalidated here): all 17 relaxations converged (fmax 0.1 eV/Å, 28–106 steps) with the substrate intact; the 14 two-molecule minima span 1.23 eV, which is enough range for the rank test; the three legacy arrangements are the three lowest. Four pool minima are flagged `reactive_rearrangement` by the final-state classifier; in all four both melamines are still intact C₃H₆N₆ fragments, and what they share is short contacts to the surface (N–H···I 2.42–2.57 Å, H···Pb 2.64–2.79 Å), so the flag most likely marks those contacts rather than a reaction. They stay in the audit, labelled. None of this is evidence about the real ordering until the DFT numbers exist.

Inputs: `examples/scripts/adsearch_v1_pool.py` (pool, model energies and forces), `prepare_static_audit.py` (VASP inputs and a manifest carrying the thresholds above, written before any DFT number exists; `--analyze` applies them to a returned package), `stage_static_audit.py` (cluster package). The staged package was dry-run locally with a mock VASP. It waits for a package number and submission by the dispatch desk.

**Tranche V2, relaxations (only if V1 is encouraging or a fine-tune is planned).** 6–8 frozen-far-field relaxes on a DFT-relaxed slab in the common cell (that slab: 42–234 node-h in the v3 plan, shared with label set v3b). The first package-53 A4 relax took 97 ionic steps at about 1 node-h per step for 532 atoms; scaled by (2778/2220)² that is about 1.5 node-h per step, so **90–230 node-h per structure, 550–1,800 node-h** for the tranche. This is the binding-mode-stability test and the source of fine-tuning frames; the audit structures are chosen before any of it is used for fitting. Not prepared.

## 8a. V1 result (LONI package 110, 2026-11-11): the model ranks correctly but fails as an energy and force model

19 of 20 audited structures returned (EDIFF reached, 27–38 SCF steps, at most 4.2 h on 2 nodes); `asbuilt_02` was still running at export, so the pre-set gate, which requires every structure, is formally **not passed** on that ground alone. The numbers below would not pass with it either. Analysis: `prepare_static_audit.py --analyze`, thresholds from the manifest written before the run.

| Test (two-melamine composition, 16 structures incl. 2 as-built) | Threshold | Result | |
|---|---|---|---|
| Relative energy, MAE | ≤ 0.10 eV | 0.23 eV | fail |
| Relative energy, worst | ≤ 0.20 eV | 0.49 eV | fail |
| Spearman rank correlation | ≥ 0.8 | 0.97 | pass |
| DFT top 3 in model top 3 | ≥ 2 | 2 | pass |
| Force RMSE near the molecules (686 atoms) | ≤ 0.05 eV/Å | 0.21 eV/Å | fail |
| Force direction, mean cos for \|F\| > 0.05 eV/Å | ≥ 0.9 | 0.11 | fail |

Single-molecule group (3 structures): MAE 0.18 eV, worst 0.26 eV (fail; too few for a rank test). Far-field force RMSE 0.057 eV/Å.

What the failure looks like, on the 14 relaxed two-melamine structures alone (MAE 0.22, worst 0.44 eV, Spearman 0.95):

- **The order is right, the scale is compressed.** DFT spans 2.05 eV where the model spans 1.23 eV (slope 0.60, Pearson 0.97). Both put `legacy_MelMel_C` lowest and `legacy_MelMel` second; DFT's third is the random `pool_00`, the model's is `legacy_MelMel_B` (DFT fourth).
- **Strongly bound arrangements are accurate, weakly bound ones are overbound.** Fixed-geometry interaction energy against slab and molecule: ring-N end-on −1.02 eV in both DFT and model; `legacy_MelMel_C` −2.31 vs −2.32 eV; but flat −0.48 vs −0.72 eV, amine-N end-on +0.09 vs −0.43 eV, and `pool_04` −0.26 vs −1.10 eV. The model does not see how unfavourable the non-coordinating contacts are.
- **At the model's minima DFT still has sizeable forces on the molecules** (rms 0.18–0.26 eV/Å, up to 0.6 eV/Å on a nitrogen), which is why the direction test fails: the model's forces there are near zero by construction. Far from the minimum the two agree in direction (as-built: correlation 0.92 on the molecules, 0.96 on iodide), with the model's forces about 15–25 % smaller.

Consequences. (1) The foundation model plus D3 is usable to *order* arrangements of this kind and to pre-relax them; it is not usable for energy differences at the 0.10 eV level, nor for forces near the molecule. (2) These DFT energies are at model geometries, not DFT minima, so they are audit points, not adsorption energies. (3) A fine-tune is warranted: package 110 gives 19 r2SCAN frames with forces on this interface (about 11,300 atoms), to be combined with the package-53 frames only after the `LREAL` difference is handled, with whole arrangements held out for testing. Not started.

## 8b. Fine-tune prepared, not run (2026-11-11)

- **Data** (`export_audit_frames.py`, git-ignored `data/cache/adsearch/ft110/dataset_v1`): one frame per finished static of package 110, target = DFT free energy minus D3(BJ, r2SCAN), with a per-element energy offset to the foundation model removed (fitted on the training frames; it cancels within a composition). Split by whole arrangement, so nothing descended from a test start is trained on: **train 13** (slab, molecule, two single-molecule modes, legacy A and C, six random arrangements, one unrelaxed start), **valid 2** (random 03, 09), **test 6** (legacy B, random 00 and its unrelaxed start, random 04, random 06, amine-N single molecule). `asbuilt_02` is absent until it returns.
- **Baseline on the test split** (`score_audit_model.py --stored`, foundation model, no dispersion on either side): energy differences within the two-melamine group off by 0.24 eV on average and 0.55 eV at worst (5 structures spanning 1.66 eV); force RMSE 0.075 eV/Å over all atoms and 0.21 eV/Å near the molecules.
- **Configuration** (`examples/adsearch/mel2_ft110_finetune.yaml`): naive fine-tune of the foundation checkpoint, foundation E0s, energy weight 1000 and force weight 100, learning rate 1e-4, batch 1, up to 150 epochs with patience 30, float32 (float64 training does not fit the 8 GB GPU).
- **Preflight:** one epoch ran end to end in about 9 s on the laptop GPU (the whole run is under half a GPU-hour). That single epoch already halved the all-atom force RMSE on the validation frames (0.076 to 0.037 eV/Å); it is a plumbing check, not a result.
- **What will count as success**, on the test split only: the audit's own thresholds (relative energy MAE ≤ 0.10 eV and worst ≤ 0.20 eV; near-molecule force RMSE ≤ 0.05 eV/Å, mean cosine ≥ 0.9). With six test frames this is a pilot: a pass justifies a cross-validated repeat and re-relaxation of the pool with the new model, not a claim that the model is validated.
- **Known weaknesses:** 13 training geometries, all at or near *model* minima plus one far-from-minimum start, so the model sees few displaced configurations; no replay of the foundation's training data, so behaviour away from this interface may degrade; package-53 frames are not included (their `LREAL = Auto` offset is unquantified here).

## 8c. Fine-tune pilot result (2026-11-11): three of four test thresholds met, the fourth missed by 0.001 eV/Å

Run: 150 epochs (the cap; the validation loss was still falling slowly), 12.4 minutes on the laptop GPU. Model: `data/cache/adsearch/ft110/run_v1/run/pvpass_mel2_ft110_v1.model` (git-ignored), sha256 `99fdd158…e903d3`. Scored in float64 with `score_audit_model.py` on the six held-out frames, no dispersion on either side.

| Test split (6 frames, never trained on) | Threshold | Foundation | Fine-tuned | |
|---|---|---|---|---|
| Relative energy, MAE (5 two-melamine frames spanning 1.66 eV) | ≤ 0.10 eV | 0.24 | 0.06 | pass |
| Relative energy, worst | ≤ 0.20 eV | 0.55 | 0.13 | pass |
| Force RMSE near the molecules (215 atoms) | ≤ 0.05 eV/Å | 0.208 | 0.051 | marginal fail |
| Force direction near the molecules, mean cos | ≥ 0.9 | 0.14 | 0.97 | pass |
| Force RMSE, all atoms | (none) | 0.075 | 0.014 | |

- The order of the five held-out two-melamine frames matches DFT exactly, and the energy scale is no longer compressed: DFT 0, 0.26, 0.95, 1.43, 1.66 eV against 0, 0.17, 0.81, 1.18, 1.55 eV (foundation: 0, −0.06, 0.61, 0.58, 1.68).
- The near-molecule force error is carried by the unrelaxed start (`asbuilt_00`, 0.090 eV/Å) and legacy B (0.056); the three other complexes are at 0.02–0.04. Only one far-from-minimum frame was in training.
- The held-out amine-N single molecule is still 0.14 eV too low against the aligned DFT energy (0.33 before).
- Train/valid for comparison: force RMSE near 0.024 / 0.030 eV/Å, relative energy MAE 0.08 / 0.06 eV. Test errors are about twice the training ones: some overfitting, as expected from 13 frames.

What this does and does not show. It shows that 13 single points on this interface are enough to remove most of the foundation model's error on unseen arrangements of the same kind. It does not validate the model: six test frames, one split, one seed, all geometries produced by the foundation model, and nothing tested away from this slab and molecule. Next, in order: (1) repeat as a leave-family-out cross-validation so all 16 arrangements are tested once; (2) re-relax the pool with the fine-tuned model plus D3 and see how far the minima move and whether the order holds; (3) DFT single points at a few of the *new* minima, which is the first test at geometries this model chose. Step 3 needs a go-ahead; 1 and 2 are local.

## 9. Reproduce

```
python examples/scripts/adsearch_pilot.py demo  examples/adsearch/melamine2_pbi2.json --out data/cache/adsearch/demo_melamine2
python examples/scripts/adsearch_pilot.py bench examples/adsearch/melamine2_pbi2.json --out data/cache/adsearch/bench_synth --evaluator synthetic
python examples/scripts/adsearch_pilot.py bench examples/adsearch/melamine2_pbi2.json --out OUT --evaluator mace --model FOUNDATION.model   # maceh env; not run at full budget
python examples/scripts/acat_surface_probe.py SLAB_CONTCAR --complex COMPLEX_CONTCAR --out report.json                                   # ACAT env
```

## 10. Limits of what is claimed

- Low structural energy, even after DFT validation, says nothing by itself about electronic passivation, assembly kinetics or whether the arrangement forms experimentally.
- Energies are comparable only at fixed composition, charge and constraints. Other coverages, protonation states or added water need formation free energies with declared reservoirs first; an energy/force MLIP is not a constant-potential model. Water-related extensions are outside this branch.
- The site list is under-coordinated Pb only; halide and hollow sites, and adsorption into vacancies, are not in the pilot.
- The 30° / three-tilt grid and three binding modes are a starting library; relaxation, not the grid, decides the final geometry.
- The relaxation stopping rule (fmax 0.1 eV/Å) is too loose for the 0.10 eV resolution: two optimizers run to that threshold from the same starts in the same model end 0.01–0.34 eV apart (`docs/alchemi_feasibility.md` §7). It must be tightened, to a value not yet determined, before the equal-budget benchmark; the model energies of the V1 pool carry the same uncertainty as minima, though not as fixed-geometry audit points.
