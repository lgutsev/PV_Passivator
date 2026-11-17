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

## 8d. Cross-validation and re-relaxation (2026-11-12)

**Package 110 complete (22/22).** With `asbuilt_02` the pre-set gate is evaluated on all 20 structures and is not passed by the foundation model: two-melamine relative energy MAE 0.23 eV, worst 0.48 eV; Spearman 0.97; top-3 overlap 2; near-molecule force RMSE 0.22 eV/Å, mean cos 0.15 (`data/cache/adsearch/v1/audit_110_full.json`). Receipt filed with the desk.

**Leave-arrangements-out cross-validation** (`cv_audit_finetune.py`, five fine-tunes with the §8b configuration, every arrangement predicted once by a model that never saw it or anything from the same start; dataset_v1, so without `asbuilt_02`):

| Out of fold | Threshold | Foundation | Fine-tuned |
|---|---|---|---|
| Relative energy within each fold, MAE / worst | ≤ 0.10 / ≤ 0.20 eV | | 0.06–0.09 / 0.06–0.18 (all five folds pass) |
| Relative energy, 16 two-melamine frames pooled over the five models, MAE / worst | ≤ 0.10 / ≤ 0.20 eV | 0.23 / 0.49 | 0.10 / 0.21 |
| Spearman, same 16 | ≥ 0.8 | 0.97 | 0.97 |
| Force RMSE near the molecules, 686 atoms pooled | ≤ 0.05 eV/Å | 0.21 | 0.042 |
| Mean cos near the molecules | ≥ 0.9 | 0.11 | 0.97 |

Forces pass out of sample. Energies sit at the thresholds: inside a fold they pass, pooled they are at the limit (pooling compares five different models on one scale, which adds error of its own). The fine-tuned models still compress the scale somewhat (highest relaxed arrangement 1.77 eV in DFT, 1.37 eV out of fold). Single-molecule modes: DFT order reproduced, MAE 0.09 eV.

**Re-relaxation of the pool with the fine-tuned model** (`rerelax_pool.py`, model `ft110_v1` + D3, started from the foundation model's minima, same free atoms, fmax 0.05 eV/Å): all 17 converged (31–194 steps), all molecules intact, frozen atoms unmoved.

- **The foundation minima were not minima of the better model.** Every structure dropped by 0.47–1.08 eV; molecule atoms moved 0.08–0.90 Å and slab atoms up to 0.46 Å. About 0.5 eV of each drop is common to all structures (the surface region relaxing, which the bare slab would share; the slab was not re-relaxed here). The rest, up to 0.6 eV, is specific to the weakly bound arrangements, which moved most.
- **Pb–N contacts shortened everywhere:** 2.57–2.59 → 2.52–2.55 Å for ring-N bonds, and by 0.04–0.17 Å for the looser ones.
- **Order after re-relaxation:** legacy C stays lowest; legacy A and random 00 are now tied (0.17 and 0.18 eV); then legacy B (0.50). The spread shrinks from 1.79 eV (this model at the old geometries) to 1.23 eV, because the high-energy arrangements gained most. Rank correlation with the foundation order 0.95.
- **Reading of the audit in this light:** part of the 2.05 eV DFT spread at the foundation geometries was the foundation model leaving weakly bound molecules in poor positions, not only an energy-scale error.

**Not validated:** these new geometries are up to 0.9 Å from anything the fine-tuned model was trained or tested on. Whether its energies there are right needs DFT single points at the new minima (same protocol as package 110). **Prepared, not submitted (owner: "Prepare 8 statics", 2026-11-12):** eight statics, the four lowest plus four that moved most (legacy C, legacy A, random 00, legacy B; random 02, 03, 04, 07), about 60–65 node-h. Staged as `acat_v1b_vasp_pvpass_mel2_rerelax_statics` with a hand-off note; mock run 8/8. The bare slab and the molecule are not rerun (unchanged, in package 110); the fine-tuned model's float64 energies and forces at these geometries are stored with them (`rerelax_to_audit_pool.py`). With eight structures the rank test (needs ten) is not evaluated; relative energies and forces are. The larger option (all 17 plus a slab relaxed with the same model, about 140 node-h) is not prepared.

## 8e. DFT at the fine-tuned model's own minima (LONI package 120, 2026-11-13)

Eight r2SCAN+D3(BJ) single points at geometries relaxed by `pvpass_mel2_ft110_v1` + D3, all DONE in 35–38 SCF steps (4.0–4.5 h on 2 nodes, about 68 node-h). Analysis as for package 110 (`data/cache/adsearch/v1b/audit_120.json`); the model is the one that chose the geometries, so this is the first test where it was not handed its inputs.

| Test (8 two-melamine structures spanning 1.37 eV in DFT) | Threshold | Result | |
|---|---|---|---|
| Relative energy, MAE | ≤ 0.10 eV | 0.053 eV | pass |
| Relative energy, worst | ≤ 0.20 eV | 0.103 eV | pass |
| Rank agreement | needs ≥ 10 structures | not evaluated (order identical on these 8; slope 0.88) | n/a |
| Force RMSE near the molecules (307 atoms) | ≤ 0.05 eV/Å | 0.070 eV/Å | fail |
| Force direction near the molecules, mean cos | ≥ 0.9 | 0.13 | fail (see below) |

| Structure | DFT, eV above lowest | Model, eV above lowest | Near-molecule force RMSE, eV/Å |
|---|---|---|---|
| `rr_legacy_MelMel_C` | 0.000 | 0.000 | 0.058 |
| `rr_legacy_MelMel` | 0.149 | 0.173 | 0.057 |
| `rr_pool_00` | 0.257 | 0.178 | 0.090 |
| `rr_legacy_MelMel_B` | 0.616 | 0.497 | 0.083 |
| `rr_pool_02` | 0.661 | 0.612 | 0.057 |
| `rr_pool_03` | 0.837 | 0.745 | 0.073 |
| `rr_pool_07` | 1.267 | 1.088 | 0.072 |
| `rr_pool_04` | 1.372 | 1.233 | 0.057 |

- **Energies hold at the model's own minima.** Same order as DFT for all eight, errors 0.02–0.12 eV, and the energy scale is nearly right (slope 0.88, against 0.60 for the foundation model at its minima).
- **The re-relaxation was real.** Against package 110 at the foundation geometries, DFT itself is lower by 0.37–1.04 eV per structure (the model predicted 0.51–1.08 eV, an overestimate of 0.02–0.21 eV). The weakly bound arrangements gained most in DFT too.
- **Forces: DFT's residual at these minima is 0.070 eV/Å near the molecules** (0.014 eV/Å in the far field), three times smaller than at the foundation model's minima (0.22) but above the 0.05 threshold. The direction test is not informative here and would fail for any model at its own minimum: the model's forces are below 0.05 eV/Å by construction, so the cosine compares DFT's residual with numerical noise. The meaningful number is the RMSE, and it says the model's minima are still slightly off DFT's.
- **Status of the pre-set gate:** not passed (force RMSE; rank test not evaluable with eight). Relative energies pass with margin.

Reading: the fine-tuned model is now good enough to rank arrangements of this kind and to pre-relax them close to the DFT minimum, on this slab and molecule. It is not a validated energy model in general. The eight new frames are the first training data away from the foundation model's minima; adding them (holding out whole arrangements again) is the obvious next round and is local. A DFT relaxation from one or two of these minima (tranche V2) remains the only test of binding-mode stability, and has not been prepared.

## 8f. Fine-tune round 2: packages 110 + 120 (2026-11-13)

Dataset `dataset_v2` (`export_audit_frames.py --also`): all 30 frames of the two packages (22 + 8), same target and alignment as before; a re-relaxed structure `rr_X` is in the family of `X`. Same five folds as §8d, same configuration, so each arrangement, in all its versions (foundation minimum, unrelaxed start, re-relaxed), is predicted by a model that saw none of them. 1 h 40 min of GPU for the five folds.

| Out of fold | Threshold | Round 1 (110 only) | Round 2 (110 + 120) |
|---|---|---|---|
| Relative energy, two-melamine frames pooled, MAE / worst | ≤ 0.10 / ≤ 0.20 eV | 0.104 / 0.209 (16 frames) | **0.073 / 0.175** (25 frames) |
| Same, the 14 relaxed package-110 frames only | | (0.10 / 0.21 with the two as-built) | 0.073 / 0.157, slope 0.88 |
| Same, the 8 re-relaxed frames only | | not in round 1 | 0.069 / 0.169 |
| Spearman, pooled | ≥ 0.8 | 0.97 | 0.99 |
| Force RMSE near the molecules, 17 relaxed package-110 frames | ≤ 0.05 eV/Å | 0.033 | 0.035 |
| Force RMSE near the molecules, 8 re-relaxed frames | ≤ 0.05 eV/Å | (0.070 for ft110_v1 at its own minima, not out of fold) | **0.043** |
| Force RMSE near the molecules, unrelaxed starts | ≤ 0.05 eV/Å | 0.088, 0.072 (two frames) | 0.079, 0.068, **0.283** (`asbuilt_02`) |
| Per fold: relative energy MAE / worst | | 0.06–0.09 / 0.06–0.18 | 0.03–0.06 / 0.04–0.15 |

- **Energies now pass out of sample with margin**, and the scale is no longer compressed (slope 0.88 against 0.77 in round 1). Single-molecule modes: order right, MAE 0.07 eV.
- **Forces pass on every relaxed geometry**, including the re-relaxed ones the model family had never been tested on in round 1.
- **Forces fail far from equilibrium.** `asbuilt_02`, an unrelaxed start with DFT forces up to 11 eV/Å, is at 0.28 eV/Å near the molecules when held out; the two milder starts are at 0.07–0.08. Pooled over everything including these three, the near-molecule RMSE is 0.070 eV/Å, above the threshold; without them it is 0.03–0.04. The training set has three such frames in total, one per fold at most.

**Round-2 model for use:** `data/cache/adsearch/ft110/run_v2/run/pvpass_mel2_ft_v2.model` (git-ignored; sha256 `d5054b1c16b34232…`), trained on 27 frames with 3 for validation (random 03 and its re-relaxed version, random 09). It has no held-out test of its own; the cross-validation above is the estimate of its error.

**Where this leaves the model question.** For two melamine on this face, near relaxed geometries: energy differences to about 0.07 eV (worst case under 0.2), forces to about 0.04 eV/Å near the molecules, order as DFT. That meets the thresholds fixed in §8 for the ranking use. Not established: accuracy during the early steps of a relaxation from an as-built start (the far-from-equilibrium frames), other coverages or molecules, and binding-mode stability under DFT relaxation (tranche V2).

## 8g. Relaxation-threshold study (2026-11-13): the force threshold is not the control; the optimizer is

Question: how tight must a relaxation be before energy differences stop changing. `fmax_study.py` relaxes each structure through a ladder of thresholds (0.1, 0.05, 0.03, 0.02, 0.01 eV/Å) in one trajectory, once with FIRE and once with LBFGS, same free atoms, round-2 model + D3 in float32, and gives every saved geometry one float64 energy. Four two-melamine arrangements: legacy C (lowest), random 00, 02 and 04.

**Run 1, pool geometries as they were (cap 250 steps).** No relaxation got below 0.03 eV/Å. At the same threshold of 0.05 eV/Å LBFGS is 0.74–0.85 eV below FIRE for all three arrangements that reached it. In the LBFGS runs surface formamidinium cations reorient (38 H, 7–10 C and N atoms by more than 0.5 Å, up to 2.1 Å), surface iodides shift by up to 1.1 Å and the molecules by 0.5–0.7 Å rms; FIRE leaves everything within 0.45 Å of the start. Gaps between arrangements at 0.05 eV/Å agree between the optimizers to 0.06–0.11 eV; at 0.1 eV/Å one gap differs by 0.85 eV and two arrangements swap.

**Bare slab (cap 400).** The same happens without molecules: LBFGS converges to 0.01 eV/Å in 214 steps, 0.50 eV below where FIRE stops (0.02 eV/Å after 49 steps, never 0.01), with surface atoms moved by up to 1.7 Å. The slab every pool structure was built on is not a minimum of the model.

**Run 2, the four arrangements placed on the LBFGS-relaxed slab** (`transplant_slab.py`; closest molecule–slab contact 1.97 Å; cap 400). Pre-relaxing the slab does not remove the effect:

| Arrangement | FIRE end point (fmax reached) | LBFGS end point (fmax reached) | LBFGS − FIRE, round-2 model | same, foundation model | Largest molecule-atom move, LBFGS |
|---|---|---|---|---|---|
| legacy C | −9311.094 (0.02) | −9311.802 (0.02) | −0.71 eV | −0.80 | 3.3 Å |
| random 00 | −9310.836 (0.03) | −9311.487 (0.03) | −0.65 | −0.86 | 2.0 Å |
| random 02 | −9310.435 (0.03) | −9310.508 (0.05) | −0.07 | −0.10 | 1.5 Å |
| random 04 | −9309.869 (0.02) | −9311.150 (0.05) | −1.28 | −1.46 | 5.0 Å |

- No run reached 0.01 eV/Å in 400 steps; the LBFGS runs were still descending at the cap.
- The deep geometries are chemically ordinary: every C–N, C–H, N–H bond list is unchanged, Pb–I coordination is 5–6 as before, the movers are surface FA hydrogens (18 by more than 1 Å in three of four) and the molecules.
- The foundation model, which never saw the fine-tune data, puts the same sign and nearly the same size on every drop. That argues against a hole dug by the fine-tune, but both models share a parent and neither has a DFT label near these geometries.
- The order changes at the weak end: shallow legacy C < 00 (+0.26) < 02 (+0.66) < 04 (+1.22); deep legacy C < 00 (+0.32) < 04 (+0.65) < 02 (+1.29). The two lowest keep their order and gap (0.26 against 0.32 eV).

**What follows.**
1. "The relaxed energy of an arrangement" is not defined at the 0.1 eV level by a FIRE run to any threshold tried here. Earlier rankings (§8, §8d, packages 110 and 120) compare shallow FIRE end points; they stand as model-against-DFT tests at fixed geometries, not as a ranking of true minima.
2. A search benchmark with the §7 relaxation (FIRE, fmax 0.1, 150 steps) would mostly measure how far each candidate slid. It needs LBFGS to 0.02 eV/Å or tighter with several hundred steps per candidate (about 12 min each on the laptop GPU, D3 dominating), which is cluster work at search scale, or a cheaper dispersion term.
3. The deep minima need DFT before anything is built on them. **Prepared, not submitted:** `acat_v1c_vasp_pvpass_mel2_deepmin_statics`, ten r2SCAN+D3(BJ) statics (the four shallow/deep pairs above and the bare slab shallow/deep; `geoms_to_audit_pool.py`, then the package-120 path), about 75–85 node-h, mock run 10/10, hand-off note marked not authorized.

## 8h. DFT at the deep minima (LONI package 127, 2026-11-14): the deeper minima are real, and the model holds there

Package `127_vasp_pvpass_mel2_acat_deepmin_statics` (job 1181419): 10/10 done, 32-37 SCF steps, 3.5-4.2 h each on 2 nodes. Outputs under `D:\MLIP_Work_Folder\hpc_smoke_tests\127_...`; report `data/cache/adsearch/v1c/audit_127.json`.

| Structure | LBFGS end point minus FIRE end point, DFT | same, round-2 model | Model force RMSE near the molecules, shallow / deep |
|---|---|---|---|
| legacy C | -0.822 eV | -0.708 | 0.026 / 0.038 eV/A |
| random 00 | -0.771 | -0.651 | 0.023 / 0.031 |
| random 02 | -0.047 | -0.074 | 0.029 / 0.039 |
| random 04 | -1.459 | -1.281 | 0.026 / 0.066 |
| bare slab | -0.426 | -0.496 | (all atoms) 0.003 / 0.037 |

- **DFT puts every LBFGS end point below its FIRE end point**, by nearly the amount the model said. The deeper minima are not a model artefact. The model underestimates the drop for the three large cases by 0.11-0.18 eV and overestimates it for the bare slab by 0.07 eV.
- **Relative energies of the eight two-melamine frames: MAE 0.069 eV, worst 0.109 eV** against thresholds 0.10 / 0.20. None of these geometries, nor anything within 1.5 A of the deep ones, was in the training data.
- **Forces near the molecules: RMSE 0.037 eV/A pooled** (threshold 0.05); seven of eight frames are at 0.02-0.04, deep random 04 (molecules moved 5 A) at 0.066. The cosine criterion (0.20) is not meaningful here for the reason given in 8e: these are the model's own near-minima, where its forces are close to zero. The report's overall flag is "fail" for that criterion alone.
- **Order in DFT, shallow set:** legacy C < 00 (+0.26) < 02 (+0.68) < 04 (+1.26). **Deep set:** legacy C < 00 (+0.31) < 04 (+0.62) < 02 (+1.46). The model gives the same two orders (+0.26/+0.66/+1.23 and +0.32/+0.65/+1.29). The swap of 02 and 04 between the sets is real: how deep a relaxation goes changes the ranking of the weakly bound arrangements, in DFT as in the model.
- DFT forces at these geometries reach 0.23-0.30 eV/A on some atom; none is a DFT minimum.

**Consequences.** (1) A search must relax with LBFGS (or equivalent) well past where FIRE stalls, or it ranks stalled states. (2) The round-2 model is good enough to drive those relaxations on this system: its energy differences at geometries it chose, far from its training set, are within thresholds. (3) The ten frames are new training data from a region the model had not seen (a round 3 would add them). (4) Still not done: no relaxation here reached 0.01 eV/A, so the true minima are lower still by an unknown amount.

## 8i. Fine-tune round 3: packages 110 + 120 + 127 (2026-11-14)

Dataset `dataset_v3`: all 40 frames. `sh_X` and `dp_X` join the family of `X`; the two relaxed bare slabs are a family of their own (`slab_relaxed`, held out in fold 3). Same configuration and folds otherwise, so every arrangement in all its versions (foundation minimum, unrelaxed start, re-relaxed, shallow, deep) is predicted by a model that saw none of them.

| Out of fold | Threshold | Round 2 (30 frames) | Round 3 (40 frames) |
|---|---|---|---|
| Relative energy, two-melamine frames pooled, MAE / worst | ≤ 0.10 / ≤ 0.20 eV | 0.073 / 0.175 (25 frames) | **0.061 / 0.143** (33 frames) |
| Spearman, pooled | ≥ 0.8 | 0.990 | 0.994 |
| Per fold: relative energy MAE / worst | | 0.03–0.06 / 0.04–0.15 | 0.03–0.05 / 0.05–0.13 |
| Deep minus shallow, error against DFT − D3 (legacy C, 00, 02, 04) | | 0.11, 0.12, 0.03, 0.18 (model trained without them, §8h) | 0.02, 0.00, 0.01, 0.09 |
| Bare slab, deep minus shallow, error | | 0.07 (§8h) | 0.09 |
| Force RMSE near the molecules: 17 relaxed package-110 frames | ≤ 0.05 eV/Å | 0.035 | 0.033 |
| same, 8 re-relaxed frames | ≤ 0.05 | 0.043 | 0.039 |
| same, 4 shallow / 4 deep frames | ≤ 0.05 | 0.026 / 0.046 pooled 0.037 (§8h, not out of fold by family) | 0.034 / **0.047** (deep random 04: 0.071) |
| same, unrelaxed starts | ≤ 0.05 | 0.079, 0.068, 0.283 | 0.077, 0.069, **0.269** |
| Single-molecule modes, relative MAE | | 0.071 | 0.080 |

- Energies improve everywhere two-melamine frames are concerned; the systematic underestimate of the deep-minimum gain is gone for three of four arrangements.
- Forces are unchanged within noise. Far-from-equilibrium frames still fail (`asbuilt_02`), as do the molecules of deep random 04 (0.07 eV/Å). Ten more frames near minima do not teach the model about strongly distorted geometries.
- Single-molecule modes did not improve (three frames, no new data of that kind).

**Round-3 model for use:** `data/cache/adsearch/ft110/run_v3/run/pvpass_mel2_ft_v3.model` (git-ignored), trained on all 40 frames less the validation families (random 03, random 09). No held-out test of its own; the cross-validation is the estimate.

Practical note: `score_audit_model.score` now releases the scoring model's GPU memory; before, the cross-validation parent kept about 2 GB while the next fold trained, which put an 8 GB card at 7.6 GB.

## 8j. Search benchmark, laptop-sized pilot (2026-11-14): plumbing and cost; no claim about the arms

`adsearch_bench_small.py`: both arms, seeds 0 and 1, 16 attempts each (first 8 shared), round-3 model + D3 in float32, substrate = the LBFGS-relaxed common-cell slab of §8g. Two stages: **screen** every valid proposal with LBFGS to 0.1 eV/Å, at most 80 steps (the energy the search ranks by); **refine** the six lowest unique screened structures over all runs with LBFGS to 0.03 eV/Å, at most 400 steps, then one float64 energy. 96 minutes on the laptop GPU at 4 CPU threads; relaxations cached, so the 14 shared proposals were computed once.

| Run | Rejected before relaxing | Duplicate proposals | Relaxed | Ranked | Force calls | Best screened energy |
|---|---|---|---|---|---|---|
| random, seed 0 | 2 | 0 | 14 | 11 | 762 | −9310.885 |
| evolutionary, seed 0 | 1 | 2 | 13 | 10 | 695 | −9310.899 |
| random, seed 1 | 3 | 0 | 13 | 9 | 688 | −9310.855 |
| evolutionary, seed 1 | 2 | 1 | 13 | 12 | 671 | −9310.871 |

Screening: 51 steps and 75–85 s per candidate on average; 3 of 53 relaxations hit the 80-step cap and 7 ended as `reactive_rearrangement` (not ranked). In both seeds the evolutionary arm's best is 0.01–0.02 eV below the random arm's, which is far inside the model error and means nothing with two seeds and 8 arm-specific attempts.

Refinement of the six lowest (all two ring-N end-on molecules):

| Screen rank | Screened energy | Refined energy (float64) | Steps | First found by |
|---|---|---|---|---|
| 0 | -9310.899 | -9311.825 | 395 | evolutionary (exchange), seed 0 |
| 1 | -9310.885 | -9311.529 | 307 | first random proposal, seed 0 |
| 2 | -9310.878 | -9311.744 | 282 | evolutionary (move), seed 0 |
| 3 | -9310.871 | -9312.001 | 360 | evolutionary (move), seed 1 |
| 4 | -9310.870 | -9311.845 | 400 (cap) | random, seed 0 |
| 5 | -9310.855 | -9311.741 | 385 | random proposal shared by both arms, seed 1 |

- **The screening order does not survive.** The six are within 0.044 eV at the screening level and spread over 0.47 eV after refinement; the fourth by screening becomes the lowest and the second becomes the highest. Every one drops by 0.64–1.13 eV and atoms move 2.9–3.7 Å.
- **The screen does find the right family.** Every structure within 0.1 eV of a run's best is ring-N + ring-N, the motif of the legacy arrangements.
- **Against the known arrangements, same model, same slab, same frozen atoms:** legacy C, continued from its deep geometry of §8g with the round-3 model and the refinement protocol (LBFGS to 0.03 eV/Å; 28 steps, moves under 0.1 Å), is at −9311.900; deep random 00, not re-relaxed, at −9311.494. One refined structure is below legacy C: −9312.001, **0.10 eV lower**, which is inside the model's worst out-of-fold error (0.14 eV). It is a different arrangement (not the same structure at any tolerance up to 2.5 Å; both molecules ring-N to Pb at 2.60–2.62 Å, molecule centres 5.7 Å apart against 6.0 Å in legacy C). No DFT number exists for it. (An earlier version of this line compared with legacy C before re-relaxation, −9311.809, and gave 0.19 eV.)

**What the pilot establishes.** The pipeline runs end to end with LBFGS; a screened candidate costs about 80 s and a refined one 8–11 min on this laptop; a loose-threshold screen is good for choosing the motif family but not for ordering candidates inside it, so the ordering must come from refinement (or the search must rank on refined energies, at about 8 times the cost per candidate). **What it does not establish:** anything about random against evolutionary. That still needs the §7 budget (3 seeds, 200 attempts), now with refinement of the top of each run: of the order of 600 screens and 60 refinements per arm, 25–35 GPU-hours on a card like this one, which is cluster work.

**DFT check prepared, not submitted (owner: "Yes, prepare them", 2026-11-15):** `acat_v1d_vasp_pvpass_mel2_searchbest_statics`, two r2SCAN+D3(BJ) statics on the package-127 protocol: `search_best` (the −9312.001 structure) and `legacyC_v3` (legacy C as re-relaxed above). About 15–17 node-h; mock run 2/2; hand-off note marked not authorized. The model predicts 0.10 eV in favour of `search_best`; with an expected error of 0.06 eV (worst 0.14) on such differences, DFT may well reverse it.

## 8k. DFT check of the search result (LONI package 131, 2026-11-16): the search's structure is lower

Package `131_vasp_pvpass_mel2_acat_searchbest_statics` (job 1183184): 2/2 done, 35 and 37 SCF steps, 3.9 and 4.0 h on 2 nodes. Report `data/cache/adsearch/v1d/audit_131.json`.

| Structure | DFT E0 (eV) | Round-3 model (eV) | Model force RMSE near the molecules |
|---|---|---|---|
| `search_best` (search pilot, refined) | −9337.6260 | −9312.001 | 0.039 eV/Å |
| `legacyC_v3` (legacy C, same model and protocol) | −9337.4858 | −9311.900 | 0.025 eV/Å |
| difference | **−0.140** | −0.101 | |

- **DFT confirms the search found a lower arrangement than legacy C**, by 0.14 eV; the model said 0.10 eV. The error on the difference is 0.04 eV, inside the cross-validated 0.06 eV.
- `search_best` is the lowest DFT energy on record for two melamine on this slab: 0.27 eV below the deep legacy C frame of package 127 (−9337.3552), and legacy C itself came down 0.13 eV in DFT when continued with the round-3 model (the model said 0.09).
- Forces near the molecules are within threshold on both frames (pooled 0.033 eV/Å). The cosine criterion is not meaningful at the model's own minima (§8e).
- Limits: two single points at model geometries. Neither is a DFT minimum (DFT forces up to 0.15–0.21 eV/Å on some atom), the pilot sampled 64 attempts, and nothing says a still lower arrangement does not exist. The result is "a search with this model found, and correctly ranked, a structure better than the hand-built ones", not "the global minimum".

## 8l. DPA-4 trial (LONI package 138, 2026-11-16/17): blocked by the cluster module version

Same-data comparison of a pretrained DPA-4 fine-tuned on the 40 frames of §8i (same target, same five folds) against the round-3 MACE model. Generator `examples/scripts/prepare_dpa4_trial.py`. Owner authorized the download and a one-hour smoke job only; the fold array waits for the smoke report.

Pretrained checkpoint (OMat24, PBE-level labels; licence CC-BY-NC-4.0 per the model card), fetched once into the cluster model store:

| File | Bytes | sha256 |
|---|---|---|
| `DPA4-Neo-OMat24-v20260805.pt` | 4,583,819 | `fd7f34ae28f921201e4a0328ddf56179892752084f8363e7796038201471c989` |
| `DPA4-Neo-OMat24-v20260805.json` | 3,654 | `3ca5ec05a3d3f626034e463f09ebf1673fe377aca40381062efb50825061409b` |

Smoke job 1186734 tests whether the cluster's deepmd 3.2.0 beta module loads this checkpoint, gives zero-shot predictions on all 40 frames, and times 100 fine-tune steps.

**Smoke result (2026-11-17): the module cannot use the released checkpoint.** Job 1186734 (A100 80 GB, torch 2.11.0+cu126, deepmd 3.2.0b0) stopped after 38 s:
- Zero-shot load fails with size mismatches: the beta builds a narrower DPA-4 than the released one (`so3_linear_1` [4, 32, 192] against [4, 32, 576] in the checkpoint; `so3_linear_2` [4, 96, 32] against [4, 288, 32]; same in both blocks and the output FFN).
- The fine-tune is rejected at input parsing: the beta requires `model/descriptor/type`, which the released config does not have (it sets `model.type: "SeZM"`). The local deepmd 3.2.0 release accepted a config of the same shape and got as far as building the neighbour list, so this is a beta-versus-release difference.
- No accuracy, step time or memory figure exists. Running the trial needs a deepmd 3.2.0 release environment on the cluster (with `vesin` or `nvalchemiops` for DPA-4's neighbour list). Not prepared.


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
- See §8g: tightening the threshold alone does not fix the following; the optimizer matters more. The relaxation stopping rule (fmax 0.1 eV/Å) is too loose for the 0.10 eV resolution: two optimizers run to that threshold from the same starts in the same model end 0.01–0.34 eV apart (`docs/alchemi_feasibility.md` §7). It must be tightened, to a value not yet determined, before the equal-budget benchmark; the model energies of the V1 pool carry the same uncertainty as minima, though not as fixed-geometry audit points.
