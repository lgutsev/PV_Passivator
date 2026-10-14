# Representation-based passivant discovery: audit, design, pilot (14 Oct 2026)

Status: **prototype**. Everything below that says "implemented" is unit-tested (`tests/test_discovery.py`). Nothing
has been evaluated with DFT or an MLIP. The only optimisation runs used a synthetic oracle and say nothing about
passivation. No calculations were launched.

Tags: **[R]** read from code/files, **[I]** inferred, **[P]** proposed, not implemented.

## 1. What the reference work actually implements

Lei, Brosa Planella, Baird, Cooper, *From Prompt to Protocol: Fast Charging Batteries with Large Language Models*,
arXiv:2601.09626 (v1, 14 Jan 2026). The DOI given in the brief (10.1002/advs.78012) redirects to Wiley, which refused
automated access, so the journal version was not read. Code: github.com/tldr-group/prompt-to-protocol at
6f02ab5 (4 Feb 2026). It uses a **non-commercial licence**, so no code was copied; only its design was studied.

| Item | Finding |
|---|---|
| P2O (`src/p2o`) | [R] The LLM writes Python code for small neural-network charging policies (<35 parameters, fixed output range), plus a symbolic PyBaMM twin. The LLM generates the *function class*; it does not generate solutions. |
| Inner optimiser | [R] Ax `Models.SOBOL` × 30, then `Models.SAASBO` 60 batches × 3 = **210 simulator calls per proposed representation**. It tunes the network's continuous weights. |
| Outer loop | [R] 50 iterations × 3 children. Parents come from a top-10 pool by tournament selection (k=3). The LLM recombines two parent *codes* into a child. Feedback reaches the LLM only through parent selection; the prompt contains the parent code, not the losses. |
| P2P (`src/p2p`) | [R] The LLM writes an explicit charging function with scalar parameters, which are then optimised. The paper reports P2P under matched evaluation budgets. |
| Evaluator | [R] PyBaMM simulations: seconds each, deterministic. |
| Order of total cost | [I] Up to 150 × 210 ≈ 3×10⁴ evaluations per campaign. |

**Transferable to PV_Passivator:**
- The split between the LLM, which designs the space, and a numerical optimiser, which searches inside it.
- Selection pressure on representations.
- Hard output constraints on the LLM.

**Not transferable as is:**
- The evaluation budget. One r2SCAN slab relaxation is about 10²–10³ node-hours, against 3×10⁴ evaluations per campaign.
- SAASBO on continuous weights. Our variables are discrete molecular identities.
- An evaluator with no domain-of-validity problem. MLIPs on unseen organic–perovskite interfaces have one.

The pilot below therefore uses ~10 evaluations per representation, not 210, and puts all optimisation over
molecular identity in a GP with a molecular kernel.

## 2. Repository audit and gap analysis

| Need | Exists | Gap |
|---|---|---|
| Surface models, defects | [R] `surface.py` (faces, sites, halide classes), `poses.py` (`remove_fragments`, `fill_schottky`, Schottky V_FA+V_I slab in the label set) | none for the first benchmark |
| Adsorption poses | [R] `poses.generate_poses`, clash checks, post-relaxation classification | No automatic pose search per new SMILES: 3D embedding → `Passivator` is missing [P] |
| Charge/stoichiometry consistency | [R] `ledger.py` balanced reactions, declared charges | none |
| DFT inputs, consistent profile | [R] `vaspprep.py`, `prepare_label_set.py`, `stage_label_set.py`; r2SCAN+D3(BJ) common-cell label sets (package 53) | Per-candidate job generation from a SMILES is missing [P] |
| DFT results | [R] `labels.py` / `pvpass labels` (E_ads with balance, settings and k-point checks) | No electronic descriptors (in-gap state, η, dipole/work-function shift) from returned runs [P]. `density.py` exists on a synthetic fixture only. |
| MLIP | [R] `benchmark.py` (archive single points), `training_data.py` and the MACE pilot scripts | No validated passivation MLIP yet; domain-of-validity check per candidate is missing [P] |
| Molecular representation, Tier 0 | none before this work | **added**: `discovery/chem.py`, `discovery/space.py` |
| Objectives, Pareto, optimiser, loop, store | none | **added**: `discovery/objectives.py`, `optimize.py`, `loop.py`, `store.py`, `proposer.py` |
| Device-level aggregation | none | **added**: `discovery/score.py` (inputs needed: a SCAPS sweep table and calibration) |

**Reusable from InterfaceForge** (`../InterfaceForge`, inspected at file-name and README level only, so these are candidates, not commitments):
- `committee.py`, `mlip_compare.py`: committee uncertainty, which supplies the `sigma` of MLIP descriptors in `score.py`.
- `audit.py`, `dft_evidence.py`, `vasp_provenance.py`: hash-bound DFT evidence.
- `scheduler.py`, `campaign.py`: dry-run-by-default submission.
- `surface/` phosphonate docking: an anchor-docking analogue for sulfonate/phosphonate anions.

None were imported; the discovery package has no InterfaceForge dependency.

**Prior data that can be reused:**
- r2SCAN label set v2 (package 53): nine salts on the FAI-Schottky slab. Results are still pending.
- The legacy PBE+D3 spreadsheet. It is **not usable as labels**: the archive could not reproduce its slab energies (`WORKFLOW_AUDIT.md` §4).
- The group's published experimental device data for PEAI, PEATFA, PEAAc, TFPACl, PFPACl and benzocaine-HCl. These are a calibration set for the score (§4). Device architectures differ between papers, so they must be recorded per data point.

The cation identities in the r2SCAN reference geometries were confirmed by bond perception:
- PFPA⁺ = 2,2,3,3,3-pentafluoropropylammonium;
- TFPA⁺ = 3,3,3-trifluoropropylammonium;
- TH⁺ = 4-hydroxyphenethylammonium (tyrammonium).

The PEAAc and PEATFA references are neutral amine·acid pairs, not ion pairs. That changes the reference state for those two reactions.

## 3. Scientific problem for the first benchmark

| Item | Choice | Note |
|---|---|---|
| Surface | α-FAPbI₃(100), FAI-terminated, 3×3 in-plane, the r2SCAN common cell of label set v2 | Reuses the cell and k-mesh of package 53 so energies are comparable. |
| Defect | One V_FA+V_I Schottky pair (the existing FAI_Schot slab) | Chosen as a benchmark, not as the dominant defect of every sample. The PbI₂-terminated Pb5c site is the second benchmark. |
| Reaction | salt(ion pair or neutral pair, declared) + V_FAI slab → complex, one per cell, balanced by `ledger.py` | Coverage = 1/9 of surface A sites. All rows use one cell, one profile and one reference type. |
| Charge state | Neutral cells only for the screening tier | Charged defects need the truncation/correction work of package 55. Not used until it is analysed. |
| Electrostatics | Dipole correction for asymmetric slabs (open item in `WORKFLOW_AUDIT.md`) | Required before any work-function or band-offset descriptor is trusted. |
| Poses | Several poses per candidate (`poses.generate_poses`), lowest relaxed one kept, all recorded | Molecule selection (optimiser) and pose search (inside the oracle) are separate. |

**Objectives.** The spec is in `examples/discovery/objectives_fapbi3_physical.json` and is a draft. Thresholds marked "PI-to-set" must be fixed by the PI.

| Criterion | Role | Computed from |
|---|---|---|
| device score `pce_scaps` | objective (max) | §4: coverage, trap removal and new traps fed into SCAPS |
| synthetic practicality `sa_score` | objective (min) | RDKit SA score (Tier 0) + vendor availability [P] |
| exchange free energy `dg_exchange_eV` | constraint | MLIP screening → DFT |
| surface integrity `pb_coordination_loss` | constraint | relaxed geometry (MLIP → DFT) |
| no new deep states `new_deep_states` | constraint | DFT DOS / defect-level analysis |
| chemical robustness (moisture, light, heat) | **not computed** | Separate tests in Tier 3. Clean-surface binding is not operational stability. |

Adsorption energy alone is never the objective. `ObjectiveSpec` refuses a spec with fewer than two objectives.

## 4. Composite score: device-referenced, not weighted

A weighted sum of E_ads, gap states and dipole needs arbitrary weights. Instead, every electronic or energetic
descriptor enters through a physical mechanism, and the drift–diffusion device model is the only aggregator
(`discovery/score.py`):

```
θ     = 1 / (1 + exp((ΔG_bind − μ) / kT))           coverage of the targeted defect site (saturating)
N_it  = N_res + N0 (1 − θ η) + θ N_new              interface trap density
PCE   = SCAPS(N_it [, Δχ])                          interpolated from a SCAPS-1D sweep, never extrapolated
```

| Symbol | Meaning | Source | Tier |
|---|---|---|---|
| ΔG_bind | Binding free energy against a declared reference state | MLIP screening → r2SCAN DFT | 1 → 2 |
| η | Fraction of the defect's in-gap states removed when the site is occupied | DFT electronic structure (PDOS/IPR); MLIP geometry proxy (restored Pb coordination) only for pre-screening | 2 (1 proxy) |
| N_new | Deep states the passivant adds | DFT | 2 |
| Δχ | Interfacial dipole → band offset | DFT planar-averaged potential with dipole correction | 2 |
| N0, N_res, μ, T | Device/processing calibration | SCAPS fit to measured devices; the group's experimental set | calibration |

**Properties:**
- **Saturation.** Binding much stronger than μ adds no score, so "more negative E_ads" is not rewarded without limit.
- **Electronic effect required.** A molecule that binds strongly but leaves the trap in the gap (η≈0) scores like the bare surface.
- **Penalty for new traps.** A passivant that adds deep states is penalised through the same device physics.
- **Uncertainty.** It propagates by Monte Carlo, so MLIP numbers (larger σ) produce wide p10–p90 bands and cannot pass as DFT.
- **Separate axes.** SA score, stability and integrity stay as their own Pareto axes or constraints. The device model does not contain them.

**Validation before use** [P]:
1. Fit μ, N0 and N_res on part of the experimental set.
2. Predict ΔV_oc/ΔPCE for the held-out molecules.
3. Report the rank correlation.

With fewer than ten molecules from different device stacks this is a consistency check, not a validated model. That limit must be stated.

## 5. Architecture (implemented, `src/pvpassivator/discovery/`)

| Module | Responsibility | Key guarantees |
|---|---|---|
| `chem.py` | Tier 0: parse, eligibility (`Eligibility`; `Eligibility.anion()` for salt anions), descriptors, Morgan fingerprints, Tanimoto | Never repairs a structure. Protonation state must be declared. Forbidden reactive groups. SA-score cap. One shared rule set, hashed. |
| `space.py` | Representations `pvpass.searchspace/1`: `library` or `combinatorial` (scaffold `[*:k]` × substituent lists); deterministic enumeration and dedup by InChIKey | A representation **cannot** set objectives, constraints, thresholds, eligibility, budget, oracle, weights or holdout (refused). Size caps. Validity yield and rejection reasons reported. |
| `objectives.py` | Hash-locked spec, feasibility, Pareto mask, hypervolume | ≥2 objectives. The store refuses to mix specs. |
| `optimize.py` | `RandomOptimizer`, `FixedOrderOptimizer`, `GPParEGO` (Tanimoto GP; amplitude and noise by evidence; ParEGO random Chebyshev scalarisation; constraint GPs → P(feasible); Tanimoto-diverse batches) | Sees only the search split. |
| `oracles.py` | Declared id, tier, cost and status. `SyntheticOracle` (engineering only, `syn_*` names). `CachedDFTOracle` (returns completed results, never invents) | Cost ≥ 1 DFT-equivalent raises `ApprovalRequired` unless approved. |
| `store.py` | Append-only JSONL; statuses hypothesis / prediction / computed / validated; dedup; budget; holdout split; resume | Holdout is excluded from optimizer and proposer views. |
| `proposer.py` | `ReplayProposer`, `CallableProposer(fn)` (any provider); evidence-only feedback prompt | No provider SDK imported. Usage/tokens logged per proposal. |
| `loop.py` | `run_pool` (arms A/B), `run_representation_loop` (arm C) | Failed proposals cost no evaluations and are logged separately. |
| `score.py` | §4 | No extrapolation beyond the SCAPS table. |

The FAPbI₃ specifics live only in the example JSONs (`examples/discovery/`). The modules know nothing about perovskites except the default eligibility.

## 6. Three-arm benchmark design and pilot result

| Arm | Pool | Optimiser |
|---|---|---|
| A | `expert_library.json`: 22 cations in priority order | fixed order |
| B | `conventional_space.json`: generic NH₃⁺–(CH₂)₁₋₄–R grid, 92 valid | GPParEGO |
| B0 | same as B | random (checks whether the GP helps) |
| C | union of four recorded proposals (`representations/rep_0*.json`), 119 valid, one per round | GPParEGO |

All arms share:
- the same eligibility, spec, oracle and budget (30 evaluations);
- batch size 3 and no prior labels.

B and C use the same optimiser, so their difference isolates the pool. Pool overlap: A∩B 14, A∩C 9, B∩C 18; 6 expert molecules lie outside both.

**Pilot** (`python examples/scripts/run_discovery_pilot.py --out <dir>`, 10 oracle seeds, about 11 s). Metric: hypervolume
of truly feasible evaluated points / hypervolume of the true feasible front of the union of all pools (199 molecules).

| Arm | HV ratio (mean ± s.e.) | evaluations spent on infeasible | true-front members found (of ~5.2) |
|---|---|---|---|
| A expert | 0.74 ± 0.06 | 68 % | 0.4 (22 evaluations; library exhausted) |
| B0 random | 0.73 ± 0.05 | 73 % | 0.6 |
| B GP | 0.80 ± 0.06 | 67 % | 2.0 |
| C representations + GP | 0.85 ± 0.04 | 58 % | 1.5 |

Paired C − B: +0.04 ± 0.09, C better on 4 of 10 seeds. **No detectable difference.**

**What this establishes:** the loop runs end to end, and budgets, dedup, failed-proposal handling and holdout isolation behave as specified. On this landscape the GP adds about +0.08 over random on the same pool.

**What it does not establish:**
- anything about passivation;
- any LLM benefit. The four proposals were written once by the same session that wrote the code, without oracle feedback (ReplayProposer ignores the prompt), and the synthetic landscape is a random function of generic descriptors.

## 7. Scientific risk register

| Risk | Effect | Mitigation |
|---|---|---|
| Surface model: one termination, one Schottky pair, 1/9 coverage | Ranking may not transfer to other terminations or coverages | Second benchmark (PbI₂-rich Pb5c). Coverage series for the top candidates. |
| Reference states (ion pair vs neutral pair vs solvated ion) | E_ads shifts of order 1 eV between conventions | One declared reference per campaign (`ledger.py`). PEAAc/PEATFA neutral pairs flagged. Exchange free energy, not raw E_ads. |
| Legacy PBE spreadsheet values | Irreproducible slab energies (`WORKFLOW_AUDIT.md` §4) | Never used as labels. r2SCAN set v2 only. |
| Charged defects, finite size | Spurious interactions in charged slabs | Neutral cells for screening. Package 55 must be analysed first. |
| Dipole correction missing in archived runs | Work function / Δχ wrong | Dipole-corrected statics before any Δχ descriptor. |
| MLIP transferability | Unseen organics on perovskite outside the training domain | Committee σ in the score. Domain check (element/fragment coverage, committee disagreement). DFT on everything that enters the front. |
| η from geometry proxies | Restored Pb coordination ≠ trap removal | Proxy only for pre-screening. η for any reported candidate from DFT electronic structure. |
| SOC / band edges | PBE/r2SCAN without SOC misplace Pb-derived states | Relative trap-depth criteria only. Targeted SOC/hybrid checks on finalists (Tier 3). |
| Score calibration | Fewer than 10 molecules, mixed device stacks | Leave-one-out rank check reported as a consistency check. Architecture recorded per point. |
| Molecular feasibility / novelty | Non-purchasable or already-reported molecules | SA score now. Vendor and literature lookup before any novelty claim [P]. |
| LLM leakage or goalpost moving | Inflated benefit | Protected fields refused. Hash-locked spec. Holdout excluded from prompts. Proposals logged with usage. |

## 8. Roadmap and resource estimates

| Stage | Content | Compute | Gate |
|---|---|---|---|
| 0 (done) | Prototype, synthetic pilot, tests | laptop | — |
| 1 | Package 53 returns → `CachedDFTOracle`. Pose generator from SMILES. Electronic descriptors (η, N_new, Δχ) from returned r2SCAN statics. | Analysis only | Package 53 outputs |
| 2 | Score calibration: SCAPS sweep table for the reference device; fit μ/N0/N_res on the experimental set; leave-one-out check | Laptop (SCAPS) | **PI review of calibration data and thresholds** |
| 3 | Tier-1 MLIP screen of the A/B/C pools (~230 molecules × ~5 poses), committee σ | ~1–3 GPU-days [I] | Validated MLIP on an independent bundle (`TRAINING_PILOT.md`); **PI approval** |
| 4 | Matched-budget A/B/C with a live LLM proposer and DFT on the top ~10 per arm, ≥3 replicates where affordable | ~30 r2SCAN relax+static ≈ 10⁴ node-hours [I] | **PI approval; LONI desk** |
| 5 | Tier 3 on 2–3 finalists: tighter settings, alternate poses, SOC check, moisture/thermal stress models | ~10³–10⁴ node-hours [I] | Stage-4 evidence |

Stop conditions from the brief apply: no campaign, change of physical model or thresholds, or production
integration without PI approval.
