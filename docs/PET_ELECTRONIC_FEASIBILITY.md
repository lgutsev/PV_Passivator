# PET electronic-property feasibility (2 Nov 2026)

Question: can the methods of Abbott et al., *Unconstrained learning of equivariant atomistic properties*
([arXiv:2610.31183](https://arxiv.org/abs/2610.31183), v1 22 Oct 2026), add electronic observables to the
passivation/degradation workflow more cheaply than DFT on every frame?

Tags: **[D]** demonstrated in the paper, **[P]** proposed or future work in the paper, **[R]** read from our files,
**[M]** measured here, **[I]** inferred.

## 1. What the paper does and does not cover

| Item | Paper | Relevance to us |
|---|---|---|
| Targets | [D] molecular dipoles and polarizabilities (SPICE-α), NMR shielding tensors (ShiftML-molecules), auxiliary-basis density coefficients (SCFBench, PBE/def2-SVP, PySCF) | All on **isolated molecules**. Only periodic example: a basis-incompleteness test on 64-atom bulk FCC Au. |
| Equivariance | [D] learned by rotational augmentation, per-irrep heads, O(3) quadrature averaging at inference | Useful when the full tensor is defined. For a periodic slab only the normal component of the dipole is defined. |
| Elements | [D] H C N O F P S Cl (+Na Mg Ca, Au in side tests). SPICE has I; **no Pb anywhere** | No released checkpoint covers Pb. |
| Periodic dipole / polarization | not addressed (no Berry phase, slabs, surfaces, ESP or work function) | Our target must be defined by us (Sect. 4). |
| Charge, spin | [D] no total-charge input: densities need an a-posteriori electron-count fix; errors concentrate on charged dimers. No spin input | Charged slabs (q+1 hole states, package 55) are out of scope. |
| Density → SCF acceleration | [D] ~40 % fewer SCF cycles in PySCF (Gaussian aux. basis) | Not transferable to VASP plane waves/PAW without a projection layer that does not exist. |
| Extensions | [P] charge-aware models, partial charges, spin density, ESP, LDOS, Hamiltonians | ESP and LDOS are exactly what we would want. They are proposals, not results. |
| Code, checkpoints | metatrain/metatensor/metatomic. Paper checkpoints: "DOI to be confirmed" | metatrain's PET supports `cartesian` rank-1/2 and `spherical` targets per system or per atom (docs). Not installed here. |
| Cost | [D] PET 4–6 µs/atom inference on H100; training 3–23 h on 16–48 GPUs for 0.06–1.7 M molecules | Inference is negligible. The real cost is labels. |

PET-MAD (separate work) is a universal PET energy model. MAD-1.5 / PET-MAD-1.5 is reported at r2SCAN with 102
elements; it is a plausible backbone to fine-tune a new head on, not an electronic model. Not verified locally.

## 2. What we already have

| Data | Content | Electronic label |
|---|---|---|
| Package 53 (`D:\MLIP_Work_Folder\hpc_smoke_tests\53_vasp_pvpass_r2scan_eads_v1`) | [R] r2SCAN+D3(BJ) relaxations of the 504-atom FAI-Schottky slab and 9 salt complexes, 3 OUTCAR segments each, plus final statics | **Yes: LDIPOL/IDIPOL=1 on every SCF**, so each ionic step prints the converged slab-normal dipole. No LOCPOT/CHGCAR kept (`lib_pvpass.sh` deletes CHG/CHGCAR; LOCPOT not exported). |
| Package 55 (PBE truncation test) | [R] same slab, fixed geometry, LOCPOT planar averages | Used here to **calibrate** the dipole ↔ vacuum-step relation. |
| `data/prerelax_a0/dataset_v2` | [R] energies/forces of package 53 frames; production split: test = TFPAI, THCl | No dipoles stored (added by the new extractor). |
| Legacy AIMD (`GIEC_Projects/VASP/NAMD_ML`) | [R] 324-atom bare FAPbI₃ slab, PBE 400 eV, no vdW, no dipole correction, 2 × 2000 frames, with NAMD eigenvalue/NAC text | Excited-state/NAC data on a different profile; no ground-state dipole. Geometry source only. |
| Hydrated surfaces | none at r2SCAN | A hydration pilot needs new structures and labels. |

## 3. Question and observable

**Question.** How does passivant binding, and the surface restructuring during it, change the electrostatic potential
step across the FAPbI₃ slab (equivalently the work-function shift of the passivated face)? Can a model predict it
for an unseen passivant without a DFT calculation per frame?

**Observable:** the slab-normal dipole p (e·Å), reported as the vacuum-level step
ΔV = V(high-x vacuum) − V(low-x vacuum) = p·e/(ε₀A) = 180.951·p/A eV.

We chose it for three reasons:
- It is the one dipole component that is well defined for a periodic slab with vacuum. In-plane components are
  Berry-phase quantities and are not labelled.
- Package 53 already labels it on every frame.
- It maps directly onto the interface-dipole and work-function language used for passivation.

Density-derived descriptors (Δn, λ(z), `density.py`) are deferred: no CHGCARs are kept.

Band alignment and defect states are separate validation problems and are **not** addressed here. Nothing here is a
ground-state substitute for excited-state calculations or nonadiabatic couplings. The NAMD/NAC chain stays as it is.

### Reference convention [M]

| Item | Convention |
|---|---|
| Slab normal | x (IDIPOL=1, DIPOL = 0.5019 0.5136 0.4809 fractional). The passivant sits on the **low-x face**: the ammonium sits in the A-site vacancy and the tail points to −x (cation atoms at x ≈ 19–28 Å). The complex cells place the whole slab 2.5–4 Å higher in x than the bare-slab cell; this does not change the dipole of a neutral cell. |
| Periodicity | Neutral cell with 70 Å along x, LDIPOL dipole correction, no Coulomb truncation. |
| Calibration | Package 55 (same slab, PBE): printed p = −0.07744 / −0.07748 / −0.07744 e·Å at 55 / 70 / 90 Å. Predicted ΔV = −0.0380 eV; LOCPOT plateaus give −0.0380 eV (high − low) at both 55 and 70 Å. **The label is cell-length independent and matches the potential to 0.1 meV.** |
| Truncated runs | Print a different p (−0.0589 e·Å with truncation + dipole at 70 Å), so only LDIPOL-only runs are labels. |
| Sign | ΔV < 0 means the passivated (low-x) face's vacuum level lies above the back face's. ΔV_ads = ΔV(complex) − ΔV(slab), so to first order the work-function shift of the passivated face is −ΔV_ads [I: assumes the back face is unchanged]. |

## 4. Labels extracted (no new DFT) [M]

`examples/scripts/extract_slab_dipoles.py` → `data/cache/pet_dipole/pkg53_dipoles.extxyz` (3091 relaxation frames
plus 7 statics; all frames have a converged dipole; molecular references carry no IDIPOL and are skipped).

| Job | ΔV first → last relax frame (eV) | ΔV_ads (eV) |
|---|---|---|
| slab_FAI_Schot | −0.038 → −0.039 (swings −0.09…+0.08 during the relax) | 0 |
| PEAAc | +0.271 → +0.194 | +0.233 |
| THCl | +0.141 → +0.036 | +0.075 |
| PEACl | +0.157 → +0.016 | +0.055 |
| PEAI | +0.036 → +0.007 | +0.046 |
| PEATFA | +0.076 → −0.032 | +0.007 |
| PFPACl | −0.027 → −0.054 (**unfinished**: no DONE, export of 1 Nov ends mid-SCF; not an endpoint) | (−0.015) |
| PFPAI | −0.045 → −0.118 | −0.079 |
| TFPACl | −0.159 → −0.174 | −0.135 |
| TFPAI | −0.179 → −0.205 | −0.166 |

Static single points reproduce the last relax frame to ≤ 1 meV (e.g. TFPACl −0.1744 vs −0.1740).

**Findings:**
- **Passivants differ by about 0.4 eV.** The ordering is chemical. The fluorinated PFPA and TFPA salts *raise* the
  passivated face's work function (by up to ~0.17 eV), consistent with their C–F tails pointing out. The PEA salts and
  THCl lower it. (An earlier draft of this note had the face, and so this sign, reversed.)
- **The anion matters as much as the cation.** The four PEA salts share one cation, yet ΔV_ads spans +0.007 (TFA)
  to +0.233 eV (acetate). Cl vs I with the same cation differs by 0.01–0.03 eV.
- **On the surface every salt is ionic.** Each bound cation keeps three H on N, and acetate and TFA stay
  deprotonated. This contrasts with the gas-phase references below.
- **Restructuring during one relaxation moves ΔV by up to 0.14 eV** (PEACl, PEATFA, THCl). That is comparable to the
  differences between passivants.
- So the target resolves both passivant binding and restructuring. The precision a model needs is about 20–30 meV.

### Gas-phase salt dipoles: not available, and not ion pairs [R]

**No DFT dipole exists for the salts.** The `ref_*` statics ran without LDIPOL/IDIPOL and kept no CHGCAR, so the
extractor skips them.

**Getting one is cheap but not yet run.** A re-run of the 9 statics with IDIPOL=4 and DIPOL at the molecular centre
would supply it. Each is 15–30 atoms at Γ in a ~25 Å box.

**The relaxed gas-phase references are mostly not ion pairs.** The table below uses the closest H on the static
CONTCAR, with the distance to the anion and to N (Å).

| Salt | H–anion | H–N | State |
|---|---|---|---|
| PEAAc | 1.03 (O) | 1.66 | proton on the acid: neutral amine···acid |
| PEATFA | 1.08 (O) | 1.50 | proton on the acid: neutral amine···acid |
| PFPACl | 1.36 (Cl) | 1.63 | proton on Cl: amine···HCl (HCl bond length is 1.27) |
| TFPACl | 1.39 (Cl) | 1.55 | proton on Cl |
| PEACl, THCl | 1.44–1.45 | 1.42–1.43 | proton shared |
| PEAI, PFPAI, TFPAI | 1.96–2.10 (I) | 1.13–1.23 | ammonium iodide with a stretched N–H |

Consequences:
- **A formal ion-pair estimate is meaningless for PFPACl.** Charges of ±1 at the N and Cl positions would give 14 D.
  The real dipole belongs to a neutral H-bonded complex, so it is smaller and depends on where the proton sits on a
  flat potential. That makes it a poor descriptor of ΔV_ads, because on the surface the cation and anion are
  separated ions. It is also a hard target for any model.
- **Isolated molecules are the paper's demonstrated domain.** H, C, N, O, F and Cl are all in SPICE, so a released
  SPICE-α PET dipole model could predict these dipoles zero-shot. Its reference, however, is ωB97M-D3(BJ), not
  r2SCAN.
- **For E_ads [I]:** the salt reference energy includes a proton back-transfer that differs by family (acids, Cl,
  I). This is atom-balanced and valid as a gas-phase reference, but it is not an "ion pair" reference. It should be
  kept in mind when comparing E_ads across anion families.

### Sensitivity to counter-ion position [M]

The owner's earlier Gaussian runs on PFPACl showed the gas-phase dipole changing sharply with small counter-ion
moves. Physically this is dp/dr, the Born (atomic polar tensor) charge. It is largest for a proton on the flat
N–H···Cl coordinate, where a 0.3 Å shift moves about one charge across the bond.

**On the slab, Cl⁻ is a separate ion** (closest H–Cl 2.23–2.33 Å, no proton transfer), but the observable is just as
sensitive. Along the four Cl relaxations:

| Job | Cl moves along x | ΔV moves | dp/dx_Cl (e, r) |
|---|---|---|---|
| PFPACl | 0.06 Å | 68 meV | +0.56 (0.38) |
| TFPACl | 0.09 Å | 103 meV | −0.44 (−0.20) |
| PEACl | 0.15 Å | 161 meV | −0.74 (−0.54) |
| THCl | 0.15 Å | 111 meV | −0.44 (−0.30) |

On this cell, 0.1 Å of a full charge along the normal is 49 meV. Chloride alone explains only part of each swing:
the motion is collective and electronically screened.

Consequences:
- **Gas-phase salt dipoles are not passivant descriptors.** A single geometry is arbitrary; at best use a thermal
  average.
- **Geometry error propagates directly.** Hitting 25 meV needs normal-direction ion positions to ≲ 0.05 Å. Pilot A
  evaluates on DFT geometries for this reason; MLIP-MD geometries add their own error, which must be measured
  separately.
- **Report MD averages ⟨ΔV⟩ with their spread**, not snapshots. The pass criterion for MD use is the ranking of
  averages, not per-frame error.
- **Prefer the per-atom charge head.** There dp/dx_i is a learned effective charge, which is the right structure for
  this sensitivity.
- **Gaussian check (gas phase).** A frequency job prints APT charges, i.e. Born charges. An H-transfer scan along
  N···Cl with the dipole at each step would quantify the effect for PFPACl directly. The reference would be the
  owner's Gaussian level, not r2SCAN.

### Gas-phase descriptors: what the paper's domain offers, and a test [M]

**What the paper's gas-phase models cover:**

| Property | Status |
|---|---|
| Dipole / polarizability | Demonstrated (SPICE-α). |
| NMR shielding | ¹H and ¹³C only, PBE0. ¹⁹F is not reported; no iodide; nothing bound to Pb or I. |
| Partial charges | Not demonstrated. |

**Limits for our passivants:**
- Ions are uncertain: there is no total-charge input, and an ion's dipole depends on the origin.
- Salt pairs are unsafe: see the proton-transfer table above.
- For a handful of molecules Gaussian is cheaper than any model. A model pays off only for screening at scale or for
  averaging over thermal conformers.

**The test.** `examples/scripts/cation_dipole_test.py prepare` cuts each bound cation from its final r2SCAN complex.
- **Frame:** x is the slab normal and the ammonium N is the origin. All cations are +1.
- **Inputs (`data/cache/pet_dipole/cation_gaussian/`):** a Gaussian input per cation, plus a neutral cation + anion
  pair input where the anion can be identified (Cl, acetate, TFA). The pair dipole is origin-independent.
- **Not yet run:** there is no local DFT engine (no PySCF, Psi4, xTB or WSL).
- **Afterwards:** `analyze` reads the `.log` files and ranks p_x against ΔV_ads.

Geometry-only proxies, computed now:

| Proxy | Spearman vs ΔV_ads | Comment |
|---|---|---|
| F_out = Σ x(F) − x(N) | +0.77 (tie-aware) | Only the fluorinated/non-fluorinated split (5 of 9 tie at 0). Within the 4 fluorinated salts it is −0.60, i.e. reversed: PFPA has more outward F, yet TFPA shifts ΔV more. |
| Tail length (centroid − N) | −0.75 | The same split; −0.20 within the 5 non-fluorinated salts. |

**What it can and cannot settle:**
- A cation-only descriptor cannot capture the 0.23 eV anion spread within the PEA salts. The pair dipoles are the
  part of the test that can.
- A good rank would make the cation or pair dipole a cheap screening descriptor. It is not a ΔV predictor: the
  calculation has no slab, no screening and no thermal averaging.

## 5. Offline comparison: point-charge baseline [M]

`examples/scripts/fit_dipole_baseline.py`.

**Model:** p = Σ q_i (x_i − ⟨x⟩). Each atom carries an element charge plus a correction for its bonded-environment
class. Measuring from the mean position ⟨x⟩ keeps every frame neutral. The fit is ridge regression; an unseen
environment falls back to the element charge.

**Data split:**
- Production split (`dataset_v2`): test = TFPAI and THCl as whole jobs; train = the other 7 complexes plus the slab.
- Leave-one-job-out (LOJO) over all 10 jobs, scored on the final ΔV.

| Ridge λ | Train MAE | Test MAE (TFPAI / THCl) | Change-from-frame-0 MAE, test (persistence) | LOJO final-ΔV MAE | LOJO Spearman ΔV / ΔV_ads |
|---|---|---|---|---|---|
| 0.01 | 14 meV | 777 / 4132 meV | 40 / 82 (38 / 61) meV | 2.19 eV | 0.03 / 0.13 |
| 1 | 15 meV | 367 / 1482 meV | 17 / 49 (38 / 61) meV | 0.82 eV | 0.16 / 0.22 |
| 100 | 20 meV | 71 / 53 meV | 31 / 53 (38 / 61) meV | 0.27 eV | 0.44 / 0.48 |

An earlier non-neutral variant (charges measured from the cell centre) gave 13 meV train, 275 / 558 meV test and
Spearman 0.53. Its origin dependence makes it unphysical.

PFPACl's "final" frame in the LOJO scores is its last exported frame, not a converged one. Its frames remain valid
training data, since every frame has a converged SCF.

**Symmetry [M]:**
- p is unchanged by a 37° rotation about the normal.
- p changes sign under the mirror x → −x.
- Both hold by construction. They are the right rotation tests for any learned model here. Full O(3) tests do not
  apply, because only the normal component is labelled.

**Reading:**
- Within one chemistry, fixed charges fit to 15–20 meV.
- Across passivants they do not transfer. The best λ still errs by 50–70 meV on held-out jobs, 0.27 eV on LOJO, and
  ranks passivants at Spearman ≤ 0.5.
- Along a trajectory, it beats "no change" (persistence) by only 7–8 meV.
- The dipole is not a geometric function of fixed charges: polarization and charge transfer at the interface carry
  it. This is exactly where a learned model has to earn its place, and also where 9 chemistries give little data to
  learn from. **Effective sample size is 10 jobs, not 3091 frames:** frames within one relaxation are strongly
  correlated.

## 6. What PET would and would not add

- **Equivariance.** The paper's key result, learned equivariance for vectors and tensors, has little to act on here.
  The labelled quantity is one component in a fixed slab frame. Training PET on a Cartesian rank-1 target with
  rotational augmentation would need in-plane components we do not have and that are ill-defined for a periodic
  slab. Zero-filling them would teach a false label.
- **Workable formulations.**
  - (a) A scalar head in the slab frame.
  - (b) Better: a per-atom charge head with p = Σ q_i(x_i − ⟨x⟩). This is the "displaced partial charges" form the
    paper mentions, and it is exactly rotation-consistent about the normal.
  - Both are available in metatrain for PET and also for MACE, so the choice of backbone is open. Our MACE fine-tune
    infrastructure (`training_data.py`, `prepare_mace_finetune.py`) already handles the same frames.
- **Needs.** No new labels for the first test. It does need a new head (fine-tuning) on a pretrained backbone; no
  released checkpoint predicts our observable. PET-MAD-1.5 (r2SCAN) is the natural PET backbone [not verified
  locally].
- **Out of scope:** density prediction and SCF acceleration (Gaussian auxiliary basis, PySCF only), and charged slabs.

## 7. Pilot specification

### Pilot A: zero new DFT (recommended now)

| Item | Specification |
|---|---|
| Structures | Package 53 r2SCAN+D3(BJ) relaxation frames. Train/valid: the 7 complexes PEAAc, PEACl, PEAI, PEATFA, PFPACl, PFPAI, TFPACl plus slab_FAI_Schot (valid = last 15 % of each job, as in `dataset_v2`). Test: 16 frames from the held-out jobs, listed below, plus all of their frames. |
| Test frames, THCl (raw-index / ΔV eV / max displacement in the passivated 6 Å, low-x face) | 0 / +0.141 / 0.00; 42 / +0.054 / 0.09; 84 / +0.128 / 0.32; 126 / +0.107 / 0.40; 168 / +0.072 / 0.42; 210 / +0.040 / 0.45; 252 / +0.073 / 0.52; 294 / +0.036 / 0.56 |
| Test frames, TFPAI | 0 / −0.179 / 0.00; 69 / −0.258 / 0.40; 139 / −0.190 / 1.00; 208 / −0.210 / 1.18; 277, 346, 416, 485 / −0.209 … −0.205 / 1.23 (converged tail) |
| Observable | ΔV = 180.951·p/A (eV), with the convention in Sect. 3 |
| Reference | The OUTCAR dipole of the converged SCF (r2SCAN+D3(BJ), 520 eV, Γ, LDIPOL IDIPOL=1) |
| Models | (0) the point-charge baseline above; (1) PET-MAD-1.5 backbone with a per-atom charge head (metatrain), and optionally (2) the same head on MACE. Energy heads are not touched; production models are not modified. |
| Metrics | Per-frame ΔV MAE on test; MAE of the change from frame 0 vs persistence; LOJO final-ΔV MAE and Spearman over 10 jobs; ΔV_ads sign. Symmetry: rotation about the normal (invariance) and mirror (sign flip) on all test frames, with the O(3)-averaged vs single-orientation spread reported for PET. |
| Go / no-go (set in advance) | Go if LOJO final-ΔV MAE ≤ 30 meV, Spearman ≥ 0.8 and ΔV_ads signs correct for ≥ 9 of 10 jobs. Otherwise reject for this use. |
| Cost | 0 DFT. About 1–3 GPU-hours per fine-tune [I: about 1.5 M atom-environments per epoch]; LOJO means 10 fine-tunes, so about 10–30 GPU-hours. Plus a one-time metatrain environment install. |

### Pilot B: hydration (deferred; needs new DFT and the owner's approval)

- **Structures:** PEAI and TFPAI final complexes with 1, 2 and 4 waters at the passivant and at exposed Pb, pre-relaxed
  with the A0 MACE pre-relaxer. That gives 16 frames, kept as a separate chemical condition (test only).
- **DFT:** r2SCAN+D3(BJ) static, LDIPOL IDIPOL=1, LVHAR, LOCPOT kept.
- **Measured cost of one static (package 53):** about 5400 s on 2 × 64 cores, ≈ 3 node-hours or 190 core-hours.
  Sixteen frames come to ≈ 48 node-hours, about 3,100 core-hours. **Not launched.**

The legacy AIMD trajectories (PBE-400, no vdW, no dipole correction) are another profile. Mixing them with r2SCAN
labels is ruled out.

## 8. Recommendation

**Proceed with Pilot A only. Defer Pilot B. Reject the density/SCF-acceleration and excited-state uses.**

- **Pilot A.** The labels already exist, they are calibrated against LOCPOT, and they resolve a 0.4 eV passivant
  spread and 0.1 eV restructuring changes. The physical baseline fails to transfer across passivants. Testing a
  learned model is therefore a well-posed, zero-DFT question with a pre-set go/no-go.
- **What PET's novelty buys.** Learned tensor equivariance on molecules is not what matters here. The decisive
  factors are the backbone's chemical transfer and the 10-job effective sample size.
- **Pilot B.** Spend DFT on hydration only if Pilot A passes.
- **Rejected uses.** Density/SCF acceleration has no plane-wave path. A ground-state dipole cannot stand in for
  excited-state calculations or NACs.
