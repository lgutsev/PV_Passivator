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
| Slab normal | x (IDIPOL=1, DIPOL = 0.5019 0.5136 0.4809 fractional). The passivant sits on the high-x face (atoms reach x ≈ 50–52 Å; the bare slab ends at 47.9 Å). |
| Periodicity | Neutral cell with 70 Å along x, LDIPOL dipole correction, no Coulomb truncation. |
| Calibration | Package 55 (same slab, PBE): printed p = −0.07744 / −0.07748 / −0.07744 e·Å at 55 / 70 / 90 Å. Predicted ΔV = −0.0380 eV; LOCPOT plateaus give −0.0380 eV (high − low) at both 55 and 70 Å. **The label is cell-length independent and matches the potential to 0.1 meV.** |
| Truncated runs | Print a different p (−0.0589 e·Å with truncation + dipole at 70 Å), so only LDIPOL-only runs are labels. |
| Sign | ΔV < 0 means the passivated face's vacuum level is lower than the back face's. To first order the change relative to the bare slab, ΔV_ads = ΔV(complex) − ΔV(slab), is the work-function shift of the passivated face [I: assumes the back face is unchanged]. |

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
| PFPACl | −0.027 → −0.054 | −0.015 |
| PFPAI | −0.045 → −0.118 | −0.079 |
| TFPACl | −0.159 → −0.174 | −0.135 |
| TFPAI | −0.179 → −0.205 | −0.166 |

Static single points reproduce the last relax frame to ≤ 1 meV (e.g. TFPACl −0.1744 vs −0.1740).

**Findings:**
- **Passivants differ by about 0.4 eV.** The ordering is chemical: the fluorinated PFPA and TFPA salts lower the
  passivated face's vacuum level, while the PEA salts and THCl raise it.
- **Restructuring during one relaxation moves ΔV by up to 0.14 eV** (PEACl, PEATFA, THCl). That is comparable to the
  differences between passivants.
- So the target resolves both passivant binding and restructuring. The precision a model needs is about 20–30 meV.

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
| Test frames, THCl (raw-index / ΔV eV / max displacement in the outer 6 Å) | 0 / +0.141 / 0.00; 42 / +0.054 / 0.10; 84 / +0.128 / 0.38; 126 / +0.107 / 0.48; 168 / +0.072 / 0.58; 210 / +0.040 / 0.75; 252 / +0.073 / 1.03; 294 / +0.036 / 1.26 |
| Test frames, TFPAI | 0 / −0.179 / 0.00; 69 / −0.258 / 0.24; 139 / −0.190 / 0.70; 208 / −0.210 / 1.19; 277, 346, 416, 485 / −0.209 … −0.205 / 1.26 (converged tail) |
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
