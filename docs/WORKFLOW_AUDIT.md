# Workflow audit: stage 1 (11 Oct 2026)

All research files were read only. Archives were extracted only into `data/cache/` (gitignored).
Tags: **[R]** recovered from files, **[I]** inferred, **[?]** unresolved, needs the owner.

## 1. Data sets and calculation profiles

| Profile | Where | Settings |
|---|---|---|
| `pbe_d3zero_335` (spreadsheet source) | `VASP/2025/X_OutPackLite_PbI2_Surf_335_PBE_Mar15_2025.zip`, `VASP/2024/X_OutPackLite_FAI_Surf_335_PBE_Dec26_2024.zip`, `VASP/2025/X_OutPackLite_REAG_NEW_Mar15_2025.zip` | [R] PBE, IVDW=11, ENCUT=520, PREC=Accurate, LASPH, ISMEAR=0/σ=0.05, ISIF=4, NSW=99. Slab KPOINTS: PbI2 1×1×1 M, FAI 2×2×1. |
| `r2scan_d3bj_335` | `VASP/SCAN/{PbI2,FAI}_Surf_335_SCAN` | [R] METAGGA=R2SCAN, IVDW=12, s8=0.7898, a1=0.4948, a2=5.7308, ENCUT=520, ISIF=4 (2 in IVac_ISIF2). |
| `pbe_d3zero_reag2024` | `VASP/2024/REAG_Jan22_2024_OutPackLite` | [R] PBE+D3(zero), Γ-only. **Not compatible with the r2SCAN slabs.** |
| `namd_scf_pbe400` | `NAMD_Project/Step2.../NAMD_Template/INCAR` | [R] PBE, 400 eV, PREC=Normal, NBANDS=960, Γ, no SOC, no vdW. |
| `aimd_pbe400` | `NAMD_Project/Step1_MD_Instructions` | [R] VASP AIMD, 400 eV, POTIM=1 fs, 300 K. Equilibration uses SMASS=-1 rescaling; then 5000 steps of NVT (Nosé, MDALGO=2) or NVE (Andersen p=0). No constraints, no vdW. |

The common settings are as follows:
- [R] No archive keeps OUTCAR, vasprun.xml, CHGCAR or POTCAR. Energies come from the last OSZICAR line (F).
- [I] Convergence can only be inferred: fewer steps than NSW suggests convergence; reaching NSW means unconverged.
- [R] EDIFF and EDIFFG are unset, so VASP defaults apply.
- [R] ISIF=4 relaxes cell shape. Slab and complex cells can therefore differ; this is a convergence concern.
- [R] No selective dynamics. No dipole correction (IDIPOL unset) despite asymmetric slabs that the notes say carry an internal dipole.
- [I] Neutral electron counts assume standard PAW_PBE valences (Pb_d 14, I 7, N 5, C 4, H 1). Folders with a `_N` suffix use NELECT=N. Examples:
  - PbI2-rich 2682 → `_2681` = +1, `_2683` = −1.
  - FAI-rich 2178.
- [?] The POTCAR variants still need confirming.

## 2. Surfaces (alpha-FAPbI3, "333_x" = 3×3 in-plane, surface normal along **a/x**)

| Termination | Composition | Formal charge | Notes |
|---|---|---|---|
| PbI2-rich bare | C45H225I153N90Pb54 = 45 FA⁺ 54 Pb²⁺ 153 I⁻ | 0 | [R] ≈22 Å vacuum [I] |
| FAI-rich bare | C45H225I117N90Pb36 = 45 FA⁺ 36 Pb²⁺ 117 I⁻ | 0 | [R] |
| FAI-rich with one FAI Schottky pair | C44H220I116N88Pb36 | 0 | [R] Salt passivators fill V_FA+V_I here, which makes the process an exchange-type reaction. |

## 3. Energy conventions (`VASP_Adsorption_Energy.xlsx`)

The adsorption energy is defined as follows [R]:
- E_ads = E(complex) − E(slab) − Σ n·E(ref).
- Negative values mean binding. Units are eV per passivator.
- The total is divided by n only for multi-adsorbate rows.
- No two-face normalisation is applied, because adsorption is one-sided.

How the references are handled [R]:
- Salts are referenced as one neutral ion-pair cell (for example PEAI). This keeps atoms and charge balanced.
- Halide-in-vacancy rows use μ_X, taken from X₂(solid)/8 or from a charged anion (NELECT+1).
- [?] No Fermi-level or electrostatic correction is visible for the charged anion rows.

Spreadsheet errors found by the audit agent:
- BIG sheet cells G22 and G23 mix 333 and 335 energies (+936 eV).
- In the FINAL sheet, the PEATFA and PEAAc labels are swapped in column H.
- In the BIG sheet, the "Schot PEAI" and "FAVac PEA" rows have identical energies.

## 4. Reproduction from raw data (`pvpass reproduce examples/recovered_cases_pbe335.json`)

| Case | Raw E_ads (eV) | Sheet (eV) | Result |
|---|---|---|---|
| PbI2-rich + PEAI / PEAAc / FAI | n/a | −1.770 / −0.960 / −1.242 | **Bare-slab OSZICAR is empty.** The sheet slab value (−2624.69515) is not found in any checked archive. |
| FAI-rich Schot + PEAAc | **+0.071** | −1.583 | Complex hit NSW=99. The archived Schottky slab (−2418.032, 68 steps) is ~2.06 eV lower than the sheet slab (−2415.975). |
| FAI-rich Schot + PEATFA | **+0.044** | −1.707 | Same slab issue. |
| FAI-rich Schot + TFPACl | n/a | −2.191 | TFPACl reference folder not found. |

All complexes balance atom-for-atom against slab + neutral salt (checked from POSCARs).

The sheet value −1.583 is reproduced exactly from (−2586.155) − (−2415.975) − (−168.597). However, neither −2586.155 nor −2415.975 exists in any archive checked, so **the spreadsheet rests on an earlier, unarchived set of runs**. The archived re-relaxed slab makes these salts appear unbound. This could reflect a lower-energy slab reconstruction or ISIF=4 cell changes, and it is the decisive open question.

## 5. Charge displacement and work function

**Charge displacement:**
- [R] `Summary_ChargeDisp.xlsx` holds only 3 scalar "avg del Q" values (PEAI 0.062, PEATFA 0.054, PEAAc 0.050) and an area of 365.4 Å².
- [?] It contains no profiles, and no integration region or sign is recorded.
- [?] No CHGCARs are archived. The density-difference case cannot be reproduced until fragment CHGCARs are located or recomputed.

**Work function:**
- [R] Computed with LVHAR, without dipole correction.
- [I] `locpot.dat` looks like VASPKIT planar-average output. WF = vacuum plateau minus E_F.
- Reported values: PbI2-rich 5.68 eV, FAI-rich 4.95 eV, FAI Schottky 5.07 eV.
- With asymmetric slabs, the two vacuum levels differ. Both sides should be reported.

## 6. NAMD chain

**FolderPrep.py** [R]:
- Reads `XDATCAR_FINAL` and takes the last 2000 frames by negative indexing, with no length check.
- Writes folders `0001..2000` (zero-pad width = len(str(NSCF))).
- `pvpassivator.namd_adapter` reproduces this layout and adds:
  - window validation
  - duplicate-frame and continuity checks
  - one global atom map
  - a manifest that keeps geometry, trajectory, electronic and NAC statuses separate.

**Frame-count inconsistency** [R]:
- FolderPrep writes 2000 frames, `inp` NSW is 1998, and `train_N2AMD_prep.py` uses 1999 frames.
- NAC rows are aligned to frames by row index, with the column choice uncertain.
- `Convert_N2AMD.py` keys do not match the training keys (`nac_scalars` vs `nac_scalar`).

**Hard-coded values** [R]:
- `inp` BMIN/BMAX = 744/745 and NBANDS=960 are fixed. They must be recomputed whenever passivators change the electron count.

**LAMMPS converter notebook** [R]:
- Hard-codes an orthorhombic cell [25.02, 17.69, 50.32] and a type→element map that includes Ni/O (a NiO interface).
- Sorts atoms alphabetically, which loses atom order.
- Measures distances without PBC.
- Not reused. No secrets were found.

## 7. Questions for the owner (scientifically consequential)

1. Which runs produced the spreadsheet slab energies, −2624.69515 (PbI2-rich bare) and −2415.97515 (FAI Schottky)? Is the archived −2418.032 Schottky slab the intended reference?
2. Which is the target profile for labels: PBE+D3(zero) or r2SCAN+D3(BJ)? The two must not be mixed.
3. Where is the TFPACl/PFPACl/THCl reference set for PBE 335?
4. Is `FolderPrep.py` the intended NAMD preparation script, and should the canonical window be 2000 frames or 1999?
