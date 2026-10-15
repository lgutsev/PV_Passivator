# 2D Coulomb kernel truncation for charged slabs: PBE test results (package 55, 14 Oct 2026)

Generator `examples/scripts/prepare_truncation_test.py`; analyzer `examples/scripts/analyze_truncation_test.py`
(run on the returned outputs; JSON in `data/cache/pkg55/analysis.json`, not committed).

- **Slab:** 504-atom FAI-terminated FAPbI₃(100) Schottky slab, fixed geometry, normal along x.
- **Cells:** 55, 70 and 90 Å along the normal; in-plane area A = 19.42 × 18.91 Å².
- **Runs:** PBE, Γ only, 19 single points. No LDIPOL on charged runs.

## Results

| Setup | 55 Å | 70 Å | 90 Å | Verdict |
|---|---|---|---|---|
| neutral, dipole correction | −2375.6904 | −2375.7398 | −2375.7103 | spread 49 meV, not monotonic |
| neutral, truncation | −2375.6872 | −2375.7264 | −2375.7075 | spread 39 meV; equals dipole correction within 3–13 meV |
| q+1, uncorrected | −2373.3013 | −2372.4597 | −2371.5539 | drifts +50 meV/Å (expected) |
| q+1, truncation | −2371.7908 | −2371.8296 | −2371.8100 | spread 39 meV, the same pattern as neutral |
| q−1, uncorrected | −2375.2798 | −2375.8326 | −2376.0630 | drifts |
| q−1, truncation | −2380.4054 | −2381.1589 | −2382.2386 | **falls 1.83 eV** |

Energies in eV (σ→0).

- **Truncation with the dipole correction** (neutral, 70 Å) is identical to truncation alone (0.0 meV), so the two can be
  combined without double counting.
- **Neutral and q+1 spread.** The 39–49 meV spread with cell length is the same, and not monotonic, for neutral and
  q+1 runs. It is a numerical baseline: the FFT grid changes with the cell. It cancels in differences at fixed length:
  IP(ΔSCF) = E(q+1) − E(q0) = 3.896 / 3.897 / 3.898 eV, flat to 2 meV.
- **q+1 field.** The fitted vacuum field equals the charged-sheet value e/(2ε₀A) = 0.2449 eV/Å on both sides at every
  length, so the whole hole sits in the slab.

## Diagnosis of q−1 (no rerun needed)

The vacuum field of the truncated q−1 runs is only part of the sheet value:

| L (Å) | 55 | 70 | 90 |
|---|---|---|---|
| field / sheet value | 0.63 | 0.37 | 0.25 |

By Gauss's law this ratio is the fraction of the added electron still enclosed by the slab region; the rest has moved
into the vacuum.

**Mechanism.** An uncompensated −1 sheet makes the electron potential energy fall linearly into the vacuum by
e²/(2ε₀A) per Å. Across half the vacuum that drop is about 3.5, 5.4 and 7.8 eV at 55, 70 and 90 Å. Every one of those
exceeds the binding of the slab's lowest empty state, vacuum − LUMO = 3.12 eV. So the electron lowers its energy by
moving toward the cell boundary, and it does so more as the vacuum grows. That is why E(q−1) keeps falling with L, and
why the ΔSCF "EA" (4.7 → 6.5 eV) is meaningless.

**Interpretation.**
- This is not a truncation bug. An isolated, negatively charged slab is ill-posed in any vacuum large enough to
  converge: the electron is unbound in its own field.
- Truncation is validated for cations (q+1) on this slab.

**Rule for anions (q < 0):** use a counter-charge, or keep (field × half-vacuum) < (vacuum − LUMO). The field is
e|q|/(2ε₀A), so a larger lateral cell relaxes the limit; the analyzer's vacuum-field check detects leakage directly.

**Smaller cells.** A 2×2 cell (area ≈ 160 Å²) at q = −2 has a field about 4.5× larger than here: ≈ 1.1 eV/Å per face.
Anion frames in such cells leak at any vacuum that converges the energy.
