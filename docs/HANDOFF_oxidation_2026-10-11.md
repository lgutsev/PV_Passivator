# Handoff: surface iodide oxyanions (`oxidation.py`), 2026-10-11

Status: **uncommitted** working-tree changes. Do not commit or push without the owner's OK. Note that
`main` tracks an upstream that no longer exists (`git status` says "upstream is gone").

## What was added

| File | Change |
|---|---|
| `src/pvpassivator/oxidation.py` | New. Replaces one surface iodide by IO⁻/IO₂⁻/IO₃⁻/IO₄⁻ and enumerates orientations. |
| `src/pvpassivator/ledger.py` | `identify_units`: a halide with an O within `HALIDE_OXO_BOND` (2.3 Å) is treated as covalent, so `IO3` is recognised as one unit instead of an ionic I⁻ plus three unknown O fragments. Pb–I stays ionic. |
| `tests/test_oxidation.py` | New: idealised bond lengths/angles for n = 1–4, iodate poses on the 5-layer `toy_slab`, composition, `atom_id`/`fragment_id`/`origin` arrays, formal charge preserved, balanced reaction, minimum Pb–O, non-halide site rejected. |
| `README.md` | Status row for `oxidation.py`. |

`pytest`: 34 passed (system Python 3.14, ase + numpy). The local MACE env has ase and MACE but no pytest.

## Design decisions (keep these)

- **End states only.** The product keeps the halide's −1 charge, so the cell stays formally neutral. The ledger entry
  is `slab + n/2 O2(g) -> slab(IO_n-)` via `oxidation_reaction`, with `O2_REFERENCE` as a triplet gas-phase reservoir.
  The module deliberately builds **no O₂–surface encounter geometries or O₂ reaction paths**: triplet→singlet chemistry
  is outside what the downstream potentials are meant to describe. Don't add O₂ adsorption poses here without the owner.
- **Idealised starting geometry**, relaxed downstream: `IODINE_OXYANIONS` holds X–O length, O–X–O angle, label and
  oxidation state (IO₃⁻: 1.81 Å, 99°, C₃v). Only iodine is parameterised; Br/Cl raise `ValueError`.
- **Orientation = lone-pair direction L** plus rotation φ about it, plus an optional lift of the halide along the outward
  normal. `default_lone_pairs` gives out, in, and 45° tilts toward ±(nearest Pb) and ±(normal × that). φ = (0, 60°)
  covers staggered/eclipsed for C₃v.
- **Filters**: covalent-radius clash test × 0.85 against all substrate atoms except the oxidised halide, plus an explicit
  **minimum Pb–O of 2.05 Å** (an O bridging to surface Pb is an intended motif; closer is not). The explicit Pb–O check was
  added after a bug: the general clash test only fires below ~1.80 Å for O–Pb, so 1.86–1.97 Å contacts slipped through.
- Arrays follow `poses.py`: the halide keeps its `atom_id`; new O atoms get ids ≥ `ADSORBATE_ID0`, `origin = 1`, and the
  halide's `fragment_id`.

## First real use (for context)

A 204-atom PbI₂-terminated FAPbI₃(100) 2×2 slab (bottom half fixed), iodate on one in-plane surface iodide: 11 of 24
orientations pass the filters. Each was pre-relaxed with a universal MLIP (MACE-MP-0 medium, float32, FIRE, fmax 0.05):
about 1.2 s per step on 16 CPU threads, a few hundred steps each. Several orientations converge to the same minimum, which
motivates the dedup item below.

## Next tasks, in priority order

1. **Deduplicate relaxed poses.** Fingerprint by sorted O–Pb / I–Pb / O–H(N) distances around the oxyanion (or a
   SOAP/RMSD after `atom_id` alignment) and group within a tolerance; report a representative and the multiplicity.
2. **`classify_final_state` support for oxyanions**: intact IOₙ, I–O cleavage, O transferred to Pb (Pb–O < 2.4 Å with
   I–O > 2.3 Å), oxyanion lifted out of the layer (halide height above the outer Pb plane), proton transfer to an oxyanion O.
3. **Pb(IO₃)₂ unit builder**: two in-plane iodides that share one surface Pb → both oxidised; a record that declares the
   Pb(IO₃)₂ unit. A companion removal that yields the PbI₂ vacancy through `remove_fragments` conventions, with the
   reaction `slab(Pb(IO3)2) -> slab(V_PbI2) + Pb(IO3)2(bulk)` and a declared bulk reference (never a default energy).
4. **Halide-site classifier**: in-plane (bridging two Pb) vs apical vs subsurface, from Pb coordination and height, so
   callers stop passing raw atom indices.
5. **VASP bundle for the oxidation reaction** in `vaspprep.py`: slab, product and an O₂ reference job (ISPIN = 2,
   NUPDOWN = 2, ≥ 15 Å box), with a fingerprint check that all three share functional/cutoff/PAW settings.
6. **Declared O₂ reference correction** in the ledger: an optional, explicitly named per-O₂ energy shift with its source,
   applied in `delta_e` only when passed. Never a built-in default.
7. **Water next to an oxyanion**: reuse `generate_poses` with the oxidised slab as substrate.
8. **CLI**: `pvpass oxidize <POSCAR> --halide <idx> --n 3 --out <dir>` writing start structures plus a JSON of specs.

## Repository conventions to keep

- Open repository: keep docs, code comments and commit messages free of grant names, proposal wording or funding context.
- Never guess charges or multiplicities; refuse instead (see `load_passivator`). Ledger entries must pass `assert_balanced`.
- Tests use the synthetic `toy_slab` from `tests/test_surface_poses_prep.py`; no proprietary structures in `tests/` or `data/`.
