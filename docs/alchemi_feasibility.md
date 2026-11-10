# NVIDIA ALCHEMI as a batched-relaxation backend: feasibility

Branch `acat-adsorption-search`, 2026-11-10. Scope: can `nvalchemi-toolkit` 0.3.0 run the MLIP relaxations of the
adsorbate-arrangement search (`pvpassivator.adsearch`) faster than the native ASE path, with the same numbers?
This note is about **backend correctness and speed only**. Whether the model is right for the chemistry is a
separate question, answered by the DFT audit in `docs/acat_feasibility.md` §8 and not by anything here.

## 1. Recommendation: defer

**Do not adopt ALCHEMI for this workload on this workstation. Keep the adapter as an optional, tested backend.**

- Correctness gates pass: same model, same numbers, through both interfaces (§5).
- The speed target does not: end-to-end throughput is **0.93×** native in double precision and **1.10–1.14×** in
  single precision, against a 1.5× adoption target (§6).
- The reason is physical, not a tuning problem: one 597-atom slab already saturates the GPU, so time per system
  does not fall with batch size, and an 8 GB card holds one such system in double precision or three in single.
- The large speed-up that *is* available is precision: single precision is 6.5× faster than double in the native
  path, with relative energies inside one composition changed by ≤ 0.002 eV (§7). That needs no new dependency.

Revisit ALCHEMI if (a) systems become small enough that several fit and the GPU is not saturated by one, (b) a
24 GB-class GPU is available, or (c) batched MD replicas or NEB are wanted, where it offers features the native
path lacks.

## 2. Readiness

| Item | State |
|---|---|
| Pinned toolkit | `nvalchemi-toolkit` 0.3.0 (tag commit 77d3c9e930a3eab5f44b61a7ffaf161609e13be2), public beta |
| Platform | Windows 11; not a documented platform, works with two workarounds (§4) |
| Model wrapper | `MACEWrapper`, documented upstream; used with mace-torch 0.3.16 (the toolkit's extra pins 0.3.15) |
| Checkpoint | `MACE-matpes-r2scan-omat-ft.model`, sha256 `8f147ecf…d9bba9`, ScaleShiftMACE, 2 interactions, cutoff 6.0 Å, 89 elements, 9.06 M parameters, stored in float64, one head |
| Why this checkpoint | It is the model the adsorbate search runs natively today, so both paths use the *same* weights. It is a foundation model, **not validated for this interface**. No GRACE checkpoint was substituted or bridged. |
| Dispersion | Excluded on both sides. The search adds D3(BJ) through torch-dftd; the toolkit's D3 needs a parameter download that was not made. All comparisons here are MACE-only against MACE-only. |
| cuEquivariance, `torch.compile` | Off on both sides (not installed; no MSVC compiler). |
| Adapter | `src/pvpassivator/alchemi_backend.py`, optional import; the native path is untouched and remains the default |
| Production dependencies | Unchanged. Separate environment (§3). |

## 3. Stack

Environment `D:/MLIP_Work_Folder/envs/alchemi_v030`: a venv with system site packages over the existing `defects`
environment, installed with constraints pinning `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`,
`numpy==2.5.3` (an unconstrained install replaces the CUDA build of torch with a CPU build of 2.15).

| Package | Version | | Package | Version |
|---|---|---|---|---|
| Python | 3.12.14 | | mace-torch | 0.3.16 |
| torch | 2.11.0+cu128 (CUDA 12.8) | | e3nn | 0.4.4 |
| nvalchemi-toolkit | 0.3.0 | | ase | 3.29.0 |
| nvalchemi-toolkit-ops | 0.5.0 | | numpy | 2.5.3 |
| warp-lang | 1.19.0 | | tensordict | 0.15.5 |
| physicsnemo | 2.2.2 | | torchvision | 0.26.0+cu128 |

GPU: NVIDIA GeForce RTX 5070 Laptop, 8151 MiB, driver 591.44.

## 4. What had to be worked around in 0.3.0

1. **`from_checkpoint` fails on Windows drive-letter paths.** The adapter loads the model with `torch.load` and
   wraps it.
2. **Dynamics call `torch.compile`, which needs a C compiler.** The adapter sets `TORCHDYNAMO_DISABLE=1`.
3. **`map_location` leaves four e3nn constant buffers on the CPU** for this checkpoint; the first GPU call fails
   with a device mismatch. The adapter moves the model explicitly, as MACE's own calculator does.
4. **A standalone optimizer keeps stepping systems that have converged** until the slowest one in the batch is
   done. The adapter records each system at its own first convergence, so results equal a run alone, but the extra
   evaluations are wasted (14 % of them in the batch-2 run of §6).
5. **Out of memory does not fail on Windows.** Past the card's capacity the driver spills into shared system
   memory and a batch runs 2–17× slower per system. A real CUDA out-of-memory error, when it does occur, leaves the process
   unusable, so the benchmark runs every timing point in its own process.

## 5. Correctness (passes)

Tolerances were fixed before the GPU runs, from the CPU toy check (1e-13 eV, 7e-12 eV/Å): **1e-6 eV and
1e-6 eV/Å in float64**. Float32 is reported, not gated, and judged against the 0.10 eV per cell the search must
resolve.

Panel: eight real structures from the V1 audit pool on the PbI₂-terminated FAPbI₃ (100) slab, 582–597 atoms: two
single-molecule modes, two legacy two-melamine arrangements, three random arrangements, and one unrelaxed
start with forces up to 11 eV/Å. (The bare slab is not in the panel.)

| Check (float64, GPU) | Largest difference |
|---|---|
| ALCHEMI batch 1 vs native MACE calculator | 5×10⁻¹² eV, 1.3×10⁻¹⁴ eV/Å |
| Batches of 2 and 4 vs batch 1 | 4.5×10⁻¹¹ eV, 1.4×10⁻¹⁴ eV/Å |
| Reordered batch | 4.4×10⁻¹¹ eV |
| Same structure twice in a batch; different companion | 3.8×10⁻¹¹ eV; 3.6×10⁻¹² eV |
| Repeat evaluation | 1.8×10⁻¹² eV |
| Periodicity off along the normal (37 Å vacuum, 6 Å cutoff) vs full periodicity | 0 eV, 1×10⁻¹⁴ eV/Å |

Float32, same panel: forces agree with the native float32 calculator to 8×10⁻⁶ eV/Å. **Total energies of a
batched float32 evaluation depend on the batch by up to 0.03 eV** (the same structure twice in one batch differs
by 0.026 eV). The adapter therefore recomputes each final energy and the raw forces one system at a time; with
that, batch 2 and batch 1 relaxations end within 0.003 eV and 2.5×10⁻⁵ Å of each other.

Relaxation, all arms (§6): frozen atoms moved by exactly 0 at every step in every run; atom ids, roles and
molecule ids preserved; every input has one terminal record; the same step counts alone and in a batch. The
step-limit and broken-structure cases are covered by `tests/test_alchemi_backend.py` on CPU fixtures (4 checks),
not by the GPU runs, where nothing failed.

## 6. Speed (does not reach the target)

Per evaluation, largest structure (597 atoms), median of 5 warm calls, each point in a fresh process. ALCHEMI
times include conversion, neighbour list, model call and transfer back.

| Precision | Backend, systems per batch | s per system | Peak allocated / reserved, MiB | Cold first call, s |
|---|---|---|---|---|
| float64 | native, 1 | 1.15 | 3658 / 4924 | 1.7 |
| float64 | ALCHEMI, 1 | 1.14 | 3656 / 4922 | 3.4 |
| float64 | ALCHEMI, 2 | 2.41 | 7168 / 9688 (over) | 5.5 |
| float64 | ALCHEMI, 3 | 3.46 | 10702 / 14474 (over) | 10.1 |
| float64 | ALCHEMI, 4 | 4.38 | 14247 / 19266 (over) | 15.8 |
| float32 | native, 1 | 0.15 | 1838 / 2452 | 1.1 |
| float32 | ALCHEMI, 1 | 0.14 | 1837 / 2452 | 2.3 |
| float32 | ALCHEMI, 2 | 0.15 | 3596 / 4868 | 6.7 |
| float32 | ALCHEMI, 3 | 0.14 | 5363 / 7978 | 3.0 |
| float32 | ALCHEMI, 4 | 1.03 | 7135 / 9672 (over) | 4.5 |
| float32 | ALCHEMI, 6 | 1.92 | 10682 / 14470 (over) | 10.0 |
| float32 | ALCHEMI, 8 | 2.36 | 14204 / 19240 (over) | 18.3 |

Relaxation of eight structures from their unrelaxed starts, fixed cell, the search's face-wide mobile set,
fmax 0.1 eV/Å, 150-step cap. The pool is 8, not the 128 proposed: no larger pool of validated candidates exists.
One run per arm; not repeated.

| Precision | Arm | Wall, s | Converged | Per hour | vs native | Peak reserved, MiB |
|---|---|---|---|---|---|---|
| float64 | native ASE FIRE | 429.5 | 8/8 | 67 | 1.00 | 5770 |
| float64 | ALCHEMI FIRE2, batch 1 | 461.0 | 8/8 | 62 | 0.93 | 5852 |
| float32 | native ASE FIRE | 66.1 | 8/8 | 436 | 1.00 | 2892 |
| float32 | ALCHEMI FIRE2, batch 1 | 60.0 | 8/8 | 480 | 1.10 | 2928 |
| float32 | ALCHEMI FIRE2, batch 2 | 58.0 | 8/8 | 496 | 1.14 | 5156 |

Model setup is 3 s native and under 0.1 s more for the wrapper; it is not in the wall times. Batch 2 in float64
was not run as a relaxation because it does not fit (table above). **Inflight batching was not run:** it can
recover at most the 14 % of evaluations that static batch 2 wastes on finished systems, and time per system does
not fall with batch size, so it cannot reach 1.5× here.

Not matched between the arms: the optimizer (ASE FIRE vs the toolkit's FIRE2, both default parameters). FIRE2
used 381 steps in total against 364, so the 0.93–1.14× figures mix optimizer and backend. Batch 1 vs batch 2 is
the same engine and is the clean batching comparison: 1.03×.

## 7. Two findings that matter more than the backend

**Single precision.** Native float32 against native float64 on ten pool structures: total energies shift by
−0.013 to −0.017 eV, almost uniformly; differences *within* the two-molecule composition change by at most
**0.002 eV**; forces by 1.4×10⁻⁴ eV/Å. Relaxations take the same number of steps. Throughput rises from 67 to
436 structures per hour. A reasonable protocol is float32 for relaxation and one float64 evaluation of the final
geometry; this is a proposal, not yet the search default.

**The stopping rule is too loose for the resolution the search needs.** From identical starts and to the same
fmax of 0.1 eV/Å, FIRE2 ended 0.01–0.34 eV *lower* than ASE FIRE in the same model, with atoms up to 0.30 Å
apart; five of eight structures differ by more than 0.10 eV. Single-point agreement is 10⁻¹¹ eV, so this is not a
wrapper error: on this soft surface a 0.1 eV/Å residual leaves several tenths of an eV unrelaxed. The lowest
arrangement is the same in both, but the order of the others changes (relative energies of the six two-molecule
structures: native 0.12, 0.29, 0, 0.44, 0.75, 0.64 eV; FIRE2 0.02, 0.45, 0, 0.39, 0.58, 0.79 eV). Before the
random-versus-evolutionary benchmark is run, fmax must be tightened until energies are stable to well under
0.10 eV; the value has not been determined. The V1 DFT audit is unaffected: it compares model and DFT at fixed
geometries.

## 8. Integration with the adsorbate search

Nothing needs to change now. If the backend is taken up later:

- `adsearch.run_search` evaluates one candidate at a time. A batched evaluator needs the proposal loop to hand
  over a generation (evolutionary arm) or a chunk (random arm) at once, without changing the attempt budget,
  selection or deduplication; rejected and duplicate proposals must still consume budget.
- `alchemi_backend.relax` already returns what `AseRelaxEvaluator` stores (final atoms with ids and roles,
  `model_energy`, `model_forces`, status, steps), so the manifest format is unchanged.
- D3 must be added identically on both sides first (the toolkit's D3, after its parameters are fetched and
  checked against torch-dftd at fixed geometries), or the two paths are not the same model.

## 9. Reproduce

```
PY=D:/MLIP_Work_Folder/envs/alchemi_v030/Scripts/python.exe ; M=<MACE checkpoint> ; POOL=data/cache/adsearch/v1_pool
PVPASS_MACE_MODEL=$M $PY -m pytest tests/test_alchemi_backend.py
$PY examples/scripts/alchemi_benchmark.py singlepoint $POOL --model $M --out OUT --names <8 names>
$PY examples/scripts/alchemi_benchmark.py steptime    $POOL --model $M --out OUT --dtypes float64 --max-batch 4 --repeats 5
$PY examples/scripts/alchemi_benchmark.py steptime    $POOL --model $M --out OUT --dtypes float32 --max-batch 8 --repeats 5
$PY examples/scripts/alchemi_benchmark.py relax       $POOL --model $M --out OUT --dtype float64 --batch 1 --names <8 names>
$PY examples/scripts/alchemi_benchmark.py relax       $POOL --model $M --out OUT --dtype float32 --batch 2 --names <8 names>
```
The pool comes from `examples/scripts/adsearch_v1_pool.py`. Reports are JSON files in `OUT` (git-ignored here:
`data/cache/adsearch/alchemi_bench*`).

## 10. Not done

- Pool of 128 and repeated timed runs; mixed-size workload; inflight batching (reasons in §6).
- D3 on the ALCHEMI side.
- Any larger GPU. Every number here is from one 8 GB laptop card.
- MD replicas and NEB (optional stage; not started).
- Agreement of final binding motifs between the optimizers beyond energies and displacements.
