# Package 53: status of the returned outputs and a proposed revision 4

Written 2026-10-16. **Proposal only.** Nothing has been staged, sent to the desk or submitted.

## Update 2026-10-19

- **Rev-2 gate is now local** (re-export 20261019_1632, desk 2ffd5b5).
  - `outputs/bulk/relax`: DONE, reached the required accuracy in 26 ionic steps, TOTEN −181.14799 eV.
  - Lattice 6.3789 → 6.3857, 6.4734 → 6.4641, 6.3025 → 6.3055 Å, a max change of 0.144 %, so `check_bulk_gate.sh` PASS.
  - `outputs/smoke/static`: DONE, 134–157 s per SCF step, consistent with rev 1.
- **Slab relaxes:**
  - The walltime kill on tasks 0–5 left nothing in `outputs/`, because outputs were copied only after VASP returned. The progress sits in `runs/` on LONI only.
  - On resubmission `run_step` continues from `runs/<job>/relax/CONTCAR` and archives the killed segment as `OUTCAR.segN` / `vasprun.xml.segN.gz`. The 72 h segments are not lost, but they can't be measured here until they are exported.
  - Tasks 6–9 run until about 2026-10-19 18:02.
- **Revision 4 now has three items,** all script-level. Inputs, INCARs and geometries are unchanged, so restarted segments stay comparable.
  1. **Walltime-safe outputs (commit ff1370a, pushed):** VASP runs in the background, outputs are saved every 20 min, and `--signal=B:USR1@1800` triggers STOPCAR plus an immediate save. This fixes the "nothing to export after a walltime kill" problem. It also applies to 48, 55 and 60, which use the same pattern.
  2. **Wrapper-guard wait** of 10 min, with a transient exit code (item 1 below).
  3. **Optional relax-only EDIFF 1E-5** (item 2 below).

  The order of the sections below is unchanged. Revision 4 waits for the owner's decision at the desk.

## What came back

The only export is `20261014_0951`, at `D:\MLIP_Work_Folder\hpc_smoke_tests\53_vasp_pvpass_r2scan_eads_v1`.

| Unit | State in the export |
|---|---|
| 9 reference relaxes (`ref_*`, 15–28 atoms) | DONE, "reached required accuracy", 18–99 ionic steps, 0.4–2.4 h each, no SCF warnings |
| Rev-2 bulk relax and smoke (jobs 1093085, 1093084) | not exported, because of the path-only exporter bug that the desk fixed in 9039f8f. The desk reports the gate passed on LONI, but no local copy exists. `outputs_rev1/` holds only the rev-1 runs. |
| Slab array 1093393, tasks 0–5 | logs only (snapshot 2026-10-13 18:02); `outputs/complex_*/relax/` holds only `POTCAR.used` + `vasp_binary.txt` |
| Slab tasks 6–9, all statics, k-checks | not started |

`pvpass labels` on this folder: 0/19 statics usable, so **no E_ads yet**. That is expected: no slab relax is DONE, and no static has run.

## The "6 failed" tasks

The desk table shows `11/42, 6 failed`, but only one of the six actually failed.

- **Task 0, `slab_FAI_Schot`:** a real failure, but a transient one. It exited at start with `vasp_gam ... is not a complete script after 3 min`. That message comes from the module-wrapper guard in `lib_pvpass.sh`, which waits 12 × 15 s. Tasks 1–5 got a complete wrapper at the same moment. The desk has already resubmitted it (manifest note, 2026-10-14 10:26).
- **Tasks 1–5, `complex_PEAAc/PEATFA/PEAI/PEACl/PFPACl`:** not failures. Their logs stop at the module help banner because they were exported about 16 h into a 72 h run. The desk's own note from the same export says "slab relaxes running". The export counter scored them as failed because they had no DONE marker.

These tasks started on 2026-10-13 at 18:02, so their 72 h walltime ends **today at about 18:02 (cluster time)**. Each will then exit `did not reach EDIFFG; resubmit`, unless it converged earlier. A resubmit continues from CONTCAR, and revision 3 keeps the XML segments.

## Cost check

The rev-1 smoke run measured about 145 s per SCF step on 2 nodes for the 532-atom complex. The relaxes use CG (IBRION 2) with EDIFF 1E-6, so an ionic step probably needs 10–20 SCF steps, or about 25–50 min. That gives roughly 85–170 ionic steps per 72 h segment. A 500-atom CG relax to EDIFFG −0.02 often needs 150–300 steps, so expect **2–3 segments per slab**. That works out to about 2900–4300 node-h for the ten slabs, above the owner-confirmed 1000–2000 node-h. These numbers are an estimate [P]; today's walltime exit will give the measured ionic-step rate.

## Proposed revision 4 (not staged)

Revision 4 changes only scripts and relax settings. Geometry, cell, k-mesh, references and static settings stay the same, so energies remain comparable with everything already run.

1. **Wrapper guard (`stage_label_set.py` → `lib_pvpass.sh`):** wait up to 10 min (40 × 15 s) instead of 3 min. When the guard does give up, exit with a distinct code (75, transient) and write `outputs/<job>/<step>/TRANSIENT`, so the export counts it as "retry", not "failed". This prevents another task-0-style loss.
2. **Optional (owner decision), relax-only EDIFF 1E-6 → 1E-5** in the 10 slab-type relax INCARs. Statics keep 1E-6. This cuts SCF steps per ionic step by an estimated 20–35 %. Forces at EDIFFG −0.02 eV/Å are not limited by a 1E-5 SCF on these systems, and the labels come from the statics. Resubmitted segments of tasks 0–5 and the unstarted tasks 6–9 would pick it up, so geometries would mix the two relax tolerances. Labels are unaffected, but the training exporter would see relax frames from both. Its per-frame SCF check accepts both, as long as each frame is consistent with its own INCAR.
3. **Not proposed:** switching the optimizer (IBRION 1/FIRE) mid-campaign, changing NSW/walltime, or touching tasks that are running.

**Recommended order:**
1. Wait for today's walltime exit and a fresh export.
2. Read the ionic steps per segment and the max force trend from the tasks 1–5 OUTCARs.
3. Decide on item 2.
4. Stage revision 4 with the desk (numbering and intake are the desk's).

Separately, the desk should correct the status line to read "1 transient (resubmitted), 5 running at export".
