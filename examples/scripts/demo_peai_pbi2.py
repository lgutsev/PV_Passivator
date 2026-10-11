"""Demo: PEAI ion pair on the r2SCAN PbI2-rich slab -> poses -> linked r2SCAN job bundle."""
import json, sys
from collections import Counter
from pathlib import Path
from pvpassivator.surface import load_surface, find_sites
from pvpassivator.poses import load_passivator, tag_substrate, generate_poses
from pvpassivator.vaspprep import adsorption_bundle

SCAN = Path("C:/Users/lguts/OneDrive/Desktop/Research/2026/Perovskites/GIEC_Projects/VASP/SCAN")
REAG = Path("data/cache/X_OutPackLite_REAG_NEW_Mar15_2025/X_OutPackLite_REAG_NEW_Mar15_2025")
out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cache/demo_jobs")

slab, rec = load_surface(SCAN / "PbI2_Surf_335_SCAN/FAPI_333_x_PbI2_Surf/CONTCAR",
                         surface_id="FAPbI3_100_PbI2rich_333x_r2scan")
slab = tag_substrate(slab)
peai = load_passivator(REAG / "PEAI/CONTCAR", "PEAI", total_charge=0, multiplicity=1,
                       component_charges={"C8H12N": +1, "I": -1}, chemical_form="ion_pair_salt",
                       protonation_note="PEA+ = C6H5CH2CH2NH3+ (protonated amine); I- counterion kept")
print("components:", [(c.label, len(c.indices)) for c in peai.components])
sites = [s for s in find_sites(slab, rec, "top") if s.kind == "undercoordinated_Pb"]
# anchor = the iodide of the ion pair over a 5-coordinate Pb (restores octahedral coordination)
iod = [i for i in peai.donor_indices if peai.atoms[i].symbol == "I"]
poses = generate_poses(slab, rec, peai, sites, heights=(3.2,), tilts=(0, 30), n_azimuth=4,
                       anchors=iod, max_poses=24, seed=1)
print(f"{len(poses)} clash-free poses from {poses[0][1].extra['n_candidates']} candidates;",
      Counter(p[1].site_id for p in poses).most_common(3))
cplx, spec = poses[0]
plan = adsorption_bundle(out, "PbI2rich_PEAI_pose0", slab, rec, cplx,
                         [("PEAI", peai.atoms, 0, 1, 1)], n_units=1, overwrite=True)
print(plan["equation"]); print(json.dumps({k: v["static"]["kpoints"] for k, v in plan["jobs"].items()}))
print(open(out / "PbI2rich_PEAI_pose0/complex/relax/INCAR").read())
