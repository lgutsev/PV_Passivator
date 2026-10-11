"""Export the structures behind the status page (real archive slabs + generated poses)."""
import json
from pathlib import Path
import numpy as np
from pvpassivator.surface import load_surface, find_sites, unwrap_along
from pvpassivator.poses import (load_passivator, tag_substrate, generate_poses, remove_fragments,
                                place, clash_report, min_image_distances)
from pvpassivator.surface import Site
from pvpassivator.vaspprep import adsorption_bundle

SCAN = Path("C:/Users/lguts/OneDrive/Desktop/Research/2026/Perovskites/GIEC_Projects/VASP/SCAN")
REAG = Path("data/cache/X_OutPackLite_REAG_NEW_Mar15_2025/X_OutPackLite_REAG_NEW_Mar15_2025")
OUT = Path("data/cache/viz"); OUT.mkdir(parents=True, exist_ok=True)

def xyz(a, ax):
    # rotate so the surface normal is vertical (y in viewer): map normal axis -> y
    order = {0: [1, 0, 2], 1: [0, 1, 2], 2: [0, 2, 1]}[ax]
    p = a.positions[:, order]
    lines = [str(len(a)), "x"] + [f"{s} {x:.3f} {y:.3f} {z:.3f}" for s, (x, y, z) in zip(a.get_chemical_symbols(), p)]
    return "\n".join(lines), np.array(a.cell.array)[:, order][order].tolist()

def entry(key, title, a, ax, caption, hl=None, ghosts=None, facts=None):
    s, cell = xyz(a, ax)
    order = {0: [1, 0, 2], 1: [0, 1, 2], 2: [0, 2, 1]}[ax]
    g = [{"elem": e, "pos": [float(p[i]) for i in order]} for e, p in (ghosts or [])]
    return {"key": key, "title": title, "xyz": s, "cell": cell, "caption": caption,
            "highlight": [int(i) for i in (list(hl) if hl is not None else [])], "ghosts": g, "facts": facts or {}}

pas = load_passivator(REAG / "PEAI/CONTCAR", "PEAI", 0, 1, {"C8H12N": 1, "I": -1}, "ion_pair_salt")
iod = [i for i, s in enumerate(pas.atoms.get_chemical_symbols()) if s == "I"][0]
nidx = [i for i, s in enumerate(pas.atoms.get_chemical_symbols()) if s == "N"][0]
out = []

# 1. PbI2-rich slab + sites
slab, rec = load_surface(SCAN / "PbI2_Surf_335_SCAN/FAPI_333_x_PbI2_Surf/CONTCAR", "FAPbI3(100) PbI2-rich r2SCAN")
slab = tag_substrate(slab)
sites = find_sites(slab, rec, "top")
upb = [s for s in sites if s.kind == "undercoordinated_Pb"]
out.append(entry("pbi2_slab", "PbI2-rich slab, r2SCAN CONTCAR", slab, rec.normal_axis,
    "Archived relaxed slab. Highlighted: the 9 five-coordinate Pb on the top face, used as anchor sites.",
    [s.atom_index for s in upb],
    facts={"composition": "45 FA+ 54 Pb2+ 153 I-", "formal charge": "0", "normal": "a (x)",
           "thickness": f"{rec.slab_thickness:.1f} A", "vacuum": f"{rec.vacuum:.1f} A",
           "faces": "PbI2 / PbI2 (symmetric)", "top-face sites": "9 Pb(5-coord), 18 I, 9 FA"}))

# 2. PEAI poses
poses = generate_poses(slab, rec, pas, upb, heights=(3.2,), tilts=(0, 30), n_azimuth=4,
                       anchors=[iod], max_poses=24, seed=1)
for k, (full, spec) in enumerate(poses[:3]):
    ads = np.where(full.arrays["origin"] > 0)[0]
    out.append(entry(f"pose{k}", f"PEAI pose {k + 1} on PbI2-rich", full, rec.normal_axis,
        f"Starting geometry, not relaxed. I- placed {spec.height} A above 5-coordinate Pb; tilt {spec.tilt_deg:g} deg, "
        f"azimuth {spec.azimuth_deg:g} deg. PEA+ kept with its counterion.", ads,
        facts={"pose id": spec.pose_id, "site": spec.site_id, "clash-free poses": f"{len(poses)} of {spec.extra['n_candidates']}",
               "reaction": "bare slab + PEAI -> complex", "electrons (r2SCAN PAW)": "2682 + 56 = 2738"}))
adsorption_bundle("data/cache/demo_jobs", "PbI2rich_PEAI_pose0", slab, rec, poses[0][0],
                  [("PEAI", pas.atoms, 0, 1, 1)], n_units=1, overwrite=True)

# 3. FAI-rich slab, Schottky pair, PEAI exchange start
fslab, frec = load_surface(SCAN / "FAI_Surf_335_SCAN/FAPI_333_x_FAI_Surf/CONTCAR", "FAPbI3(100) FAI-rich r2SCAN")
fslab = tag_substrate(fslab)
fs = find_sites(fslab, frec, "top")
fa = [s for s in fs if s.kind == "FA"][0]
ia = min((s for s in fs if s.kind == "exposed_I"),
         key=lambda s: min_image_distances(fslab.cell.array, fslab.pbc, np.array([s.position]), np.array([fa.position]))[0, 0])
out.append(entry("fai_slab", "FAI-rich slab, r2SCAN CONTCAR", fslab, frec.normal_axis,
    "Archived relaxed slab. Highlighted: the FA+ and I- chosen for the Schottky pair (nearest top-face pair).",
    [i for i in range(len(fslab)) if fslab.arrays["fragment_id"][i] == fslab.arrays["fragment_id"][fa.atom_index]] + [ia.atom_index],
    facts={"composition": "45 FA+ 36 Pb2+ 117 I-", "formal charge": "0", "thickness": f"{frec.slab_thickness:.1f} A",
           "vacuum": f"{frec.vacuum:.1f} A", "faces": "FAI / FAI (symmetric)"}))
ghosts = [(fslab[i].symbol, fslab.positions[i]) for i in range(len(fslab))
          if fslab.arrays["fragment_id"][i] == fslab.arrays["fragment_id"][fa.atom_index] and fslab[i].symbol != "H"] + \
         [("I", fslab.positions[ia.atom_index])]
vac, vrec = remove_fragments(fslab, frec, [fa.atom_index, ia.atom_index], "FAI_Schottky_top")
out.append(entry("schottky", "FAI Schottky pair (V_FA + V_I)", vac, frec.normal_axis,
    "Removed FA+ and I- shown as translucent spheres. Formal charge stays 0, so no electronic compensation is implied.",
    [], ghosts, facts={"removed": ", ".join(vrec.defects[-1]["removed_units"]), "formal charge": str(vrec.formal_charge),
                       "composition": "C44 H220 I116 N88 Pb36"}))
cat = [i for i, c in enumerate(pas.components) if c.formula == "C8H12N"][0]
cidx = pas.components[cat].indices
cation = pas.atoms[cidx]
best = None
for tilt in (0, 20, 40):
    for az in range(0, 360, 30):
        for h in (0.0, 0.5, 1.0):
            site = Site("vac", "FA", "top", fa.position, None)
            m = place(vac, vrec, pas, site, cidx.index(nidx), h, tilt, az, mol_atoms=cation)
            from ase import Atoms
            i_atom = Atoms("I", positions=[fslab.positions[ia.atom_index]], cell=vac.cell, pbc=True)
            i_atom.set_array("atom_id", np.array([m.arrays["atom_id"].max() + 1]))
            i_atom.set_array("fragment_id", np.array([m.arrays["fragment_id"].max() + 1]))
            i_atom.set_array("origin", np.array([2]))
            mm = m + i_atom
            d = min_image_distances(vac.cell.array, vac.pbc, mm.positions, vac.positions).min()
            if not clash_report(vac.cell.array, vac.pbc, vac.get_chemical_symbols(), vac.positions,
                                mm.get_chemical_symbols(), mm.positions, 1.0):
                if best is None or d > best[0]:
                    best = (d, vac + mm, tilt, az, h)
if best:
    full = best[1]
    out.append(entry("exchange", "PEAI filling the Schottky pair", full, frec.normal_axis,
        f"Starting geometry, not relaxed. PEA+ NH3 group placed at the old FA+ site (tilt {best[2]} deg, azimuth {best[3]} deg); "
        "I- returned to the vacancy. Reaction: Schottky slab + PEAI -> complex (exchange-type).",
        np.where(full.arrays["origin"] > 0)[0], ghosts=[],
        facts={"min contact": f"{best[0]:.2f} A", "balance": "atoms and charge balanced"}))
else:
    print("no clash-free exchange pose")

# 4. reference
ref = pas.atoms.copy(); ref.cell = np.eye(3) * 25; ref.pbc = True; ref.center()
out.append(entry("ref", "PEAI reference (ion pair)", ref, 2,
    "Archived PBE CONTCAR, made whole across the periodic boundary. Needs an r2SCAN re-relaxation (job prepared).",
    [], facts={"components": "C8H12N+ (PEA+), I-", "total charge": "0", "multiplicity": "1", "box": "25 A cube"}))
json.dump(out, open(OUT / "structures.json", "w"))
print([(o["key"], len(o["xyz"])) for o in out])
