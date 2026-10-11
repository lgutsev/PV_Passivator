"""Versioned VASP preparation profiles and linked job bundles (dry-run; never submits).

POTCARs are licensed: we write ``POTCAR.spec`` (ordered symbols + variant names) and a
``potcar_check`` that a user runs on the cluster to assemble and hash them locally.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from .vasp_io import normalized_fingerprint

# PAW variants recovered from electron counts in archived runs (Pb_d=14 e) [inferred]
POTCAR_VARIANTS = {"Pb": "Pb_d", "I": "I", "C": "C", "N": "N", "H": "H", "O": "O", "F": "F",
                   "Cl": "Cl", "Br": "Br", "S": "S", "Cs": "Cs_sv", "Na": "Na_pv", "K": "K_sv"}
ZVAL = {"Pb_d": 14, "I": 7, "C": 4, "N": 5, "H": 1, "O": 6, "F": 7, "Cl": 7, "Br": 7, "S": 6,
        "Cs_sv": 9, "Na_pv": 7, "K_sv": 9}

_COMMON_R2SCAN = {"METAGGA": "R2SCAN", "LASPH": ".TRUE.", "IVDW": 12, "VDW_S8": 0.7898,
                  "VDW_A1": 0.4948, "VDW_A2": 5.7308, "ENCUT": 520, "PREC": "Accurate",
                  "ISMEAR": 0, "SIGMA": 0.05, "LREAL": "Auto"}

PROFILES: dict[str, dict] = {
    # exactly the recovered SCAN-set settings (for continuity / re-running legacy cases)
    "r2scan_d3bj_recovered/relax": {**_COMMON_R2SCAN, "IBRION": 2, "ISIF": 4, "NSW": 99,
                                     "LVHAR": ".TRUE.", "LORBIT": 11, "LWAVE": ".TRUE."},
    # proposed label profile v1: fixed common cell, tight convergence, dipole correction
    "r2scan_d3bj_v1/relax": {**_COMMON_R2SCAN, "IBRION": 2, "ISIF": 2, "NSW": 300, "EDIFF": 1e-6,
                             "EDIFFG": -0.02, "NELM": 200, "LWAVE": ".FALSE.", "LCHARG": ".FALSE.",
                             "LDIPOL": ".TRUE.", "IDIPOL": None},
    "r2scan_d3bj_v1/static": {**_COMMON_R2SCAN, "IBRION": -1, "NSW": 0, "EDIFF": 1e-6, "NELM": 200,
                              "LVHAR": ".TRUE.", "LORBIT": 11, "LCHARG": ".TRUE.", "LAECHG": ".FALSE.",
                              "LDIPOL": ".TRUE.", "IDIPOL": None},
    # frozen-fragment density for charge displacement: identical grid enforced via NGX/Y/Z
    "r2scan_d3bj_v1/fragment_density": {**_COMMON_R2SCAN, "IBRION": -1, "NSW": 0, "EDIFF": 1e-6,
                                        "NELM": 200, "LCHARG": ".TRUE.", "LDIPOL": ".TRUE.",
                                        "IDIPOL": None, "NGX": None, "NGY": None, "NGZ": None},
    "r2scan_d3bj_v1/molecule_relax": {**_COMMON_R2SCAN, "IBRION": 2, "ISIF": 2, "NSW": 300,
                                      "EDIFF": 1e-6, "EDIFFG": -0.02, "NELM": 200,
                                      "LWAVE": ".FALSE.", "LCHARG": ".FALSE."},
    "r2scan_d3bj_v1/molecule_static": {**_COMMON_R2SCAN, "IBRION": -1, "NSW": 0, "EDIFF": 1e-6,
                                       "NELM": 200, "LCHARG": ".FALSE."},
    # legacy, for comparison only
    "pbe_d3zero_legacy/relax": {"IVDW": 11, "ENCUT": 520, "PREC": "Accurate", "LASPH": ".TRUE.",
                                "ISMEAR": 0, "SIGMA": 0.05, "LREAL": "Auto", "IBRION": 2, "ISIF": 4,
                                "NSW": 99, "LVHAR": ".TRUE.", "LORBIT": 11},
}
PROFILE_VERSION = "2026-10-11"
PROFILE_NOTES = {
    "r2scan_d3bj_v1": [
        "ISIF=2: complex, bare slab and defect slabs share one fixed in-plane cell (required for "
        "a direct E_ads comparison; recovered runs used ISIF=4).",
        "LDIPOL/IDIPOL along the surface normal; recovered runs had none.",
        "EDIFF/EDIFFG explicit (recovered runs used VASP defaults 1e-4 / 1e-3 energy criterion).",
        "Relaxation and final static are separate linked jobs; label energy = static energy_sigma0.",
        "Spin: closed-shell default; ISPIN=2 is set automatically when multiplicity > 1 or the "
        "electron count is odd.",
    ]}


def kpoints_for(atoms: Atoms, normal_axis: int | None, density: float = 0.25,
                gamma: bool = False) -> list[int]:
    """VASP KSPACING convention: N_i = ceil(|b_i| / density), |b_i| including 2pi (A^-1).
    1 along the surface normal and for molecules. Treat as a convergence parameter."""
    rec = 2 * np.pi * np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    k = [max(1, int(np.ceil(r / density))) if not gamma else 1 for r in rec]
    if normal_axis is not None:
        k[normal_axis] = 1
    return k


@dataclass
class JobSpec:
    job_id: str
    role: str                         # bare_slab | complex | reference | defect_slab | fragment
    profile: str
    structure_hash: str
    parents: list[str]
    nelect: float
    cell_charge: int
    multiplicity: int | None
    kpoints: list[int]
    species_order: list[str]
    potcars: list[str]
    incar: dict
    atom_map: list[int]               # written index -> source atom_id
    fingerprint: str
    links: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _order(atoms: Atoms) -> tuple[list[int], list[str]]:
    syms = atoms.get_chemical_symbols()
    species = list(dict.fromkeys(syms))
    perm = sorted(range(len(syms)), key=lambda i: (species.index(syms[i]), i))
    return perm, species


def structure_hash(atoms: Atoms) -> str:
    h = hashlib.sha256()
    h.update(np.round(atoms.cell.array, 5).tobytes())
    h.update(np.round(atoms.positions, 4).tobytes())
    h.update(" ".join(atoms.get_chemical_symbols()).encode())
    return h.hexdigest()[:16]


def write_job(atoms: Atoms, out_dir: str | Path, profile: str, role: str, *, cell_charge: int = 0,
              multiplicity: int | None = None, normal_axis: int | None = None,
              kpoints: list[int] | None = None, fixed_atom_ids: list[int] | None = None,
              parents: list[str] | None = None, job_id: str | None = None,
              overrides: dict | None = None, overwrite: bool = False) -> JobSpec:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    incar = dict(PROFILES[profile])
    perm, species = _order(atoms)
    a = atoms[perm]
    ids = (atoms.arrays["atom_id"][perm].tolist() if "atom_id" in atoms.arrays else perm)
    pots = [POTCAR_VARIANTS[s] for s in species]
    n_neutral = sum(ZVAL[POTCAR_VARIANTS[s]] for s in a.get_chemical_symbols())
    nelect = n_neutral - cell_charge
    notes = []
    if cell_charge != 0:
        incar["NELECT"] = nelect
        notes.append("Charged cell: requires declared electrostatic treatment and reservoir; "
                     "outside the neutral screening domain.")
    odd = int(round(nelect)) % 2 == 1
    if (multiplicity and multiplicity > 1) or odd:
        incar["ISPIN"] = 2
        if multiplicity:
            incar["NUPDOWN"] = multiplicity - 1
        notes.append(f"Open shell: {nelect} electrons, multiplicity {multiplicity}.")
    if "IDIPOL" in incar:
        if normal_axis is None:
            incar.pop("IDIPOL"); incar.pop("LDIPOL", None)
        else:
            incar["IDIPOL"] = normal_axis + 1
            com = a.get_scaled_positions().mean(axis=0)
            incar["DIPOL"] = " ".join(f"{x:.4f}" for x in com)
    for g in ("NGX", "NGY", "NGZ"):
        if incar.get(g) is None:
            incar.pop(g, None)
    incar.update(overrides or {})
    if kpoints is None:
        kpoints = kpoints_for(a, normal_axis, gamma=(role == "reference"))
    if fixed_atom_ids:
        fixed = [k for k, i in enumerate(ids) if i in set(fixed_atom_ids)]
        a.set_constraint(FixAtoms(indices=fixed))
    from ase.io import write
    write(out / "POSCAR", a, format="vasp", direct=True, sort=False)
    (out / "INCAR").write_text("".join(f"{k} = {_fmt(v)}\n" for k, v in incar.items()))
    (out / "KPOINTS").write_text(f"pvpassivator {profile}\n0\n{'Gamma' if all(k == 1 for k in kpoints) else 'Gamma'}\n"
                                 f"{kpoints[0]} {kpoints[1]} {kpoints[2]}\n0 0 0\n")
    (out / "POTCAR.spec").write_text("\n".join(pots) + "\n")
    fp = normalized_fingerprint({k: str(v) for k, v in incar.items()}, " ".join(map(str, kpoints)))
    fp["POTCARS"] = pots
    fph = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    sh = structure_hash(a)
    spec = JobSpec(job_id or f"{role}-{sh}", role, profile, sh, parents or [], nelect, cell_charge,
                   multiplicity, kpoints, species, pots, incar, [int(i) for i in ids], fph,
                   notes=notes + PROFILE_NOTES.get(profile.split("/")[0], []))
    (out / "job.json").write_text(json.dumps({"schema_version": "job/1", "profile_version": PROFILE_VERSION,
                                              **asdict(spec)}, indent=1))
    return spec


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.0E}"
    return str(v)


# ----------------------------------------------------------------------------- bundles
def adsorption_bundle(root: str | Path, case_id: str, slab: Atoms, slab_rec, complex_: Atoms,
                      references: list[tuple[str, Atoms, int, int, float]], *, n_units: float,
                      profile_family: str = "r2scan_d3bj_v1", fix_bottom_A: float | None = None,
                      ref_box: float = 25.0, overwrite: bool = False) -> dict:
    """Write linked jobs for one balanced adsorption reaction:
        bare_slab/{relax,static}, complex/{relax,static}, ref_<name>/{relax,static}
    ``references`` = [(name, atoms, cell_charge, multiplicity, stoich_coeff)].
    Atom/charge balance is checked first; an unbalanced case writes nothing."""
    from .ledger import Species, Reaction
    root = Path(root) / case_id

    def comp(a):
        c = {}
        for s in a.get_chemical_symbols():
            c[s] = c.get(s, 0) + 1
        return c
    rx = Reaction([(1, Species("bare_slab", comp(slab)))] +
                  [(nu, Species(n, comp(a), cell_charge=q)) for n, a, q, m, nu in references],
                  [(1, Species("complex", comp(complex_)))], energy_kind="adsorption_relaxed",
                  n_units=n_units, label=case_id)
    rx.assert_balanced()
    ax = slab_rec.normal_axis
    if not np.allclose(slab.cell.array, complex_.cell.array, atol=1e-4):
        raise ValueError("complex and bare slab cells differ; a direct E_ads comparison needs one cell")
    fixed = None
    if fix_bottom_A:
        from .surface import unwrap_along
        h = unwrap_along(complex_, ax)
        ids = complex_.arrays.get("atom_id", np.arange(len(complex_)))
        org = complex_.arrays.get("origin", np.zeros(len(complex_), int))
        fixed = [int(i) for i, z, o in zip(ids, h, org) if o == 0 and z - h.min() < fix_bottom_A]
    kp = kpoints_for(slab, ax)
    jobs = {}
    for role, a in (("bare_slab", slab), ("complex", complex_)):
        r = write_job(a, root / role / "relax", f"{profile_family}/relax", role, normal_axis=ax,
                      kpoints=kp, fixed_atom_ids=fixed, overwrite=overwrite)
        s = write_job(a, root / role / "static", f"{profile_family}/static", role, normal_axis=ax,
                      kpoints=kp, parents=[r.job_id], overwrite=overwrite)
        s.links = {"geometry_from": f"{role}/relax/CONTCAR"}
        jobs[role] = {"relax": asdict(r), "static": asdict(s)}
    for n, a, q, m, nu in references:
        b = a.copy()
        b.pbc = True
        b.cell = np.eye(3) * ref_box
        b.center()
        r = write_job(b, root / f"ref_{n}" / "relax", f"{profile_family}/molecule_relax", "reference",
                      cell_charge=q, multiplicity=m, kpoints=[1, 1, 1], overwrite=overwrite)
        s = write_job(b, root / f"ref_{n}" / "static", f"{profile_family}/molecule_static", "reference",
                      cell_charge=q, multiplicity=m, kpoints=[1, 1, 1], parents=[r.job_id], overwrite=overwrite)
        jobs[f"ref_{n}"] = {"relax": asdict(r), "static": asdict(s), "stoich": nu}
    plan = {"schema_version": "bundle/1", "case_id": case_id, "equation": rx.equation(),
            "energy_kind": "adsorption_relaxed", "n_units": n_units,
            "energy_definition": "E_ads = [E(complex/static) - E(bare_slab/static) - sum nu E(ref/static)] / n_units; "
                                 "energy_sigma0; negative = exothermic",
            "fixed_bottom_A": fix_bottom_A, "kpoints": kp, "jobs": jobs,
            "cache_note": "bare_slab and ref jobs are shared across cases with identical fingerprint + structure hash",
            "status": "prepared (not submitted)"}
    (root / "bundle.json").write_text(json.dumps(plan, indent=1))
    return plan
