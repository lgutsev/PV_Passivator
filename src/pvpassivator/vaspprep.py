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

# PAW variants: Lavrenty's POTCAR_DEFS.txt (POTCAR_gen default on LONI, /home/lgutsev/bin/POTCAR_DEFS.txt;
# copy shipped in data/). Never choose a variant independently; a missing element is an error.
def load_potcar_defs(path: str | Path | None = None) -> dict[str, str]:
    path = Path(path) if path else Path(__file__).with_name("data") / "POTCAR_DEFS.txt"
    out = {}
    for line in path.read_text().splitlines():
        if "|" in line:
            el, var = line.strip().split("|", 1)
            out[el] = var
    return out


POTCAR_VARIANTS = load_potcar_defs()
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
    # electrostatics test (charged slabs, Coulomb kernel truncation vs dipole correction); fixed geometry, PBE
    "pbe_electrostatics_test/static": {"GGA": "PE", "LASPH": ".TRUE.", "ENCUT": 400, "PREC": "Accurate",
                                       "ISMEAR": 0, "SIGMA": 0.05, "LREAL": "Auto", "IBRION": -1, "NSW": 0,
                                       "EDIFF": 1e-6, "NELM": 200, "LVHAR": ".TRUE.", "LWAVE": ".FALSE.",
                                       "LCHARG": ".FALSE.", "LDIPOL": ".TRUE.", "IDIPOL": None},
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
    missing = [s for s in species if s not in POTCAR_VARIANTS]
    if missing:
        raise KeyError(f"{missing} not in POTCAR_DEFS.txt; ask before choosing a POTCAR variant")
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
    import io
    buf = io.StringIO()
    write(buf, a, format="vasp", direct=True, sort=False)
    _write_lf(out / "POSCAR", buf.getvalue())
    _write_lf(out / "INCAR", "".join(f"{k} = {_fmt(v)}\n" for k, v in incar.items()))
    _write_lf(out / "KPOINTS", f"pvpassivator {profile}\n0\n{'Gamma' if all(k == 1 for k in kpoints) else 'Gamma'}\n"
                                 f"{kpoints[0]} {kpoints[1]} {kpoints[2]}\n0 0 0\n")
    _write_lf(out / "POTCAR.spec", "\n".join(pots) + "\n")
    fp = normalized_fingerprint({k: str(v) for k, v in incar.items()}, " ".join(map(str, kpoints)))
    fp["POTCARS"] = pots
    fph = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    sh = structure_hash(a)
    spec = JobSpec(job_id or f"{role}-{sh}", role, profile, sh, parents or [], nelect, cell_charge,
                   multiplicity, kpoints, species, pots, incar, [int(i) for i in ids], fph,
                   notes=notes + PROFILE_NOTES.get(profile.split("/")[0], []))
    _write_lf(out / "job.json", json.dumps({"schema_version": "job/1", "profile_version": PROFILE_VERSION,
                                              **asdict(spec)}, indent=1))
    return spec


def _write_lf(path: Path, text: str) -> None:
    """VASP inputs go to Linux clusters: always LF line endings, whatever the local platform."""
    Path(path).write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


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
    _write_lf(root / "bundle.json", json.dumps(plan, indent=1))
    return plan


O2_BOND_A = 1.21          # starting O=O length; relaxed in the reference job


def _hamiltonian(spec: JobSpec) -> dict:
    """Settings that must match across the jobs of one reaction energy (functional, cutoff, PAW, vdW,
    smearing). ISPIN/NUPDOWN may differ (triplet O2) and so may k-points (molecule vs slab)."""
    from .vasp_io import HAMILTONIAN_KEYS, INCAR_DEFAULTS, _norm_val
    h = {}
    for k in HAMILTONIAN_KEYS:
        if k == "ISPIN":
            continue
        v = spec.incar.get(k, INCAR_DEFAULTS.get(k))
        if v is not None:
            h[k] = _norm_val(v)
    h["POTCARS"] = dict(zip(spec.species_order, spec.potcars))
    return h


def check_matched_settings(specs: dict[str, JobSpec]) -> list[str]:
    """Problems if the static jobs of one reaction differ in functional, cutoff or PAW variant per element."""
    problems = []
    items = list(specs.items())
    ref_name, ref = items[0]
    h0 = _hamiltonian(ref)
    for name, sp in items[1:]:
        h = _hamiltonian(sp)
        for k in sorted((set(h0) | set(h)) - {"POTCARS"}):
            if h0.get(k) != h.get(k):
                problems.append(f"{name} {k}={h.get(k)} differs from {ref_name} {k}={h0.get(k)}")
        for el in set(h0["POTCARS"]) & set(h["POTCARS"]):
            if h0["POTCARS"][el] != h["POTCARS"][el]:
                problems.append(f"{name} uses POTCAR {h['POTCARS'][el]} for {el}, {ref_name} uses {h0['POTCARS'][el]}")
    return problems


def oxidation_bundle(root: str | Path, case_id: str, slab: Atoms, slab_rec, product: Atoms, n_oxygen: int, *,
                     profile_family: str = "r2scan_d3bj_v1", fix_bottom_A: float | None = None,
                     o2_box: float = 15.0, overwrite: bool = False) -> dict:
    """Linked jobs for   slab + n/2 O2(g, triplet) -> slab(oxyanion):
        bare_slab/{relax,static}, product/{relax,static}, ref_O2/{relax,static} (ISPIN=2, NUPDOWN=2).
    Atom balance is checked first; the static jobs must share functional, cutoff and PAW variants, or
    nothing is kept. O2 enters only as a reservoir; any O2 binding-energy correction must be declared
    later through ``ledger.ReferenceCorrection``."""
    from .ledger import Species
    from .oxidation import O2_REFERENCE, oxidation_reaction
    if o2_box < 15.0:
        raise ValueError("O2 reference box must be at least 15 A")

    def comp(a):
        c = {}
        for s_ in a.get_chemical_symbols():
            c[s_] = c.get(s_, 0) + 1
        return c
    rx = oxidation_reaction(Species("bare_slab", comp(slab)), Species("product", comp(product)), n_oxygen)
    ax = slab_rec.normal_axis
    if not np.allclose(slab.cell.array, product.cell.array, atol=1e-4):
        raise ValueError("product and bare slab cells differ; the reaction energy needs one cell")
    root = Path(root) / case_id
    fixed = None
    if fix_bottom_A:
        from .surface import unwrap_along
        h = unwrap_along(product, ax)
        ids = product.arrays.get("atom_id", np.arange(len(product)))
        org = product.arrays.get("origin", np.zeros(len(product), int))
        fixed = [int(i) for i, z, o in zip(ids, h, org) if o == 0 and z - h.min() < fix_bottom_A]
    kp = kpoints_for(slab, ax)
    jobs, statics = {}, {}
    for role, a in (("bare_slab", slab), ("product", product)):
        r = write_job(a, root / role / "relax", f"{profile_family}/relax", role, normal_axis=ax,
                      kpoints=kp, fixed_atom_ids=fixed, overwrite=overwrite)
        st = write_job(a, root / role / "static", f"{profile_family}/static", role, normal_axis=ax,
                       kpoints=kp, parents=[r.job_id], overwrite=overwrite)
        st.links = {"geometry_from": f"{role}/relax/CONTCAR"}
        jobs[role] = {"relax": asdict(r), "static": asdict(st)}
        statics[role] = st
    # slightly anisotropic box so the molecule axis is not degenerate with the cell
    o2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, O2_BOND_A]], cell=np.diag([o2_box, o2_box + 0.5, o2_box + 1.0]),
               pbc=True)
    o2.center()
    m = O2_REFERENCE.spin_multiplicity
    r = write_job(o2, root / "ref_O2" / "relax", f"{profile_family}/molecule_relax", "reference",
                  multiplicity=m, kpoints=[1, 1, 1], overwrite=overwrite)
    st = write_job(o2, root / "ref_O2" / "static", f"{profile_family}/molecule_static", "reference",
                   multiplicity=m, kpoints=[1, 1, 1], parents=[r.job_id], overwrite=overwrite)
    st.links = {"geometry_from": "ref_O2/relax/CONTCAR"}
    jobs["ref_O2"] = {"relax": asdict(r), "static": asdict(st), "stoich": n_oxygen / 2}
    statics["ref_O2"] = st
    problems = check_matched_settings(statics)
    if problems:
        import shutil
        shutil.rmtree(root)
        raise ValueError("static jobs are not comparable:\n  " + "\n  ".join(problems))
    plan = {"schema_version": "bundle/1", "case_id": case_id, "equation": rx.equation(), "energy_kind": "reaction",
            "energy_definition": f"dE = E(product/static) - E(bare_slab/static) - {n_oxygen / 2:g} E(ref_O2/static); "
                                 "energy_sigma0; negative = exothermic. Any O2 correction must be declared "
                                 "explicitly (ledger.ReferenceCorrection), never assumed.",
            "o2_reference": {"multiplicity": m, "ISPIN": 2, "NUPDOWN": m - 1, "box_A": o2_box,
                             "start_bond_A": O2_BOND_A},
            "fixed_bottom_A": fix_bottom_A, "kpoints": kp, "jobs": jobs, "status": "prepared (not submitted)"}
    _write_lf(root / "bundle.json", json.dumps(plan, indent=1))
    return plan


def to_common_cell(atoms: Atoms, normal_axis: int, cell: np.ndarray, max_strain: float = 0.03,
                   max_shear_deg: float = 7.0, rigid_fragments: bool = True) -> tuple[Atoms, dict]:
    """Map a slab onto ``cell`` (one fixed cell for every member of a reaction).

    Lattice ions follow their in-plane fractional coordinates; positions along the surface normal keep their absolute
    distance (the vacuum length may differ). With ``rigid_fragments`` every covalent fragment (FA+, a passivator)
    moves as a rigid body with its centroid, so undoing a cell shear does not bend molecules. Refuses an in-plane
    length change above ``max_strain`` or an in-plane angle change above ``max_shear_deg``. Returns the mapped atoms
    (arrays carried over) and a report of the strain and shear removed."""
    old = atoms.cell.array
    new = np.asarray(cell, float)
    ip = [k for k in range(3) if k != normal_axis]

    def plane_normal(c):
        # the true surface normal is perpendicular to both in-plane vectors; the third lattice vector may be tilted
        n = np.cross(c[ip[0]], c[ip[1]])
        n /= np.linalg.norm(n)
        return n if np.dot(n, c[normal_axis]) > 0 else -n
    n_old, n_new = plane_normal(old), plane_normal(new)
    strain = {}
    for k in ip:
        strain[k] = float(np.linalg.norm(new[k]) / np.linalg.norm(old[k]) - 1)
        if abs(strain[k]) > max_strain:
            raise ValueError(f"in-plane axis {k}: strain {strain[k]:+.3%} exceeds {max_strain:.0%}")

    def angle(c):
        u, v = c[ip[0]], c[ip[1]]
        return float(np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))))
    shear = angle(new) - angle(old)
    if abs(shear) > max_shear_deg:
        raise ValueError(f"in-plane angle changes by {shear:+.2f} deg (> {max_shear_deg} deg)")

    B_old = np.array([old[k] for k in ip])                 # in-plane basis rows

    def mapped(pos):
        h = pos @ n_old                                    # height along the true normal
        par = pos - h[:, None] * n_old                     # in-plane part, expressed in the in-plane basis
        f, *_ = np.linalg.lstsq(B_old.T, par.T, rcond=None)
        return f.T @ np.array([new[k] for k in ip]) + h[:, None] * n_new

    pos = mapped(atoms.positions)
    n_rigid = 0
    if rigid_fragments:
        from .poses import fragments
        ionic = {"Pb", "Sn", "I", "Br", "Cl", "Cs", "K", "Na", "Rb"}
        sym = atoms.get_chemical_symbols()
        for g in fragments(atoms):
            if len(g) < 2 or all(sym[i] in ionic for i in g):
                continue
            v = atoms.get_distances(g[0], g, mic=True, vector=True)        # whole fragment around its first atom
            whole = atoms.positions[g[0]] + v
            c = whole.mean(axis=0)
            pos[g] = mapped(c[None])[0] + (whole - c)
            n_rigid += 1
    out = atoms.copy()
    out.set_cell(new, scale_atoms=False)
    out.positions = pos
    out.wrap()
    return out, {"inplane_strain": [strain[k] for k in ip], "inplane_shear_removed_deg": shear,
                 "rigid_fragments": n_rigid}
