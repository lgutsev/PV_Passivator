"""Minimal, dependency-light VASP output parsing with explicit convergence checks.

POTCAR contents are never read or copied; pseudopotential identity is taken from the
TITEL / ZVAL lines echoed in OUTCAR.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

ENERGY_KEYS = ("energy_sigma0", "energy_free", "energy_no_entropy")

# INCAR keys that change the Hamiltonian / energy reference. Mismatch => incompatible labels.
HAMILTONIAN_KEYS = ("GGA", "METAGGA", "LHFCALC", "HFSCREEN", "IVDW", "VDW_S6", "VDW_S8",
                    "VDW_A1", "VDW_A2", "VDW_SR", "LSORBIT", "ISPIN", "ENCUT", "PREC",
                    "LASPH", "LDAU", "ISMEAR", "SIGMA", "LREAL", "ADDGRID")


def electronic_state_problems(incar: dict, n_electrons: float | None, cell_charge: float | None) -> list[str]:
    """Label-acceptance rules for the electronic state (user decision 2026-10-11):
    an odd-electron cell computed without spin polarisation is not a valid label (the unpaired
    electron/hole is forced into a fractionally occupied, artificially delocalised state)."""
    bad = []
    ispin = int(float(incar.get("ISPIN", 1) or 1))
    ncl = str(incar.get("LSORBIT", "")).upper().strip(".") in ("TRUE", "T") or         str(incar.get("LNONCOLLINEAR", "")).upper().strip(".") in ("TRUE", "T")
    if n_electrons is not None and not ncl and ispin == 1 and int(round(n_electrons)) % 2 == 1:
        bad.append(f"odd electron count ({n_electrons:g}) computed with ISPIN=1: spin state not represented")
    if cell_charge not in (None, 0) and abs(cell_charge) > 1e-6:
        isif = int(float(incar.get("ISIF", 2) or 2))
        if isif >= 3:
            bad.append(f"charged cell (q={cell_charge:+g}) with cell relaxation ISIF={isif}: "
                       "cell responds to the compensating background")
    return bad


@dataclass
class VaspRun:
    path: str
    incar: dict = field(default_factory=dict)
    potcars: list[str] = field(default_factory=list)       # TITEL strings in species order
    zval: dict = field(default_factory=dict)               # element -> ZVAL
    ions_per_type: list[int] = field(default_factory=list)
    species: list[str] = field(default_factory=list)
    nelect: float | None = None
    kpoints: str | None = None
    energy_free: float | None = None          # F  (free energy TOTEN)
    energy_no_entropy: float | None = None
    energy_sigma0: float | None = None
    ionic_converged: bool | None = None       # 'reached required accuracy'
    electronic_converged_last: bool | None = None
    nelm_hit: bool = False
    max_force: float | None = None
    n_ionic_steps: int = 0
    finished: bool = False                    # timing block present
    warnings: list[str] = field(default_factory=list)

    @property
    def cell_charge(self) -> float | None:
        if self.nelect is None or not self.zval:
            return None
        tot = sum(self.zval[s] * n for s, n in zip(self.species, self.ions_per_type))
        return round(tot - self.nelect, 6)

    def fingerprint(self) -> dict:
        h = {k: self.incar.get(k) for k in HAMILTONIAN_KEYS if k in self.incar}
        h["POTCARS"] = sorted(set(self.potcars))
        h["KPOINTS"] = self.kpoints
        return h

    def fingerprint_hash(self, exclude_kpoints: bool = False) -> str:
        fp = self.fingerprint()
        if exclude_kpoints:
            fp.pop("KPOINTS", None)
        return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]

    def energy(self, kind: str = "energy_sigma0") -> float:
        v = getattr(self, kind)
        if v is None:
            raise ValueError(f"{self.path}: {kind} not found")
        return v

    def acceptance(self, ediffg_force: float | None = None, require_relax: bool = True) -> list[str]:
        """Return reasons this run should NOT be a trusted label; [] means accepted."""
        bad = []
        if not self.finished:
            bad.append("run did not finish (no timing block in OUTCAR)")
        if self.energy_sigma0 is None:
            bad.append("no final energy")
        if self.electronic_converged_last is False or self.nelm_hit:
            bad.append("last electronic step hit NELM without converging")
        nsw = int(float(self.incar.get("NSW", 0) or 0))
        if require_relax and nsw > 0 and int(float(self.incar.get("IBRION", -1))) in (1, 2, 3) \
                and not self.ionic_converged:
            bad.append("ionic relaxation did not reach required accuracy")
        if ediffg_force is not None and self.max_force is not None and self.max_force > ediffg_force * 1.05:
            bad.append(f"max force {self.max_force:.3f} eV/A > {ediffg_force}")
        bad += electronic_state_problems(self.incar, self.nelect, self.cell_charge)
        return bad

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cell_charge"] = self.cell_charge
        return d


def parse_incar(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.split("#")[0].split("!")[0]
        for part in line.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip().upper()] = v.strip()
    return out


_FLOAT = r"[-+]?\d+\.\d*(?:[eE][-+]?\d+)?"


def parse_run(folder: str | Path, outcar: str = "OUTCAR") -> VaspRun:
    folder = Path(folder)
    run = VaspRun(path=str(folder))
    if (folder / "INCAR").exists():
        run.incar = parse_incar(folder / "INCAR")
    if (folder / "KPOINTS").exists():
        run.kpoints = " ".join((folder / "KPOINTS").read_text(errors="replace").split()[1:])
    oc = folder / outcar
    if not oc.exists():
        run.warnings.append("OUTCAR missing")
        return run
    nelm = int(float(run.incar.get("NELM", 60) or 60))
    titel, zvals = [], []
    scf_count = 0
    last_scf_count = 0
    forces_block = None
    with open(oc, errors="replace") as fh:
        lines = iter(fh)
        for line in lines:
            if "TITEL" in line:
                titel.append(line.split("=", 1)[1].strip())
            elif line.lstrip().startswith("POMASS") and "ZVAL" in line:
                m = re.search(r"ZVAL\s*=\s*(" + _FLOAT + ")", line)
                if m:
                    zvals.append(float(m.group(1)))
            elif "ions per type" in line:
                run.ions_per_type = [int(x) for x in line.split("=")[1].split()]
            elif re.search(r"^\s*NELECT\s*=", line):
                run.nelect = float(line.split("=")[1].split()[0])
            elif "Iteration" in line and "(" in line:
                scf_count += 1
            elif "free  energy   TOTEN" in line:
                run.energy_free = float(line.split("=")[1].split()[0])
            elif "energy  without entropy" in line:
                nums = re.findall(_FLOAT, line)
                run.energy_no_entropy, run.energy_sigma0 = float(nums[0]), float(nums[-1])
                run.n_ionic_steps += 1
                last_scf_count, scf_count = scf_count, 0
            elif "TOTAL-FORCE" in line:
                next(lines)
                forces_block = []
                for fl in lines:
                    if fl.strip().startswith("---"):
                        break
                    forces_block.append([float(x) for x in fl.split()[3:6]])
            elif "reached required accuracy" in line:
                run.ionic_converged = True
            elif "General timing and accounting" in line:
                run.finished = True
    # species from TITEL e.g. 'PAW_PBE Pb_d 06Sep2000'
    run.potcars = titel[: len(run.ions_per_type)] if run.ions_per_type else titel
    run.species = [t.split()[1].split("_")[0] for t in run.potcars]
    run.zval = {s: z for s, z in zip(run.species, zvals)}
    if run.ionic_converged is None:
        run.ionic_converged = False
    run.nelm_hit = last_scf_count >= nelm
    run.electronic_converged_last = not run.nelm_hit if run.n_ionic_steps else None
    if forces_block:
        import numpy as np
        f = np.array(forces_block)
        # NB: forces on fixed atoms are printed too; selective dynamics must be applied by caller
        run.max_force = float(np.linalg.norm(f, axis=1).max())
    return run


def compatible(runs: list[VaspRun], allow_kpoint_difference: bool = True) -> list[str]:
    """Compare Hamiltonian fingerprints across all runs in one reaction."""
    probs = []
    ref = runs[0].fingerprint()
    for r in runs[1:]:
        fp = r.fingerprint()
        for k in set(ref) | set(fp):
            if k == "POTCARS":
                continue  # different species sets are fine; same element must use same POTCAR
            if k == "KPOINTS" and allow_kpoint_difference:
                continue
            if ref.get(k) != fp.get(k):
                probs.append(f"{k}: {runs[0].path}={ref.get(k)!r} vs {r.path}={fp.get(k)!r}")
    # same element -> same POTCAR
    seen: dict[str, tuple[str, str]] = {}
    for r in runs:
        for s, t in zip(r.species, r.potcars):
            if s in seen and seen[s][0] != t:
                probs.append(f"POTCAR for {s}: {seen[s][1]}={seen[s][0]!r} vs {r.path}={t!r}")
            seen.setdefault(s, (t, r.path))
    return probs


# --------------------------------------------------------------------------------------
# "OutPackLite" archives keep only INCAR/KPOINTS/POSCAR/CONTCAR/OSZICAR. Energies come from
# the last OSZICAR ionic line; convergence can only be *inferred* (steps < NSW, last dE).
# --------------------------------------------------------------------------------------

# VASP defaults applied when a tag is absent, so "unset" and "explicit default" compare equal.
# GGA=PE is the default for PBE POTCARs (assumes PAW_PBE POTCARs; verify from POTCAR TITEL).
INCAR_DEFAULTS = {"GGA": "PE", "IVDW": "0", "LSORBIT": "FALSE", "ISPIN": "1", "LASPH": "FALSE",
                  "ISMEAR": "1", "SIGMA": "0.2", "PREC": "NORMAL", "LREAL": "FALSE"}


def _norm_val(v: str) -> str:
    v = str(v).strip().upper().strip(".")
    if v in ("T", "TRUE"):
        return "TRUE"
    if v in ("F", "FALSE"):
        return "FALSE"
    try:
        f = float(v)
        return repr(int(f)) if f.is_integer() else repr(f)
    except ValueError:
        return v


def normalized_fingerprint(incar: dict, kpoints: str | None) -> dict:
    fp = {}
    for k in HAMILTONIAN_KEYS:
        v = incar.get(k, INCAR_DEFAULTS.get(k))
        if v is not None:
            fp[k] = _norm_val(v)
    if "METAGGA" in fp:
        fp.pop("GGA", None)
    fp["KPOINTS"] = kpoints
    return fp


@dataclass
class LiteRun:
    path: str
    incar: dict
    kpoints: str | None
    species: list[str]
    counts: list[int]
    n_ionic: int
    energy_F: float | None
    energy_E0: float | None
    last_dE: float | None
    status: str          # converged_inferred | hit_nsw | single_point | empty | missing

    @property
    def composition(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for s, n in zip(self.species, self.counts):
            c[s] = c.get(s, 0) + n
        return c

    def nelect_override(self) -> float | None:
        v = self.incar.get("NELECT")
        return float(v) if v is not None else None

    def cell_charge(self, zval: dict[str, float]) -> float:
        """q = sum ZVAL - NELECT; 0 if NELECT not set (VASP default neutral)."""
        ne = self.nelect_override()
        if ne is None:
            return 0.0
        return round(sum(zval[s] * n for s, n in self.composition.items()) - ne, 6)

    def fingerprint(self) -> dict:
        return normalized_fingerprint(self.incar, self.kpoints)

    def n_electrons(self, zval: dict[str, float]) -> float:
        ne = self.nelect_override()
        return ne if ne is not None else sum(zval[s] * n for s, n in self.composition.items())

    def acceptance(self, zval: dict[str, float]) -> list[str]:
        """Reasons this archived run is not a trusted label; [] means accepted (as far as an
        OutPackLite archive allows: no forces, convergence inferred from step counts)."""
        bad = []
        if self.status in ("empty", "missing"):
            bad.append(f"no energy ({self.status})")
        if self.status == "hit_nsw":
            bad.append("relaxation stopped at NSW")
        if self.status == "converged_inferred" and self.n_ionic <= 3 and                 int(float(self.incar.get("NSW", 0) or 0)) > 3:
            bad.append(f"only {self.n_ionic} ionic steps in OSZICAR: likely a restart fragment; "
                       "convergence of the full relaxation unknown")
        bad += electronic_state_problems(self.incar, self.n_electrons(zval), self.cell_charge(zval))
        return bad


def parse_lite(folder: str | Path) -> LiteRun:
    folder = Path(folder)
    incar = parse_incar(folder / "INCAR") if (folder / "INCAR").exists() else {}
    kp = None
    if (folder / "KPOINTS").exists():
        kp = " ".join((folder / "KPOINTS").read_text(errors="replace").splitlines()[1:4]).split()
        kp = " ".join(kp)
    species, counts = [], []
    pos = folder / "POSCAR"
    if pos.exists():
        lines = pos.read_text(errors="replace").splitlines()
        species, counts = lines[5].split(), [int(x) for x in lines[6].split()]
    n, F, E0, dE = 0, None, None, None
    osz = folder / "OSZICAR"
    if not osz.exists():
        status = "missing"
    else:
        for line in osz.read_text(errors="replace").splitlines():
            if " F=" in line:
                m = re.search(r"^\s*(\d+)\s+F=\s*(\S+)\s+E0=\s*(\S+)\s+d E\s*=\s*(\S+)", line)
                if m:
                    n, F, E0, dE = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        nsw = int(float(incar.get("NSW", 0) or 0))
        if F is None:
            status = "empty"
        elif nsw == 0 or int(float(incar.get("IBRION", -1))) == -1:
            status = "single_point"
        elif n >= nsw:
            status = "hit_nsw"
        else:
            status = "converged_inferred"
    return LiteRun(str(folder), incar, kp, species, counts, n, F, E0, dE, status)
