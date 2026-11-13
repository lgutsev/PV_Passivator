"""Export a returned static-audit package (prepare_static_audit.py) as a small fine-tuning set, split by FAMILY.

    python examples/scripts/export_audit_frames.py PACKAGE --pool POOL --out DIR --test F1 F2 ... --valid F3 ...
           [--subtract-d3 bj r2scan] [--align-to FOUNDATION.model] [--device cuda]

One frame per finished static (DONE + complete vasprun.xml): the DFT free energy and raw forces at a geometry where a
model's numbers are already stored.  A FAMILY is everything that descends from one starting arrangement: ``pool_NN``
and ``asbuilt_NN`` are one family, and structures the model relaxed into the same minimum (energies within
``--same-minimum`` eV and positions within 0.05 A) are merged into one.  Families never straddle splits.  ``--test`` and
``--valid`` take family names (the name of any member); everything else trains.  The bare slab and the isolated
molecule always train: they are references, not arrangements.

Targets, as in export_prerelax_frames.py:
  --subtract-d3   REF_* = DFT minus D3 (torch-dftd; its parameters are checked against the INCAR), because the
                  foundation model was trained without dispersion.  Add the same D3 back at inference.
  --align-to      additionally removes a composition-linear energy offset between DFT-minus-D3 and the given model
                  (least squares over the TRAIN frames, per element, minimum norm), stored in dataset.json.  Total
                  energies of the two differ by a per-element constant (PAW reference energies); with the foundation's
                  E0s kept, that constant would otherwise dominate the energy loss.  It cancels in every difference
                  between structures of one composition, so rankings and forces are untouched.

These frames ran with LREAL = .FALSE.; do not pool them with sets run with LREAL = Auto without checking the offset.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from ase.io import read, write


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_frame(d: Path):
    """(atoms with DFT free energy + forces) from one finished static, or raise FileNotFoundError."""
    if not (d / "DONE").exists():
        raise FileNotFoundError(f"{d.parent.name}: no DONE marker")
    f = d / "vasprun.xml.gz"
    with (gzip.open(f, "rt") if f.exists() else open(d / "vasprun.xml")) as fh, \
            tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as tmp:
        tmp.write(fh.read())
    a = read(tmp.name, format="vasp-xml", index=-1)
    Path(tmp.name).unlink()
    return a, float(a.get_potential_energy(force_consistent=True)), a.get_forces().copy(), sha256(f)[:16]


def families(names, pool: Path, man: dict, same_minimum: float) -> dict:
    """name -> family id.  pool_NN/asbuilt_NN share a family; same-minimum structures are merged."""
    # asbuilt_NN is the unrelaxed start of pool_NN; rr_X is X relaxed again by another model: one family each
    fam = {n: n.removeprefix("rr_").replace("asbuilt_", "pool_") for n in names}
    cplx = [n for n in names if man["jobs"][n]["role"] == "complex" and man["jobs"][n].get("relaxed_by_model")
            and not n.startswith("rr_")]
    for i, a in enumerate(cplx):
        for b in cplx[:i]:
            ja, jb = man["jobs"][a], man["jobs"][b]
            if ja["group"] != jb["group"] or abs(ja["model_energy_eV"] - jb["model_energy_eV"]) > same_minimum:
                continue
            pa, pb = read(pool / ja["source"]), read(pool / jb["source"])
            if np.abs(pa.positions - pb.positions).max() < 0.05:
                old, new = fam[a], fam[b]
                fam = {k: (new if v == old else v) for k, v in fam.items()}
    return fam


def export(package, pool, out, test, valid, d3=None, align=None, device="cpu", same_minimum=0.01, also=()):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be new or empty")
    sources = [(Path(package).resolve(), Path(pool))] + [(Path(a).resolve(), Path(b)) for a, b in also]
    man = {"jobs": {}}
    where = {}
    for r, pl in sources:                       # later packages add jobs; names must be unique across packages
        mm = json.loads((r / "manifest.json").read_text())
        dup = set(mm["jobs"]) & set(man["jobs"])
        if dup:
            raise ValueError(f"job names repeated across packages: {sorted(dup)}")
        man["jobs"].update(mm["jobs"])
        where.update({n: (r, pl) for n in mm["jobs"]})
    root, pool = sources[0]
    frames, missing = {}, []
    hamiltonian = None
    for name, j in man["jobs"].items():
        root, pool = where[name]
        d = root / "outputs" / name / "static"
        try:
            a, e, f, src = read_frame(d)
        except FileNotFoundError as ex:
            missing.append(str(ex))
            continue
        jj = json.loads((root / "inputs" / name / "static" / "job.json").read_text())
        inc = {k.upper(): v for k, v in jj["incar"].items()}
        h = {k: inc.get(k) for k in ("METAGGA", "IVDW", "VDW_S8", "VDW_A1", "VDW_A2", "ENCUT", "PREC", "ISMEAR", "SIGMA",
                                     "LASPH", "LREAL")}
        if hamiltonian is None:
            hamiltonian = h
        elif h != hamiltonian:
            raise ValueError(f"{name}: Hamiltonian differs from the other jobs: {h} vs {hamiltonian}")
        src_atoms = read(pool / j["source"])
        mol = np.zeros(len(a), int)
        if "mol_id" in src_atoms.arrays and isinstance(jj["atom_map"], list):
            idx = {int(i): k for k, i in enumerate(src_atoms.arrays["atom_id"])}
            mol = src_atoms.arrays["mol_id"][[idx[int(i)] for i in jj["atom_map"]]]
        elif j["role"] == "reference":
            mol = np.ones(len(a), int)
        a.calc = None
        a.set_array("DFT_forces", f)
        a.set_array("mol_id", mol)
        a.info.update(DFT_energy=e, job=name, source="vasprun.xml.gz", source_sha256=src, frame=0,
                      energy_kind="vasp_free_energy", config_type=j["role"], group=j["group"],
                      selection=j.get("selection", j["role"]), label_grade=False)
        frames[name] = a
    root, pool = sources[0]
    fam = families(list(frames), pool, man, same_minimum)
    known = set(fam.values())
    resolve = lambda xs: {fam[x] if x in fam else x for x in xs}
    test_f, valid_f = resolve(test), resolve(valid)
    if (test_f | valid_f) - known:
        raise ValueError(f"unknown families: {sorted((test_f | valid_f) - known)}")
    if test_f & valid_f:
        raise ValueError("a family is in both test and valid")
    part = {"train": [], "valid": [], "test": []}
    for name, a in frames.items():
        role = man["jobs"][name]["role"]
        split = "train" if role != "complex" else "test" if fam[name] in test_f else "valid" if fam[name] in valid_f else "train"
        if role != "complex" and fam[name] in (test_f | valid_f):
            raise ValueError(f"{name} is a reference and always trains")
        a.info.update(split=split, family=fam[name])
        part[split].append(a)
    report = {"schema": "pvpass_audit_frames/1", "label_grade": False,
              "purpose": "fine-tune and test an MLIP on audit single points; geometries are MODEL geometries, not DFT minima",
              "packages": [{"package": str(r), "manifest_sha256": sha256(r / "manifest.json"), "pool": str(pl)}
                           for r, pl in sources], "hamiltonian": hamiltonian,
              "missing": missing, "families": {f: sorted(n for n in fam if fam[n] == f) for f in sorted(known)},
              "splits": {s: sorted(a.info["job"] for a in ats) for s, ats in part.items()}}
    every = [a for ats in part.values() for a in ats]
    if d3:
        import torch
        from torch_dftd.dftd3_xc_params import get_dftd3_default_params
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        p = get_dftd3_default_params(*d3)
        want = {"rs6": float(hamiltonian["VDW_A1"]), "s18": float(hamiltonian["VDW_S8"]), "rs18": float(hamiltonian["VDW_A2"])}
        if any(abs(p[k] - v) > 1e-4 for k, v in want.items()):
            raise ValueError(f"torch-dftd {d3} parameters {p} differ from the INCAR {want}")
        # float64 D3 on ~600 atoms needs more than 8 GB on a GPU; the CPU is fast enough for a few dozen frames
        calc = TorchDFTD3Calculator(damping=d3[0], xc=d3[1], device="cpu", dtype=torch.float64)
        for a in every:
            c = a.copy()
            c.calc = calc
            e, f = float(c.get_potential_energy()), c.get_forces()
            a.info["D3_energy"] = e
            a.info["REF_energy"] = a.info["DFT_energy"] - e
            a.set_array("REF_forces", a.arrays["DFT_forces"] - f)
        report["target"] = "DFT free energy minus D3(%s,%s) [torch-dftd]" % tuple(d3)
        report["add_back_at_inference"] = "D3(%s,%s)" % tuple(d3)
    else:
        for a in every:
            a.info["REF_energy"] = a.info["DFT_energy"]
            a.set_array("REF_forces", a.arrays["DFT_forces"].copy())
        report["target"] = "DFT free energy (incl. D3)"
    if align:
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=str(align), device=device, default_dtype="float64")
        zs = sorted({int(z) for a in every for z in a.numbers})
        for a in every:
            c = a.copy()
            c.calc = calc
            a.info["align_model_energy"] = float(c.get_potential_energy())
            a.set_array("align_model_forces", c.get_forces())
        A = np.array([[int((a.numbers == z).sum()) for z in zs] for a in part["train"]], float)
        y = np.array([a.info["REF_energy"] - a.info["align_model_energy"] for a in part["train"]])
        shift, *_ = np.linalg.lstsq(A, y, rcond=None)            # minimum-norm solution
        for a in every:
            off = float(sum(s * (a.numbers == z).sum() for s, z in zip(shift, zs)))
            a.info["REF_energy_unaligned"] = a.info["REF_energy"]
            a.info["REF_energy"] = a.info["REF_energy"] - off
            a.info["align_offset"] = off
        report["alignment"] = {"model": str(align), "model_sha256": sha256(align), "fitted_on": "train frames",
                               "per_element_shift_eV": {str(z): float(s) for z, s in zip(zs, shift)},
                               "train_residual_max_eV": float(np.abs(A @ shift - y).max()),
                               "note": "REF_energy = (DFT - D3) - sum_Z n_Z shift_Z; cancels within one composition"}
    out.mkdir(parents=True, exist_ok=True)
    report["files"] = {}
    for split, ats in part.items():
        path = out / f"{split}.extxyz"
        write(path, ats, format="extxyz") if ats else path.write_text("")
        report["files"][split] = {"file": path.name, "sha256": sha256(path), "frames": len(ats),
                                  "atoms": int(sum(len(a) for a in ats))}
    (out / "dataset.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--valid", nargs="*", default=[])
    ap.add_argument("--subtract-d3", nargs=2, metavar=("DAMPING", "XC"))
    ap.add_argument("--align-to")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--same-minimum", type=float, default=0.01)
    ap.add_argument("--also", nargs=2, action="append", default=[], metavar=("PACKAGE", "POOL"),
                    help="a further returned package and its pool; repeatable")
    a = ap.parse_args(argv)
    r = export(a.package, a.pool, a.out, a.test, a.valid, a.subtract_d3, a.align_to, a.device, a.same_minimum, a.also)
    print(json.dumps({k: r[k] for k in ("splits", "families", "missing", "target", "files") if k in r}
                     | ({"alignment": r["alignment"]} if "alignment" in r else {}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
