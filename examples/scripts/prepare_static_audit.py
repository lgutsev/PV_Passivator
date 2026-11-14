"""DFT single points at fixed geometries, to audit an MLIP (inputs only; never submits).

    python examples/scripts/prepare_static_audit.py --pool data/cache/adsearch/v1_pool --out PACKAGE_SRC
           [--config audit.json] [--overwrite]
    python examples/scripts/prepare_static_audit.py --analyze RETURNED_PACKAGE --pool POOL [--out report.json]

Input is the folder written by ``adsearch_v1_pool.py``: structures already on the common cell, each with the model's
energy and forces at exactly that geometry.  For every structure this writes ``<name>/static`` with the project's
static profile, plus the bare slab (same cell) and the isolated molecule in a vacuum box.  No geometry is changed and
nothing is relaxed: the DFT numbers are comparable with the stored model numbers atom by atom.

Differences from the label profile, and why:
  LREAL = .FALSE.   the audit is a force comparison, and LREAL = Auto shifts forces by up to ~12 meV/A at 520 eV on
                    these elements (I, Pb, C; package 108).  Energies of this set must therefore not be combined with
                    label sets run with LREAL = Auto.
  one DIPOL centre  the bare slab's, for every slab-type job, as in the label sets.

The acceptance thresholds are written into the manifest when the inputs are made, before any DFT number exists.
``--analyze`` reads a returned package (outputs/<job>/static/{OUTCAR,vasprun.xml.gz}) and applies them.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pvpassivator.vaspprep import write_job, check_matched_settings, PROFILES  # noqa: E402
from pvpassivator.poses import min_image_distances  # noqa: E402

THRESHOLDS = {
    "relative_energy_mae_eV": 0.10, "relative_energy_max_eV": 0.20,
    "spearman_min": 0.8, "spearman_needs": "at least 10 same-composition structures spanning >= 0.5 eV in DFT",
    "top3_overlap_min": 2,
    "force_rmse_near_eV_A": 0.05, "force_cos_min": 0.9, "force_cos_only_above_eV_A": 0.05,
    "near_definition": "passivant atoms and substrate atoms within 4 A of any passivant atom",
    "energy_reference": "differences within one composition group (same atoms, same cell); never across groups",
    "fixed_on": "2026-11-09, before any DFT single point of this set existed",
}
DEFAULTS = {"profile_family": "r2scan_d3bj_v1", "min_interatomic_A": 0.75, "kpoints": [1, 1, 1], "ref_box": 25.0,
            "parallel": {"slab": {"NCORE": 16, "KPAR": 1}, "ref": {"NCORE": 8, "KPAR": 1}},
            "static_overrides": {"LREAL": ".FALSE.", "LCHARG": ".FALSE.", "LVHAR": ".FALSE."},
            # False: do not run the bare slab and the molecule again; name the package that already has them
            "references": True, "references_in": None}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def prepare(pool: Path, out: Path, cfg: dict, overwrite: bool) -> dict:
    summ = json.loads((pool / "pool.json").read_text())
    fam = cfg["profile_family"]
    slab = read(pool / summ["references"]["slab"]["file"])
    ax = int(np.argmax(slab.cell.lengths()))
    cell = slab.cell.array
    ov_slab = {**cfg["parallel"]["slab"], **cfg["static_overrides"]}
    man = {"schema": "static_audit/1", "title": "MLIP audit: r2SCAN+D3(BJ) single points at model geometries",
           "profile": f"{fam}/static", "overrides": cfg["static_overrides"], "common_cell_A": cell.tolist(),
           "normal_axis": ax, "thresholds": THRESHOLDS, "model": summ["provenance"].get("evaluator"),
           "pool_provenance": {k: summ["provenance"][k] for k in ("slab", "slab_mapping", "molecule", "code_commit")
                               if k in summ["provenance"]},
           "jobs": {}, "groups": {}, "notes": []}
    specs = {}
    s0 = write_job(slab, out / "slab" / "static", f"{fam}/static", "bare_slab", normal_axis=ax, kpoints=cfg["kpoints"],
                   overrides=ov_slab, overwrite=overwrite)
    dipol = s0.incar["DIPOL"]
    specs["slab"] = s0
    if cfg.get("references", True):
        man["jobs"]["slab"] = {"static": "slab/static", "role": "bare_slab", "kpoints": cfg["kpoints"], "natoms": len(slab),
                               "nelect": s0.nelect, "group": "slab", "source": summ["references"]["slab"]["file"],
                               "model_energy_eV": summ["references"]["slab"]["model_energy_eV"],
                               "structure_hash": s0.structure_hash}
    else:                                   # written only to fix the dipole centre; the static already exists elsewhere
        import shutil
        shutil.rmtree(out / "slab")
        man["notes"].append(f"No reference statics here: the bare slab (structure_hash {s0.structure_hash}) and the "
                            f"isolated molecule are in {cfg.get('references_in') or 'an earlier package'}; the DIPOL "
                            "centre is that slab's.")
    for rec in summ["structures"]:
        f = pool / rec["file"]
        a = read(f)
        if not np.allclose(a.cell.array, cell, atol=1e-6):
            raise ValueError(f"{rec['name']}: not on the common cell")
        if rec.get("relaxed") and not (rec.get("substrate_ok") and rec.get("final_state") == "intact_adsorption"):
            man["notes"].append(f"{rec['name']}: model final state '{rec.get('final_state')}', substrate_ok="
                                f"{rec.get('substrate_ok')}; kept and labelled (a model-domain test, not a ranked "
                                "structure)")
        d = min_image_distances(cell, a.pbc, a.positions, a.positions)
        np.fill_diagonal(d, np.inf)
        if d.min() < cfg["min_interatomic_A"]:
            raise ValueError(f"{rec['name']}: atoms closer than {cfg['min_interatomic_A']} A")
        s = write_job(a, out / rec["name"] / "static", f"{fam}/static", "complex", normal_axis=ax,
                      kpoints=cfg["kpoints"], overrides={**ov_slab, "DIPOL": dipol}, overwrite=overwrite)
        specs[rec["name"]] = s
        grp = f"{rec['formula']}"
        man["groups"].setdefault(grp, []).append(rec["name"])
        man["jobs"][rec["name"]] = {
            "static": f"{rec['name']}/static", "role": "complex", "kpoints": cfg["kpoints"], "natoms": len(a),
            "nelect": s.nelect, "group": grp, "selection": rec["group"], "relaxed_by_model": bool(rec.get("relaxed")),
            "model_final_state": rec.get("final_state"), "model_converged": rec.get("converged"),
            "model_energy_eV": rec["model_energy_eV"], "source": rec["file"], "source_sha256_16": _sha(f),
            "structure_hash": s.structure_hash, "min_interatomic_A": round(float(d.min()), 3)}
    if cfg.get("references", True):
        mol = read(pool / summ["references"]["molecule"]["file"])
        sm = write_job(mol, out / "ref_molecule" / "static", f"{fam}/molecule_static", "reference", kpoints=[1, 1, 1],
                       overrides={**cfg["parallel"]["ref"], "LREAL": cfg["static_overrides"]["LREAL"]}, overwrite=overwrite)
        man["jobs"]["ref_molecule"] = {"static": "ref_molecule/static", "role": "reference", "kpoints": [1, 1, 1],
                                       "natoms": len(mol), "nelect": sm.nelect, "group": "molecule",
                                       "model_energy_eV": summ["references"]["molecule"]["model_energy_eV"],
                                       "source": summ["references"]["molecule"]["file"],
                                       "structure_hash": sm.structure_hash}
    problems = check_matched_settings(specs)
    if problems:
        raise SystemExit("slab-type statics are not comparable:\n  " + "\n  ".join(problems))
    man["notes"] += [
        "Statics only: every POSCAR is the geometry at which the model energy and forces were stored. Nothing relaxes.",
        "LREAL = .FALSE. (force audit); do not combine these energies with label sets run with LREAL = Auto.",
        "The slab is the archived r2SCAN slab mapped onto the common cell (face region then relaxed with the audited "
        "MODEL, if the pool's provenance has slab_prerelax), not a slab relaxed with DFT in that cell: these are "
        "model-audit data, not adsorption-energy labels.",
        "ref_molecule is the molecule at its start geometry in a vacuum box; with 'slab' it gives fixed-geometry "
        "interaction energies for model-vs-DFT comparison only.",
    ]
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8", newline="\n")
    return man


# ----------------------------------------------------------------------------- analysis of a returned package
def read_static(d: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """(E0 eV, positions, forces) of a finished static, from vasprun.xml(.gz); refuses an unfinished step."""
    if not (d / "DONE").exists():
        raise FileNotFoundError(f"{d}: no DONE marker")
    f = d / "vasprun.xml.gz"
    src = gzip.open(f, "rt") if f.exists() else open(d / "vasprun.xml")
    import tempfile
    with src as fh, tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as tmp:
        tmp.write(fh.read())
    a = read(tmp.name, format="vasp-xml", index=-1)
    Path(tmp.name).unlink()
    return float(a.get_potential_energy(force_consistent=False)), a.positions, a.get_forces()


def _spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze(pkg: Path, pool: Path, reader=None) -> dict:
    reader = reader or read_static
    man = json.loads((pkg / "manifest.json").read_text())
    th = man["thresholds"]
    rows, near_dft, near_ml, far_dft, far_ml = {}, [], [], [], []
    missing = []
    for name, j in man["jobs"].items():
        if j["role"] != "complex":
            continue
        try:
            e, pos, f = reader(pkg / "outputs" / name / "static")
        except FileNotFoundError as ex:
            missing.append(str(ex))
            continue
        a = read(pool / j["source"])
        order = [int(i) for i in json.loads((pkg / "inputs" / name / "static" / "job.json").read_text())["atom_map"]]
        idx = {int(i): k for k, i in enumerate(a.arrays["atom_id"])}
        perm = np.array([idx[i] for i in order])            # DFT row -> pool row
        # VASP writes positions wrapped into the cell; compare modulo lattice vectors
        frac = np.linalg.solve(a.cell.array.T, (a.positions[perm] - pos).T).T
        dev = np.abs((frac - np.round(frac)) @ a.cell.array).max()
        if dev > 1e-3:
            raise ValueError(f"{name}: returned geometry differs from the audited one (max {dev:.4f} A)")
        fm = a.arrays["model_forces"][perm]
        ads = (a.arrays["mol_id"] > 0)[perm]
        if ads.any():
            d = min_image_distances(a.cell.array, a.pbc, pos, pos[ads]).min(axis=1)
            near = ads | (d <= 4.0)
        else:                                   # a bare slab audited as a structure: no "near the molecules" region
            near = np.zeros(len(pos), bool)
        near_dft.append(f[near]), near_ml.append(fm[near]), far_dft.append(f[~near]), far_ml.append(fm[~near])
        moving = np.linalg.norm(f, axis=1) > 1e-8       # the model's forces are stored for every atom; compare all
        rows[name] = {"group": j["group"], "selection": j["selection"], "E_dft": e, "E_model": j["model_energy_eV"],
                      "force_rmse_near": float(np.sqrt(((f - fm)[near] ** 2).mean())) if near.any() else None,
                      "force_rmse_all": float(np.sqrt(((f - fm)[moving] ** 2).mean())),
                      "dft_fmax": float(np.linalg.norm(f, axis=1).max())}
    out = {"schema": "static_audit_report/1", "thresholds": th, "missing": missing, "structures": rows, "groups": {}}

    def force_stats(D, M):
        D, M = np.concatenate(D), np.concatenate(M)
        big = np.linalg.norm(D, axis=1) > th["force_cos_only_above_eV_A"]
        cos = (D[big] * M[big]).sum(1) / (np.linalg.norm(D[big], axis=1) * np.linalg.norm(M[big], axis=1) + 1e-12)
        return {"rmse_eV_A": float(np.sqrt(((D - M) ** 2).mean())), "mean_cos": float(cos.mean()) if big.any() else None,
                "n_atoms": int(len(D))}
    if rows:
        out["forces_near"], out["forces_far"] = force_stats(near_dft, near_ml), force_stats(far_dft, far_ml)
    ok = bool(rows) and not missing
    for grp in sorted({r["group"] for r in rows.values()}):
        names = [n for n, r in rows.items() if r["group"] == grp]
        ed = np.array([rows[n]["E_dft"] for n in names])
        em = np.array([rows[n]["E_model"] for n in names])
        rel_d, rel_m = ed - ed.mean(), em - em.mean()
        g = {"n": len(names), "dft_span_eV": float(np.ptp(ed)), "relative_energy_mae_eV": float(np.abs(rel_d - rel_m).mean()),
             "relative_energy_max_eV": float(np.abs(rel_d - rel_m).max())}
        if len(names) >= 10 and np.ptp(ed) >= 0.5:
            g["spearman"] = _spearman(ed, em)
            g["top3_overlap"] = len(set(np.argsort(ed)[:3]) & set(np.argsort(em)[:3]))
            ok &= g["spearman"] >= th["spearman_min"] and g["top3_overlap"] >= th["top3_overlap_min"]
        else:
            g["spearman"] = None
            g["rank_note"] = "rank agreement not tested: " + th["spearman_needs"]
        if len(names) >= 2:
            ok &= g["relative_energy_mae_eV"] <= th["relative_energy_mae_eV"] and \
                g["relative_energy_max_eV"] <= th["relative_energy_max_eV"]
        out["groups"][grp] = g
    if rows:
        ok &= out["forces_near"]["rmse_eV_A"] <= th["force_rmse_near_eV_A"] and \
            (out["forces_near"]["mean_cos"] or 0) >= th["force_cos_min"]
    out["passes"] = bool(ok)
    out["not_tested_here"] = "binding-mode stability under DFT relaxation (tranche V2)"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True), ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None), ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyze", default=None, help="returned package folder")
    a = ap.parse_args(argv)
    if a.analyze:
        rep = analyze(Path(a.analyze), Path(a.pool))
        if a.out:
            Path(a.out).write_text(json.dumps(rep, indent=1))
        print(json.dumps({k: v for k, v in rep.items() if k != "structures"}, indent=1))
        return 0 if rep["passes"] else 1
    cfg = {**DEFAULTS, **(json.loads(Path(a.config).read_text()) if a.config else {})}
    assert f"{cfg['profile_family']}/static" in PROFILES
    man = prepare(Path(a.pool), Path(a.out), cfg, a.overwrite)
    print(f"{len(man['jobs'])} statics in {len(man['groups'])} composition groups -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
