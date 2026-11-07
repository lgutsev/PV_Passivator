"""local_settle.py (package-53 rebuild plan, step A3): constrained settle of a rebuilt complex."""
import json
import sys
from pathlib import Path

import numpy as np
from ase.calculators.lj import LennardJones
from ase.io import write

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "examples" / "scripts"))
import test_rebuild_on_slab as TR  # noqa: E402
import rebuild_on_slab as R  # noqa: E402
import local_settle as S  # noqa: E402


def lj():
    return LennardJones(sigma=1.2, epsilon=0.01, rc=4.0, smooth=True)    # weak and short-ranged: a gentle settle


def rebuilt(tmp_path):
    s, cx = TR.archived()
    rs = TR.relaxed(s)
    write(tmp_path / "cx.vasp", cx, format="vasp")
    write(tmp_path / "slab.vasp", rs, format="vasp")
    assert R.main(["--slab", str(tmp_path / "slab.vasp"), "--case", f"X={tmp_path / 'cx.vasp'}",
                   "--out", str(tmp_path / "rb")]) == 0
    return tmp_path / "rb"


def test_mobile_mask_keeps_molecules_whole_and_far_field_fixed():
    s, cx = TR.archived()
    out, rep = R.rebuild(TR.relaxed(s), cx)
    mob = S.mobile_mask(out, rep["salt_indices"], 6.0, rep["moved_indices"])
    assert mob[rep["salt_indices"]].all() and 0 < mob.sum() < len(out)
    sym = np.array(out.get_chemical_symbols())
    for i in np.flatnonzero(sym == "N"):                        # each N-H unit moves whole or not at all
        h = np.argmin([out.get_distance(i, j, mic=True) if sym[j] == "H" else 9e9 for j in range(len(out))])
        assert mob[i] == mob[h]


def test_settle_moves_only_mobile_atoms_and_reports(tmp_path):
    rb = rebuilt(tmp_path)
    assert S.main(["--rebuild-dir", str(rb), "--out", str(tmp_path / "st"), "--radius", "6", "--steps", "20"],
                  calc=lj()) == 0
    r = json.loads((tmp_path / "st" / "X" / "settle.json").read_text())
    assert r["settle"]["frozen_max_change_A"] == 0.0 and r["checks"]["frozen_unchanged"]["pass"]
    assert r["settle"]["steps"] <= 20 and r["settle"]["n_mobile"] > 0
    assert r["status"] in ("pass", "review") and "sites" in r["checks"]
    summ = json.loads((tmp_path / "st" / "summary.json").read_text())
    assert summ["model"] == "injected (test)" and summ["cases"]["X"]["status"] == r["status"]


def test_failed_rebuild_is_skipped(tmp_path):
    rb = rebuilt(tmp_path)
    rep = json.loads((rb / "X" / "rebuild.json").read_text())
    rep["status"] = "fail"
    (rb / "X" / "rebuild.json").write_text(json.dumps(rep))
    S.main(["--rebuild-dir", str(rb), "--out", str(tmp_path / "st")], calc=lj())
    assert json.loads((tmp_path / "st" / "summary.json").read_text())["cases"]["X"]["status"] == "skipped"
