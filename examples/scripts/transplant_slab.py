"""Put the arrangements of a pool onto a re-relaxed slab.

    python examples/scripts/transplant_slab.py POOL SLAB_RELAXED OUT --names a,b,c

POOL/structures/NAME.extxyz are slab + molecules with the slab atoms first and in the order of SLAB_RELAXED (checked
through ``atom_id``).  The slab atoms take the positions of SLAB_RELAXED; the molecule atoms keep theirs.  ``mobile``
is kept, ``model_forces`` is zeroed (stale).  Prints the shortest molecule-slab distance before and after, so a start
that the new surface has made a clash is visible before any relaxation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase.geometry import get_distances
from ase.io import read, write


def main(argv=None) -> int:
    pool, slab_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    names = sys.argv[sys.argv.index("--names") + 1].split(",")
    slab = read(slab_file)
    n = len(slab)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    s = slab.copy()
    for name in names:
        a = read(pool / "structures" / f"{name}.extxyz")
        assert (a.numbers[:n] == slab.numbers).all(), name
        if "atom_id" in slab.arrays:
            assert (a.arrays["atom_id"][:n] == slab.arrays["atom_id"]).all(), name
        mol = np.arange(len(a)) >= n

        def closest(x):
            d = get_distances(x.positions[mol], x.positions[~mol], cell=x.cell, pbc=x.pbc)[1]
            i, j = np.unravel_index(d.argmin(), d.shape)
            return f"{d.min():.2f} A ({x.symbols[n + i]}-{x.symbols[j]})"
        before = closest(a)
        moved = np.linalg.norm(slab.positions - a.positions[:n], axis=1)
        a.positions[:n] = slab.positions
        a.arrays["model_forces"] = np.zeros((len(a), 3))
        write(out / "structures" / f"{name}.extxyz", a, format="extxyz")
        print(f"{name}: closest molecule-slab contact {before} -> {closest(a)}; slab atoms moved up to {moved.max():.2f} A, "
              f"{int((moved > 0.5).sum())} by more than 0.5 A")
        s.arrays["mobile"] = a.arrays["mobile"][:n].copy()
    s.arrays["model_forces"] = np.zeros((n, 3))
    write(out / "structures" / "slab.extxyz", s, format="extxyz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
