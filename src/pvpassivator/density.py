"""Frozen-fragment charge-displacement analysis.

Delta n(r) = n_complex(r) - sum_f n_f(r)          (electrons / A^3; >0 = accumulation)
lambda(z)  = ∬ Delta n dx dy                   (electrons / A)
CD(z)      = ∫_{zmin}^{z} lambda(z') dz'          (electrons)

Raw CHGCAR values are rho*V_cell; ase's VaspChargeDensity already divides by the volume,
returning electrons/A^3. Grids must be identical (shape, cell, atom positions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    density: np.ndarray        # electrons / A^3, shape (n0, n1, n2)
    cell: np.ndarray           # 3x3, A
    positions: np.ndarray | None = None
    nelect: float | None = None
    label: str = ""

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.cell)))

    def total_electrons(self) -> float:
        return float(self.density.mean() * self.volume)


def read_chgcar(path: str, label: str | None = None) -> Grid:
    from ase.calculators.vasp import VaspChargeDensity
    ch = VaspChargeDensity(path)
    atoms = ch.atoms[-1]
    return Grid(np.asarray(ch.chg[-1]), np.array(atoms.cell), atoms.get_positions(),
                label=label or path)


class GridMismatch(ValueError):
    pass


def check_compatible(complex_: Grid, fragments: list[Grid], pos_tol: float = 1e-3) -> None:
    for f in fragments:
        if f.density.shape != complex_.density.shape:
            raise GridMismatch(f"{f.label}: FFT grid {f.density.shape} != complex {complex_.density.shape}")
        if not np.allclose(f.cell, complex_.cell, atol=1e-5):
            raise GridMismatch(f"{f.label}: cell differs from complex")
    if complex_.positions is not None and all(f.positions is not None for f in fragments):
        cat = np.concatenate([f.positions for f in fragments])
        if len(cat) != len(complex_.positions):
            raise GridMismatch("fragment atom total != complex atom count (missing/duplicated atoms)")
        # fragments must reproduce complex coordinates as a set (frozen geometry)
        inv = np.linalg.inv(complex_.cell)
        for p in cat:
            d = (complex_.positions - p) @ inv
            d -= np.round(d)
            if np.min(np.linalg.norm(d @ complex_.cell, axis=1)) > pos_tol:
                raise GridMismatch("fragment coordinates are not the frozen complex coordinates")


@dataclass
class CDResult:
    axis: int
    z: np.ndarray               # A, along the surface-normal axis (fractional * |c|)
    lam: np.ndarray             # electrons / A
    cd: np.ndarray              # electrons, cumulative from z[0]
    area: float                 # A^2, in-plane cell area
    integral_total: float       # electrons; expected (N_complex - sum N_frag)
    expected_total: float | None
    warnings: list[str]

    def region(self, z0: float, z1: float) -> float:
        m = (self.z >= z0) & (self.z < z1)
        dz = self.z[1] - self.z[0]
        return float(self.lam[m].sum() * dz)


def charge_displacement(complex_: Grid, fragments: list[Grid], axis: int = 2,
                        check: bool = True, tol: float = 1e-2) -> CDResult:
    """Planar-integrated rearrangement along ``axis`` (index of the lattice vector
    normal to the surface). Requires the other two lattice vectors to be
    perpendicular to ``axis`` (standard slab setup); otherwise raises."""
    if check:
        check_compatible(complex_, fragments)
    cell = complex_.cell
    others = [i for i in range(3) if i != axis]
    for i in others:
        if abs(np.dot(cell[i], cell[axis])) > 1e-6 * np.linalg.norm(cell[i]) * np.linalg.norm(cell[axis]):
            raise ValueError("surface-normal lattice vector is not perpendicular to the in-plane vectors")
    dn = complex_.density - sum(f.density for f in fragments)
    area = float(np.linalg.norm(np.cross(cell[others[0]], cell[others[1]])))
    L = float(np.linalg.norm(cell[axis]))
    n = dn.shape[axis]
    dz = L / n
    # plane average * area = plane integral
    lam = dn.mean(axis=tuple(others)) * area           # electrons / A
    cd = np.cumsum(lam) * dz
    z = np.arange(n) * dz
    total = float(cd[-1])
    warnings = []
    expected = None
    if complex_.nelect is not None and all(f.nelect is not None for f in fragments):
        expected = complex_.nelect - sum(f.nelect for f in fragments)
        if abs(total - expected) > tol:
            warnings.append(f"integrated Delta n = {total:.4f} e but expected {expected:.4f} e "
                            "(grid/normalisation problem)")
        if abs(expected) > tol:
            warnings.append(f"fragments do not partition the electrons (net {expected:+.3f} e): "
                            "CD reflects electron exchange, not pure redistribution")
    for g in [complex_] + fragments:
        if g.nelect is not None and abs(g.total_electrons() - g.nelect) > max(tol, 1e-4 * g.nelect):
            warnings.append(f"{g.label}: density integrates to {g.total_electrons():.3f}, NELECT={g.nelect}")
    return CDResult(axis, z, lam, cd, area, total, expected, warnings)
