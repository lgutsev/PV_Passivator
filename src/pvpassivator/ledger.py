"""Atom / charge / electron ledger and balanced-reaction checks.

Conventions
-----------
* ``cell_charge`` (q_cell) is the net charge of a simulation cell in units of +e.
  For VASP: q_cell = sum_i ZVAL_i - NELECT.  A default-NELECT calculation has q_cell = 0.
* ``formal_charge`` is the sum of *formal* ionic charges assigned to chemical units
  (e.g. FA+ , Pb2+, I-). It is a bookkeeping label, not a DFT observable.
* If formal_charge != cell_charge, the difference is carried by electronic carriers
  (holes if formal < cell ... see ``electronic_carriers``). This is recorded, never hidden.
* Partial charges (force field, Bader, DDEC) are *not* stored here; see analysis modules.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ase.data import atomic_numbers

# Formal charges of recognised chemical units, keyed by Hill-ish sorted formula.
# Extend through ``register_unit``; never guess for unknown molecules.
KNOWN_UNITS: dict[str, tuple[str, int]] = {
    "CH5N2": ("FA+", +1),      # formamidinium HC(NH2)2+
    "CH6N": ("MA+", +1),       # methylammonium CH3NH3+
    "Cs": ("Cs+", +1),
    "Pb": ("Pb2+", +2),
    "I": ("I-", -1),
    "Br": ("Br-", -1),
    "Cl": ("Cl-", -1),
}


HALIDE_OXO_BOND = 2.3   # A, halide-O distance treated as a covalent oxyanion bond (I-O in IO3- ~1.8)


def register_unit(formula: Mapping[str, int] | str, label: str, charge: int) -> None:
    key = formula if isinstance(formula, str) else formula_key(formula)
    KNOWN_UNITS[key] = (label, int(charge))


def formula_key(comp: Mapping[str, int]) -> str:
    """Hill order string, e.g. {'C':1,'H':5,'N':2} -> 'CH5N2'."""
    comp = {k: v for k, v in comp.items() if v}
    els = sorted(comp)
    if "C" in comp:
        els = ["C"] + (["H"] if "H" in comp else []) + [e for e in els if e not in ("C", "H")]
    return "".join(f"{e}{comp[e] if comp[e] != 1 else ''}" for e in els)


@dataclass(frozen=True)
class Species:
    """A chemically defined energy-bearing entity in a reaction.

    ``composition`` counts atoms. ``cell_charge`` is the DFT net charge (+e units).
    ``formal_units`` optionally lists the formal ionic decomposition, e.g.
    {'FA+': 18, 'Pb2+': 9, 'I-': 36}; used to detect electronic compensation.
    """
    name: str
    composition: Mapping[str, int]
    cell_charge: int = 0
    spin_multiplicity: int | None = None
    formal_units: Mapping[str, int] | None = None
    kind: str = "species"          # slab | molecule | ion | salt | reservoir | electron
    notes: str = ""

    def formal_charge(self) -> int | None:
        if self.formal_units is None:
            return None
        tot = 0
        for label, n in self.formal_units.items():
            q = _label_charge(label)
            tot += q * n
        return tot

    def electronic_carriers(self) -> int | None:
        """cell_charge - formal_charge: >0 means electrons removed beyond the formal
        ionic picture (holes), <0 means excess electrons. None if no formal units."""
        f = self.formal_charge()
        return None if f is None else self.cell_charge - f

    def valence_electrons(self, zval: Mapping[str, float]) -> float:
        return sum(zval[e] * n for e, n in self.composition.items()) - self.cell_charge


ELECTRON = Species("e-", {}, cell_charge=-1, kind="electron",
                   notes="electron reservoir; energy = Fermi level / mu_e, must be declared")


def _label_charge(label: str) -> int:
    for _, (lab, q) in KNOWN_UNITS.items():
        if lab == label:
            return q
    # parse trailing charge e.g. 'PEA+', 'SO4 2-', 'Pb2+'
    s = label.rstrip()
    sign = 0
    if s.endswith("+"):
        sign = 1
    elif s.endswith("-"):
        sign = -1
    else:
        return 0
    body = s[:-1]
    digits = ""
    while body and body[-1].isdigit():
        digits = body[-1] + digits
        body = body[:-1]
    # treat digits as magnitude only if label is e.g. 'Pb2+'; ambiguous for 'X2+' formulas
    return sign * (int(digits) if digits else 1)


def composition_of(symbols: Iterable[str]) -> dict[str, int]:
    return dict(Counter(symbols))


def identify_units(atoms, mult: float = 1.15) -> tuple[dict[str, int], list[dict]]:
    """Split an ASE Atoms (periodic) into covalently bonded fragments and map them to
    known formal units. Returns (unit_counts, unknown_fragments). Pb-I bonds are NOT
    treated as covalent (ionic lattice), so each Pb / I is its own unit. A halide with an O
    within ``HALIDE_OXO_BOND`` is covalent instead, so oxyanions (IO3- etc.) stay whole.
    """
    from ase.neighborlist import natural_cutoffs, NeighborList, neighbor_list
    import numpy as np

    syms = atoms.get_chemical_symbols()
    cut = natural_cutoffs(atoms, mult=mult)
    ionic = {"Pb", "I", "Br", "Cl", "Cs", "Sn"}
    oxo = set()
    if "O" in syms:
        ii, jj = neighbor_list("ij", atoms, HALIDE_OXO_BOND)
        oxo = {int(i) for i, j in zip(ii, jj) if syms[i] in ("I", "Br", "Cl") and syms[j] == "O"}
    cut = [0.0 if (s in ionic and k not in oxo) else c for k, (s, c) in enumerate(zip(syms, cut))]
    nl = NeighborList(cut, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)
    n = len(atoms)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in nl.get_neighbors(i)[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    counts: Counter = Counter()
    unknown = []
    for idx in groups.values():
        key = formula_key(Counter(syms[i] for i in idx))
        if key in KNOWN_UNITS:
            counts[KNOWN_UNITS[key][0]] += 1
        else:
            unknown.append({"formula": key, "indices": sorted(idx)})
    return dict(counts), unknown


@dataclass
class Reaction:
    """sum_reactants nu_i S_i -> sum_products nu_i S_i, energies in eV."""
    reactants: list[tuple[float, Species]]
    products: list[tuple[float, Species]]
    label: str = ""
    energy_kind: str = "reaction"   # adsorption_relaxed | interaction_frozen | incremental | exchange | defect | reaction
    n_units: float | None = None     # normalisation (adsorbed units), required for adsorption kinds

    def equation(self) -> str:
        def side(terms):
            return " + ".join(f"{_fmt(nu)}{s.name}" for nu, s in terms)
        return f"{side(self.reactants)} -> {side(self.products)}"

    def atom_balance(self) -> dict[str, float]:
        bal: Counter = Counter()
        for nu, s in self.products:
            for e, n in s.composition.items():
                bal[e] += nu * n
        for nu, s in self.reactants:
            for e, n in s.composition.items():
                bal[e] -= nu * n
        return {e: v for e, v in bal.items() if abs(v) > 1e-9}

    def charge_balance(self) -> float:
        return (sum(nu * s.cell_charge for nu, s in self.products)
                - sum(nu * s.cell_charge for nu, s in self.reactants))

    def check(self) -> list[str]:
        """Return list of human-readable problems; empty list == balanced."""
        problems = []
        ab = self.atom_balance()
        if ab:
            parts = ", ".join(f"{e}: {v:+g}" for e, v in sorted(ab.items()))
            problems.append(f"Atoms not conserved (products minus reactants): {parts}. "
                            "A species or reservoir is missing, e.g. a counterion or released FA/H.")
        cb = self.charge_balance()
        if abs(cb) > 1e-9:
            problems.append(f"Net charge not conserved: products - reactants = {cb:+g} e. "
                            "Add the compensating ion or an explicit electron reservoir (e-) with a declared mu_e.")
        if self.energy_kind.startswith("adsorption") and not self.n_units:
            problems.append("Adsorption energy requires n_units (number of adsorbed units, all faces).")
        return problems

    def assert_balanced(self) -> None:
        p = self.check()
        if p:
            raise UnbalancedReaction(self.equation() + "\n  " + "\n  ".join(p))

    def delta_e(self, energies: Mapping[str, float]) -> dict:
        """Reaction energy from a {species name: energy (eV)} map. Refuses to run if
        unbalanced or a reference energy is missing (never fabricates one)."""
        self.assert_balanced()
        missing = [s.name for _, s in self.reactants + self.products if s.name not in energies]
        if missing:
            raise MissingReference(f"No energy for: {missing}")
        de = (sum(nu * energies[s.name] for nu, s in self.products)
              - sum(nu * energies[s.name] for nu, s in self.reactants))
        out = {"equation": self.equation(), "energy_kind": self.energy_kind,
               "delta_E_total_eV": de, "sign_convention": "negative = exothermic"}
        if self.n_units:
            out["n_units"] = self.n_units
            out["delta_E_per_unit_eV"] = de / self.n_units
        return out


class UnbalancedReaction(ValueError):
    pass


class MissingReference(KeyError):
    pass


def _fmt(x: float) -> str:
    return "" if x == 1 else (f"{int(x)} " if float(x).is_integer() else f"{x:g} ")


def adsorption_reaction(complex_: Species, slab: Species,
                        adsorbates: list[tuple[float, Species]], **kw) -> Reaction:
    """E_ads = E(complex) - E(slab) - sum n_j E(ref_j). n_units defaults to sum n_j."""
    n = kw.pop("n_units", sum(nu for nu, _ in adsorbates))
    return Reaction([(1, slab)] + list(adsorbates), [(1, complex_)],
                    energy_kind=kw.pop("energy_kind", "adsorption_relaxed"), n_units=n, **kw)
