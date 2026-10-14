"""Tier-0 chemistry: deterministic parsing, eligibility rules, descriptors and fingerprints (RDKit).

Eligibility is defined once, in ``Eligibility``, and is shared by every search arm. A representation can narrow
the space but can never relax these rules (``space.validate_spec`` refuses fields that try to).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
import os
import sys

import numpy as np

try:
    from rdkit import Chem, RDConfig, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
except ImportError as e:          # pragma: no cover - exercised only without the extra
    raise ImportError("pvpassivator.discovery needs RDKit: pip install -e .[discovery]") from e

_SA = None


def _sascorer():
    """Ertl & Schuffenhauer synthetic-accessibility score, from RDKit Contrib (1 easy .. 10 hard)."""
    global _SA
    if _SA is None:
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer  # type: ignore
        _SA = sascorer
    return _SA


# Cationic anchors understood by the pilot. Patterns are matched on the protonated, declared form.
ANCHORS = {
    "primary_ammonium": "[NX4+;H3][CX4]",
    "anilinium": "[NX4+;H3]c",
    "secondary_ammonium": "[NX4+;H2]([CX4])[CX4]",
    "amidinium": "[NX3;H2][CX3](=[NX3+;H2])",          # formamidinium-like
    "guanidinium": "[NX3][CX3](=[NX3+])[NX3]",
    "pyridinium_NH": "[nX3+;H1]",
}
# Anionic anchors (e.g. the oxide-side half of a co-passivating salt).
ANION_ANCHORS = {
    "sulfonate": "[SX4](=O)(=O)[O-]",
    "carboxylate": "[CX3](=O)[O-]",
    "phosphonate": "[PX4](=O)([OX2H1,O-])[O-]",
}

# Reactive or implausible groups excluded from every arm (kept short and explicit; extend by review, not by LLM).
FORBIDDEN = {
    "peroxide": "[OX2][OX2]",
    "azide": "[N-]=[N+]=N",
    "acyl_halide": "C(=O)[Cl,Br,I]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "isocyanate": "N=C=O",
    "diazo": "[#6]=[N+]=[N-]",
    "free_thiol": "[SX2H1]",
    "alkyl_iodide": "[CX4]I",
    "N_N_single": "[NX3,NX4+][NX3,NX4+]",
    "hemiaminal_like": "[OX2H][CX4][NX3,NX4+]",
}
_PATS = {k: Chem.MolFromSmarts(v) for k, v in {**ANCHORS, **ANION_ANCHORS, **FORBIDDEN}.items()}


@dataclass(frozen=True)
class Eligibility:
    """Shared eligibility for the pilot: one organic monocation, counter-ion declared separately.
    ``anion()`` gives the matching rules for the anionic half of a salt."""
    elements: tuple[str, ...] = ("C", "H", "N", "O", "F", "S", "Cl", "Br")
    net_charge: int = 1
    min_heavy: int = 3
    max_heavy: int = 20
    max_rotatable: int = 8
    max_sa_score: float = 6.0
    anchors: tuple[str, ...] = tuple(ANCHORS)
    forbidden: tuple[str, ...] = tuple(FORBIDDEN)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def anion(cls, **kw) -> "Eligibility":
        return cls(net_charge=-1, anchors=tuple(ANION_ANCHORS), **kw)


@dataclass
class Candidate:
    smiles: str                      # canonical, protonated as declared
    inchikey: str
    anchors: list[str]
    descriptors: dict[str, float]
    source: str = ""                 # representation id / library id that produced it
    origin: dict = field(default_factory=dict)


def canonical(smiles: str) -> tuple[Chem.Mol | None, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "unparseable SMILES"
    return mol, ""


def check(smiles: str, rules: Eligibility = Eligibility()) -> tuple[Candidate | None, list[str]]:
    """Return (candidate, []) when eligible, else (None, reasons). Never repairs a structure silently."""
    mol, err = canonical(smiles)
    if mol is None:
        return None, [err]
    why = []
    if Chem.GetFormalCharge(mol) != rules.net_charge:
        why.append(f"net charge {Chem.GetFormalCharge(mol)} != {rules.net_charge}")
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        why.append("radical")
    if len(Chem.GetMolFrags(mol)) != 1:
        why.append("more than one fragment (counter-ions are declared separately)")
    els = {a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()}
    bad = els - set(rules.elements)
    if bad:
        why.append(f"elements not allowed: {sorted(bad)}")
    if any(a.GetIsotope() for a in mol.GetAtoms()) or any(a.GetSymbol() == "*" for a in mol.GetAtoms()):
        why.append("isotope label or unfilled attachment point")
    nh = mol.GetNumHeavyAtoms()
    if not rules.min_heavy <= nh <= rules.max_heavy:
        why.append(f"{nh} heavy atoms outside [{rules.min_heavy}, {rules.max_heavy}]")
    anchors = [k for k in rules.anchors if mol.HasSubstructMatch(_PATS[k])]
    if not anchors:
        why.append("no declared ionic anchor")
    sign = 1 if rules.net_charge > 0 else -1
    if any(a.GetFormalCharge() * sign < 0 for a in mol.GetAtoms()):
        why.append("zwitterion (opposite charges in one ion)")
    hits = [k for k in rules.forbidden if mol.HasSubstructMatch(_PATS[k])]
    if hits:
        why.append(f"forbidden groups: {hits}")
    if rdMolDescriptors.CalcNumRotatableBonds(mol) > rules.max_rotatable:
        why.append("too flexible")
    sa = _sascorer().calculateScore(mol)
    if sa > rules.max_sa_score:
        why.append(f"SA score {sa:.2f} > {rules.max_sa_score}")
    if why:
        return None, why
    smi = Chem.MolToSmiles(mol)
    return Candidate(smi, Chem.MolToInchiKey(mol), anchors, descriptors(mol, sa)), []


def descriptors(mol: Chem.Mol, sa: float | None = None) -> dict[str, float]:
    """Generic, conventional descriptors. Used by every arm and by the synthetic oracle alike."""
    hal = sum(a.GetSymbol() in ("F", "Cl", "Br") for a in mol.GetAtoms())
    return {
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "mol_wt": Descriptors.MolWt(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "rotatable": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "n_fluorine": float(sum(a.GetSymbol() == "F" for a in mol.GetAtoms())),
        "n_halogen": float(hal),
        "n_heteroatoms": float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        "sa_score": float(sa if sa is not None else _sascorer().calculateScore(mol)),
    }


_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles: list[str]) -> np.ndarray:
    """Binary Morgan r=2, 2048 bits (a conventional, well-established molecular representation)."""
    out = np.zeros((len(smiles), 2048), dtype=np.float64)
    for i, s in enumerate(smiles):
        out[i] = _FPGEN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
    return out


def tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ab = A @ B.T
    na = A.sum(1)[:, None]
    nb = B.sum(1)[None, :]
    den = na + nb - ab
    return np.where(den > 0, ab / np.maximum(den, 1e-12), 1.0)
