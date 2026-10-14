"""Search representations: versioned, machine-readable descriptions of a region of passivant space.

Two kinds are supported (JSON, schema ``pvpass.searchspace/1``):

``library``      an explicit list of SMILES (the expert baseline, or any fixed screen)
``combinatorial`` scaffolds with numbered attachment points ``[*:k]`` and one substituent list per point::

    {"schema": "pvpass.searchspace/1", "kind": "combinatorial", "id": "aryl_ethylammonium_v1",
     "author": "...", "rationale": "...", "design_variables": ["ring substitution", "linker length"],
     "scaffolds": [{"id": "PEA", "smiles": "[NH3+]CCc1ccc([*:1])cc1"}],
     "substituents": {"1": ["[*][H]", "[*]F", "[*]C(F)(F)F"]}}

A substituent is a SMILES with exactly one ``[*]`` dummy; ``[*][H]`` means "unsubstituted".
A representation may only *narrow* the shared eligibility rules; ``validate_spec`` rejects any attempt to
define objectives, thresholds, budgets or eligibility, so a proposer cannot move the goalposts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
import re

from rdkit import Chem

from .chem import Candidate, Eligibility, check

SCHEMA = "pvpass.searchspace/1"
ALLOWED_KEYS = {"schema", "kind", "id", "author", "rationale", "design_variables", "scaffolds",
                "substituents", "smiles", "provenance", "notes"}
# keys a proposer must never set; listed so the refusal message is explicit
PROTECTED_KEYS = {"objectives", "constraints", "thresholds", "eligibility", "budget", "acceptance", "oracle",
                  "weights", "holdout"}
MAX_ENUMERATION = 5000
MAX_SCAFFOLDS = 12
MAX_SUBSTITUENTS = 30


@dataclass
class SpecReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    n_enumerated: int = 0
    n_valid: int = 0
    n_duplicates: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_enumerated if self.n_enumerated else 0.0


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def load_spec(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_spec(spec: dict) -> list[str]:
    """Structural checks before any enumeration. Empty list means the spec may be enumerated."""
    err = []
    if spec.get("schema") != SCHEMA:
        err.append(f"schema must be {SCHEMA!r}")
    bad = set(spec) & PROTECTED_KEYS
    if bad:
        err.append(f"representation may not set protected fields {sorted(bad)}")
    extra = set(spec) - ALLOWED_KEYS - PROTECTED_KEYS
    if extra:
        err.append(f"unknown fields {sorted(extra)}")
    if not spec.get("id"):
        err.append("missing id")
    kind = spec.get("kind")
    if kind == "library":
        if not isinstance(spec.get("smiles"), list) or not spec["smiles"]:
            err.append("library needs a non-empty smiles list")
        elif len(spec["smiles"]) > MAX_ENUMERATION:
            err.append(f"library larger than {MAX_ENUMERATION}")
    elif kind == "combinatorial":
        scs = spec.get("scaffolds") or []
        subs = spec.get("substituents") or {}
        if not scs or len(scs) > MAX_SCAFFOLDS:
            err.append(f"need 1..{MAX_SCAFFOLDS} scaffolds")
        for k, lst in subs.items():
            if not isinstance(lst, list) or not lst or len(lst) > MAX_SUBSTITUENTS:
                err.append(f"substituent slot {k}: need 1..{MAX_SUBSTITUENTS} entries")
                continue
            for s in lst:
                m = Chem.MolFromSmiles(s)
                if m is None or sum(a.GetSymbol() == "*" for a in m.GetAtoms()) != 1:
                    err.append(f"substituent {s!r} in slot {k} must parse with exactly one [*]")
        for sc in scs:
            m = Chem.MolFromSmiles(sc.get("smiles", ""))
            if m is None:
                err.append(f"scaffold {sc.get('id')!r} does not parse")
                continue
            slots = {str(a.GetAtomMapNum()) for a in m.GetAtoms() if a.GetSymbol() == "*"}
            if any(a.GetSymbol() == "*" and not a.GetAtomMapNum() for a in m.GetAtoms()):
                err.append(f"scaffold {sc.get('id')!r}: every attachment point needs a map number [*:k]")
            missing = slots - set(subs)
            if missing:
                err.append(f"scaffold {sc.get('id')!r}: no substituents for slots {sorted(missing)}")
        if not err and _size(spec) > MAX_ENUMERATION:
            err.append(f"enumeration size {_size(spec)} exceeds {MAX_ENUMERATION}")
    else:
        err.append("kind must be 'library' or 'combinatorial'")
    return err


def _slots(smiles: str) -> list[str]:
    m = Chem.MolFromSmiles(smiles)
    return sorted({str(a.GetAtomMapNum()) for a in m.GetAtoms() if a.GetSymbol() == "*"})


def _size(spec: dict) -> int:
    n = 0
    for sc in spec["scaffolds"]:
        k = 1
        for s in _slots(sc["smiles"]):
            k *= len(spec["substituents"][s])
        n += k
    return n


def _attach(scaffold: str, choice: dict[str, str]) -> str | None:
    """Join substituents onto the scaffold's mapped dummies with RDKit molzip."""
    parts = [scaffold]
    for slot, sub in choice.items():
        parts.append(sub.replace("[*]", f"[*:{slot}]", 1))
    mol = Chem.MolFromSmiles(".".join(parts))
    if mol is None:
        return None
    try:
        z = Chem.molzip(mol)
        z = Chem.RemoveHs(z)
        Chem.SanitizeMol(z)
        return Chem.MolToSmiles(z)
    except Exception:
        return None


def enumerate_spec(spec: dict, rules: Eligibility = Eligibility()) -> tuple[list[Candidate], SpecReport]:
    """Deterministic enumeration + Tier-0 filtering. Duplicates (same InChIKey) keep their first origin."""
    errs = validate_spec(spec)
    if errs:
        return [], SpecReport(False, errs)
    raw: list[tuple[str, dict]] = []
    if spec["kind"] == "library":
        raw = [(s, {"index": i}) for i, s in enumerate(spec["smiles"])]
    else:
        for sc in spec["scaffolds"]:
            slots = _slots(sc["smiles"])
            for combo in itertools.product(*(spec["substituents"][s] for s in slots)):
                choice = dict(zip(slots, combo))
                smi = _attach(sc["smiles"], choice)
                raw.append((smi or "", {"scaffold": sc["id"], "substituents": choice}))
    out: list[Candidate] = []
    seen: set[str] = set()
    rep = SpecReport(True, n_enumerated=len(raw))
    for smi, origin in raw:
        cand, why = check(smi, rules) if smi else (None, ["attachment failed"])
        if cand is None:
            for w in why:
                key = re.sub(r"-?\d+(\.\d+)?", "#", w)          # group reasons that differ only in numbers
                rep.rejections[key] = rep.rejections.get(key, 0) + 1
            continue
        if cand.inchikey in seen:
            rep.n_duplicates += 1
            continue
        seen.add(cand.inchikey)
        cand.source = spec["id"]
        cand.origin = origin
        out.append(cand)
    rep.n_valid = len(out)
    return out, rep
