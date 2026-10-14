"""Representation-based passivant discovery (prototype; see docs/DISCOVERY_P2O.md).

Layers, each replaceable on its own:
  space.py       machine-readable search representations (what an LLM may propose) and their enumeration
  chem.py        deterministic Tier-0 chemistry: parsing, eligibility, descriptors, fingerprints
  objectives.py  frozen, hash-locked objective/constraint specification and Pareto analysis
  oracles.py     physical (or explicitly synthetic) evaluators with declared cost and tier
  optimize.py    numerical optimizers that select candidates from an enumerated pool
  store.py       append-only evaluation store with status labels, dedup, budget and held-out split
  proposer.py    provider-independent representation proposers and evidence-only feedback summaries
  loop.py        the closed loop and the three-arm benchmark

Requires the optional extra ``pip install -e .[discovery]`` (RDKit).
"""
