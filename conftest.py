"""Pytest bootstrap: make the --basetemp parent exist on a fresh clone.

pyproject.toml sets ``--basetemp=data/cache/pytest``; ``data/cache/`` is
git-ignored, so a clean checkout lacks it and every tmp_path test errors.
"""
from pathlib import Path


def pytest_configure(config):
    base = config.getoption("basetemp", default=None)
    if base:
        Path(base).parent.mkdir(parents=True, exist_ok=True)
