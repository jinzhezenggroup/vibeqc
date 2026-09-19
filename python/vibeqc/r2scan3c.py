"""Pinned data access for the canonical first r2SCAN-3c support domain."""

from pathlib import Path

from .basis import load_basis

_DATA = Path(__file__).resolve().parent / "data" / "r2scan3c"


def load_r2scan3c_basis():
    """Load the exact H-Ar def2-mTZVPP snapshot defining canonical r2SCAN-3c."""

    return load_basis(_DATA / "def2-mtzvpp-h-ar.json")
