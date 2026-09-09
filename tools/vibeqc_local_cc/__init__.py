"""Experimental restricted local spaces; no local CC energy or gradient API."""

from .localization import OccupiedLocalization, localize_occupied
from .spaces import PairSpace, ProjectedVirtualSpace, projected_virtual_space

__all__ = [
    "OccupiedLocalization",
    "PairSpace",
    "ProjectedVirtualSpace",
    "localize_occupied",
    "projected_virtual_space",
]
