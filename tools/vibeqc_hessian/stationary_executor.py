"""Compatibility shim for the production-owned second-order executor.

The canonical implementation lives in :mod:`vibeqc.second_order` so installed
runtime code no longer depends on the repository-only Hessian tools package.
This module intentionally contains no second implementation.
"""

from vibeqc.second_order import (
    StationaryHVPContext,
    StationaryHVPContributor,
    StationaryPerturbationProvider,
    StationaryResponseDriver,
    StationarySecondOrderExecutor,
    StationarySecondOrderResult,
)

__all__ = [
    "StationaryHVPContext",
    "StationaryHVPContributor",
    "StationaryPerturbationProvider",
    "StationaryResponseDriver",
    "StationarySecondOrderExecutor",
    "StationarySecondOrderResult",
]
