"""Shared validation for exact rational scientific parameters."""

from __future__ import annotations

from fractions import Fraction
from typing import Any


def require_fraction(
    value: Any,
    label: Any,
    error_type: type[Exception],
    *,
    role: str,
) -> Fraction:
    """Require an exact Fraction while preserving the caller's domain error."""
    if not isinstance(value, Fraction):
        raise error_type(f"{label} requires an exact Fraction {role}")
    return value
