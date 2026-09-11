"""Owned immutable finite FP64 arrays shared by method and grid consumers."""

import numpy as np


def immutable(value, *, shape=None):
    """Copy finite real FP64 values into irreversibly read-only storage."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("complex references are unsupported")
    array = np.asarray(raw, dtype=np.float64, order="C")
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("reference arrays must be finite")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)
