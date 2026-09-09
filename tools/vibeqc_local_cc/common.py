"""Shared identity and validation rules for owned local-space records."""

from hashlib import sha256
from numbers import Real

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.reference import ReferenceSnapshot


def checked_reference(snapshot):
    """Require #147's validated canonical reference and its exact Hamiltonian."""
    if not isinstance(snapshot, ReferenceSnapshot):
        raise TypeError("local spaces require a validated ReferenceSnapshot")
    if snapshot.algorithm != "RHF" or snapshot.frozen_mask:
        raise ValueError("local spaces currently require unfrozen closed-shell RHF")
    return snapshot.nocc, snapshot.nmo - snapshot.nocc


def number(value, name, *, positive=False):
    """Reject nonfinite tolerances and implicit string/bool conversions."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {name}")
    return value


def fingerprint(metadata, **arrays):
    """Hash exact portable values; numerical subspace equivalence is separate."""
    return canonical_hash(
        {
            **metadata,
            **{
                k: sha256(np.asarray(v, dtype="<f8", order="C").tobytes()).hexdigest()
                for k, v in arrays.items()
            },
        }
    )


def orthogonality(columns, metric=None):
    """Maximum Gram-matrix error, including the valid empty pair space."""
    if columns.shape[1] == 0:
        return 0.0
    gram = columns.T @ columns if metric is None else columns.T @ metric @ columns
    return float(np.max(np.abs(gram - np.eye(columns.shape[1]))))


def checked_budget(budget_bytes):
    """Bound declared numeric storage before expensive transformations."""
    if type(budget_bytes) is not int or not 0 < budget_bytes < 2**63:
        raise ValueError("numeric budget must be a positive int64 byte count")
    return budget_bytes
