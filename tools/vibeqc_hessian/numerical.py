"""Numerical Hessian oracle for the analytic HF Hessian work (issue #180).

This is the step-2 oracle of slice A: a finite-difference-of-analytic-gradient
Hessian. It is deliberately independent of the analytic assembly in step 4 in
the useful direction -- it consumes only the analytic first derivatives, so a
sign or factor error shared between the skeleton and the relaxation assembly
cannot hide from it.

The discipline mirrors
:func:`vibeqc_compiler.common.evidence.finite_difference`:

* at least three distinct positive step sizes, and every one is reported --
  there is no best-step gate and no post-hoc selection;
* the method policy is frozen through a JSON round-trip before each sample so an
  evaluator cannot silently retune itself between steps;
* the analytic input is the energy *gradient*, never the force.

A Hessian produced here is a **numerical Hessian**. It is an oracle and an
early utility; it does not constitute analytic Hessian support.
"""

from __future__ import annotations

import json
import math
import typing

import numpy as np
from vibeqc.profiles import canonical_hash

__all__ = [
    "forces_to_gradient",
    "hessian_difference",
    "hessian_symmetry_error",
    "hessian_translation_error",
    "numerical_hessian",
]

DEFAULT_STEPS = (1e-2, 3e-3, 1e-3)


def forces_to_gradient(forces: typing.Any) -> typing.Any:
    """Convert native forces (``-dE/dR``) to the energy gradient (``dE/dR``).

    Native VibeQC forces are the negative energy derivative. The numerical
    oracle compares against gradients, so the sign conversion happens here, at
    one named boundary, rather than being applied implicitly by callers.
    """
    return -np.asarray(forces, dtype=np.float64)


def _as_hessian(values: typing.Any, *, name: str) -> typing.Any:
    """Return ``values`` as a ``(natom, 3, natom, 3)`` array, or raise.

    The axis order is part of the contract every consumer relies on: a matrix
    stored as ``(3N, 3N)`` or as ``(natom, natom, 3, 3)`` would be reduced
    against the wrong elements by the checks below rather than rejected.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4 or array.shape[1] != 3 or array.shape[3] != 3:
        raise ValueError(
            f"{name} must have shape (natom, 3, natom, 3), got {array.shape}"
        )
    if array.shape[0] != array.shape[2]:
        raise ValueError(
            f"{name} must be square in its atom indices, got {array.shape}"
        )
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def hessian_symmetry_error(hessian: typing.Any) -> float:
    """Return the largest raw asymmetry, ``max |H - H^T|``.

    This is evaluated on the raw assembled matrix. Any later presentation
    symmetrization would hide exactly the errors this check exists to expose, so
    callers must assert on the unsymmetrized array.
    """
    values = _as_hessian(hessian, name="hessian")
    return float(np.max(np.abs(values - values.transpose(2, 3, 0, 1))))


def hessian_translation_error(hessian: typing.Any) -> float:
    """Return ``max |sum_a H[a, c, b, d]|``, the translation zero-mode residual.

    Translational invariance of the energy gives ``sum_a dE/dR[a, c] = 0`` for
    every Cartesian direction ``c``. Differentiating once more with respect to
    ``R[b, d]`` gives the identity checked here. It holds for every geometry,
    stationary or not, so it needs no stationarity condition.
    """
    values = _as_hessian(hessian, name="hessian")
    return float(np.max(np.abs(values.sum(axis=0))))


def hessian_difference(actual: typing.Any, reference: typing.Any) -> dict:
    """Report elementwise statistics of ``actual - reference``.

    Both arrays are ``(natom, 3, natom, 3)``. No tolerance is applied and no
    element is filtered out: the full error distribution is returned so a
    caller cannot promote on a favourable subset.

    The layout is validated the same way the symmetry and translation helpers
    validate it, and the shapes must match exactly. Two Hessians both stored as
    ``(3N, 3N)`` would otherwise subtract happily and return a statistic that
    does not describe the layout this function documents; a broadcastable but
    different shape would silently reduce against the wrong elements.
    """
    left = _as_hessian(actual, name="actual")
    right = _as_hessian(reference, name="reference")
    if left.shape != right.shape:
        raise ValueError(
            f"cannot compare Hessians of shape {left.shape} and {right.shape}"
        )
    difference = left - right
    error = np.abs(difference)
    return {
        "max_absolute_error": float(error.max()) if error.size else 0.0,
        "rms_error": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        "shape": list(difference.shape),
    }


def _gradient_at(
    gradient: typing.Any,
    coordinates: typing.Any,
    policy: str,
    expected_shape: typing.Any,
) -> typing.Any:
    """Evaluate the gradient under a freshly decoded copy of the frozen policy.

    Decoding a new copy per evaluation is what prevents a stateful evaluator
    from adapting its settings between the plus and minus displacements, or
    between step sizes.

    The returned shape is checked rather than assumed. The differencing below
    writes into a ``(natom, 3)`` slot, so a callback returning a smaller but
    broadcastable array -- a scalar, or a bare ``(3,)`` -- would be silently
    replicated across atoms and produce a plausible-looking matrix from
    meaningless input. Non-finite values are rejected for the same reason: they
    would propagate into the report as if they were measurements.
    """
    values = np.asarray(gradient(coordinates, json.loads(policy)), dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError(
            f"gradient callback returned shape {values.shape}, "
            f"expected {expected_shape} (natom, 3)"
        )
    if not np.isfinite(values).all():
        raise ValueError("gradient callback returned a non-finite value")
    # A callback may return a view of a reusable native work buffer. Snapshot
    # it before the next evaluation can overwrite the plus-displacement result.
    return values.copy()


def numerical_hessian(
    gradient: typing.Any,
    coordinates: typing.Any,
    *,
    settings: dict,
    steps: typing.Any = DEFAULT_STEPS,
) -> dict:
    """Report the whole central-difference Hessian curve under one frozen policy.

    ``gradient(coordinates, settings)`` must return the analytic energy gradient
    ``dE/dR`` with shape ``(natom, 3)``. Use :func:`forces_to_gradient` if the
    available endpoint reports forces.

    The returned Hessian is indexed ``H[a, c, b, d] = d(grad[b, d]) / d(R[a, c])``
    with shape ``(natom, 3, natom, 3)`` -- the same axis order the analytic
    assembly reports, so the two can be compared without reshaping.

    Every requested step size is evaluated and recorded. The result carries no
    pass/fail verdict: choosing a step, or a tolerance, is the caller's
    acceptance decision and stays visible at the call site.
    """
    step_sizes = tuple(float(step) for step in steps)
    if len(set(step_sizes)) < 3 or any(
        not math.isfinite(step) or step <= 0 for step in step_sizes
    ):
        raise ValueError("at least three distinct positive finite step sizes required")

    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError(f"coordinates must have shape (natom, 3), got {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("coordinates must be finite")

    policy = json.dumps(settings, allow_nan=False)
    samples = []
    for step in step_sizes:
        hessian = np.empty(xyz.shape + xyz.shape, dtype=np.float64)
        evaluations = 0
        for index in np.ndindex(xyz.shape):
            plus, minus = xyz.copy(), xyz.copy()
            plus[index] += step
            minus[index] -= step
            hessian[index] = (
                _gradient_at(gradient, plus, policy, xyz.shape)
                - _gradient_at(gradient, minus, policy, xyz.shape)
            ) / (2.0 * step)
            evaluations += 2
        samples.append(
            {
                "step_bohr": step,
                "hessian": hessian.tolist(),
                "gradient_evaluations": evaluations,
                "symmetry_error": hessian_symmetry_error(hessian),
                "translation_error": hessian_translation_error(hessian),
                "max_absolute": float(np.max(np.abs(hessian))) if hessian.size else 0.0,
            }
        )

    # The record stores the round-tripped policy, not the caller's object: that
    # is what the evaluator actually received, and it cannot be mutated after
    # the call to disagree with settings_hash. The hash is unchanged by the
    # round trip because canonical_hash encodes through JSON as well.
    recorded = json.loads(policy)
    return {
        "settings": recorded,
        "settings_hash": canonical_hash(recorded),
        "coordinates": xyz.tolist(),
        "samples": samples,
    }
