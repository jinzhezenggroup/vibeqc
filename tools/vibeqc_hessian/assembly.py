"""Component assembly for the frozen-density RHF Hessian skeleton.

The second-derivative integral providers expose weighted tensors, while the
molecular Hessian also contains orbital-response terms.  This module only
assembles the four frozen-density components and keeps the response boundary
explicit: callers must supply the already signed overlap/Pulay contribution
and cannot accidentally receive a partial tensor labelled as a complete
analytic Hessian.
"""

from __future__ import annotations

import typing

import numpy as np

from .numerical import hessian_symmetry_error, hessian_translation_error

__all__ = ["assemble_frozen_skeleton", "validate_hessian_component"]


def validate_hessian_component(values: typing.Any, *, name: str) -> np.ndarray:
    """Return a finite Cartesian Hessian component with the canonical layout.

    Components use ``(atom, xyz, atom, xyz)`` ordering.  Requiring this shape
    at the assembly boundary prevents a broadcastable but transposed tensor
    from silently contaminating the total Hessian.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4 or array.shape[1] != 3 or array.shape[3] != 3:
        raise ValueError(
            f"{name} must have shape (natom, 3, natom, 3), got {array.shape}"
        )
    if array.shape[0] != array.shape[2] or array.shape[0] == 0:
        raise ValueError(f"{name} must be square and nonempty in atom indices")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array.copy()


def assemble_frozen_skeleton(
    *,
    nuclear_repulsion: typing.Any,
    one_electron_skeleton: typing.Any,
    overlap_pulay_skeleton: typing.Any,
    two_electron_skeleton: typing.Any,
) -> dict:
    """Assemble and report the four frozen-density RHF Hessian components.

    ``overlap_pulay_skeleton`` is expected to include its graph sign
    (``-Tr[W S^{{RR'}}]``).  The two-electron input likewise already contains
    the folded ``1/2`` and ``1/4`` density factors from
    :func:`two_electron_weight`; no additional prefactor is applied here.

    The return value contains independent copies under ``components``, their
    sum under ``skeleton``, and raw symmetry/translation diagnostics.  Response
    and relaxation terms are deliberately absent and are not represented as
    zeros, so a consumer cannot mistake this slice for a complete Hessian.
    """

    names = {
        "nuclear_repulsion": nuclear_repulsion,
        "one_electron_skeleton": one_electron_skeleton,
        "overlap_pulay_skeleton": overlap_pulay_skeleton,
        "two_electron_skeleton": two_electron_skeleton,
    }
    components = {
        name: validate_hessian_component(value, name=name)
        for name, value in names.items()
    }
    shapes = {value.shape for value in components.values()}
    if len(shapes) != 1:
        raise ValueError(f"Hessian components must have identical shapes, got {shapes}")
    skeleton = sum(
        components.values(), start=np.zeros_like(next(iter(components.values())))
    )
    return {
        "components": components,
        "skeleton": skeleton,
        "symmetry_error": hessian_symmetry_error(skeleton),
        "translation_error": hessian_translation_error(skeleton),
        "includes_response": False,
    }
