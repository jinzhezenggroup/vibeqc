"""Declared issue-#633 frontend subset; unsupported behavior fails closed."""

from __future__ import annotations

FRONTEND_VERSION = 1

SUPPORTED_FUNCTIONS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "negative",
        "pow",
        "exp",
        "log",
        "sqrt",
        "sum",
        "reshape_explicit_indices",
        "permute_dims",
        "reshape",
        "broadcast_to",
        "slice",
        "take",
        "broadcast_to_explicit_indices",
        "slice_static",
        "take_static",
        "matmul_rank2",
        "einsum_extension",
    }
)


def capabilities() -> dict[str, object]:
    """Return a detached capability description for diagnostics/tests."""
    return {
        "frontend_version": FRONTEND_VERSION,
        "surface": "array-api-shaped-internal-preview",
        "array_api_version": None,
        "array_namespace_protocol": False,
        "implicit_broadcast": False,
        "reshape_requires_explicit_indices": True,
        "broadcast_requires_explicit_indices_and_axes": True,
        "slice_ranges": "static-half-open-unit-step",
        "take_indices": "static-int-tuple",
        "dtype_promotion": False,
        "dynamic_shapes": False,
        "python_control_flow": False,
        "functions": tuple(sorted(SUPPORTED_FUNCTIONS)),
        "tensorir_metadata": (
            "index_spaces",
            "representation",
            "symmetry",
            "exact_coefficients",
            "role",
            "differentiability",
        ),
    }
