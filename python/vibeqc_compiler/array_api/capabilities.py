"""Declared issue-#633 frontend subset; unsupported behavior fails closed."""

from __future__ import annotations

from .interop import DLPACK_INTEROP_VERSION

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
        "reference_execution": {
            "numpy_reference": "independent",
            "alternate_namespace": "validation-only",
            "production_native_dispatch": False,
        },
        "interoperability_disposition": {
            "reference_backends": "internal-validation",
            "dlpack": "internal-same-device-zero-copy",
            "array_api_conformance": False,
            "public_api": False,
        },
        "dlpack_interop": {
            "version": DLPACK_INTEROP_VERSION,
            "import": "same-device-zero-copy",
            "device_transfer": False,
            "stream_handoff": "consumer-owned-protocol",
            "raw_capsule_ownership": "not-retained",
        },
        "tensorir_metadata": (
            "index_spaces",
            "representation",
            "symmetry",
            "exact_coefficients",
            "role",
            "differentiability",
        ),
    }
