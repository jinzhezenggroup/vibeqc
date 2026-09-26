"""Bounded radial batching shared by ECP CUDA emission and resource planning."""

import typing

RADIAL_TILE = 4
MAX_BATCHED_AOS = 16
MAX_BATCHED_POLAR = 44


def cuda_radial_tile(nao: typing.Any, polar: typing.Any) -> typing.Any:
    """Retain the single-layer schedule outside the measured small-grid domain."""
    return (
        RADIAL_TILE
        if 0 < nao <= MAX_BATCHED_AOS and 0 < polar <= MAX_BATCHED_POLAR
        else 1
    )


def emit_ecp_schedule_cpp() -> typing.Any:
    return [
        "inline constexpr unsigned ecp_cuda_radial_tile(unsigned nao, unsigned polar) {",
        f"  return nao > 0 && nao <= {MAX_BATCHED_AOS} && polar > 0 &&",
        f"      polar <= {MAX_BATCHED_POLAR} ? {RADIAL_TILE} : 1;",
        "}",
    ]
