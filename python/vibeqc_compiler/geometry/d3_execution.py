"""Checked NumPy reference execution for fixed-topology D3 compiler programs."""

from __future__ import annotations

import numpy as np

from vibeqc_compiler.tensor import execute

from .d3 import D3GeometryProgram


def execute_d3_bj(
    compiled: D3GeometryProgram,
    coordinates: object,
    *,
    gradient: bool = False,
) -> dict[str, np.ndarray]:
    """Validate a coordinate snapshot before evaluating energy and optional dE/dR.

    This is the checked compiler-reference boundary, not a native/CUDA production
    executor. Raw ``compiled.program`` and ``coordinate_vjp().program`` are
    unchecked lowering interfaces; their callers own equivalent preflight checks.
    The returned mapping contains the primal outputs and, when requested, a
    ``gradient`` array with the same shape as the Cartesian coordinate input.
    """
    if not isinstance(compiled, D3GeometryProgram):
        raise TypeError("compiled must be a D3GeometryProgram")
    if type(gradient) is not bool:
        raise TypeError("gradient must be a Boolean")
    xyz = np.array(coordinates, dtype=np.float64, copy=True)
    compiled.validate_coordinates(xyz)
    coordinate_name = compiled.geometry.coordinate_name
    inputs = {coordinate_name: xyz}
    outputs = dict(execute(compiled.program, inputs).outputs)
    if gradient:
        reverse = compiled.coordinate_vjp()
        seeds = {
            name: np.ones_like(outputs[source])
            for name, source in reverse.input_map.items()
        }
        adjoints = execute(reverse.program, {**inputs, **seeds}).outputs
        gradient_name = next(
            name
            for name, source in reverse.output_map.items()
            if source == coordinate_name
        )
        outputs["gradient"] = adjoints[gradient_name]
    return outputs
