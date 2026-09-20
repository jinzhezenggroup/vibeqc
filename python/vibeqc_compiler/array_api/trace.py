"""Capture ordinary symbolic array expressions as ordinary TensorIR Programs."""

from __future__ import annotations

import typing
from collections.abc import Callable, Mapping

from vibeqc_compiler.tensor.ir import input_tensor
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import TensorSpec

from .array import VibeArray
from .capabilities import FRONTEND_VERSION


def input_array(name: str, spec: TensorSpec) -> VibeArray:
    """Construct one symbolic input while preserving the complete TensorSpec."""
    if not isinstance(spec, TensorSpec):
        raise TypeError("input_array requires a TensorSpec")
    return VibeArray(input_tensor(name, spec))


def _outputs(result: object, *, output_name: str) -> dict[str, typing.Any]:
    if isinstance(result, VibeArray):
        if not isinstance(output_name, str) or not output_name.isidentifier():
            raise ValueError("output_name must be an identifier")
        return {output_name: result.node}
    if isinstance(result, Mapping):
        outputs = {}
        for name, value in result.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("trace output names must be identifiers")
            if not isinstance(value, VibeArray):
                raise TypeError("trace output values must be symbolic VibeArrays")
            outputs[name] = value.node
        if not outputs:
            raise ValueError("trace requires at least one output")
        return outputs
    raise TypeError("traced function must return a VibeArray or mapping of VibeArrays")


def trace(
    function: Callable[..., object],
    inputs: Mapping[str, TensorSpec],
    *,
    output_name: str = "output",
    provenance: Mapping[str, object] | None = None,
) -> Program:
    """Run a pure symbolic function once and return the lowered TensorIR Program."""
    if not callable(function):
        raise TypeError("trace requires a callable")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("trace requires a nonempty mapping of input TensorSpecs")
    arrays = {}
    for name, spec in inputs.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("trace input names must be identifiers")
        arrays[name] = input_array(name, spec)
    result = function(**arrays)
    metadata = {} if provenance is None else dict(provenance)
    previous = metadata.setdefault("array_frontend_version", FRONTEND_VERSION)
    if previous != FRONTEND_VERSION:
        raise ValueError("array_frontend_version provenance is incompatible")
    return Program(
        _outputs(result, output_name=output_name),
        provenance=metadata,
    )
