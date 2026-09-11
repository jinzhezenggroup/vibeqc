"""Immutable shell constants shared across CUDA component/consumer lowering."""

from __future__ import annotations

from ..shell_spec import (
    AXES,
    DPPP_SPEC,
)

DpppComponent = tuple[str, str, str, str]

_AXIS_INDEX = {axis: index for index, axis in enumerate(AXES)}

_COMPONENT_COUNT = DPPP_SPEC.component_count
