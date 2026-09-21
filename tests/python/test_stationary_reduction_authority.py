"""The native sum must consume the shared reduction program, not a second formula."""

from dataclasses import replace

import pytest
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.stationary_cuda import emit_stationary_scientific_kernels
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.tensor import Program


def plan(ecp: bool = False) -> StationaryGradientPlan:
    return StationaryGradientPlan(
        resolve_method("PBE", spin="polarized"),
        StationaryMeanField(
            SCF_POINT_MODEL,
            hamiltonian="scalar-semilocal-ecp" if ecp else "all-electron",
        ),
    )


def test_native_reduction_binds_the_authoritative_tensor_program() -> None:
    selected = plan()
    program = selected.reduction_program(atoms=1)
    assert (
        f"stationary-reduction-program: {program.logical_hash}"
        in emit_stationary_scientific_kernels(selected)
    )


def test_native_reduction_cannot_ignore_changed_program_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = plan()
    before = emit_stationary_scientific_kernels(selected)
    original = StationaryGradientPlan.reduction_program

    def changed(self: StationaryGradientPlan, **kwargs: object) -> Program:
        program = original(self, **kwargs)
        root = program.outputs["gradient"]
        coefficients = (*root.attrs["coefficients"][:-1], (1, 2))
        return Program(
            {"gradient": replace(root, attributes=(("coefficients", coefficients),))}
        )

    monkeypatch.setattr(StationaryGradientPlan, "reduction_program", changed)
    assert emit_stationary_scientific_kernels(selected) != before


def test_native_reduction_rejects_unsupported_external_source_inventory() -> None:
    from vibeqc_compiler.method.stationary_cuda import emit_stationary_cuda

    source = emit_stationary_cuda("", functional=1, plan=plan(ecp=True))
    assert "constexpr bool stationary_native_reduction_supported = false;" in source
