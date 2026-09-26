"""Reject unsupported precision before emitting fixed-FP64 SCF CUDA kernels."""

from dataclasses import replace

import pytest
from vibeqc_compiler.tensor import scf_cuda
from vibeqc_compiler.tensor.ir import einsum, input_tensor, multiply
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.scf import (
    density_input_specs,
    weighted_density_input_specs,
)


def _float32_program(weighted: bool) -> Program:
    specs = (weighted_density_input_specs if weighted else density_input_specs)(
        1, 3, spin_count=2, orbital_count=2
    )
    inputs = {
        name: input_tensor(name, replace(spec, dtype="float32"))
        for name, spec in specs.items()
    }
    weights = inputs["occupations"]
    if weighted:
        weights = multiply(weights, inputs["orbital_energies"])
    result = einsum(
        "bspi,bsi,bsqi->bspq",
        inputs["coefficients"],
        weights,
        inputs["coefficients"],
    )
    return Program({"weighted_density" if weighted else "density": result})


@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("emit", (False, True))
def test_cuda_density_rejects_non_fp64_ir_before_hash_or_emission(
    monkeypatch: pytest.MonkeyPatch, weighted: bool, emit: bool
) -> None:
    # The tensor equation/topology is valid, but the AOT ABI always uses double.
    # Silently emitting identical FP64 bytes for FP32 IR misrepresents its dtype.
    program = _float32_program(weighted)
    builder = "weighted_density_program" if weighted else "density_program"
    monkeypatch.setattr(scf_cuda, builder, lambda *args, **kwargs: program)
    target = (
        scf_cuda.emit_density_cuda
        if emit
        else scf_cuda.weighted_density_template_hash
        if weighted
        else scf_cuda.density_template_hash
    )
    with pytest.raises(ValueError, match="float64"):
        target()
