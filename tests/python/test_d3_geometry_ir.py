"""D3(BJ) GeometryIR/PairIR/TensorIR qualification."""

import json
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.geometry import D3CompilerSpec, compile_d3_bj
from vibeqc_compiler.tensor import execute

from tools.vibeqc_d3.reference import gfn1_compatibility, make_spec

_GOLDENS = json.loads(
    (Path(__file__).parents[1] / "data/d3_bj_reference.json").read_text()
)["fixtures"]


@pytest.mark.parametrize("case", _GOLDENS, ids=lambda case: case["name"])
def test_d3_pair_ir_matches_independent_simple_dftd3_goldens(
    case: typing.Any,
) -> None:
    spec = make_spec(**case["parameters"])
    coordinates = np.asarray(case["positions"], dtype=np.float64)
    compiled = compile_d3_bj(spec, case["numbers"], coordinates)
    result = execute(
        compiled.program,
        {"coordinates": coordinates},
    ).outputs
    assert result["energy"].item() == pytest.approx(
        case["energy"],
        abs=3e-13,
        rel=0,
    )
    reverse = execute(
        compiled.coordinate_vjp().program,
        {
            "coordinates": coordinates,
            "bar_energy": np.array(1.0),
        },
    ).outputs["bar_coordinates"]
    np.testing.assert_allclose(
        reverse,
        case["gradient"],
        atol=8e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        reverse.sum(axis=0),
        0.0,
        atol=2e-13,
        rtol=0,
    )


def test_compiler_spec_identity_matches_method_spec_without_import_cycle() -> None:
    spec = make_spec(
        s6=1.0,
        s8=0.7875,
        a1=0.4289,
        a2=4.4407,
    )
    compiler_spec = D3CompilerSpec.from_spec(spec)
    assert compiler_spec.identity == spec.identity
    assert compiler_spec.to_payload() == spec.to_payload()


def test_switch_state_is_explicit_and_generated_gradient_is_finite_difference() -> None:
    spec = gfn1_compatibility()
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [49.975, 0.0, 0.0]],
        dtype=np.float64,
    )
    compiled = compile_d3_bj(spec, (6, 8), coordinates)
    assert compiled.pair_state.energy_regions == ("switch",)
    assert compiled.pair_state.cn_active == (False,)

    output = execute(
        compiled.program,
        {"coordinates": coordinates},
    ).outputs
    gradient = execute(
        compiled.coordinate_vjp().program,
        {
            "coordinates": coordinates,
            "bar_energy": np.array(1.0),
        },
    ).outputs["bar_coordinates"]
    assert output["energy"].item() < 0.0

    step = 1e-5
    plus = coordinates.copy()
    minus = coordinates.copy()
    plus[1, 0] += step
    minus[1, 0] -= step
    compiled.validate_coordinates(plus)
    compiled.validate_coordinates(minus)
    ep = execute(
        compiled.program,
        {"coordinates": plus},
    ).outputs["energy"].item()
    em = execute(
        compiled.program,
        {"coordinates": minus},
    ).outputs["energy"].item()
    assert gradient[1, 0] == pytest.approx(
        (ep - em) / (2 * step),
        abs=2e-11,
        rel=0,
    )

    outside = coordinates.copy()
    outside[1, 0] = 50.0
    with pytest.raises(
        ValueError,
        match="stale D3 pair topology/switch state",
    ):
        compiled.validate_coordinates(outside)


def test_d3_primal_and_generated_vjp_lower_through_shared_cuda_tensorir() -> None:
    from vibeqc_compiler.common.cuda_target import CUDA_TARGETS
    from vibeqc_compiler.tensor.cuda_emit import emit_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    spec = make_spec(
        s6=1.0,
        s8=0.7875,
        a1=0.4289,
        a2=4.4407,
    )
    coordinates = np.array(
        [
            [0.1, -0.1, 0.2],
            [1.5, 0.3, 0.4],
            [-0.5, 1.4, 0.1],
        ],
        dtype=np.float64,
    )
    compiled = compile_d3_bj(spec, (8, 1, 1), coordinates)
    target = CUDA_TARGETS["sm_80"]
    primal = plan_cuda(compiled.program, target)
    reverse = plan_cuda(compiled.coordinate_vjp().program, target)
    primal_source = emit_cuda(primal)
    reverse_source = emit_cuda(reverse)
    assert "tensor_create" in primal_source
    assert "tensor_run" in primal_source
    assert "tensor_create" in reverse_source
    assert "tensor_run" in reverse_source
    assert reverse.program.provenance["mode"] == "vjp"
