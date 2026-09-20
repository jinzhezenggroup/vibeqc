"""D3(BJ) GeometryIR/PairIR/TensorIR qualification."""

import json
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.geometry import D3CompilerSpec, compile_d3_bj, execute_d3_bj

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
    result = execute_d3_bj(compiled, coordinates, gradient=True)
    assert result["energy"].item() == pytest.approx(
        case["energy"],
        abs=3e-13,
        rel=0,
    )
    reverse = result["gradient"]
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

    output = execute_d3_bj(compiled, coordinates, gradient=True)
    gradient = output["gradient"]
    assert output["energy"].item() < 0.0

    step = 1e-5
    plus = coordinates.copy()
    minus = coordinates.copy()
    plus[1, 0] += step
    minus[1, 0] -= step
    ep = execute_d3_bj(compiled, plus)["energy"].item()
    em = execute_d3_bj(compiled, minus)["energy"].item()
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


@pytest.mark.parametrize("gradient", [False, True])
@pytest.mark.parametrize(
    ("before", "after", "cutoffs"),
    [
        (
            49.975,
            50.1,
            {"cn_cutoff": 25.0, "pair_cutoff": 50.0, "pair_switch_width": 0.05},
        ),
        (24.9, 25.1, {"cn_cutoff": 25.0, "pair_cutoff": 50.0}),
        (7.9, 8.1, {"pair_cutoff": 10.0, "pair_switch_width": 2.0}),
        (9.9, 10.1, {"pair_cutoff": 10.0}),
    ],
)
def test_checked_execution_rejects_stale_pair_cn_and_switch_state(
    before: float,
    after: float,
    cutoffs: dict[str, float],
    gradient: bool,
) -> None:
    spec = make_spec(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407, **cutoffs)
    coordinates = np.array([[0.0, 0.0, 0.0], [before, 0.0, 0.0]])
    compiled = compile_d3_bj(spec, (6, 8), coordinates)
    coordinates[1, 0] = after
    with pytest.raises(ValueError, match="stale D3 pair topology/switch state"):
        execute_d3_bj(compiled, coordinates, gradient=gradient)


def test_checked_execution_preserves_custom_coordinate_name_and_input() -> None:
    spec = make_spec(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407)
    coordinates = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    original = coordinates.copy()
    compiled = compile_d3_bj(spec, (1, 1), coordinates, coordinate_name="positions")
    identity = compiled.identity
    output = execute_d3_bj(compiled, coordinates, gradient=True)
    step = 1e-5
    plus, minus = coordinates.copy(), coordinates.copy()
    plus[1, 0] += step
    minus[1, 0] -= step
    finite_difference = (
        execute_d3_bj(compiled, plus)["energy"].item()
        - execute_d3_bj(compiled, minus)["energy"].item()
    ) / (2 * step)
    assert output["gradient"][1, 0] == pytest.approx(finite_difference, abs=2e-10)
    np.testing.assert_array_equal(coordinates, original)
    assert compiled.identity == identity


@pytest.mark.parametrize("atomic_numbers", [(1,), (6, 8)])
def test_checked_execution_handles_empty_pair_state(
    atomic_numbers: tuple[int, ...],
) -> None:
    spec = make_spec(
        s6=1.0,
        s8=0.7875,
        a1=0.4289,
        a2=4.4407,
        cn_cutoff=25.0,
        pair_cutoff=50.0,
    )
    coordinates = np.zeros((len(atomic_numbers), 3))
    if len(atomic_numbers) == 2:
        coordinates[1, 0] = 100.0
    compiled = compile_d3_bj(spec, atomic_numbers, coordinates)
    output = execute_d3_bj(compiled, coordinates, gradient=True)
    assert output["energy"].item() == 0.0
    np.testing.assert_array_equal(output["gradient"], np.zeros_like(coordinates))


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
