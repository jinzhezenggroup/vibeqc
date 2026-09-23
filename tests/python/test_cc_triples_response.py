"""Compiler-generated standard-(T) response tests for #154 A."""

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import execute, optimize
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

from tools.vibeqc_cc.triples import triples_energy
from tools.vibeqc_cc.triples_response import (
    TRIPLES_RESPONSE_INPUTS,
    accumulate_tile_triples_vjp,
    build_tile_triples_vjp,
    full_triples_vjp,
    tile_triples_vjp,
)
from tools.vibeqc_cc.triples_tiles import build_tile_triples_program

INPUT_NAMES = ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")


def _random_case(nocc: int, nvir: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    ovvv = rng.normal(size=(nocc, nvir, nvir, nvir))
    ovoo = rng.normal(size=(nocc, nvir, nocc, nocc))
    ovov = rng.normal(size=(nocc, nvir, nocc, nvir))
    fov = rng.normal(size=(nocc, nvir))
    t1 = rng.normal(size=(nocc, nvir))
    t2 = rng.normal(size=(nocc, nocc, nvir, nvir))
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    eps_o = np.linspace(-1.0, -0.5, nocc)
    eps_v = np.linspace(0.5, 1.5, nvir)
    return ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v


@pytest.mark.parametrize("v,chunk", [(2, 1), (2, 2), (3, 2)])
def test_disjoint_tile_vjps_sum_to_untiled_vjp_without_double_counting(
    v: int, chunk: int
) -> None:
    """The derivative of the disjoint energy partition equals the full VJP."""

    o = 2
    arrays = _random_case(o, v, 15401)
    tiled = accumulate_tile_triples_vjp(
        o,
        v,
        *arrays,
        vir_chunk_size=chunk,
    )
    full = full_triples_vjp(o, v, *arrays)
    assert set(tiled) == set(full) == set(TRIPLES_RESPONSE_INPUTS)
    for name in TRIPLES_RESPONSE_INPUTS:
        assert np.linalg.norm(full[name]) > 1e-8
        np.testing.assert_allclose(
            tiled[name],
            full[name],
            rtol=2e-11,
            atol=2e-11,
            err_msg=f"tile accumulation mismatch for {name}",
        )


def test_native_tile_response_prewarms_exact_executed_set() -> None:
    o, v = 2, 3
    arrays = _random_case(o, v, 15406)
    selected = ("t1", "t2")

    class RecordingExecutor:
        def __init__(self) -> None:
            self.programs = ()
            self.executed = []

        def prewarm(self, programs: object) -> None:
            assert not self.programs
            self.programs = tuple(programs)

        def execute(self, program: object, feeds: object) -> object:
            assert any(program is entry for entry in self.programs)
            self.executed.append(program)
            return execute(program, feeds).outputs

    executor = RecordingExecutor()
    actual = accumulate_tile_triples_vjp(
        o, v, *arrays, vir_chunk_size=1, inputs=selected, executor=executor
    )
    expected = accumulate_tile_triples_vjp(
        o, v, *arrays, vir_chunk_size=1, inputs=selected
    )

    assert len(executor.programs) == len(executor.executed) == v
    assert len({program.logical_hash for program in executor.programs}) == v
    assert executor.programs == tuple(executor.executed)
    for name in selected:
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize(
    "name",
    ["t1", "t2", "ovvv", "ovoo", "ovov", "fov", "eps_o", "eps_v"],
)
def test_generated_response_matches_recomputed_energy_finite_difference(
    name: str,
) -> None:
    """Each generated source differentiates the actual audited (T) energy."""

    # nocc=1 makes r3 identically zero and cannot validate derivative equations.
    o, v = 2, 2
    arrays = [np.array(x, copy=True) for x in _random_case(o, v, 15402)]
    by_name = dict(zip(INPUT_NAMES, arrays, strict=True))
    gradient = accumulate_tile_triples_vjp(
        o,
        v,
        *arrays,
        vir_chunk_size=1,
        inputs=(name,),
    )[name]
    assert np.linalg.norm(gradient) > 1e-8

    rng = np.random.default_rng(15403 + INPUT_NAMES.index(name))
    direction = rng.normal(size=by_name[name].shape)
    if name == "t2":
        direction = (direction + direction.transpose(1, 0, 3, 2)) / 2
    direction /= max(1.0, float(np.linalg.norm(direction)))
    analytic = float(np.vdot(gradient, direction))
    assert abs(analytic) > 1e-8

    for step in (2e-5, 2e-6):
        samples = []
        for sign in (-1.0, 1.0):
            changed = [np.array(x, copy=True) for x in arrays]
            changed[INPUT_NAMES.index(name)] += sign * step * direction
            samples.append(triples_energy(o, v, *changed))
        finite_difference = (samples[1] - samples[0]) / (2 * step)
        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=2e-6,
            atol=2e-7,
        )


def test_denominator_response_is_present_and_not_frozen() -> None:
    """eps_o/eps_v cotangents must carry reciprocal-denominator response."""

    o, v = 2, 2
    arrays = _random_case(o, v, 15404)
    response = accumulate_tile_triples_vjp(
        o,
        v,
        *arrays,
        vir_chunk_size=1,
        inputs=("eps_o", "eps_v"),
    )
    assert np.linalg.norm(response["eps_o"]) > 0
    assert np.linalg.norm(response["eps_v"]) > 0


def test_optimized_reverse_graph_preserves_response_and_never_grows_live_dag() -> None:
    """CSE/dead cleanup may share work but cannot change a response source."""

    o, v = 2, 2
    arrays = _random_case(o, v, 15405)
    raw = build_tile_triples_vjp(o, v, vir_chunk=(0, v))
    optimized = optimize(raw.program)
    assert len(optimized.live_nodes) <= len(raw.program.live_nodes)

    before = tile_triples_vjp(o, v, *arrays, inputs=("t1", "eps_v"))
    after = tile_triples_vjp(
        o,
        v,
        *arrays,
        inputs=("t1", "eps_v"),
        optimize_graph=True,
    )
    for name in before:
        assert np.linalg.norm(before[name]) > 1e-8
        np.testing.assert_allclose(after[name], before[name], rtol=1e-12, atol=1e-12)


def test_generated_vjp_keeps_primal_identity_and_is_cuda_plannable() -> None:
    """#154 A remains a compiler derivative, not a second equation stack."""

    o, v = 1, 2
    primal = build_tile_triples_program(o, v, vir_chunk=(0, v))
    reverse = build_tile_triples_vjp(o, v, vir_chunk=(0, v))
    assert reverse.primal_logical_hash == primal.logical_hash
    assert reverse.program.provenance["generation"] == "demand-driven"
    assert reverse.program.provenance["primal_logical_hash"] == primal.logical_hash

    target = cuda_target_info("sm_80")
    retained = plan_cuda(reverse.program, target)
    recomputed = plan_cuda(
        reverse.program,
        target,
        schedule=TensorSchedule(recompute=True),
    )
    assert recomputed.estimated_flops >= retained.estimated_flops
    assert recomputed.schedule.recompute is True

    # Recompute is an explicit schedule/cost choice, not a promise that every
    # multi-output reverse graph has a smaller arena.  The triples response
    # bounds memory by demand-driving only the requested cotangent blocks.
    selected = build_tile_triples_vjp(o, v, vir_chunk=(0, v), inputs=("t1", "t2"))
    selected_plan = plan_cuda(selected.program, target)
    assert selected_plan.arena_bytes < retained.arena_bytes
    assert selected_plan.peak_bytes < retained.peak_bytes
