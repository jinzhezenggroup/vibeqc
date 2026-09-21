"""Generated CC response graphs execute through the generic native TensorIR CPU backend."""

import typing
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.tensor import PackedLayout, execute
from vibeqc_compiler.tensor.cpu import NativeTensorProgram, emit_cpu

from tools.vibeqc_cc.gradient_equations import build_hamiltonian_programs
from tools.vibeqc_cc.lambda_equations import build_lambda_programs
from tools.vibeqc_cc.oracle import dense_feeds, random_case
from tools.vibeqc_cc.triples_response import build_tile_triples_vjp


def _native(program: typing.Any, tmp_path: Path) -> NativeTensorProgram:
    return NativeTensorProgram(
        program,
        compiler=CppCompilerAdapter(Path("c++")),
        cache=tmp_path,
        max_bytes=256 << 20,
        max_work=1_000_000_000,
    )


def test_lambda_transpose_action_matches_interpreter(tmp_path: Path) -> None:
    o = v = 1
    f, g, t1, t2 = random_case(o, v, 152)
    feeds = {
        **dense_feeds(f, g, t1, t2),
        "bar_singles_residual": np.ones_like(t1),
        "bar_doubles_residual": np.ones_like(t2),
    }
    program = build_lambda_programs(o, v).residual_vjp.program
    expected = execute(program, feeds).outputs
    actual = _native(program, tmp_path).execute(feeds)
    for output_name in program.outputs:
        np.testing.assert_array_equal(actual[output_name], expected[output_name])


def test_hamiltonian_pullback_with_dense_symmetry_matches_interpreter(
    tmp_path: Path,
) -> None:
    program = build_hamiltonian_programs(1, 1).weights
    rng = np.random.default_rng(15501)
    feeds = {}
    for node in program.live_nodes:
        if node.op != "input":
            continue
        layout = PackedLayout.from_spec(node.spec)
        feeds[node.attrs["name"]] = layout.unpack(rng.normal(size=layout.size))
    expected = execute(program, feeds).outputs
    actual = _native(program, tmp_path).execute(feeds)
    for output_name in program.outputs:
        np.testing.assert_allclose(
            actual[output_name], expected[output_name], atol=5e-15, rtol=5e-15
        )


def test_triples_tile_vjp_matches_interpreter(tmp_path: Path) -> None:
    o = v = 1
    rng = np.random.default_rng(15502)
    feeds = {
        "ovvv": rng.normal(size=(o, v, v, v)),
        "ovoo": rng.normal(size=(o, v, o, o)),
        "ovov": rng.normal(size=(o, v, o, v)),
        "fov": rng.normal(size=(o, v)),
        "t1": rng.normal(size=(o, v)),
        "t2": rng.normal(size=(o, o, v, v)),
        "eps_o": np.array([-1.0]),
        "eps_v": np.array([0.8]),
        "bar_triples_energy": np.asarray(1.0),
    }
    program = build_tile_triples_vjp(o, v, vir_chunk=(0, v)).program
    expected = execute(program, feeds).outputs
    actual = _native(program, tmp_path).execute(feeds)
    for output_name in program.outputs:
        np.testing.assert_array_equal(actual[output_name], expected[output_name])


def test_production_shape_response_graphs_are_native_cpu_lowerable() -> None:
    o, v = 2, 3
    programs = build_lambda_programs(o, v)
    candidates = (
        programs.primal,
        programs.energy_vjp.program,
        programs.residual_vjp.program,
        build_hamiltonian_programs(o, v).weights,
        build_tile_triples_vjp(o, v, vir_chunk=(0, 2)).program,
    )
    for program in candidates:
        source, resources = emit_cpu(
            program,
            max_bytes=2 << 30,
            max_work=100_000_000_000,
        )
        assert "tensor_cpu" in source
        assert resources["required_bytes"] > 0
        assert resources["scalar_work"] > 0
