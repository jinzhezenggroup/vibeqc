"""Native runtime-shape DF actions agree with independently executed TensorIR."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.cc.df_equations import build_df_virtual_response_programs
from generativeqc_compiler.tensor import execute

from tools import generate_df_ccsd_native as generator

ROOT = Path(__file__).resolve().parents[2]


def test_df_generation_has_no_runtime_or_reference_dependency() -> None:
    """Generation must work before a native library or NumPy is installed."""
    script = r"""
import importlib.abc
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "python")]
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"numpy", "pyscf", "torch", "cupy", "generativeqc"}:
            raise ImportError("compiler imported runtime dependency: " + fullname)
        return None
sys.meta_path.insert(0, BlockRuntime())
from tools.generate_df_ccsd_native import cpu_header, cuda_header, cuda_source
assert "run_virtual_cpu" in cpu_header()
assert "run_factor_vjp_cuda" in cuda_header()
assert "run_amplitude_vjp_cuda" in cuda_source()
from tools.generate_df_ccsd_hoisted import cpu_header, cuda_header, cuda_source
assert "run_prepare_cpu" in cpu_header()
assert "run_auxiliary_cuda" in cuda_header()
assert "run_iteration_cuda" in cuda_source()
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture(scope="module", params=("cpu", "cuda"))
def df_native_probe(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    backend = request.param
    if sys.platform == "win32":
        pytest.skip("requires a POSIX host compiler")
    if backend == "cuda" and os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1":
        pytest.skip("explicit real-GPU qualification is required")
    compiler = shutil.which("c++") if backend == "cpu" else shutil.which("nvcc")
    cache = shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires the selected native compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True, text=True)
    directory = tmp_path_factory.mktemp("df-cc-" + backend)
    for name, producer in (
        ("generated_df_ccsd_cpu.hpp", generator.cpu_header),
        ("generated_df_ccsd_cuda.cuh", generator.cuda_header),
        ("generated_df_ccsd_cuda.cu", generator.cuda_source),
    ):
        (directory / name).write_text(producer())
    executable = directory / ("probe-" + backend)
    command = [
        cache,
        compiler,
        "-std=c++20",
        "-O2",
        "-I" + str(directory),
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "include"),
    ]
    source = ROOT / "tests/native/df_cc_actions_probe.cpp"
    sources = [source]
    if backend == "cuda":
        command += ["-x", "cu", "-arch=sm_120", "-DDF_PROBE_CUDA=1"]
        sources.append(directory / "generated_df_ccsd_cuda.cu")
    objects = []
    for source in sources:
        obj = directory / (source.name + ".o")
        subprocess.run(
            [*command, "-c", str(source), "-o", str(obj)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        )
        objects.append(str(obj))
    subprocess.run(
        [compiler, *objects, "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return executable


def _inputs(o: int, v: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1853 + 11 * o + v)
    shapes = (
        (o, v),
        (v, v),
        (o, v),
        (o, o, v, v),
        (o, v),
        (o, o, v, v),
        (o, v),
        (o, o, v, v),
    )
    feeds = {
        name: rng.normal(scale=0.04, size=shape)
        for name, shape in zip(generator.INPUTS, shapes, strict=True)
    }
    feeds["bvv"] = (feeds["bvv"] + feeds["bvv"].T) / 2
    for name in ("t2", "d_t2"):
        feeds[name] = (feeds[name] + feeds[name].transpose(1, 0, 3, 2)) / 2
    return feeds


def _stream(
    o: int, v: int, action: int, feeds: dict[str, np.ndarray], short: bool = False
) -> str:
    return f"{o} {v} {action} {int(short)}\n" + " ".join(
        format(value, ".17g")
        for name in generator.INPUTS
        for value in feeds[name].ravel()
    )


@pytest.mark.parametrize("o,v", [(1, 1), (2, 3), (4, 2), (3, 7)])
@pytest.mark.parametrize("action", range(4))
def test_native_df_actions(df_native_probe: Path, o: int, v: int, action: int) -> None:
    feeds = _inputs(o, v)
    generated = build_df_virtual_response_programs(o, v)
    program = (
        generated.primal,
        generated.amplitude_jvp.program,
        generated.amplitude_vjp.program,
        generated.factor_vjp.program,
    )[action]
    expected = execute(program, feeds).outputs
    fields = tuple(generator.OUTPUTS.values())[action][1]
    wanted = np.concatenate([expected[key].ravel() for key in fields])
    result = subprocess.run(
        [str(df_native_probe)],
        input=_stream(o, v, action, feeds),
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    rows = result.stdout.splitlines()
    assert int(rows[0]) > 0
    np.testing.assert_allclose(
        np.asarray(rows[1:], dtype=float), wanted, atol=3e-12, rtol=1e-12
    )


@pytest.mark.parametrize("action", range(4))
def test_native_df_short_arena_refuses_before_arithmetic(
    df_native_probe: Path, action: int
) -> None:
    if df_native_probe.name.endswith("cuda"):
        pytest.skip("device arena admission belongs to the execution owner")
    result = subprocess.run(
        [str(df_native_probe)],
        input=_stream(2, 3, action, _inputs(2, 3), True),
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.startswith("refused ")


def test_composed_cuda_action_keeps_prior_arithmetic_failure(
    df_native_probe: Path,
) -> None:
    if df_native_probe.name != "probe-cuda":
        pytest.skip("sticky error state belongs to CUDA composition")
    feeds = _inputs(2, 3)
    supplied = _stream(2, 3, 0, feeds).replace("2 3 0 0\n", "2 3 0 2\n", 1)
    result = subprocess.run(
        [str(df_native_probe)],
        input=supplied,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "sticky 173"
