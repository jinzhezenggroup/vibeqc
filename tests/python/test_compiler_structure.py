"""Compiler package ownership, bootstrap and legacy class identity regressions."""

import os
import subprocess
import sys
import typing
from pathlib import Path

import pytest
from vibeqc_compiler.common.structure import audit_structure

ROOT = Path(__file__).resolve().parents[2]


def test_grid_native_generator_matches_jit_policy(tmp_path: typing.Any) -> None:
    """Native and JIT builds must compile exactly one scientific grid policy."""
    from vibeqc_compiler.dft.ao_cuda import (
        emit_grid_source,
        emit_native_xc_contraction_kernels,
    )

    output = tmp_path / "grid.cu"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_grid_kernels.py"),
            "--output",
            str(output),
        ],
        check=True,
        cwd=tmp_path,
    )
    native = emit_grid_source(native_ks=True)[0]
    assert output.read_text() == native
    # Resident KS adds only a consumer of the shared AO/ingredient policy.
    # Keep the independently compiled JIT owner's ABI free of that extension.
    source, _, headers = emit_grid_source()
    native_contractions = emit_native_xc_contraction_kernels()
    assert native == source + native_contractions + '#include "cuda_xc_kernels.cuh"\n'
    runtime = source.index('#include "cuda_grid.cu"')
    for scientific in (
        "__global__ void ao_kernel",
        "__global__ void feature_kernel",
        "__device__ vibeqc::dft::point::Value evaluate_xc_point",
        "__global__ void xc_local_potential_kernel",
    ):
        assert scientific in source
        assert source.index(scientific) < runtime
    for scientific in (
        "__global__ void density_product",
        "__global__ void density_features",
        "struct DevicePointValue",
        "__device__ inline DevicePointValue response_point",
        "__device__ inline DevicePointValue evaluate_semilocal_point",
        "__global__ void evaluate_points",
        "__global__ void assemble_potential",
        "__global__ void accumulate_totals",
    ):
        assert scientific in native_contractions
        assert scientific not in source
    native_template = (ROOT / "src/dft/cuda_grid.cu").read_text()
    assert "__global__ void ao_kernel" not in native_template
    assert (
        "__device__ vibeqc::dft::point::Value evaluate_xc_point" not in native_template
    )
    resident_template = (ROOT / "src/dft/cuda_xc_kernels.cuh").read_text()
    for retired in (
        "__global__ void density_product",
        "__global__ void density_features",
        "struct DevicePointValue",
        "__device__ inline DevicePointValue response_point",
        "__device__ inline DevicePointValue evaluate_semilocal_point",
        "__global__ void evaluate_points",
        "__global__ void assemble_potential",
        "__global__ void accumulate_totals",
    ):
        assert retired not in resident_template
    assert headers[-1] == ROOT / "include/vibeqc/vibeqc.h"


def test_dependency_directions() -> None:
    assert audit_structure()["errors"] == []


def test_method_composition_is_above_xc_and_dft(tmp_path: typing.Any) -> None:
    method = tmp_path / "method"
    method.mkdir()
    (method / "ok.py").write_text(
        "from vibeqc_compiler.xc.spec import FunctionalSpec\n"
    )
    assert audit_structure(tmp_path)["errors"] == []

    dft = tmp_path / "dft"
    dft.mkdir()
    (dft / "bad.py").write_text("import vibeqc_compiler.method\n")
    errors = audit_structure(tmp_path)["errors"]
    assert any("forbidden dft -> method import" in error for error in errors)


@pytest.mark.parametrize(
    "module,target,allowed",
    [
        ("ao_cuda", "expr", True),
        ("ao_cuda", "cuda", True),
        ("ao_cuda", "df_cuda", False),
        ("feature_policy", "expr", True),
        ("feature_policy", "cuda", True),
        ("feature_policy", "scalar_c", True),
        ("feature_policy", "df_cuda", False),
        ("spatial", "expr", False),
    ],
)
def test_ao_lowering_scalar_dependency_is_narrow(
    tmp_path: typing.Any, module: typing.Any, target: typing.Any, allowed: typing.Any
) -> None:
    dft = tmp_path / "dft"
    dft.mkdir()
    (dft / (module + ".py")).write_text(f"import vibeqc_compiler.integral.{target}\n")
    assert (not audit_structure(tmp_path)["errors"]) == allowed


def test_installed_package_does_not_consume_neighbor_checkout(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    """A wheel placed under another checkout must use its own bundled inputs."""
    from vibeqc_compiler.common import paths

    (tmp_path / "CMakeLists.txt").touch()
    foreign = tmp_path / "src/tensor/cuda_runtime.cuh"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("different checkout")
    package = tmp_path / "site/vibeqc_compiler"
    bundled = package / "assets/src/tensor/cuda_runtime.cuh"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("installed template")
    monkeypatch.setattr(paths, "PACKAGE", package)
    with pytest.raises(ValueError, match="source checkout"):
        paths.source_root()
    assert paths.asset_path("src/tensor/cuda_runtime.cuh") == bundled


@pytest.mark.parametrize(
    "code",
    [
        "from ..tensor import Program",
        "import benchmarks.aot_shell_batch_gate",
        "from vibeqc import Calculator",
        "def run():\n    import tools.generate_shell_kernels",
    ],
)
def test_generic_code_rejects_upward_dependencies(
    tmp_path: typing.Any, code: typing.Any
) -> None:
    common = tmp_path / "common"
    common.mkdir()
    (common / "bad.py").write_text(code + "\n")
    assert audit_structure(tmp_path)["errors"]


def test_all_compiler_imports_are_independent_of_runtime_and_references() -> None:
    code = f"""
import importlib, importlib.abc, pkgutil, sys
sys.path.insert(0, {str(ROOT / "python")!r})
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'vibeqc', 'tools', 'benchmarks', 'pyscf', 'torch', 'cupy'}}:
            raise AssertionError('compiler imported ' + fullname)
sys.meta_path.insert(0, RejectRuntime())
import vibeqc_compiler
for item in pkgutil.walk_packages(vibeqc_compiler.__path__, vibeqc_compiler.__name__ + '.'):
    importlib.import_module(item.name)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        cwd="/",
        capture_output=True,
        text=True,
    )


def test_checkout_generator_needs_no_installation_or_runtime(
    tmp_path: typing.Any,
) -> None:
    # Bootstrap the checkout with the existing NumPy dependency, without a
    # native library, editable installation or inherited PYTHONPATH.
    output = tmp_path / "weighted.cuh"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "tools/generate_weighted_eri_kernels.py"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.stat().st_size > 0


def test_method_custom_derivatives_may_emit_tensor_graphs_but_not_the_reverse(
    tmp_path: typing.Any,
) -> None:
    method = tmp_path / "method"
    method.mkdir()
    (method / "rule.py").write_text("from vibeqc_compiler.tensor import Program\n")
    assert audit_structure(tmp_path)["errors"] == []
    tensor = tmp_path / "tensor"
    tensor.mkdir()
    (tensor / "bad.py").write_text("import vibeqc_compiler.method\n")
    assert any(
        "forbidden tensor -> method import" in e
        for e in audit_structure(tmp_path)["errors"]
    )
