"""Compiler package ownership, bootstrap and legacy class identity regressions."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from vibeqc_compiler.common.structure import audit_structure

ROOT = Path(__file__).resolve().parents[2]


def test_dependency_directions():
    assert audit_structure()["errors"] == []


@pytest.mark.parametrize(
    "module,target,allowed",
    [
        ("ao_cuda", "expr", True),
        ("ao_cuda", "cuda", True),
        ("ao_cuda", "df_cuda", False),
        ("spatial", "expr", False),
    ],
)
def test_ao_lowering_scalar_dependency_is_narrow(tmp_path, module, target, allowed):
    dft = tmp_path / "dft"
    dft.mkdir()
    (dft / (module + ".py")).write_text(f"import vibeqc_compiler.integral.{target}\n")
    assert (not audit_structure(tmp_path)["errors"]) == allowed


def test_installed_package_does_not_consume_neighbor_checkout(tmp_path, monkeypatch):
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
def test_generic_code_rejects_upward_dependencies(tmp_path, code):
    common = tmp_path / "common"
    common.mkdir()
    (common / "bad.py").write_text(code + "\n")
    assert audit_structure(tmp_path)["errors"]


def test_all_compiler_imports_are_independent_of_runtime_and_references():
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


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("vibeqc_codegen.ir", "integral.ir"),
        ("vibeqc_codegen.lowering.fock", "integral.lowering.fock"),
        ("vibeqc_codegen.cuda_adapter", "common.cuda_adapter"),
        ("vibeqc_tensor.ir", "tensor.ir"),
        ("vibeqc_tensor.cuda_resources", "common.cuda_resources"),
        ("vibeqc_xc.spec", "xc.spec"),
        ("vibeqc_dft.grid", "dft.grid"),
    ],
)
def test_legacy_leaves_share_the_canonical_module(legacy, canonical, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    target = importlib.import_module("vibeqc_compiler." + canonical)
    assert importlib.import_module("tools." + legacy) is target
    assert importlib.import_module(legacy) is target


def test_checkout_generator_needs_no_installation_or_runtime(tmp_path):
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
