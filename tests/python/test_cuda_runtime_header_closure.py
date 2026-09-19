"""Installed JIT headers must be complete and independently includable."""

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parents[2]


def test_installed_tensor_runtime_has_complete_local_header_closure():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assets = config["tool"]["scikit-build"]["wheel"]["force-include"]
    pending = [ROOT / "src/tensor/cuda_runtime.cuh"]
    seen = set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        relative = path.relative_to(ROOT).as_posix()
        assert relative in assets, f"missing installed JIT dependency: {relative}"
        for dependency in re.findall(
            r'^\s*#include\s+"([^"]+)"', path.read_text(), re.MULTILINE
        ):
            child = path.parent / dependency
            assert child.is_file(), (
                f"header requires an undeclared include root: {dependency}"
            )
            pending.append(child)


def test_transitive_resources_participate_in_all_tensor_consumer_identities():
    # These consumers directly include the shared Tensor CUDA runtime. Their
    # source/header inventories must invalidate artifacts when ownership changes.
    modules = (
        "dft/ao_cuda.py",
        "tensor/cuda_execute.py",
        "xc/cuda_emit.py",
        "integral/second_derivatives_execute.py",
        "integral/weighted_eri_execute.py",
        "integral/first_directional_execute.py",
        "method/stationary_cuda.py",
    )
    for module in modules:
        text = (ROOT / "python/vibeqc_compiler" / module).read_text()
        for header in (
            "bounded_workspace.hpp",
            "cuda_resources.cuh",
            "resource_cuda.cuh",
            "resource_ledger.hpp",
        ):
            assert f'"src/runtime/{header}"' in text, (module, header)
