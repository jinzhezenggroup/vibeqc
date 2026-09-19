"""Pure compiler/input contracts for the CUDA directional first consumer."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tomllib
from vibeqc_compiler.integral.first_directional import (
    DirectionalMatrixTerm,
    directional_identity,
    emit_directional_matrix,
)
from vibeqc_compiler.integral.first_directional_execute import (
    DirectionalFirstAccumulator,
    compile_directional_first,
    directional_storage,
)
from vibeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
)
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_slot": -1},
        {"output_slot": 32},
        {"output_slot": True},
        {"output_pair": (0,)},
        {"output_pair": (0, 4)},
        {"output_pair": (True, 1)},
        {"weight_pair": (1, 4)},
        {"coefficient": np.inf},
        {"coefficient": 1j},
    ],
)
def test_invalid_terms_fail_closed(kwargs):
    with pytest.raises((TypeError, ValueError)):
        DirectionalMatrixTerm(**({"output_slot": 0, "output_pair": (0, 1)} | kwargs))


def test_identity_covers_contraction_and_component_semantics():
    ir = build_weighted_eri_ir((1, 0, 0, 0))
    t = DirectionalMatrixTerm(0, (0, 1), (2, 3), 1)
    key = directional_identity(ir, (0,), (t,))
    assert key == directional_identity(ir, (0,), (replace(t, coefficient=1.0),))
    for other in (
        replace(t, output_slot=1),
        replace(t, output_pair=(0, 2)),
        replace(t, weight_pair=(1, 3)),
        replace(t, coefficient=-0.5),
    ):
        assert key != directional_identity(ir, (0,), (other,))
    assert key != directional_identity(ir, (1,), (t,))


def test_reject_missing_shell_slot_and_invalid_component_inventory():
    ir = build_one_electron_derivative_ir("overlap", (0, 0))
    with pytest.raises(ValueError, match="missing shell"):
        directional_identity(ir, (0,), (DirectionalMatrixTerm(0, (0, 1), (2, 3)),))
    for indices in ((), (0, 0), (1,)):
        with pytest.raises(ValueError):
            directional_identity(ir, indices, (DirectionalMatrixTerm(0, (0, 1)),))
    with pytest.raises(ValueError):
        directional_identity(ir, (0,), ())


def test_generated_sources_declare_device_direction_and_external_weight_contraction():
    ir = build_weighted_eri_ir((0, 0, 0, 0))
    terms = (
        DirectionalMatrixTerm(0, (0, 1), (2, 3)),
        DirectionalMatrixTerm(0, (0, 2), (1, 3), -0.5),
    )
    source = emit_directional_matrix(ir, (0,), terms, runtime_identity="a" * 64)
    assert "generated_weighted_eri::first_primitive" in source
    assert "direction[3*mapping.atoms[c]+axis]" in source
    assert "weights[ao[2]*nbf+ao[3]]" in source
    assert "atomicAdd(output" in source
    assert "first_directional_runtime.cuh" in source
    assert source.count("vibeqc_directional_identity_v1()") == 1
    assert "RHF" not in source and "PySCF" not in source


@pytest.mark.parametrize(
    "dims",
    [
        (0, 2, 2, 128),
        (2, 0, 2, 128),
        (2, 2, 33, 128),
        (2, 2, 2, 2**20 + 1),
        (True, 2, 2, 128),
    ],
)
def test_storage_admission_before_device_use(dims):
    with pytest.raises(ValueError):
        directional_storage(*dims)


def test_numeric_storage_counts_host_staging_publication_and_device():
    p = directional_storage(7, 3, 2, 128)
    assert p["output_bytes"] == 2 * 7 * 7 * 8
    assert (
        p["device_bytes"]
        == p["record_bytes"] + p["output_bytes"] + p["input_bytes"] + 8
    )
    assert (
        p["numeric_peak_bytes"]
        == p["device_bytes"]
        + p["record_bytes"]
        + 3 * p["output_bytes"]
        + 2 * p["input_bytes"]
        + 192
    )
    with pytest.raises((ValueError, OverflowError)):
        directional_storage(2**63, 3, 2, 128)
    with pytest.raises(TypeError):
        DirectionalFirstAccumulator(object(), nbf=2, natoms=2)
    with pytest.raises(TypeError, match="explicit CUDA"):
        compile_directional_first(None, None, None, component_indices=(0,), terms=())


def test_runtime_is_an_installed_compiler_asset():
    root = Path(__file__).resolve().parents[2]
    manifest = tomllib.loads((root / "pyproject.toml").read_text())
    mapping = manifest["tool"]["scikit-build"]["wheel"]["force-include"]
    name = "src/integrals/first_directional_runtime.cuh"
    assert mapping[name] == "vibeqc_compiler/assets/" + name
    source = (root / name).read_text()
    assert "vibeqc_tensor::Context" in source
    assert "Program::accumulate" in source
    assert "RHF" not in source and "0.5 *" not in source


def test_generation_does_not_load_native_or_probe_cuda():
    code = r"""
import ctypes
ctypes.CDLL=lambda *a,**k: (_ for _ in ()).throw(AssertionError('native library loaded'))
from vibeqc_compiler.integral.first_directional import DirectionalMatrixTerm, emit_directional_matrix
from vibeqc_compiler.integral.one_electron_derivatives import build_one_electron_derivative_ir
source=emit_directional_matrix(build_one_electron_derivative_ir('overlap',(0,0)),(0,),(DirectionalMatrixTerm(0,(0,1)),),runtime_identity='a'*64)
assert 'vibeqc_directional_append_v1' in source
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_spherical_semantics_are_not_reinterpreted_as_cartesian():
    ir = build_one_electron_derivative_ir("overlap", (0, 0))
    spherical = replace(
        ir.signature,
        shells=tuple(
            replace(shell, convention="real_spherical") for shell in ir.signature.shells
        ),
    )
    with pytest.raises(ValueError, match="Cartesian"):
        directional_identity(
            replace(ir, spec=spherical), (0,), (DirectionalMatrixTerm(0, (0, 1)),)
        )
