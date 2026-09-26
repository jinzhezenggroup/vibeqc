"""Opt-in RTX CUDA qualification for the complete RCCSD derivative consumer."""

import os
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc.calculator import Atom, Primitive, Shell
from vibeqc.profiles import find_nvcc
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info

from tools.cc_gradient_fixtures import inputs, load, source_arguments
from tools.validate_ccsd_t_gradient import analytic_oracle
from tools.vibeqc_cc import (
    complete_ccsdt_cuda_response_gradient_validation,
    complete_ccsdt_gradient_validation,
)
from tools.vibeqc_cc.complete_gradient import (
    CCSDGradientOptions,
    complete_gradient_validation,
)
from tools.vibeqc_posthf.sources import NativeSource

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_CC_GRADIENT_CUDA_TEST") != "1",
    reason="set VIBEQC_CC_GRADIENT_CUDA_TEST=1 for real-device qualification",
)


def _source(name: typing.Any) -> typing.Any:
    return NativeSource(**source_arguments(inputs(name)))


def _dense_contract(derivatives: typing.Any, weights: typing.Any) -> typing.Any:
    ncoord = derivatives.shape[0]
    return np.einsum(
        "qi,i->q",
        derivatives.reshape(ncoord, -1),
        np.asarray(weights).reshape(-1),
        optimize=False,
    ).reshape(-1, 3)


def test_posthf_cuda_bridges_match_dense_cpu_derivatives_on_water() -> None:
    with _source("h2o") as source:
        dense = source.integral_derivatives(output_budget_bytes=128 << 20)
        rng = np.random.default_rng(153)
        ws = rng.normal(size=(source.nbf, source.nbf))
        wh = rng.normal(size=(source.nbf, source.nbf))
        wg = rng.normal(size=(source.nbf,) * 4)
        one, resources = source.one_electron_gradient_cuda(
            overlap_weights=ws,
            kinetic_weights=wh,
            attraction_weights=wh,
            device_id=0,
            schedule=0,
            stage_budget_bytes=32 << 20,
        )
        eri = source.weighted_eri_gradient_cuda(
            wg, device_id=0, stage_budget_bytes=32 << 20
        )
        np.testing.assert_allclose(
            one,
            _dense_contract(dense["overlap"], ws) + _dense_contract(dense["hcore"], wh),
            atol=2e-10,
            rtol=2e-10,
        )
        np.testing.assert_allclose(
            eri, _dense_contract(dense["eri"], wg), atol=3e-10, rtol=3e-10
        )
        assert resources["device_bytes"] > 0
        assert resources["host_to_device_bytes"] > 0
        assert resources["device_to_host_bytes"] == len(source.atoms) * 3 * 8
        assert resources["synchronous_uploads"] > 0
        assert resources["stream_synchronizations"] > 0


@pytest.mark.parametrize(
    "name",
    ("h2", "h2o", "nh3", "ch4", "h2_d_cartesian", "h2_d_spherical"),
)
def test_complete_cuda_derivative_endpoint_matches_independent_reference(
    name: typing.Any,
) -> None:
    record = load(name)
    with _source(name) as source:
        result = complete_gradient_validation(
            source,
            options=CCSDGradientOptions(
                derivative_backend="cuda",
                device_id=0,
                derivative_stage_budget_bytes=32 << 20,
                one_electron_schedule=0,
            ),
        )
    expected = np.asarray(record["gradient"], dtype=np.float64)
    np.testing.assert_allclose(result.gradient, expected, atol=1e-6, rtol=0)
    assert (
        result.diagnostics["derivative_backend"] == "cuda-generated-bounded-consumers"
    )
    assert result.diagnostics["dense_ao_derivative_oracle"] is False
    assert result.diagnostics["gpu_device_id"] == 0
    assert result.diagnostics["gpu_derivative_stage_budget_bytes"] == 32 << 20
    assert result.diagnostics["gpu_one_electron_device_bytes"] > 0
    assert result.diagnostics["gpu_eri_weight_mode"] == "dense"
    assert result.diagnostics["gpu_one_electron_calls"] == 6
    assert result.diagnostics["gpu_weighted_eri_calls"] == 4


def test_complete_ccsdt_cuda_derivative_endpoint_matches_pinned_pyscf() -> None:
    expected = np.asarray(analytic_oracle("h2o")["analytic"]["gradient"])
    with _source("h2o") as source:
        result = complete_ccsdt_gradient_validation(
            source,
            options=CCSDGradientOptions(
                derivative_backend="cuda",
                device_id=0,
                derivative_stage_budget_bytes=32 << 20,
                one_electron_schedule=0,
            ),
            vir_chunk_size=1,
        )
    np.testing.assert_allclose(result.gradient, expected, atol=1e-6, rtol=0)
    assert result.diagnostics["triples_gradient"] is True
    assert (
        result.diagnostics["derivative_backend"] == "cuda-generated-bounded-consumers"
    )
    assert result.diagnostics["dense_ao_derivative_oracle"] is False
    assert result.diagnostics["gpu_one_electron_calls"] == 10
    assert result.diagnostics["gpu_weighted_eri_calls"] == 8


def test_complete_ccsdt_cuda_response_gradient_matches_pinned_pyscf(
    tmp_path: Path,
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), (
        "CUDA RCCSD(T) response-gradient qualification requires Slurm"
    )
    architecture = os.environ.get("VIBEQC_TENSOR_ARCH", "").strip()
    if not architecture:
        pytest.fail(
            "set VIBEQC_TENSOR_ARCH to the allocated GPU architecture; "
            "CUDA RCCSD(T) qualification must not assume a device target"
        )
    nvcc = find_nvcc()
    assert nvcc is not None
    compiler = CudaCompilerAdapter(nvcc, cuda_target_info(architecture))
    expected = np.asarray(analytic_oracle("h2o")["analytic"]["gradient"])
    with _source("h2o") as source:
        result = complete_ccsdt_cuda_response_gradient_validation(
            source,
            compiler,
            tmp_path / "ccsdt-cuda-response-gradient",
            options=CCSDGradientOptions(
                derivative_backend="cuda",
                device_id=0,
                derivative_stage_budget_bytes=64 << 20,
                one_electron_schedule=0,
            ),
            vir_chunk_size=1,
            tensor_max_bytes=512 << 20,
            triples_max_bytes=512 << 20,
            lambda_host_bytes=1 << 30,
            lambda_device_bytes=2 << 30,
            jk_device_budget_bytes=512 << 20,
            response_device_budget_bytes=1 << 30,
        )

    np.testing.assert_allclose(result.gradient, expected, atol=1e-6, rtol=0)
    assert result.diagnostics["cuda_response_gradient_validation"] is True
    assert result.diagnostics["state_preparation_backend"] == (
        "native-cpu-rhf+native-cpu-rccsd"
    )
    assert result.diagnostics["lambda_backend"] == "cuda-fp64-resident-actions"
    assert (
        result.diagnostics["triples_response_backend"]
        == "cuda-fp64-resident-triples-vjp"
    )
    assert result.diagnostics["parameter_tensor_backend"] == (
        "cuda-fp64-ordinary-stream"
    )
    assert result.diagnostics["response_execution"] == "cuda-resident"
    assert result.diagnostics["resident_response_diagnostics"] is not None
    assert result.diagnostics["cpu_execution_fallback"] is False
    assert result.diagnostics["derivative_backend"] == (
        "cuda-generated-bounded-consumers"
    )


def test_h2_shell_streamed_eri_weights_match_dense_cuda_endpoint() -> None:
    results = []
    for mode in ("dense", "shell"):
        with _source("h2") as source:
            results.append(
                complete_gradient_validation(
                    source,
                    options=CCSDGradientOptions(
                        derivative_backend="cuda",
                        device_id=0,
                        derivative_stage_budget_bytes=16 << 20,
                        eri_weight_mode=mode,
                    ),
                )
            )
    np.testing.assert_allclose(
        results[0].gradient, results[1].gradient, atol=2e-11, rtol=2e-11
    )
    assert results[1].diagnostics["gpu_eri_weight_mode"] == "shell"
    quartets = 2**4
    assert results[1].diagnostics["gpu_shell_quartet_calls"] == 4 * quartets
    assert results[1].diagnostics["gpu_maximum_ao_eri_weight_block_elements"] == 1
    assert results[1].diagnostics["gpu_weighted_eri_calls"] == 4 * quartets


def test_water_two_cuda_stage_budgets_are_numerically_identical() -> None:
    results = []
    for budget in (16 << 20, 64 << 20):
        with _source("h2o") as source:
            results.append(
                complete_gradient_validation(
                    source,
                    options=CCSDGradientOptions(
                        derivative_backend="cuda",
                        device_id=0,
                        derivative_stage_budget_bytes=budget,
                        one_electron_schedule=0,
                    ),
                )
            )
    np.testing.assert_allclose(
        results[0].gradient, results[1].gradient, atol=2e-11, rtol=2e-11
    )
    assert results[0].diagnostics["gpu_derivative_stage_budget_bytes"] == 16 << 20
    assert results[1].diagnostics["gpu_derivative_stage_budget_bytes"] == 64 << 20


def test_sparse_ffff_weighted_eri_cuda_matches_cpu_finite_difference() -> None:
    """Protect the through-f reference fallback from maximum-order stack blowup."""
    base = source_arguments(inputs("h2"))
    basis = list(base["basis"])
    basis.insert(1, Shell(0, 3, (Primitive(0.5, 1.0),)))
    basis.append(Shell(1, 3, (Primitive(0.55, 1.0),)))
    base["basis"] = tuple(basis)

    def source(displacement: typing.Any = 0.0) -> typing.Any:
        arguments = dict(base)
        atoms = list(arguments["atoms"])
        first = atoms[0]
        position = list(first.position)
        position[0] += displacement
        atoms[0] = Atom(first.atomic_number, tuple(position))
        arguments["atoms"] = tuple(atoms)
        return NativeSource(**arguments)

    with source() as current:
        f_shells = [
            index
            for index, shell in enumerate(current.shells)
            if shell.angular_momentum == 3
        ]
        assert len(f_shells) == 2 and current.nbf == 22
        offsets = np.cumsum((0, *current.shell_sizes))
        ao = tuple(int(offsets[index]) for index in f_shells)
        weights = np.zeros((10, 10, 10, 10))
        weights[0, 0, 0, 0] = 1.0
        center_gradient = current.weighted_eri_shell_gradient_cuda(
            (f_shells[0], f_shells[0], f_shells[1], f_shells[1]),
            weights,
            stage_budget_bytes=128 << 20,
        )
        analytic = float(center_gradient[0, 0] + center_gradient[1, 0])
        assert abs(analytic) > 1e-3
        np.testing.assert_allclose(center_gradient.sum(axis=0), 0, atol=2e-14, rtol=0)

    for step in (1e-4, 3e-5, 1e-5):
        values = []
        for sign in (-1, 1):
            with source(sign * step) as displaced:
                values.append(
                    float(
                        displaced._read(
                            "four_center_eri",
                            (ao[0], ao[0], ao[1], ao[1]),
                            (1, 1, 1, 1),
                        ).item()
                    )
                )
        np.testing.assert_allclose(
            (values[1] - values[0]) / (2 * step),
            analytic,
            atol=3e-9,
            rtol=2e-8,
        )


def test_cuda_stage_budget_fails_without_cpu_fallback(
    monkeypatch: typing.Any,
) -> None:
    with _source("h2") as source:
        calls = {"dense": 0}
        original = source.integral_derivatives

        def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            calls["dense"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(source, "integral_derivatives", forbidden)
        with pytest.raises(
            (RuntimeError, ValueError),
            match="budget|record|storage|small|maximum_bytes",
        ):
            complete_gradient_validation(
                source,
                options=CCSDGradientOptions(
                    derivative_backend="cuda",
                    derivative_stage_budget_bytes=1,
                ),
            )
        assert calls["dense"] == 0
