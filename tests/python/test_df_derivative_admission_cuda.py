"""Qualified derivative lowerings follow capabilities, not benchmark dimensions."""

import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.fixture(scope="module")
def production_profile():
    """Query the allocated device, without adding a CuPy dependency."""
    assert os.environ.get("SLURM_JOB_ID")
    runtime = ctypes.CDLL("libcudart.so")
    runtime.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    device = ctypes.c_int()
    assert runtime.cudaGetDevice(ctypes.byref(device)) == 0
    capability = []
    # cudaDevAttrComputeCapabilityMajor and cudaDevAttrComputeCapabilityMinor.
    for attribute in (75, 76):
        value = ctypes.c_int()
        assert (
            runtime.cudaDeviceGetAttribute(ctypes.byref(value), attribute, device.value)
            == 0
        )
        capability.append(value.value)
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "python/vibeqc_compiler/integral/production_df_derivatives.json"
        ).read_text()
    )
    major, minor = capability
    return manifest["architectures"].get(f"sm_{major}{minor}", {})


@pytest.mark.parametrize("buckets", ["off", "on", "packet"])
@pytest.mark.parametrize("model", ["equal192", "practical96", "oh"])
def test_qualified_lowering_without_ao_shape_admission(
    model, buckets, production_profile, monkeypatch, tmp_path
):
    """Exercise panel/group/packet consumers against independent full forces.

    The practical basis includes auxiliary f shells, which must retain the
    manifest's unqualified-class fallback. OH is an open-shell, non-water
    control. The explicit shell consumer isolates lowering admission from
    the separately qualified automatic consumer/packet selection policy.
    """
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    case = benchmark_cases()[
        {
            "equal192": "water-octamer-s4-def2-svp-spherical",
            "practical96": "water-tetramer-def2-svp-spherical",
            "oh": "oh-def2-svp-spherical-uhf",
        }[model]
    ]
    orbital = auxiliary = cpu_orbital = cpu_auxiliary = "def2-svp"
    if model == "practical96":
        identity = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/results/issue206-practical-auxiliary/identity"
        )
        orbital, cpu_orbital = load_comparison_basis(
            identity / "cc-pvdz.json", case, role="orbital", compute_forces=True
        )
        auxiliary, cpu_auxiliary = load_comparison_basis(
            identity / "cc-pvdz-jkfit.json",
            case,
            role="auxiliary",
            compute_forces=True,
        )
    mol = gto.M(
        atom=case.atoms,
        basis=cpu_orbital,
        unit="Bohr",
        spin=case.multiplicity - 1,
        charge=case.charge,
        verbose=0,
    )
    reference = (scf.UHF if case.method == "uhf" else scf.RHF)(mol).density_fit(
        auxbasis=cpu_auxiliary
    )
    reference.conv_tol, reference.conv_tol_grad = 1e-12, 1e-10
    reference.max_cycle = 100
    reference.kernel()
    assert reference.converged
    gradient = reference.nuc_grad_method()
    gradient.auxbasis_response = True
    expected = -gradient.kernel()
    for name, value in {
        "WEIGHTED_EXECUTION": "shell",
        "SHELL_SCHEDULE": "compact",
        "PRIMITIVE_BUCKETS": buckets,
        "DERIVATIVE_PAIRS": "symmetric",
        "FORCE_SCREEN_ABS": "off",
    }.items():
        monkeypatch.setenv("VIBEQC_DF_" + name, value)
    calc = Calculator(
        method=case.method,
        basis=orbital,
        auxiliary_basis=auxiliary,
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    with calc.prepare_batch(
        [case.atoms], charges=[case.charge], multiplicities=[case.multiplicity]
    ) as batch:
        for index, policy in enumerate(("auto", "legacy", "auto")):
            trace = tmp_path / f"{index}-{policy}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_SHELL_POLICY", policy)
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            actual = batch.execute(strict=True).items[0]
            assert abs(actual.energy - reference.e_tot) < 1e-9
            np.testing.assert_allclose(actual.forces, expected, atol=1e-8, rtol=0)
            (response,) = [
                row for row in read_trace(trace) if row["operation"] == "force_response"
            ]
            counters = response["counters"]
            assert counters["three_center_shell_panels"] > 0
            choices = {
                row["class"]: row["lowering"]
                for row in production_profile.get("kernels", [])
            }
            for key, value in counters.items():
                if key.endswith("_rys_selected"):
                    expected_rys = (
                        policy == "auto"
                        and production_profile.get("qualified", False)
                        and choices.get(key.split("_")[1]) == "rys"
                    )
                    assert value == int(expected_rys), key
            if model == "practical96":
                assert counters["shell_003_rys_selected"] == 0
