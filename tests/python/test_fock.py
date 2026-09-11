"""Public independent Fock semantics, native failure atomicity, and XC use."""

import ctypes as ct
import os
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator, FockScfResult
from vibeqc.fock import (
    FockBuildSpec,
    FockPlan,
    FockTerm,
    _descriptor,
    _pointer,
    _Result,
    _ScfControls,
    _ScfResult,
)
from vibeqc.mean_field import FixedDensityMeanField
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.xc import FixedDensityXC, functional
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture

ATOMS = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
DEVICE = os.environ.get("VIBEQC_TEST_FOCK_DEVICE", "cpu")


def term(choice, coefficient):
    return FockTerm(
        choice != "absent",
        coefficient,
        approximation="exact" if choice == "absent" else choice,
    )


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize("j", ["absent", "exact", "density_fitted"])
@pytest.mark.parametrize("k", ["absent", "exact", "density_fitted"])
def test_independent_fixed_density_energy_variation_and_response(spin, j, k):
    spec = FockBuildSpec(spin=spin, coulomb=term(j, -0.7), exchange=term(k, 0.23))
    d = np.array([[0.8, 0.1], [0.1, 0.6]])
    direction = np.array([[0.17, -0.13], [-0.13, 0.11]])
    if spin == "unrestricted":
        d = np.stack((d, 0.4 * d))
        direction = np.stack((direction, -0.3 * direction))
    with NativeAO(ATOMS) as basis, FockPlan(basis, spec, device=DEVICE) as plan:
        result = plan.evaluate(d, derivative=True)
        assert (result.coulomb is None) == (j == "absent")
        assert (result.exchange is None) == (k == "absent")
        np.testing.assert_allclose(
            result.gradient[:3] + result.gradient[3:], 0, atol=2e-10
        )
        step = 1e-5
        plus = plan.evaluate(d + step * direction)
        minus = plan.evaluate(d - step * direction)
        np.testing.assert_allclose(
            (plus.energy - minus.energy) / (2 * step),
            np.sum(result.fock * direction),
            atol=2e-9,
        )
        diag = plan.diagnostics
        assert diag["backend"] == DEVICE and diag["precision"] == "float64"
        assert diag["requested"] == spec.to_dict()
        assert len(diag["native_source_identity"]) == 64
        if j == "absent":
            assert diag["resolved"]["coulomb"]["coefficient"] == 0
        if DEVICE == "cuda":
            assert diag["source_schedule"] == "cuda_independent"
        assert plus.identity != minus.identity
        with pytest.raises(ValueError):
            result.fock.setflags(write=True)
        copied = result.diagnostics
        copied["resolved"]["coulomb"]["coefficient"] = 93.0
        assert result.diagnostics == diag
        # The native source owns its scientific data independently of NativeAO.
        basis.close()
        np.testing.assert_allclose(plan.evaluate(d).fock, result.fock, atol=1e-11)


def test_native_failure_publication_and_preflight():
    with NativeAO(ATOMS) as basis, FockPlan(basis, device=DEVICE) as plan:
        d = np.eye(2)
        shared = np.full((2, 2), 79.0)
        out = _descriptor(
            _Result,
            matrix_count=4,
            coulomb=_pointer(shared),
            exchange_alpha=_pointer(shared),
            energy_one_electron=79.0,
        )
        status = plan._library.vibeqc_fock_plan_evaluate(
            plan._handle, _pointer(d), 4, None, 0, ct.byref(out)
        )
        assert status == 1
        assert (
            "overlap"
            in plan._library.vibeqc_fock_plan_last_error(plan._handle).decode()
        )
        assert np.all(shared == 79) and out.energy_one_electron == 79
        with pytest.raises(ValueError, match=r"shape|finiteness"):
            plan.evaluate(np.full((2, 2), np.nan))
        before = plan.evaluate(d)
        with pytest.raises(ValueError, match=r"short|range"):
            FockPlan(
                basis,
                replace(
                    plan.spec, exchange=FockTerm(operator="short_range", omega=0.4)
                ),
                device=DEVICE,
            )
        np.testing.assert_array_equal(plan.evaluate(d).fock, before.fock)


def test_identity_and_zero_coefficients():
    with NativeAO(ATOMS) as basis:
        zero = FockBuildSpec(
            coulomb=FockTerm(coefficient=0.0), exchange=FockTerm(coefficient=0.0)
        )
        absent = replace(zero, exchange=FockTerm(False, 0.0))
        with (
            FockPlan(basis, zero, device=DEVICE) as z,
            FockPlan(basis, absent, device=DEVICE) as a,
        ):
            rz, ra = z.evaluate(np.eye(2)), a.evaluate(np.eye(2))
            assert (
                rz.exchange is not None
                and ra.exchange is None
                and z.identity != a.identity
            )
            assert rz.energy_two_electron == ra.energy_two_electron == 0
            np.testing.assert_allclose(rz.fock, ra.fock, atol=1e-12)
        with (
            FockPlan(basis, replace(zero, derivative_order=0), device=DEVICE) as values,
            pytest.raises(ValueError, match=r"gradient|capability"),
        ):
            values.evaluate(np.eye(2), derivative=True)
    with pytest.raises(RuntimeError, match="closed"):
        values.evaluate(np.eye(2))


@pytest.mark.parametrize("approximation", ["exact", "density_fitted"])
@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
def test_semilocal_consumer_uses_common_j_and_independent_xc_fixture(
    approximation, spin, name
):
    meta, data, grid = load_integration_fixture("h2")
    separate = spin == "unrestricted"
    layout = "spin" if separate else "total"
    density = data[f"density_{layout}"]
    xc = FixedDensityXC(
        functional(name, spin="polarized" if separate else "unpolarized")
    )
    spec = FockBuildSpec(
        spin, 0, FockTerm(approximation=approximation), FockTerm(False, 0.0)
    )
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        FockPlan(basis, spec, device=DEVICE) as plan,
    ):
        consumer = FixedDensityMeanField(plan, xc)
        result = consumer.integrate(grid, density, tile_points=7)
        native = plan.evaluate(density)
        prefix = f"{name}_{layout}"
        np.testing.assert_allclose(
            result.energy - native.energy, data[f"{prefix}_energy"][0], atol=2e-11
        )
        np.testing.assert_allclose(
            result.fock - native.fock, data[f"{prefix}_potential"], atol=2e-11
        )
        assert plan.diagnostics["resolved"]["exchange"]["present"] is False
        direction = np.ones_like(density) * 0.01
        step = 1e-5
        plus = consumer.integrate(grid, density + step * direction)
        minus = consumer.integrate(grid, density - step * direction)
        np.testing.assert_allclose(
            (plus.energy - minus.energy) / (2 * step),
            np.sum(result.fock * direction),
            atol=2e-8,
        )
        assert result.fock_identity == native.identity
        with (
            FockPlan(basis, FockBuildSpec.hf(spin), device=DEVICE) as hf,
            pytest.raises(ValueError, match="absent exchange"),
        ):
            FixedDensityMeanField(hf, xc)


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize("j", ["exact", "density_fitted"])
@pytest.mark.parametrize("k", ["exact", "density_fitted"])
def test_public_scf_force_variation_replay_and_legacy_equivalence(spin, j, k):
    state = {"charge": 1, "multiplicity": 2} if spin == "unrestricted" else {}
    spec = FockBuildSpec.hf(spin, coulomb=j, exchange=k)
    controls = {"energy_tolerance": 1e-12, "density_tolerance": 1e-10}
    with (
        NativeAO(ATOMS, **state) as basis,
        FockPlan(basis, spec, device=DEVICE) as plan,
    ):
        execution = plan.execution_identity
        # Start with values to exercise derivative-capability superset reuse.
        first = plan.solve(compute_forces=False, **controls)
        assert isinstance(first, FockScfResult) and first.forces is None
        full = plan.solve(initial_density=first.density, **controls)
        assert full.initial_density_used and not first.initial_density_used
        assert full.fock_builds >= full.iterations
        assert full.diagnostics == plan.diagnostics
        assert full.energy == pytest.approx(first.energy, abs=1e-10)
        assert plan.evaluate(full.density).energy == pytest.approx(
            full.energy, abs=1e-10
        )
        np.testing.assert_allclose(full.forces.sum(axis=0), 0, atol=2e-9)
        assert plan.execution_identity == execution and full.identity != first.identity
        with pytest.raises(ValueError):
            full.density.setflags(write=True)
        if j == k:
            legacy = Calculator(
                device=DEVICE,
                method="uhf" if state else "rhf",
                density_fitting=DEVICE if j == "density_fitted" else "none",
                **controls,
            ).singlepoint(ATOMS, **state)
            assert full.energy == pytest.approx(legacy.energy, abs=2e-9)
            np.testing.assert_allclose(full.forces, legacy.forces, atol=2e-8)
        energies = []
        step = 1e-4
        for shift in (-step, step):
            moved = [ATOMS[0], ("H", (0.0, 0.0, 0.7 + shift))]
            with (
                NativeAO(moved, **state) as displaced,
                FockPlan(
                    displaced, replace(spec, derivative_order=0), device=DEVICE
                ) as target,
            ):
                assert target.identity != plan.identity
                energies.append(target.solve(compute_forces=False, **controls).energy)
        assert -(energies[1] - energies[0]) / (2 * step) == pytest.approx(
            full.forces[1, 2], abs=2e-7
        )


def test_scf_failures_do_not_publish_or_poison_sources():
    with NativeAO(ATOMS) as basis, FockPlan(basis, device=DEVICE) as plan:
        reference = plan.solve()
        density = np.full((2, 2), 73.0)
        forces = np.full((2, 3), 73.0)
        out = _descriptor(
            _ScfResult,
            density=_pointer(density),
            density_count=4,
            forces=_pointer(forces),
            force_count=6,
            energy=73.0,
        )
        controls = _descriptor(
            _ScfControls,
            max_iterations=1,
            diis_history=8,
            energy_tolerance=1e-10,
            density_tolerance=1e-8,
        )
        status = plan._library.vibeqc_fock_plan_solve(
            plan._handle, ct.byref(controls), None, 0, ct.byref(out)
        )
        assert status == 4 and out.energy == 73.0
        assert np.all(density == 73) and np.all(forces == 73)
        with pytest.raises(RuntimeError, match="converge"):
            plan.solve(max_iterations=1)
        with pytest.raises(ValueError, match=r"trace|occupation|density|electron"):
            plan.solve(initial_density=0.4 * reference.density)
        with pytest.raises(ValueError, match="positive"):
            plan.solve(energy_tolerance=float("nan"))
        replay = plan.solve(initial_density=reference.density)
        assert replay.energy == pytest.approx(reference.energy, abs=1e-11)
        with FockPlan(
            basis, FockBuildSpec.hf(derivative_order=0), device=DEVICE
        ) as values:
            with pytest.raises(ValueError, match=r"force|capability"):
                values.solve()
            assert values.solve(compute_forces=False).energy == pytest.approx(
                reference.energy
            )


def test_canonical_semantics_and_execution_identity():
    with NativeAO(ATOMS) as basis:
        empty = FockBuildSpec(coulomb=FockTerm(False), exchange=FockTerm(False))
        alternate = replace(
            empty, coulomb=FockTerm(False, -0.3, approximation="density_fitted")
        )
        with (
            FockPlan(basis, empty, device=DEVICE) as a,
            FockPlan(
                basis,
                alternate,
                device=DEVICE,
                metric_relative_threshold=1e-7,
            ) as b,
        ):
            assert a.identity == b.identity
            assert a.diagnostics["requested"] != b.diagnostics["requested"]
            assert a.diagnostics["resolved"] == b.diagnostics["resolved"]
        if DEVICE == "cuda":
            with (
                FockPlan(basis, device="cpu") as cpu,
                FockPlan(basis, device="cuda") as gpu,
            ):
                assert cpu.identity == gpu.identity
                assert cpu.execution_identity != gpu.execution_identity


@pytest.mark.parametrize("scalar", [np.float32, np.float64])
@pytest.mark.parametrize("approximation", ["exact", "density_fitted"])
def test_identity_uses_normalized_native_controls(scalar, approximation):
    """Accepted scalar inputs retain the identity of their native double values."""
    screening, cutoff = scalar(1e-12), scalar(1e-10)
    spec = FockBuildSpec.hf(coulomb=approximation, exchange=approximation)
    with (
        NativeAO(ATOMS) as basis,
        FockPlan(
            basis,
            spec,
            device=DEVICE,
            device_id=np.int64(0),
            screening_tolerance=screening,
            metric_relative_threshold=cutoff,
        ) as normalized,
        FockPlan(
            basis,
            spec,
            device=DEVICE,
            device_id=0,
            screening_tolerance=float(screening),
            metric_relative_threshold=float(cutoff),
        ) as plain,
    ):
        assert normalized.identity == plain.identity
        assert normalized.execution_identity == plain.execution_identity
        assert normalized.diagnostics == plain.diagnostics
        if DEVICE == "cuda":
            assert type(normalized.diagnostics["device_id"]) is int
        np.testing.assert_array_equal(
            normalized.evaluate(np.eye(2)).fock, plain.evaluate(np.eye(2)).fock
        )


@pytest.mark.skipif(DEVICE != "cuda", reason="CUDA execution-variant diagnostics")
def test_one_electron_execution_variant_identity_is_frozen(monkeypatch):
    with NativeAO(ATOMS) as basis:
        monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUE_MAPPING", "thread")
        with FockPlan(basis, device="cuda") as original:
            before = original.diagnostics
            assert before["one_electron_value_backend"] == "cuda-generated"
            monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUE_MAPPING", "shell_warp")
            with FockPlan(basis, device="cuda") as generated:
                assert original.identity == generated.identity
                assert original.execution_identity != generated.execution_identity
                assert original.diagnostics == before
                assert (
                    generated.diagnostics["one_electron_value_backend"]
                    == "cuda-generated"
                )
                assert (
                    generated.diagnostics["one_electron_value_mapping"] == "shell-warp"
                )
                np.testing.assert_allclose(
                    original.evaluate(np.eye(2)).fock,
                    generated.evaluate(np.eye(2)).fock,
                    atol=2e-11,
                )


@pytest.mark.skipif(DEVICE != "cuda", reason="retired CUDA value selector diagnostics")
def test_retired_value_controls_cannot_restore_handwritten_dispatch(monkeypatch):
    spec = FockBuildSpec(
        coulomb=term("density_fitted", 1.0), exchange=term("density_fitted", -0.5)
    )
    monkeypatch.delenv("VIBEQC_ONE_ELECTRON_VALUES", raising=False)
    monkeypatch.delenv("VIBEQC_DF_VALUES", raising=False)
    monkeypatch.delenv("VIBEQC_ONE_ELECTRON_VALUE_MAPPING", raising=False)
    with NativeAO(ATOMS) as basis, FockPlan(basis, spec, device="cuda") as original:
        before = original.diagnostics
        monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUES", "reference")
        monkeypatch.setenv("VIBEQC_DF_VALUES", "reference")
        with FockPlan(basis, spec, device="cuda") as replay:
            assert replay.execution_identity == original.execution_identity
            assert replay.diagnostics["one_electron_value_backend"] == "cuda-generated"
            assert replay.diagnostics["one_electron_value_mapping"] == "shell-warp"
            assert replay.diagnostics["df_value_backend"] == "generated_rys"
            assert original.diagnostics == before
            np.testing.assert_array_equal(
                replay.evaluate(np.eye(2)).fock, original.evaluate(np.eye(2)).fock
            )
