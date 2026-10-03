"""Opt-in real-device qualification of all seven stationary CUDA RKS sources.

Run through tools/run_stationary_cuda_validation.py. PySCF is an independent
oracle only in this test; its entrypoints are never called by the diagnostic.
"""

import json
import os
import typing
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="explicit Slurm CUDA gate",
)


@pytest.fixture(autouse=True)
def cooperative_becke_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select C inside full-force oracles without changing production defaults.

    Explicit matrix choices still win. Fail qualification if an eligible owner
    silently takes a generic resource fallback on the allocated NVIDIA device.
    """
    if os.environ.get("GENERATIVEQC_TEST_COOPERATIVE_BECKE") != "1":
        return
    from generativeqc._stationary_cuda import _CudaSources

    original = _CudaSources.__init__

    def initialize(self: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> None:
        requested = kwargs.setdefault("cooperative_becke", True)
        original(self, *args, **kwargs)
        if requested and 1 < self.natom <= 128:
            assert self.metrics()["becke_threads_per_point"] == 32, (
                "cooperative qualification selected the generic device fallback"
            )

    monkeypatch.setattr(_CudaSources, "__init__", initialize)


@pytest.fixture(scope="module")
def compiler() -> typing.Any:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require a Slurm allocation"
    return CudaCompilerAdapter(
        Path(os.environ["CUDACXX"]), cuda_target_info("sm_120"), compile_timeout=600
    )


@contextmanager
def no_cpu_derivatives() -> typing.Any:
    """Fail on the old scientific consumers while allowing native state proofs."""
    from generativeqc._ks_snapshot import NativeKsSnapshot
    from generativeqc._stationary_cpu import _PrimitiveExecutor
    from generativeqc_compiler.common import array_graph
    from generativeqc_compiler.dft.ao import NativeAO
    from generativeqc_compiler.tensor import interpreter
    from generativeqc_compiler.tensor.cpu import NativeTensorProgram
    from generativeqc_compiler.xc.contractions import ContractionProgram
    from generativeqc_compiler.xc.grid_native import NativeGridContraction

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise AssertionError("CPU scientific fallback entered during CUDA diagnostic")

    with pytest.MonkeyPatch.context() as patch:
        for cls, name in (
            (NativeKsSnapshot, "evaluate_xc_points"),
            (NativeAO, "evaluate"),
            (_PrimitiveExecutor, "_run"),
            (NativeTensorProgram, "execute"),
            (NativeGridContraction, "contract"),
            (ContractionProgram, "features"),
            (ContractionProgram, "geometry_from_cartesian_coefficients"),
            (interpreter, "execute"),
            (array_graph, "evaluate_array_graph"),
        ):
            patch.setattr(cls, name, forbidden)
        yield


def _calculator(method: typing.Any) -> typing.Any:
    from generativeqc import Calculator, KsOptions
    from test_dft_complete_cpu import GRID

    return Calculator(
        method=method,
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )


def _production_calculator(method: typing.Any) -> typing.Any:
    from generativeqc import Calculator, KsOptions

    return Calculator(
        method=method,
        device="cuda",
        ks_options=KsOptions(),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )


def _diagnostic(
    state: typing.Any, basis: typing.Any, compiler: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    from generativeqc._stationary_cuda import complete_rks_cuda_gradient_diagnostic

    with no_cpu_derivatives():
        return complete_rks_cuda_gradient_diagnostic(
            state,
            basis,
            compiler=compiler,
            cache=os.environ.get(
                "GENERATIVEQC_STATIONARY_CACHE", ".cache/stationary-cuda"
            ),
            **kwargs,
        )


def _evidence(name: typing.Any, payload: typing.Any) -> None:
    directory = os.environ.get("GENERATIVEQC_STATIONARY_EVIDENCE")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )


@pytest.mark.parametrize("molecule", ["h2", "water"])
@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks"])
def test_complete_cuda_independent_analytic(
    method: typing.Any, molecule: typing.Any, compiler: typing.Any
) -> None:
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.common.provenance import file_hash
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import ATOMS, independent_gradient

    atoms = (
        ATOMS
        if molecule == "water"
        else [("H", (0.1, -0.2, -0.7)), ("H", (0.2, 0.1, 0.8))]
    )
    calc = _calculator(method)
    started = perf_counter()
    with calc.prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
        # This gate executes the diagnostic explicitly below. Request only SCF
        # here so it neither runs a duplicate force nor requires packaged AOT.
        energy = batch.execute(properties=("energy",), strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        assert state._source.metadata[0] == 3
        assert state._source.grid_spec is not None
        assert np.all(state._source.atomic_weights > 0)
        # Nondivisible capacities exercise both primitive and grid tails.
        result = _diagnostic(
            state,
            basis,
            compiler,
            tile_points=137,
            primitive_tile=29,
            integral_terms=17,
        )
        # Stop endpoint timing before the independent CPU reference. Otherwise
        # the timing would silently include PySCF validation rather than only
        # CUDA SCF, explicit export and the complete diagnostic call.
        endpoint_seconds = perf_counter() - started
        ref_energy, ref_gradient, refs = independent_gradient(basis, state, method)
        source_error = {
            key: float(np.max(np.abs(value - refs[key])))
            for key, value in result.components.items()
        }
        payload = {
            "energy": energy,
            "reference_energy": ref_energy,
            "energy_error": abs(energy - ref_energy),
            "gradient": result.gradient.tolist(),
            "reference_gradient": ref_gradient.tolist(),
            "analytic_max_error": float(np.max(np.abs(result.gradient - ref_gradient))),
            "source_max_errors": source_error,
            "work": dict(result.work),
            "slurm_job": os.environ["SLURM_JOB_ID"],
            "native_library": os.environ["GENERATIVEQC_LIBRARY"],
            "native_sha256": file_hash(Path(os.environ["GENERATIVEQC_LIBRARY"])),
            "complete_scf_export_gradient_seconds": endpoint_seconds,
        }
        _evidence(f"{molecule}-{method}", payload)
        print(
            f"{molecule} {method}: energy error {payload['energy_error']:.3g}; gradient error {payload['analytic_max_error']:.3g}",
            flush=True,
        )
        assert abs(energy - ref_energy) < 2e-9
        assert max(source_error.values()) < 1e-7
        np.testing.assert_allclose(result.gradient, ref_gradient, atol=1e-7, rtol=0)
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=2e-10, rtol=0)
        assert result.work["launches"] > 0
        assert result.work["tensor_executions"] == 0
        assert (
            result.work["full_range_derivative_route"] == "prepared-native-stationary"
        )
        assert result.work["full_range_ao_task_domain_elided"] is True
        assert all(
            item["source"] not in ("coulomb", "exact_exchange")
            for item in result.work["stationary_task_executor"]["sources"]
        )
        assert (
            result.work["stationary_final_reduction"]
            == "native-plan-source-device-sum-plus-prepared-integrals-v1"
        )
        assert result.work["stationary_weight_tensor_executions"] == 0
        assert result.work["stationary_weight_roundtrip_bytes"] == 0
        assert result.work["stationary_state_dw_upload_bytes"] == 0
        assert (
            result.work["stationary_integral_derivative_route"]
            == "prepared-native-complete"
        )
        assert result.work["grid_density_source"] == "exact-final-scf-device-binding"
        assert result.work["grid_density_h2d_bytes"] == 0
        assert result.work["grid_owner_source"] == "implicit-atom-major-index"
        assert result.work["grid_owner_h2d_bytes"] == 0
        assert result.work["grid_point_source"] == "exact-native-resident-grid"
        assert result.work["grid_point_h2d_bytes"] == 0
        assert result.work["grid_weight_source"] == "exact-native-resident-grid"
        assert result.work["grid_weight_h2d_bytes"] == 0
        assert result.work["grid_atomic_measure_source"] == "exact-native-resident-grid"
        assert result.work["grid_atomic_measure_h2d_bytes"] == 0
        assert result.work["xc_points"] == len(state.grid.points)
        assert result.work["geometry_lane_capacity"] > 32
        assert (
            result.work["geometry_peak_lanes"] == result.work["geometry_lane_capacity"]
        )
        assert (
            result.work["geometry_scratch_bytes"]
            == 144 * basis.natom * result.work["geometry_lane_capacity"]
        )
        assert (
            result.work["additional_device_peak_bound"]
            <= result.work["additional_device_budget"]
        )
        if molecule == "water":
            for key in result.components:
                assert np.max(np.abs(result.components[key])) > 1e-4
                assert (
                    np.max(
                        np.abs(result.gradient - result.components[key] - ref_gradient)
                    )
                    > 1e-4
                )
        assert result.execution.startswith("cuda-seven-source/")


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks"])
def test_production_grid_cuda_energy_and_force(
    method: typing.Any, compiler: typing.Any
) -> None:
    """Production v2 default is qualified on the real-device water endpoint."""
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import ATOMS, independent_gradient

    calc = _production_calculator(method)
    with calc.prepare_batch([ATOMS]) as batch, NativeAO(ATOMS) as basis:
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        assert state._source.grid_spec.version == 2
        assert state._source.grid_provenance["policy_version"] == 2
        result = _diagnostic(
            state,
            basis,
            compiler,
            tile_points=137,
            primitive_tile=29,
            integral_terms=17,
        )
        ref_energy, ref_gradient, _ = independent_gradient(basis, state, method)
        assert energy == pytest.approx(ref_energy, abs=2e-9)
        np.testing.assert_allclose(result.gradient, ref_gradient, atol=1e-7, rtol=0)
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=2e-10, rtol=0)


@pytest.mark.parametrize("method", ["lda-uks", "pbe-uks"])
def test_complete_cuda_open_shell_uks_independent_analytic(
    method: typing.Any, compiler: typing.Any
) -> None:
    """B3 real-device closure: both spin channels share the C1 seven-source plan."""
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import ATOMS, independent_uks_gradient

    charge, multiplicity = 1, 2
    calc = _calculator(method)
    with (
        calc.prepare_batch(
            [ATOMS], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(ATOMS, charge=charge, multiplicity=multiplicity) as basis,
    ):
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        assert state.density.shape[0] == 2
        assert not np.allclose(state.density[0], state.density[1], atol=1e-12, rtol=0)
        result = _diagnostic(
            state,
            basis,
            compiler,
            tile_points=137,
            primitive_tile=29,
            integral_terms=17,
        )
        reference_energy, reference = independent_uks_gradient(basis, state, method)
        assert abs(energy - reference_energy) < 2e-9
        np.testing.assert_allclose(result.gradient, reference, atol=1e-7, rtol=0)
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=3e-10, rtol=0)
        assert result.work["xc_points"] == len(state.grid.points)
        assert result.work["geometry_lane_capacity"] > 32
        assert (
            result.work["geometry_peak_lanes"] == result.work["geometry_lane_capacity"]
        )
        assert (
            result.work["geometry_scratch_bytes"]
            == 144 * basis.natom * result.work["geometry_lane_capacity"]
        )
        assert result.work["tensor_executions"] == 0
        assert (
            result.work["stationary_final_reduction"]
            == "native-plan-source-device-sum-plus-prepared-integrals-v1"
        )
        assert result.work["stationary_weight_tensor_executions"] == 0
        assert result.work["stationary_weight_roundtrip_bytes"] == 0
        assert result.work["stationary_state_dw_upload_bytes"] == 0
        assert (
            result.work["stationary_integral_derivative_route"]
            == "prepared-native-complete"
        )
        assert result.work["grid_density_source"] == "exact-final-scf-device-binding"
        assert result.work["grid_density_h2d_bytes"] == 0
        assert (
            result.work["additional_device_peak_bound"]
            <= result.work["additional_device_budget"]
        )

        # Replay revokes both spin blocks atomically; a fresh UKS state reproduces
        # the same force without borrowing the old owner/generation.
        batch.execute(strict=True)
        with pytest.raises(ValueError, match="stale"):
            _diagnostic(state, basis, compiler)
        current = StationaryKsState.from_native(batch, basis)
        replay = _diagnostic(current, basis, compiler)
        np.testing.assert_allclose(replay.gradient, result.gradient, atol=1e-9, rtol=0)
        _evidence(
            f"water-{method}-b3",
            {
                "energy_error": abs(energy - reference_energy),
                "analytic_max_error": float(
                    np.max(np.abs(result.gradient - reference))
                ),
                "work": dict(result.work),
                "slurm_job": os.environ["SLURM_JOB_ID"],
            },
        )


@pytest.mark.parametrize(
    ("method", "charge", "multiplicity"),
    [("r2scan-rks", 0, 1), ("r2scan-uks", 1, 2)],
)
def test_complete_cuda_r2scan_independent_analytic(
    method: typing.Any,
    charge: typing.Any,
    multiplicity: typing.Any,
    compiler: typing.Any,
) -> None:
    """Qualify the generated tau geometry path against independent PySCF/libxc."""
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import (
        ATOMS,
        independent_semilocal_total_gradient,
    )

    pytest.importorskip(
        "pyscf", reason="independent r2SCAN analytic reference requires PySCF"
    )
    calc = _calculator(method)
    with (
        calc.prepare_batch(
            [ATOMS], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(ATOMS, charge=charge, multiplicity=multiplicity) as basis,
    ):
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        assert state.identity.method == method
        result = _diagnostic(
            state,
            basis,
            compiler,
            tile_points=137,
            primitive_tile=29,
            integral_terms=17,
        )
        reference_energy, reference = independent_semilocal_total_gradient(
            basis, state, method
        )
        energy_error = abs(energy - reference_energy)
        gradient_error = float(np.max(np.abs(result.gradient - reference)))
        _evidence(
            f"water-{method}-tau",
            {
                "energy_error": energy_error,
                "analytic_max_error": gradient_error,
                "work": dict(result.work),
                "slurm_job": os.environ["SLURM_JOB_ID"],
            },
        )
        assert energy_error < 2e-8
        np.testing.assert_allclose(result.gradient, reference, atol=2e-6, rtol=0)
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=2e-9, rtol=0)
        assert result.work["xc_points"] == len(state.grid.points)
        assert result.work["geometry_lane_capacity"] > 32
        assert (
            result.work["geometry_peak_lanes"] == result.work["geometry_lane_capacity"]
        )
        assert (
            result.work["geometry_scratch_bytes"]
            == 144 * basis.natom * result.work["geometry_lane_capacity"]
        )
        assert result.execution.startswith("cuda-seven-source/")


def test_cuda_r2scan_reconverged_directional_finite_difference(
    compiler: typing.Any,
) -> None:
    """Detect a missing or duplicated vtau/2 contribution after full SCF relaxation."""
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import ATOMS

    calc = _calculator("r2scan-rks")
    xyz = np.asarray([position for _, position in ATOMS], dtype=np.float64)
    direction = np.array(
        [[0.13, -0.07, 0.11], [-0.05, 0.17, 0.03], [0.09, 0.02, -0.14]]
    )
    with calc.prepare_batch([ATOMS]) as batch, NativeAO(ATOMS) as basis:
        batch.execute(strict=True)
        state = StationaryKsState.from_native(batch, basis)
        result = _diagnostic(state, basis, compiler)
        estimates = []
        for step in (3e-4, 1e-4):
            values = []
            for sign in (1, -1):
                moved = [
                    (atom[0], position)
                    for atom, position in zip(
                        ATOMS, xyz + sign * step * direction, strict=True
                    )
                ]
                values.append(calc.singlepoint(moved, properties=("energy",)).energy)
            estimates.append((values[0] - values[1]) / (2 * step))
        actual = float(np.sum(result.gradient * direction))
        _evidence(
            "fd-r2scan-rks-tau",
            {"steps": [3e-4, 1e-4], "estimates": estimates, "analytic": actual},
        )
        assert abs(estimates[-1] - estimates[-2]) < 2e-6
        assert abs(estimates[-1] - actual) < 2e-6


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks"])
def test_cuda_reconverged_finite_differences_and_replay(
    method: typing.Any, compiler: typing.Any
) -> None:
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import ATOMS

    calc = _calculator(method)
    xyz = np.array([a[1] for a in ATOMS])
    direction = np.array(
        [[0.13, -0.07, 0.11], [-0.05, 0.17, 0.03], [0.09, 0.02, -0.14]]
    )
    with calc.prepare_batch([ATOMS]) as batch, NativeAO(ATOMS) as basis:
        batch.execute(strict=True)
        state = StationaryKsState.from_native(batch, basis)
        result = _diagnostic(state, basis, compiler)
        estimates = []
        for step in (1e-3, 3e-4, 1e-4):
            energies = []
            for sign in (1, -1):
                moved = [
                    (a[0], r) for a, r in zip(ATOMS, xyz + sign * step * direction)
                ]
                energies.append(calc.singlepoint(moved).energy)
            estimates.append((energies[0] - energies[1]) / (2 * step))
        actual = float(np.sum(result.gradient * direction))
        _evidence(
            f"fd-{method}",
            {"steps": [1e-3, 3e-4, 1e-4], "estimates": estimates, "analytic": actual},
        )
        assert abs(estimates[-1] - actual) < 1e-7
        assert abs(estimates[-1] - estimates[-2]) < 1e-7
        if os.environ.get("GENERATIVEQC_STATIONARY_FULL_FD") == "1":
            coordinate_estimates = []
            for step in (1e-3, 3e-4, 1e-4):
                fd = np.empty_like(xyz)
                for a, axis in np.ndindex(xyz.shape):
                    displacement = np.zeros_like(xyz)
                    displacement[a, axis] = step
                    values = [
                        calc.singlepoint(
                            [
                                (atom[0], r)
                                for atom, r in zip(ATOMS, xyz + sign * displacement)
                            ]
                        ).energy
                        for sign in (1, -1)
                    ]
                    fd[a, axis] = (values[0] - values[1]) / (2 * step)
                coordinate_estimates.append(fd)
            extrapolated = (9 * coordinate_estimates[-1] - coordinate_estimates[-2]) / 8
            errors = [
                float(np.max(np.abs(fd - result.gradient)))
                for fd in coordinate_estimates
            ]
            extrapolated_error = float(np.max(np.abs(extrapolated - result.gradient)))
            _evidence(
                f"full-fd-{method}",
                {
                    "steps": [1e-3, 3e-4, 1e-4],
                    "max_errors": errors,
                    "richardson_max_error": extrapolated_error,
                    "finite_differences": [fd.tolist() for fd in coordinate_estimates],
                    "gradient": result.gradient.tolist(),
                },
            )
            np.testing.assert_allclose(
                coordinate_estimates[-1], coordinate_estimates[-2], atol=1e-6, rtol=0
            )
            assert errors[-1] < 1e-6
            assert extrapolated_error < 1e-7
        batch.execute(strict=True)
        with pytest.raises(ValueError, match="stale"):
            _diagnostic(state, basis, compiler)
        current = StationaryKsState.from_native(batch, basis)
        replay = _diagnostic(current, basis, compiler)
        np.testing.assert_allclose(replay.gradient, result.gradient, atol=1e-9, rtol=0)
        with pytest.raises(ValueError, match="work budget"):
            _diagnostic(current, basis, compiler, max_grid_pair_visits=1)
        with pytest.raises(ValueError, match="budget"):
            _diagnostic(current, basis, compiler, max_device_bytes=1)
        with pytest.raises(ValueError, match="current native.*snapshot"):
            _diagnostic(replace(current, _source=None), basis, compiler)


def test_cuda_source_failure_zero_tail_and_recovery(compiler: typing.Any) -> None:
    """Exercise actual source kernels, late failure and a fresh transaction."""

    from generativeqc._stationary_cuda import _checked, _CudaSources, _layout, _ptr
    from generativeqc_compiler.dft import NativeAO
    from generativeqc_compiler.dft.cuda import CudaGrid
    from generativeqc_compiler.dft.cuda import compile_cuda as compile_grid
    from generativeqc_compiler.integral.first_derivative_native import (
        emit_first_derivative_cuda,
    )
    from generativeqc_compiler.method import resolve_method
    from generativeqc_compiler.method.stationary_cuda import compile_stationary_cuda
    from generativeqc_compiler.method.stationary_gradient import (
        SCF_POINT_MODEL,
        StationaryGradientPlan,
        StationaryMeanField,
    )

    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (1.0, 0.0, 0.0)), ("H", (2.0, 0.0, 0.0))]
    cache = Path(os.environ["GENERATIVEQC_STATIONARY_CACHE"])
    with NativeAO(atoms, multiplicity=2) as basis:
        _, _, _, requests = _layout(basis)
        plan = StationaryGradientPlan(
            resolve_method("LDA_XC_PW", spin="unpolarized"),
            StationaryMeanField(SCF_POINT_MODEL),
        )
        artifact = compile_stationary_cuda(
            emit_first_derivative_cuda(requests),
            pbe=False,
            plan=plan,
            iterations=3,
            compiler=compiler,
            cache=cache,
        )
        corrupt = replace(
            artifact, metadata={**artifact.metadata, "binary_sha256": "0" * 64}
        )
        with pytest.raises(ValueError, match="hash mismatch"):
            _CudaSources(basis, corrupt, compiler, 0, 5, 7, 1 << 20)
        with pytest.raises(ValueError, match="budget"):
            _CudaSources(basis, artifact, compiler, 0, 5, 7, 1)
        with pytest.raises(RuntimeError):
            _CudaSources(basis, artifact, compiler, 1, 5, 7, 1 << 20)
        for value, shape, dtype in (
            (np.ones(2, dtype=np.float32), (2,), np.float64),
            (np.ones(2), (3,), np.float64),
            (np.ones(2, dtype=np.int32), (2,), np.int64),
        ):
            with pytest.raises(ValueError, match="requires finite"):
                _checked(value, shape, dtype)
        ga = compile_grid(compiler, cache)
        with (
            _CudaSources(basis, artifact, compiler, 0, 5, 7, 1 << 20) as sources,
            CudaGrid(
                basis,
                ga,
                order=1,
                tile_points=5,
                device_id=0,
                active_ao_capacity=3,
                ingredients=("rho",),
            ) as ao,
        ):
            with pytest.raises(RuntimeError, match="reset"):
                sources.finish()
            density = np.eye(3)[None, :, :]
            weighted_density = np.zeros_like(density)
            sources.reset(1e-12, density, weighted_density)
            ao.set_density(np.eye(3))
            # Single exact-zero factor, saturated products, vacuum tail and an
            # empty tile. Points deliberately avoid center collisions.
            points = np.array([[-1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
            owners = np.array([1, 2, 0], dtype=np.int64)
            with ao.xc_task(points, np.arange(3), "LDA_XC_PW") as task:
                sources.geometry(task, owners, np.ones(3), np.ones(3), pbe=False)
            zero = sources.finish()
            np.testing.assert_array_equal(zero["xc_weight"], 0)
            with ao.xc_task(np.empty((0, 3)), np.arange(3), "LDA_XC_PW") as task:
                sources.geometry(
                    task,
                    np.empty(0, dtype=np.int64),
                    np.empty(0),
                    np.empty(0),
                    pbe=False,
                )
            for k, v in sources.finish().items():
                np.testing.assert_array_equal(v, zero[k])
            # Invalid owner appears after valid points in a real device tile.
            bad = owners.copy()
            bad[-1] = 3
            with (
                ao.xc_task(points, np.arange(3), "LDA_XC_PW") as task,
                pytest.raises(RuntimeError, match="invalid stationary CUDA"),
            ):
                sources.geometry(task, bad, np.ones(3), np.ones(3), pbe=False)
                # Drain the deferred failure before releasing its borrowed grid lease.
                sources.finish()
            out = np.full((7, 3, 3), 42.0)
            with pytest.raises(RuntimeError, match="reset"):
                sources._call("stationary_finish", sources.handle, _ptr(out), out.size)
            np.testing.assert_array_equal(out, 42.0)
            sources.reset(1e-12, density, weighted_density)
            with ao.xc_task(points, np.arange(3), "LDA_XC_PW") as task:
                sources.geometry(task, owners, np.ones(3), np.ones(3), pbe=False)
            for k, v in sources.finish().items():
                np.testing.assert_array_equal(v, zero[k])
            # A late invalid task charge also poisons the transaction. No result
            # is copied, and reset clears previous successful accumulation.
            tasks = np.full((2, 9), -1, dtype=np.int64)
            kind = sources.kinds[
                "kinetic", (sources.components[0], sources.components[0])
            ]
            tasks[:, :4] = kind, 0, 2, -1
            tasks[:, 4:6] = 0
            tasks[:, 8] = int(sources.aos[0, 2]) ** 2
            charges = np.ones(2)
            charges[-1] = np.nan
            with pytest.raises(RuntimeError, match="invalid stationary CUDA"):
                sources._call(
                    "stationary_tasks",
                    sources.handle,
                    _ptr(tasks),
                    _ptr(charges),
                    2,
                )
            with pytest.raises(RuntimeError, match="reset"):
                sources.finish()
            sources.reset(1e-12, density, weighted_density)
            assert all(np.all(v == 0) for v in sources.finish().values())
            _evidence("source-failure-zero-tail-recovery", sources.metrics())


def test_cuda_late_owner_replay_and_geometry_replacement(
    compiler: typing.Any, monkeypatch: typing.Any
) -> None:
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc._stationary_cuda import _CudaSources
    from generativeqc_compiler.dft import NativeAO

    atoms = [("H", (0.1, 0.2, -0.7)), ("H", (-0.2, 0.1, 0.8))]
    calc = _calculator("pbe-rks")
    with calc.prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
        batch.execute(strict=True)
        state = StationaryKsState.from_native(batch, basis)
        finish = _CudaSources.finish

        def replay(sources: typing.Any) -> typing.Any:
            result = finish(sources)
            batch.execute(properties=("energy",), strict=True)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(_CudaSources, "finish", replay)
            with pytest.raises(ValueError, match="stale"):
                _diagnostic(state, basis, compiler)
        current = StationaryKsState.from_native(batch, basis)
        result = _diagnostic(current, basis, compiler)
        shifted = [(name, np.array(xyz) + [0.2, -0.1, 0.3]) for name, xyz in atoms]
        with (
            NativeAO(shifted) as other_basis,
            pytest.raises(ValueError, match="basis/geometry mismatch"),
        ):
            _diagnostic(current, other_basis, compiler)
        with (
            calc.prepare_batch([shifted]) as other_batch,
            NativeAO(shifted) as other_basis,
        ):
            other_batch.execute(strict=True)
            moved = StationaryKsState.from_native(other_batch, other_basis)
            value = _diagnostic(moved, other_basis, compiler)
            np.testing.assert_allclose(
                value.gradient, result.gradient, atol=1e-9, rtol=0
            )


@pytest.mark.parametrize(
    ("method", "charge", "multiplicity"),
    [
        ("lda-rks", 0, 1),
        ("pbe-rks", 0, 1),
        ("r2scan-rks", 0, 1),
        ("lda-uks", 1, 2),
        ("pbe-uks", 1, 2),
        ("r2scan-uks", 1, 2),
    ],
)
def test_public_cuda_calculator_forces_match_independent_gradient(
    method: typing.Any, charge: typing.Any, multiplicity: typing.Any
) -> None:
    """C2: public Calculator publishes force=-gradient from the shared CUDA plan."""
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import (
        ATOMS,
        independent_gradient,
        independent_semilocal_total_gradient,
        independent_uks_gradient,
    )

    calc = _calculator(method)
    assert calc._capabilities.supported_properties == frozenset(("energy", "forces"))
    public = calc.singlepoint(ATOMS, charge=charge, multiplicity=multiplicity)
    assert public.executed_backend == "cuda"
    assert public.forces is not None
    assert np.isfinite(public.forces).all()

    with (
        calc.prepare_batch(
            [ATOMS], charges=[charge], multiplicities=[multiplicity], warm_start=False
        ) as batch,
        NativeAO(ATOMS, charge=charge, multiplicity=multiplicity) as basis,
    ):
        energy = batch.execute(strict=True, properties=("energy",)).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        if method.startswith("r2scan-"):
            ref_energy, gradient = independent_semilocal_total_gradient(
                basis, state, method
            )
        elif method.endswith("uks"):
            ref_energy, gradient = independent_uks_gradient(basis, state, method)
        else:
            ref_energy, gradient, _ = independent_gradient(basis, state, method)
        state._source.close()
    assert public.energy == pytest.approx(ref_energy, abs=2e-9)
    assert energy == pytest.approx(ref_energy, abs=2e-9)
    np.testing.assert_allclose(public.forces, -gradient, atol=1e-7, rtol=0)


def test_public_cuda_prepared_force_replay_retains_execution(
    monkeypatch: typing.Any,
) -> None:
    """#663: warm and moved force replays reuse every generated CUDA owner."""
    import generativeqc._stationary_cuda as stationary
    from test_dft_complete_cpu import ATOMS

    calc = _calculator("pbe-rks")
    xyz = np.asarray([position for _, position in ATOMS], dtype=np.float64)
    moved = xyz.copy()
    moved[1, 0] += 2.0e-3
    with calc.prepare_batch([ATOMS], warm_start=True) as batch:
        first = batch.execute(strict=True, properties=("energy", "forces"))
        owner = batch._stationary_cuda_execution
        assert owner is not None
        identity = owner.identity
        resident = (
            id(owner.sources),
            id(owner.grid),
            tuple((name, id(value)) for name, value in sorted(owner.tensors.items())),
        )
        # The shared lease owns replay, geometry refresh and failure state.
        assert owner._lease.executions == 1
        assert not first.items[0].forces is None

        def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
            raise AssertionError("warm force replay rebuilt generated CUDA execution")

        with monkeypatch.context() as patch:
            for name in (
                "emit_first_derivative_cuda",
                "compile_stationary_cuda",
                "compile_grid",
                "compile_cuda",
                "_CudaSources",
                "CudaGrid",
                "PreparedCuda",
            ):
                patch.setattr(stationary, name, forbidden)

            second = batch.execute(strict=True, properties=("energy", "forces"))
            np.testing.assert_allclose(
                second.items[0].forces, first.items[0].forces, atol=1e-9, rtol=0
            )
            assert owner.identity == identity
            assert resident == (
                id(owner.sources),
                id(owner.grid),
                tuple(
                    (name, id(value)) for name, value in sorted(owner.tensors.items())
                ),
            )

            changed = batch.execute(
                coordinates=(moved,), strict=True, properties=("energy", "forces")
            )
            assert not np.array_equal(changed.items[0].forces, first.items[0].forces)
            assert owner._lease.refreshes == 1
            assert owner._lease.executions == 3

            finish = owner.sources.finish
            calls = 0

            def fail_once() -> typing.Any:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("injected prepared force failure")
                return finish()

            patch.setattr(owner.sources, "finish", fail_once)
            failed = batch.execute(
                coordinates=(moved,), properties=("energy", "forces")
            )
            assert not failed.items[0].succeeded
            assert owner._lease.failed
            recovered = batch.execute(
                coordinates=(moved,), strict=True, properties=("energy", "forces")
            )
            assert recovered.items[0].succeeded
            np.testing.assert_allclose(
                recovered.items[0].forces, changed.items[0].forces, atol=1e-9, rtol=0
            )
            assert owner._lease.refreshes == 2
            assert not owner._lease.failed


def test_public_cuda_grid_xc_schedules_preserve_complete_endpoint() -> None:
    """DFT09: both executable XC schedules preserve the public E+F endpoint."""
    from generativeqc import Calculator, KsOptions

    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    results = []
    for schedule in ("device_fused", "host_unfused"):
        result = Calculator(
            method="pbe-rks",
            basis="sto-3g",
            device="cuda",
            ks_options=KsOptions(xc_schedule=schedule),
            max_iterations=200,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
        ).singlepoint(atoms, properties=("energy", "forces"))
        assert result.executed_backend == "cuda"
        assert result.converged
        assert result.forces is not None
        results.append(result)

    fused, unfused = results
    assert fused.iterations == unfused.iterations
    assert fused.energy == pytest.approx(unfused.energy, abs=1e-9)
    np.testing.assert_allclose(fused.forces, unfused.forces, atol=1e-7, rtol=0)
    np.testing.assert_allclose(fused.forces.sum(axis=0), 0, atol=1e-7, rtol=0)
    np.testing.assert_allclose(unfused.forces.sum(axis=0), 0, atol=1e-7, rtol=0)


def test_profiled_xc_schedule_reaches_direct_and_resource_aware_batch_paths(
    monkeypatch: typing.Any,
) -> None:
    """DFT09: one resolved profile schedule must survive every public KS path."""
    from generativeqc import Calculator, KsOptions, ResourceBudget
    from generativeqc.ks import ProfiledKsSelection, resolve_ks_options

    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    resolved = resolve_ks_options(
        "pbe-rks", KsOptions(xc_schedule="host_unfused", tile_points=31)
    )
    calculator = Calculator(
        method="pbe-rks",
        basis="sto-3g",
        device="cuda",
        resource_budget=ResourceBudget(),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )

    def selected(
        systems: typing.Any,
        *,
        charges: typing.Any = None,
        multiplicities: typing.Any = None,
    ) -> typing.Any:
        assert len(systems) == 1
        assert tuple(charges) == (0,)
        assert tuple(multiplicities) == (1,)
        exact = systems[0][1].position[0] == pytest.approx(0.0)
        return ProfiledKsSelection(resolved, exact_profile_match=exact)

    monkeypatch.setattr(calculator, "_effective_ks_selection", selected)

    direct = calculator.singlepoint(atoms, properties=("energy",))
    assert direct.converged and direct.executed_backend == "cuda"
    assert direct.ks_diagnostic.tile_points == 31

    with calculator.prepare_batch([atoms], warm_start=True) as batch:
        first = batch.execute(strict=True, properties=("energy",)).items[0]
        second = batch.execute(strict=True, properties=("energy",)).items[0]
        assert first.converged and second.converged
        assert first.executed_backend == second.executed_backend == "cuda"
        assert first.ks_diagnostic.tile_points == 31
        assert second.ks_diagnostic.tile_points == 31
        moved = np.asarray([atom[1] for atom in atoms], dtype=np.float64)
        moved[1, 0] += 2.0e-3
        with pytest.raises(RuntimeError, match="not qualified for replay coordinates"):
            batch.execute(
                coordinates=(moved,),
                strict=True,
                properties=("energy",),
            )


@pytest.mark.parametrize(
    ("invalid_index", "invalid_kind"),
    ((0, "wrong-size"), (1, "wrong-size"), (0, "nan"), (1, "nan")),
)
def test_profiled_xc_schedule_revalidates_valid_neighbor_when_peer_is_invalid(
    monkeypatch: typing.Any,
    invalid_index: int,
    invalid_kind: str,
) -> None:
    """DFT09: one invalid row cannot suppress profile requalification of a valid peer."""
    from generativeqc import Calculator, KsOptions, ResourceBudget
    from generativeqc.ks import ProfiledKsSelection, resolve_ks_options

    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    resolved = resolve_ks_options(
        "pbe-rks", KsOptions(xc_schedule="host_unfused", tile_points=31)
    )
    calculator = Calculator(
        method="pbe-rks",
        basis="sto-3g",
        device="cuda",
        resource_budget=ResourceBudget(),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )

    def selected(
        systems: typing.Any,
        *,
        charges: typing.Any = None,
        multiplicities: typing.Any = None,
    ) -> typing.Any:
        assert len(systems) == 2
        assert tuple(charges) == (0, 0)
        assert tuple(multiplicities) == (1, 1)
        exact = all(system[1].position[0] == pytest.approx(0.0) for system in systems)
        return ProfiledKsSelection(resolved, exact_profile_match=exact)

    monkeypatch.setattr(calculator, "_effective_ks_selection", selected)

    moved = np.asarray([atom[1] for atom in atoms], dtype=np.float64)
    moved[1, 0] += 2.0e-3
    if invalid_kind == "wrong-size":
        invalid = np.asarray([0.0, 1.0], dtype=np.float64)
    else:
        invalid = np.asarray([atom[1] for atom in atoms], dtype=np.float64)
        invalid[0, 0] = np.nan
    coordinates: list[np.ndarray] = [moved.copy(), moved.copy()]
    coordinates[invalid_index] = invalid

    with (
        calculator.prepare_batch([atoms, atoms], warm_start=True) as batch,
        pytest.raises(RuntimeError, match="not qualified for replay coordinates"),
    ):
        batch.execute(
            coordinates=coordinates,
            properties=("energy",),
        )


def test_public_cuda_batch_changed_geometry_and_failure_isolation() -> None:
    """C2: rebuilt owners get fresh forces and a bad neighbor cannot poison them."""
    from test_dft_complete_cpu import ATOMS

    calc = _calculator("pbe-rks")
    xyz = np.asarray([position for _, position in ATOMS], dtype=np.float64)
    moved = xyz.copy()
    moved[1, 0] += 2.0e-3
    with calc.prepare_batch([ATOMS, ATOMS], warm_start=True) as batch:
        first = batch.execute(
            coordinates=(None, moved), strict=True, properties=("energy", "forces")
        )
        assert all(item.forces is not None for item in first.items)
        assert not np.array_equal(first.items[0].forces, first.items[1].forces)
        owner = batch._stationary_cuda_execution
        assert owner is not None
        assert owner._lease.executions == 2

        malformed = np.asarray([0.0, 1.0])
        isolated = batch.execute(
            coordinates=(malformed, moved), properties=("energy", "forces")
        )
        assert not isolated.items[0].succeeded
        assert isolated.items[0].forces is None
        assert isolated.items[1].succeeded
        assert isolated.items[1].forces is not None
        assert np.isfinite(isolated.items[1].forces).all()
        assert batch._stationary_cuda_execution is owner
        assert owner._lease.executions == 3


def test_cuda_ks_resource_plan_accounts_for_public_force_staging() -> None:
    from test_dft_complete_cpu import ATOMS

    calc = _calculator("pbe-rks")
    plan = calc.estimate_resources([ATOMS])
    request = next(r for r in plan.requests if r.name == "ks")
    assert request.identity.observables == ("energy", "forces")
    names = {
        estimate.name
        for candidate in request.candidates
        for estimate in candidate.estimates
    }
    assert "serialized generated KS force device staging cap" in names
    assert "serialized generated KS force host staging cap" in names
    assert "serialized KS snapshot grid cache cap" in names


@pytest.mark.parametrize(
    "method,functional", [("LDA_XC_PW", 0), ("PBE", 1), ("R2SCAN", 2), ("PBE0", 1)]
)
@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
@pytest.mark.parametrize("atom_count", [2, 12, 33])
def test_cuda_geometry_lane_budget_tail_and_changed_geometry_replay(
    compiler: typing.Any,
    method: str,
    functional: int,
    spin: str,
    atom_count: int,
    tmp_path: Path,
    active_layout: str = "identity",
) -> None:
    """Compare actual multi-block point work with the bounded 32-lane fallback.

    This is a schedule equivalence gate. The complete endpoint tests above retain
    independent analytic/finite-difference force oracles and unchanged tolerances.
    """
    from contextlib import ExitStack

    from generativeqc._stationary_cuda import _CudaSources
    from generativeqc_compiler.dft import NativeAO
    from generativeqc_compiler.dft.cuda import CudaGrid
    from generativeqc_compiler.dft.cuda import compile_cuda as compile_grid
    from generativeqc_compiler.method import resolve_method
    from generativeqc_compiler.method.stationary_cuda import (
        compile_stationary_cuda,
        stationary_runtime_sources,
    )
    from generativeqc_compiler.method.stationary_gradient import (
        SCF_POINT_MODEL,
        StationaryGradientPlan,
        StationaryMeanField,
    )
    from generativeqc_compiler.method.stationary_resources import (
        stationary_cuda_allocation_bytes,
    )

    atoms = [
        ("H", (0.7 * i, 0.4 * np.sin(i), 0.3 * np.cos(i))) for i in range(atom_count)
    ]
    changed = [
        (symbol, tuple(np.array(xyz) + [0.02 * np.sin(i + 1), -0.01, 0.01 * np.cos(i)]))
        for i, (symbol, xyz) in enumerate(atoms)
    ]
    pairs = atom_count * (atom_count - 1) // 2
    plan = StationaryGradientPlan(
        resolve_method(method, spin=spin), StationaryMeanField(SCF_POINT_MODEL)
    )
    # Geometry must never invoke the integral derivative provider in this gate.
    primitive = """#include <cuda_runtime.h>
__device__ bool first_derivative(unsigned, const double*, const double*, double*) { return false; }
"""
    artifact = compile_stationary_cuda(
        primitive,
        functional=functional,
        plan=plan,
        iterations=3,
        compiler=compiler,
        cache=tmp_path,
    )
    point_capacity = 67 if atom_count > 33 else 2083
    rng = np.random.default_rng(1479)
    relative_points = rng.normal(size=(point_capacity, 3)) * 0.4
    owners = np.arange(point_capacity, dtype=np.int64) % atom_count
    weights = np.full(point_capacity, 0.013)
    raw = np.full(point_capacity, 0.019)
    with ExitStack() as stack:
        basis = stack.enter_context(NativeAO(atoms))
        moved_basis = stack.enter_context(NativeAO(changed))
        active_ids = np.arange(basis.nao)
        if active_layout == "reversed":
            active_ids = np.ascontiguousarray(active_ids[::-1])
        elif active_layout == "subset":
            active_ids = np.ascontiguousarray(active_ids[1::2])
        elif active_layout == "empty":
            active_ids = active_ids[:0]
        else:
            assert active_layout == "identity"
        density = np.repeat(
            np.eye(basis.nao)[None, :, :] * 0.2, plan.spin_blocks, axis=0
        )
        # Sources borrow this stream; destroy them before their grid owner.
        grid = stack.enter_context(
            CudaGrid(
                basis,
                compile_grid(compiler, tmp_path),
                order=2,
                tile_points=point_capacity,
                active_ao_capacity=basis.nao,
                ingredients=("rho", "gradient", "tau"),
            )
        )
        sources = []
        lane_bytes = 144 * basis.natom
        maximum_lanes = min(point_capacity, 2048, (8 << 20) // lane_bytes)
        for lanes, cached, cooperative in (
            (lanes, cached, cooperative)
            for lanes in (
                (1, 17, maximum_lanes) if atom_count > 33 else (32, 256, maximum_lanes)
            )
            for cached in (False, True)
            for cooperative in (False, True)
        ):
            budget = stationary_cuda_allocation_bytes(
                atoms=basis.natom,
                aos=basis.nao,
                primitives=basis.nprimitive,
                points=point_capacity,
                tasks=1,
                spins=plan.spin_blocks,
                sources=len(stationary_runtime_sources(plan)),
                geometry_lanes=lanes,
                cache_center_geometry=cached,
            )
            owner = stack.enter_context(
                _CudaSources(
                    basis,
                    artifact,
                    compiler,
                    0,
                    point_capacity,
                    1,
                    budget,
                    spin_blocks=plan.spin_blocks,
                    source_names=stationary_runtime_sources(plan),
                    cooperative_becke=cooperative,
                )
            )
            # The planner spends spare bytes on lanes before retaining centers.
            # Only the capped panel guarantees a cached route for larger systems.
            center_bytes = 48 * pairs if cached else 0
            selected_lanes = min(maximum_lanes, lanes + center_bytes // lane_bytes)
            retained_centers = center_bytes if selected_lanes == lanes else 0
            assert owner.metrics()["geometry_lane_capacity"] == selected_lanes
            assert owner.metrics()["owned_device_bytes"] == (
                budget
                - center_bytes
                + (selected_lanes - lanes) * lane_bytes
                + retained_centers
            )
            assert owner.metrics()["center_geometry_bytes"] == retained_centers
            selected = cooperative and atom_count <= 128
            assert owner.metrics()["becke_threads_per_point"] == (32 if selected else 1)
            assert owner.metrics()["becke_shared_bytes"] == (
                16 + 64 * (pairs if atom_count <= 32 else 4 * (2 * atom_count - 5) // 2)
                if selected
                else 0
            )
            with pytest.raises((ValueError, RuntimeError)):
                owner.reset_geometry(-1.0)
            sources.append(owner)
        previous = None
        replays = (
            (basis, basis, moved_basis, basis)
            if atom_count > 33
            else (basis, basis, moved_basis, moved_basis, basis, moved_basis, basis)
        )
        for repeat, current_basis in enumerate(replays):
            centers = np.ascontiguousarray(
                current_basis.packed[: 3 * atom_count].reshape(atom_count, 3)
            )
            grid._rebind_centers(centers)
            grid.set_density(density[0] if plan.spin_blocks == 1 else density)
            points = np.ascontiguousarray(relative_points + centers[owners])
            results = []
            for owner in sources:
                owner.rebind_geometry(current_basis)
                metrics_before = owner.metrics()
                if repeat % 2:
                    owner.reset_geometry(1e-12)
                else:
                    owner.reset(1e-12, density, np.zeros_like(density))
                # Reuse the full panels for an irregular large tile, a 6-point
                # tail, and an empty tile; stale high lanes must not be reduced.
                for begin, end in (
                    (0, point_capacity - 6),
                    (point_capacity - 6, point_capacity),
                    (point_capacity, point_capacity),
                ):
                    with grid.feature_task(
                        points[begin:end],
                        active_ids,
                        ("rho",)
                        if functional == 0
                        else ("rho", "gradient")
                        if functional == 1
                        else ("rho", "gradient", "tau"),
                    ) as task:
                        owner.geometry(
                            task,
                            owners[begin:end],
                            weights[begin:end],
                            raw[begin:end],
                            functional=functional,
                        )
                result = owner.finish()
                metrics_after = owner.metrics()
                cached = metrics_after["center_geometry_bytes"] != 0
                assert metrics_after["center_distance_evaluations"] - metrics_before[
                    "center_distance_evaluations"
                ] == pairs * (1 if cached else 1 + 2 * point_capacity)
                assert metrics_after["center_geometry_preparations"] - metrics_before[
                    "center_geometry_preparations"
                ] == int(cached)
                selected = metrics_after["becke_threads_per_point"] > 1
                assert metrics_after["becke_pair_state_evaluations"] - metrics_before[
                    "becke_pair_state_evaluations"
                ] == pairs * point_capacity * (
                    1 if selected and atom_count <= 32 else 2
                )
                # Preparation is device-only and adds no center table upload.
                expected_upload = 3 * basis.natom * 8 + 3 * point_capacity * 8
                if not repeat % 2:
                    expected_upload += 2 * density.nbytes
                assert (
                    metrics_after["h2d_bytes"] - metrics_before["h2d_bytes"]
                    == expected_upload
                )
                results.append(
                    np.stack(
                        [result[name] for name in ("xc_ao", "xc_grid", "xc_weight")]
                    )
                )
                assert owner.metrics()["geometry_peak_lanes"] == min(
                    owner.resources.geometry_lanes, point_capacity - 6
                )
            for result in results[1:]:
                np.testing.assert_allclose(result, results[0], atol=2e-10, rtol=2e-12)
            if previous is not None and previous[0] == current_basis.identity:
                np.testing.assert_array_equal(results[0], previous[1])
            previous = current_basis.identity, results[0]


@pytest.mark.parametrize("atom_count", [96, 128])
def test_cuda_large_tiled_becke_bounded_geometry_probe(
    compiler: typing.Any, atom_count: int, tmp_path: Path
) -> None:
    """67 points qualify the large schedule without launching a full benchmark."""
    test_cuda_geometry_lane_budget_tail_and_changed_geometry_replay(
        compiler, "PBE0", 1, "unpolarized", atom_count, tmp_path
    )


@pytest.mark.parametrize(
    "active_layout", ["reversed", "subset", "empty", "owner-reversed"]
)
def test_cuda_ordered_ao_owner_ranges(
    compiler: typing.Any, active_layout: str, tmp_path: Path
) -> None:
    """Qualify owner routing and the public refusal of unsorted AO IDs.

    The synthetic reversed-owner case compares native scalar/cooperative
    routing, not physical forces. Independent complete oracles remain separate.
    """
    if active_layout == "reversed":
        with pytest.raises(ValueError, match="active AO IDs must be sorted unique"):
            test_cuda_geometry_lane_budget_tail_and_changed_geometry_replay(
                compiler, "PBE0", 1, "unpolarized", 96, tmp_path, active_layout
            )
        return
    if active_layout == "owner-reversed":
        from generativeqc import _stationary_cuda as runtime

        original_owners = runtime._native_ao_atoms
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                runtime,
                "_native_ao_atoms",
                lambda basis: np.ascontiguousarray(original_owners(basis)[::-1]),
            )
            test_cuda_geometry_lane_budget_tail_and_changed_geometry_replay(
                compiler, "PBE0", 1, "unpolarized", 96, tmp_path
            )
        return
    test_cuda_geometry_lane_budget_tail_and_changed_geometry_replay(
        compiler, "PBE0", 1, "unpolarized", 96, tmp_path, active_layout
    )
