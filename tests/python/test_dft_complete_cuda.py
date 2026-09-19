"""Opt-in real-device qualification of all seven stationary CUDA RKS sources.

Run through tools/run_stationary_cuda_validation.py. PySCF is an independent
oracle only in this test; its entrypoints are never called by the diagnostic.
"""

import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_DFT_CUDA_TEST") != "1", reason="explicit Slurm CUDA gate"
)


@pytest.fixture(scope="module")
def compiler():
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require a Slurm allocation"
    return CudaCompilerAdapter(
        Path(os.environ["CUDACXX"]), cuda_target_info("sm_120"), compile_timeout=600
    )


@contextmanager
def no_cpu_derivatives():
    """Fail on the old scientific consumers while allowing native state proofs."""
    from vibeqc._ks_snapshot import NativeKsSnapshot
    from vibeqc._stationary_cpu import _PrimitiveExecutor
    from vibeqc_compiler.common import array_graph
    from vibeqc_compiler.dft.ao import NativeAO
    from vibeqc_compiler.tensor import interpreter
    from vibeqc_compiler.tensor.cpu import NativeTensorProgram
    from vibeqc_compiler.xc.contractions import ContractionProgram
    from vibeqc_compiler.xc.grid_native import NativeGridContraction

    def forbidden(*args, **kwargs):
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


def _calculator(method):
    from test_dft_complete_cpu import GRID
    from vibeqc import Calculator, KsOptions

    return Calculator(
        method=method,
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )


def _diagnostic(state, basis, compiler, **kwargs):
    from vibeqc._stationary_cuda import complete_rks_cuda_gradient_diagnostic

    with no_cpu_derivatives():
        return complete_rks_cuda_gradient_diagnostic(
            state,
            basis,
            compiler=compiler,
            cache=os.environ.get("VIBEQC_STATIONARY_CACHE", ".cache/stationary-cuda"),
            **kwargs,
        )


def _evidence(name, payload):
    directory = os.environ.get("VIBEQC_STATIONARY_EVIDENCE")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )


@pytest.mark.parametrize("molecule", ["h2", "water"])
@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks"])
def test_complete_cuda_independent_analytic(method, molecule, compiler):
    from test_dft_complete_cpu import ATOMS, independent_gradient
    from vibeqc._dft_gradient import StationaryKsState
    from vibeqc_compiler.common.provenance import file_hash
    from vibeqc_compiler.dft import NativeAO

    atoms = (
        ATOMS
        if molecule == "water"
        else [("H", (0.1, -0.2, -0.7)), ("H", (0.2, 0.1, 0.8))]
    )
    calc = _calculator(method)
    started = perf_counter()
    with calc.prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
        energy = batch.execute(strict=True).items[0].energy
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
            "native_library": os.environ["VIBEQC_LIBRARY"],
            "native_sha256": file_hash(Path(os.environ["VIBEQC_LIBRARY"])),
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
        assert result.work["xc_points"] == len(state.grid.points)
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
def test_cuda_reconverged_finite_differences_and_replay(method, compiler):
    from test_dft_complete_cpu import ATOMS
    from vibeqc._dft_gradient import StationaryKsState
    from vibeqc_compiler.dft import NativeAO

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
        if os.environ.get("VIBEQC_STATIONARY_FULL_FD") == "1":
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
            _diagnostic(current, basis, compiler, max_primitive_records=1)
        with pytest.raises(ValueError, match="budget"):
            _diagnostic(current, basis, compiler, max_device_bytes=1)
        with pytest.raises(ValueError, match="current native.*snapshot"):
            _diagnostic(replace(current, _source=None), basis, compiler)


def test_cuda_source_failure_zero_tail_and_recovery(compiler):
    """Exercise actual source kernels, late failure and a fresh transaction."""

    from vibeqc._stationary_cuda import _checked, _CudaSources, _layout, _ptr
    from vibeqc_compiler.dft import NativeAO
    from vibeqc_compiler.dft.cuda import CudaGrid
    from vibeqc_compiler.dft.cuda import compile_cuda as compile_grid
    from vibeqc_compiler.integral.first_derivative_native import (
        emit_first_derivative_cuda,
    )
    from vibeqc_compiler.method.stationary_cuda import compile_stationary_cuda

    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (1.0, 0.0, 0.0)), ("H", (2.0, 0.0, 0.0))]
    cache = Path(os.environ["VIBEQC_STATIONARY_CACHE"])
    with NativeAO(atoms, multiplicity=2) as basis:
        _, _, _, requests = _layout(basis)
        artifact = compile_stationary_cuda(
            emit_first_derivative_cuda(requests),
            pbe=False,
            iterations=3,
            compiler=compiler,
            cache=cache,
        )
        corrupt = replace(
            artifact, metadata={**artifact.metadata, "binary_sha256": "0" * 64}
        )
        with pytest.raises(ValueError, match="hash mismatch"):
            _CudaSources(basis, corrupt, compiler, 0, 5, 7, 1 << 20)
        with pytest.raises(RuntimeError, match="budget"):
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
            sources.reset(1e-12)
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
            out = np.full((7, 3, 3), 42.0)
            with pytest.raises(RuntimeError, match="reset"):
                sources._call("stationary_finish", sources.handle, _ptr(out), out.size)
            np.testing.assert_array_equal(out, 42.0)
            sources.reset(1e-12)
            with ao.xc_task(points, np.arange(3), "LDA_XC_PW") as task:
                sources.geometry(task, owners, np.ones(3), np.ones(3), pbe=False)
            for k, v in sources.finish().items():
                np.testing.assert_array_equal(v, zero[k])
            # A late invalid primitive also poisons the transaction. No result
            # is copied, and reset clears previous successful accumulation.
            records = np.ones((2, 26))
            records[-1, -1] = np.nan
            maps = np.zeros((2, 4), dtype=np.int64)
            with pytest.raises(RuntimeError, match="invalid stationary CUDA"):
                sources._call(
                    "stationary_records",
                    sources.handle,
                    0,
                    0,
                    _ptr(records),
                    _ptr(maps),
                    2,
                )
            with pytest.raises(RuntimeError, match="reset"):
                sources.finish()
            sources.reset(1e-12)
            assert all(np.all(v == 0) for v in sources.finish().values())
            _evidence("source-failure-zero-tail-recovery", sources.metrics())


def test_cuda_late_owner_replay_and_geometry_replacement(compiler, monkeypatch):
    from vibeqc._dft_gradient import StationaryKsState
    from vibeqc._stationary_cuda import _CudaSources
    from vibeqc_compiler.dft import NativeAO

    atoms = [("H", (0.1, 0.2, -0.7)), ("H", (-0.2, 0.1, 0.8))]
    calc = _calculator("pbe-rks")
    with calc.prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
        batch.execute(strict=True)
        state = StationaryKsState.from_native(batch, basis)
        finish = _CudaSources.finish

        def replay(sources):
            result = finish(sources)
            batch.execute(strict=True)
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
