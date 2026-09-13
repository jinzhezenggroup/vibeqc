"""Opt-in real-GPU D/C gates; run only inside a finite Slurm allocation."""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.provenance import find_nvcc
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft import DensitySource, NativeAO, density_features
from vibeqc_compiler.dft.cuda import CudaGrid, compile_cuda
from vibeqc_compiler.dft.fixtures import NAMES, basis_arguments, load_fixture
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture
from vibeqc_compiler.xc.native import NativeContractionProgram
from vibeqc_compiler.xc.prepared import PreparedXCContractions

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_GRID_CUDA_TEST") != "1",
    reason="opt-in finite Slurm GPU gate",
)
KEYS = ("rho", "gradient", "sigma", "tau")


@pytest.fixture(scope="module")
def artifact():
    compiler = find_nvcc()
    assert compiler is not None
    return compile_cuda(
        CudaCompilerAdapter(compiler, cuda_target_info("sm_120")),
        Path(".artifacts/density-cuda-cache"),
    )


def check(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)


def factors(basis, counts=(11, 3), generation=0):
    """Nonorthogonal fractional factors stress tiles beyond a local AO count."""
    rng = np.random.default_rng(2352)
    c = tuple(rng.normal(size=(basis.nao, n)) / 3 for n in counts)
    f = tuple(np.linspace(0.9, 0.1, n) for n in counts)
    d = np.stack([(cs * fs) @ cs.T for cs, fs in zip(c, f, strict=True)])
    source = DensitySource(
        d, basis_identity=basis.identity, density_generation=generation
    )
    return source.with_orbitals(c, f, stamp=source.stamp, validation_rows=3)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("points,orbitals", [(7, 1), (13, 3)])
def test_native_routes_match_independent_features_with_partial_tiles(
    artifact, name, points, orbitals
):
    meta, data = load_fixture(name)
    with NativeAO(**basis_arguments(meta)) as basis:
        source = DensitySource(data["density"], basis_identity=basis.identity)
        source = source.with_orbitals(
            data["coefficients"], data["occupations"], stamp=source.stamp
        )
        with CudaGrid(
            basis,
            artifact,
            order=3,
            tile_points=points,
            orbital_capacity=tuple(map(len, source.occupations)),
            orbital_tile=orbitals,
        ) as cuda:
            for route in ("density_matrix", "orbitals"):
                cuda.set_source(source, stamp=source.stamp, route=route)
                for begin in range(0, len(data["points"]), points):
                    result = cuda.evaluate(
                        data["points"][begin : begin + points],
                        stamp=source.stamp,
                        download_jets=True,
                    )
                    for key, value in result.items():
                        check(value, data[key][:, begin : begin + points])
            metrics = cuda.metrics()
            assert metrics["owned_device_bytes"] == cuda.plan.allocation_bytes
            assert metrics["provider_retained_bytes"] <= cuda.plan.provider_bytes
            assert metrics["library_ms"] > 0 and metrics["packing_ms"] > 0


@pytest.mark.parametrize(
    "ingredients", [r for size in range(1, 5) for r in combinations(KEYS, size)]
)
@pytest.mark.parametrize("counts", [(11, 0), (0, 0)])
def test_requested_features_and_empty_spin_channels(artifact, ingredients, counts):
    meta, data = load_fixture("water")
    order = 0 if ingredients == ("rho",) else 1
    with NativeAO(**basis_arguments(meta)) as basis:
        source = factors(basis, counts)
        points = data["points"][:5]
        expected = density_features(
            basis.evaluate(points, order), source.density, ingredients=ingredients
        )
        with CudaGrid(
            basis,
            artifact,
            order=order,
            tile_points=5,
            orbital_capacity=counts,
            orbital_tile=3,
            ingredients=ingredients,
        ) as cuda:
            for route in ("density_matrix", "orbitals"):
                cuda.set_source(source, stamp=source.stamp, route=route)
                actual = cuda.evaluate(points, stamp=source.stamp)
                assert actual.keys() == expected.keys()
                for key in actual:
                    check(actual[key], expected[key])
                empty = cuda.evaluate(np.empty((0, 3)), stamp=source.stamp)
                assert all(v.shape[1] == 0 for v in empty.values())


def test_local_maps_keep_all_occupied_columns_and_cross_terms(artifact):
    meta, data = load_fixture("f_spherical")
    with NativeAO(**basis_arguments(meta)) as basis:
        source = factors(basis, (17, 9))
        points = data["points"][:9]
        full = basis.evaluate(points, 3)
        with CudaGrid(
            basis,
            artifact,
            order=3,
            tile_points=9,
            active_ao_capacity=basis.nao,
            orbital_capacity=(17, 9),
            orbital_tile=4,
        ) as cuda:
            for ids in (np.array([0, 2, 5]), np.array([1]), np.array([], dtype=int)):
                expected = source.features(
                    full[:, :, ids],
                    stamp=source.stamp,
                    route="density_matrix",
                    ao_ids=ids,
                )
                for route in ("density_matrix", "orbitals"):
                    cuda.set_source(source, stamp=source.stamp, route=route)
                    actual = cuda.evaluate(points, ao_ids=ids, stamp=source.stamp)
                    for key in actual:
                        check(actual[key], expected[key])
            cuda.set_source(source, stamp=source.stamp, route="orbitals")
            with cuda.task(points, np.array([0, 2, 5]), stamp=source.stamp) as lease:
                assert lease.view.nactive == 3
                with pytest.raises(RuntimeError, match="leased"):
                    cuda.set_source(source, stamp=source.stamp)
            with pytest.raises(RuntimeError, match="expired"):
                _ = lease.view


def test_current_stamp_fallback_capacity_and_changed_density(artifact):
    meta, data = load_fixture("water")
    with NativeAO(**basis_arguments(meta)) as basis:
        source = factors(basis)
        points = data["points"][:5]
        jets = basis.evaluate(points)
        with CudaGrid(
            basis, artifact, tile_points=5, orbital_capacity=(11, 3), orbital_tile=3
        ) as cuda:
            cuda.set_source(source, stamp=source.stamp)
            for stamp in (None, replace(source.stamp, density_generation=1)):
                with pytest.raises(ValueError, match="stamp"):
                    cuda.evaluate(points, stamp=stamp)
            wrong_basis = replace(source.stamp, basis_generation=1)
            with pytest.raises(ValueError, match="stale"):
                cuda.set_source(source, stamp=wrong_basis)
            newer = DensitySource(
                0.9 * source.density,
                basis_identity=basis.identity,
                density_generation=1,
            )
            rejected = newer.with_orbitals(
                source.coefficients, source.occupations, stamp=newer.stamp
            )
            assert rejected.fallback_reason == "incompatible_orbitals"
            cuda.set_source(rejected, stamp=rejected.stamp)
            assert (
                cuda.source_kind == "density_matrix"
                and cuda.fallback_reason == "incompatible_orbitals"
            )
            for key, value in cuda.evaluate(points, stamp=rejected.stamp).items():
                check(value, density_features(jets, newer.density)[key])
            with pytest.raises(ValueError, match="stamp"):
                cuda.evaluate(points, stamp=source.stamp)
            cuda.set_density(-source.density)
            assert cuda.source_stamp is None and cuda.source_kind == "density_matrix"
            for key, value in cuda.evaluate(points).items():
                check(value, density_features(jets, -source.density)[key])
        with CudaGrid(basis, artifact, tile_points=5, orbital_capacity=(1, 1)) as cuda:
            cuda.set_source(source, stamp=source.stamp)
            assert cuda.fallback_reason == "orbital_capacity_exceeded"
            with pytest.raises(ValueError, match="capacity"):
                cuda.set_source(source, stamp=source.stamp, route="orbitals")
            for key, value in cuda.evaluate(points, stamp=source.stamp).items():
                check(value, density_features(jets, source.density)[key])


def test_two_independent_owners_can_replay_concurrently(artifact):
    meta, data = load_fixture("water")
    with NativeAO(**basis_arguments(meta)) as basis:
        sources = (factors(basis, (5, 3)), factors(basis, (2, 0)))
        points = data["points"][:5]
        jets = basis.evaluate(points)
        with (
            CudaGrid(
                basis, artifact, tile_points=5, orbital_capacity=(5, 3), orbital_tile=2
            ) as first,
            CudaGrid(
                basis, artifact, tile_points=5, orbital_capacity=(2, 0), orbital_tile=1
            ) as second,
        ):

            def run(cuda, source):
                cuda.set_source(source, stamp=source.stamp)
                for _ in range(3):
                    actual = cuda.evaluate(points, stamp=source.stamp)
                    for key in actual:
                        check(actual[key], density_features(jets, source.density)[key])

            with ThreadPoolExecutor(2) as pool:
                tasks = [
                    pool.submit(run, cuda, source)
                    for cuda, source in zip((first, second), sources, strict=True)
                ]
                for task in tasks:
                    task.result()


@pytest.mark.parametrize("invalid", ["negative", "complex", "response", "missing"])
def test_unavailable_factors_execute_the_original_signed_density(artifact, invalid):
    meta, data = load_fixture("water")
    with NativeAO(**basis_arguments(meta)) as basis:
        valid = factors(basis, (3, 2))
        source = DensitySource(
            -valid.density,
            basis_identity=basis.identity,
            role="response" if invalid == "response" else "state",
        )
        if invalid != "missing":
            coefficients = list(valid.coefficients)
            occupations = list(valid.occupations)
            if invalid == "negative":
                occupations[0] = -occupations[0]
            if invalid == "complex":
                coefficients[0] = coefficients[0].astype(complex) + 1j
            source = source.with_orbitals(coefficients, occupations, stamp=source.stamp)
        with CudaGrid(basis, artifact, tile_points=5, orbital_capacity=(3, 2)) as cuda:
            cuda.set_source(source, stamp=source.stamp)
            assert cuda.source_kind == "density_matrix"
            assert cuda.fallback_reason == source.fallback_reason
            with pytest.raises(ValueError, match="unavailable"):
                cuda.set_source(source, stamp=source.stamp, route="orbitals")
            points = data["points"][:5]
            expected = density_features(basis.evaluate(points), source.density)
            for key, value in cuda.evaluate(points, stamp=source.stamp).items():
                check(value, expected[key])


def test_source_topology_and_failed_upload_cannot_publish_stale_features(
    artifact, monkeypatch
):
    meta, data = load_fixture("water")
    with NativeAO(**basis_arguments(meta)) as basis:
        source = factors(basis, (3, 2))
        with CudaGrid(basis, artifact, tile_points=5, orbital_capacity=(3, 2)) as cuda:
            cuda.set_source(source, stamp=source.stamp)
            for name in (
                *cuda._fixed,
                "source_stamp",
                "source_kind",
                "fallback_reason",
            ):
                with pytest.raises(AttributeError):
                    setattr(cuda, name, getattr(cuda, name))
            stats = cuda.source_statistics
            stats["source_kind"] = "density_matrix"
            assert cuda.source_statistics["source_kind"] == "orbitals"
            native_call = cuda._call

            def failed_upload(name, *args):
                if name == "grid_cuda_source_v1":
                    raise RuntimeError("injected transport failure")
                native_call(name, *args)

            # Exercise the public boundary's failure transition without using
            # invalid device pointers or poisoning the process CUDA context.
            monkeypatch.setattr(cuda, "_call", failed_upload)
            with pytest.raises(RuntimeError, match="transport"):
                cuda.set_source(source, stamp=source.stamp)
            assert cuda.source_stamp is None
            assert cuda.source_kind == "density_matrix"
            assert cuda.fallback_reason == "missing_orbitals"
            assert cuda.source_statistics == {}
            with pytest.raises(ValueError, match="supplied density"):
                cuda.evaluate(data["points"][:5], stamp=source.stamp)
            monkeypatch.setattr(cuda, "_call", native_call)
            cuda.set_source(source, stamp=source.stamp)
            cuda.evaluate(data["points"][:5], stamp=source.stamp)


def test_pruned_features_cannot_borrow_full_feature_task_abi(artifact):
    meta, data = load_fixture("water")
    with NativeAO(**basis_arguments(meta)) as basis:
        source = factors(basis, (3, 2))
        with CudaGrid(
            basis,
            artifact,
            tile_points=5,
            order=0,
            ingredients=("rho",),
            active_ao_capacity=basis.nao,
            orbital_capacity=(3, 2),
        ) as cuda:
            cuda.set_source(source, stamp=source.stamp)
            with (
                pytest.raises(ValueError, match="full feature layout"),
                cuda.task(data["points"][:5], np.arange(basis.nao), stamp=source.stamp),
            ):
                pytest.fail("pruned feature buffers must not be published as full")


@lru_cache(maxsize=4)
def program(name, spin):
    return NativeContractionProgram(
        functional(name, spin=spin),
        compiler=CppCompilerAdapter(Path(shutil.which("c++"))),
        cache=Path(".artifacts/density-xc-cache"),
    )


def xc_source(density, basis, generation=0):
    """Test-only factor of the independently supplied positive-definite fixture.

    Production never factorizes D to force an orbital route. This reference
    fixture has a pinned positive diagonal floor, so no clipping is involved.
    """
    source = DensitySource(
        density, basis_identity=basis.identity, density_generation=generation
    )
    c = tuple(np.linalg.cholesky(d) for d in source.density)
    return source.with_orbitals(c, (np.ones(basis.nao),) * 2, stamp=source.stamp)


@pytest.mark.parametrize("case", ["h2", "water", "f_cartesian", "f_spherical"])
@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize(
    "layout,spin",
    [("total", "unpolarized"), ("total", "polarized"), ("spin", "polarized")],
)
@pytest.mark.parametrize("device_budget", [128 << 20, 256 << 20])
def test_same_grid_xc_energy_potential_and_composed_budgets(
    artifact, case, name, layout, spin, device_budget
):
    meta, data, grid = load_integration_fixture(case)
    native = program(name, spin)
    budget = ResourceBudget(host_bytes=32 << 20, device_bytes=device_budget)
    ingredients = ("rho",) if name == "LDA_XC_PW" else ("rho", "gradient", "sigma")
    with NativeAO(**basis_arguments(meta)) as basis:
        source = xc_source(data[f"density_{layout}"], basis)
        with (
            CudaGrid(
                basis,
                artifact,
                order=native.contract.ao_order,
                tile_points=7,
                orbital_capacity=(basis.nao,) * 2,
                orbital_tile=3,
                ingredients=ingredients,
                resource_budget=budget,
            ) as cuda,
            PreparedXCContractions(
                native, basis, grid, density_grid=cuda, resource_budget=budget
            ) as endpoint,
        ):
            for route in ("density_matrix", "orbitals"):
                actual = endpoint.execute(source, stamp=source.stamp, route=route)
                check(actual["energy"], data[f"{name}_{layout}_energy"][0])
                potential = (
                    actual["potential"].mean(axis=0)
                    if layout == "total"
                    else actual["potential"]
                )
                check(potential, data[f"{name}_{layout}_potential"])
                assert endpoint.statistics["collocation_backend"] == "cuda"
                assert endpoint.statistics["xc_backend"] == "native_cpu"
                assert endpoint.resource_plan.peak_bytes["device"] <= device_budget
                assert endpoint.resource_plan.peak_bytes["host"] <= budget.host_bytes
                assert (
                    endpoint.statistics["native_metrics"]["owned_device_bytes"]
                    == cuda.plan.allocation_bytes
                )
            with pytest.raises(MemoryError):
                PreparedXCContractions(
                    native,
                    basis,
                    grid,
                    density_grid=cuda,
                    resource_budget=ResourceBudget(host_bytes=1),
                )


def test_xc_density_direction_uses_the_current_orbitals(artifact):
    meta, data, grid = load_integration_fixture("h2")
    native = program("PBE", "polarized")
    with NativeAO(**basis_arguments(meta)) as basis:
        source = xc_source(data["density_spin"], basis)
        direction = 0.03 * source.density
        with (
            CudaGrid(
                basis,
                artifact,
                tile_points=7,
                orbital_capacity=(basis.nao,) * 2,
                orbital_tile=1,
                ingredients=("rho", "gradient", "sigma"),
            ) as cuda,
            PreparedXCContractions(native, basis, grid, density_grid=cuda) as endpoint,
        ):
            base = endpoint.execute(source, stamp=source.stamp, route="orbitals")
            expected = np.sum(base["potential"] * direction)
            for step in (1e-3, 3e-4, 1e-4):
                energies = []
                for sign in (1, -1):
                    changed = xc_source(
                        source.density + sign * step * direction, basis, 1
                    )
                    energies.append(
                        endpoint.execute(
                            changed, stamp=changed.stamp, route="orbitals"
                        )["energy"]
                    )
                assert abs((energies[0] - energies[1]) / (2 * step) - expected) < 2e-9
