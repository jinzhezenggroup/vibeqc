"""Bounded generated native provider lifecycle, identity and failed-call isolation."""

import ctypes as ct
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.integral.blocks import TensorLayout, WeightTile
from vibeqc_compiler.integral.capabilities import query_integral_capability
from vibeqc_compiler.integral.ir import four_center_eri_operator
from vibeqc_compiler.integral.range_separation import CoulombKernel
from vibeqc_compiler.integral.weighted_eri import (
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)
from vibeqc_compiler.integral.weighted_eri_execute import (
    PreparedWeightedEri,
    compile_weighted_eri,
)
from vibeqc_compiler.integral.weighted_eri_inputs import (
    PRIMITIVE_RANGE_RECORD,
    prepare_weighted_eri_stream,
    weighted_eri_response,
)
from vibeqc_compiler.integral.weighted_eri_native import (
    weighted_eri_program_identity,
)

from tools.validate_weighted_eri import make_fixture

RADIAL = CoulombKernel("long_range", 0.63)


@pytest.fixture(scope="module", params=("cpu", "cuda"))
def runtime(request, tmp_path_factory):
    """Exercise the exported generated ABI, including CUDA's shared arena owner."""
    pytest.importorskip("pyscf")
    cuda = request.param == "cuda"
    if cuda and os.environ.get("VIBEQC_TEST_RANGE_CUDA") != "1":
        pytest.skip("set VIBEQC_TEST_RANGE_CUDA=1 inside a Slurm GPU job")
    if cuda and not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("native CUDA validation requires a Slurm allocation")
    compiler = shutil.which("nvcc" if cuda else "c++")
    if compiler is None:
        pytest.skip("native compiler unavailable")
    kernel = build_weighted_eri_kernel(
        build_weighted_eri_ir((1, 0, 0, 0), operator=four_center_eri_operator(RADIAL))
    )
    folder = tmp_path_factory.mktemp(f"weighted-runtime-{request.param}")
    adapter = (
        CudaCompilerAdapter(Path(compiler), cuda_target_info("sm_120"))
        if cuda
        else CppCompilerAdapter(Path(compiler))
    )
    artifact = compile_weighted_eri(kernel.integral, adapter, folder)
    lib = ct.CDLL(str(artifact.native.library))
    lib._weighted_artifact = artifact
    lib._weighted_compiler = adapter
    lib._weighted_cache = folder
    lib.vibeqc_weighted_identity_v2.restype = ct.c_char_p
    assert lib.vibeqc_weighted_identity_v2().decode() == weighted_eri_program_identity(
        kernel, request.param
    )
    lib.vibeqc_weighted_create_v2.argtypes = (
        [ct.c_int] * 3
        + [ct.c_size_t] * 3
        + [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_size_t]
    )
    lib.vibeqc_weighted_destroy_v2.argtypes = [ct.c_void_p]
    lib.vibeqc_weighted_destroy_v2.restype = None
    lib.vibeqc_weighted_run_v2.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_void_p,
        ct.c_int,
        ct.c_char_p,
        ct.c_size_t,
    ]
    lib.vibeqc_weighted_storage_v2.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_uint64),
        ct.c_char_p,
        ct.c_size_t,
    ]
    return lib, cuda


def stream_for(variant="spherical", tile=0):
    """Return normalized tagged records and an independent contracted reference."""
    fixture = make_fixture("psss", variant, coulomb_kernel=RADIAL)
    stream = prepare_weighted_eri_stream(
        fixture["request"],
        fixture["primitives"],
        fixture["centers"],
        lambda descriptor, _: WeightTile(descriptor.layout, fixture["weights"].ravel()),
        projections=fixture["projections"],
        output_tile=tile,
    )
    return stream, fixture["reference"]


def records_for(variant="spherical", tile=0):
    stream, reference = stream_for(variant, tile)
    return list(stream.records()), reference


def create(runtime, *, capacity=3, tiles=2, budget=4096):
    lib, cuda = runtime
    handle = ct.c_void_p()
    error = ct.create_string_buffer(1024)
    status = lib.vibeqc_weighted_create_v2(
        0,
        12 if cuda else 0,
        0,
        capacity,
        tiles,
        budget,
        ct.byref(handle),
        error,
        len(error),
    )
    return status, handle, error.value.decode()


def run(runtime, handle, records, output, *, profile=0):
    lib, _ = runtime
    # NumPy owns aligned, contiguous bytes, including for an empty chunk.
    inputs = np.frombuffer(b"".join(records), dtype=np.uint8).copy()
    error = ct.create_string_buffer(1024)
    status = lib.vibeqc_weighted_run_v2(
        handle,
        inputs.ctypes.data,
        len(records),
        len(output),
        output.ctypes.data,
        profile,
        error,
        len(error),
    )
    return status, error.value.decode()


def test_retained_chunks_match_independent_ragged_tiles_and_empty_replay(runtime):
    first, expected_first = records_for()
    second, expected_second = records_for("changed", 1)
    status, handle, detail = create(runtime)
    assert status == 0, detail
    try:
        inputs = first + second
        actual = np.zeros((2, 13))
        for begin in range(0, len(inputs), 3):
            output = np.full((2, 13), 123.0)
            status, detail = run(
                runtime, handle, inputs[begin : begin + 3], output, profile=1
            )
            assert status == 0, detail
            actual += output
        np.testing.assert_allclose(
            actual, [expected_first, expected_second], atol=1e-11, rtol=1e-10
        )
        output = np.full((2, 13), 123.0)
        assert run(runtime, handle, [], output)[0] == 0
        np.testing.assert_array_equal(output, 0)
    finally:
        runtime[0].vibeqc_weighted_destroy_v2(handle)


def test_budget_is_checked_before_publication_and_capacity_is_enforced(runtime):
    status, handle, detail = create(runtime)
    assert status == 0, detail
    try:
        amounts = (ct.c_uint64 * 2)()
        error = ct.create_string_buffer(1024)
        assert (
            runtime[0].vibeqc_weighted_storage_v2(handle, amounts, error, len(error))
            == 0
        )
        assert amounts[0] > 0 and bool(amounts[1]) == runtime[1]
        failed, unpublished, detail = create(runtime, budget=sum(amounts) - 1)
        assert failed == 1 and not unpublished.value and "budget" in detail
        records, _ = records_for()
        assert len(records) > 3
        output = np.full((2, 13), 123.0)
        assert run(runtime, handle, records, output)[0] == 1
        np.testing.assert_array_equal(output, 123)
    finally:
        runtime[0].vibeqc_weighted_destroy_v2(handle)


def test_record_identity_and_numerical_failures_leave_output_and_plan_reusable(runtime):
    records, _ = records_for()
    fields = list(PRIMITIVE_RANGE_RECORD.unpack(records[0]))
    status, handle, detail = create(runtime)
    assert status == 0, detail
    try:
        mutations = [
            (33, np.nextafter(RADIAL.omega, 1), 1),
            (34, 1, 1),
            (0, 2 << 8, 1),
            (1, 2, 1),
            (2, 2, 1),
            (14, float("nan"), 1),
            (18, float("inf"), 1),
            (30, float("nan"), 1),
            (18, 1e308, 5),  # Finite input whose squared separation overflows.
        ]
        healthy = np.full((2, 13), 123.0)
        assert run(runtime, handle, records[:1], healthy)[0] == 0
        for field, value, expected_status in mutations:
            changed = fields.copy()
            changed[field] = value
            output = np.full((2, 13), 123.0)
            status, detail = run(
                runtime, handle, [PRIMITIVE_RANGE_RECORD.pack(*changed)], output
            )
            assert status == expected_status, detail
            np.testing.assert_array_equal(output, 123)
            replay = np.full((2, 13), 123.0)
            assert run(runtime, handle, records[:1], replay)[0] == 0
            np.testing.assert_allclose(replay, healthy, rtol=1e-13, atol=1e-13)
    finally:
        runtime[0].vibeqc_weighted_destroy_v2(handle)


def test_python_prepared_chunks_detach_results_and_track_geometry_identity(runtime):
    artifact = runtime[0]._weighted_artifact
    first, expected_first = stream_for()
    second, expected_second = stream_for("changed", 1)
    with PreparedWeightedEri(artifact, record_capacity=3, tile_capacity=2) as plan:
        result = plan.contract([first, second], profile=True)
        np.testing.assert_allclose(
            result.values, [expected_first, expected_second], atol=1e-11, rtol=1e-10
        )
        assert result.diagnostics["chunks"] == 3
        assert (
            result.diagnostics["stream_identities"][0]
            != result.diagnostics["stream_identities"][1]
        )
        before = result.values.copy()
        again = plan.contract(first)
        np.testing.assert_allclose(
            again.values[0], expected_first, atol=1e-11, rtol=1e-10
        )
        np.testing.assert_array_equal(result.values, before)
        scalar = replace(first, fused_weights=None)
        np.testing.assert_allclose(
            plan.contract(scalar).values[0], expected_first, atol=1e-11, rtol=1e-10
        )
        result.diagnostics["operator"]["omega"] = 999
        assert plan.contract(first).diagnostics["operator"]["omega"] == RADIAL.omega
        if runtime[1]:
            assert result.diagnostics["device_timing"]["kernel_ms"] > 0
        else:
            assert result.diagnostics["device_timing"] is None
        with pytest.raises(AttributeError):
            plan.record_capacity = 100
    np.testing.assert_array_equal(result.values, before)
    assert not result.values.flags.writeable
    with pytest.raises(RuntimeError, match="closed"):
        plan.contract(first)


def test_python_preflight_rejects_wrong_operator_and_native_failure_allows_replay(
    runtime,
):
    first, expected = stream_for()
    wrong = replace(
        first,
        request=replace(
            first.request,
            integral=replace(
                first.request.integral,
                operator=four_center_eri_operator(
                    CoulombKernel("short_range", RADIAL.omega)
                ),
            ),
        ),
    )
    centers = list(first.centers)
    centers[0] = (1e308, *centers[0][1:])
    overflow = replace(first, centers=tuple(centers))
    with PreparedWeightedEri(runtime[0]._weighted_artifact, record_capacity=2) as plan:
        with pytest.raises(ValueError, match="identity"):
            plan.contract(wrong)
        with pytest.raises(FloatingPointError, match="nonfinite"):
            plan.contract(overflow)
        np.testing.assert_allclose(
            plan.contract(first).values[0], expected, atol=1e-11, rtol=1e-10
        )


def test_python_budget_preflight_precedes_library_loading(runtime, monkeypatch):
    def forbidden(*_, **__):
        raise AssertionError("library loaded before budget preflight")

    monkeypatch.setattr(ct, "CDLL", forbidden)
    with pytest.raises(ValueError):
        PreparedWeightedEri(
            runtime[0]._weighted_artifact,
            budget=ResourceBudget(host_bytes=0, device_bytes=0),
        )


def test_compiled_omega_invalidates_cache_but_stream_coefficients_apply_once(runtime):
    lib, _ = runtime
    artifact = lib._weighted_artifact
    changed = replace(
        artifact.requested,
        operator=four_center_eri_operator(CoulombKernel("long_range", 0.64)),
    )
    other = compile_weighted_eri(changed, lib._weighted_compiler, lib._weighted_cache)
    assert other.program_identity != artifact.program_identity
    assert other.native.metadata["key"] != artifact.native.metadata["key"]
    fixture = make_fixture("psss", "spherical", coulomb_kernel=RADIAL)
    request = fixture["request"]
    consumer = request.consumer
    consumer = replace(
        consumer, output_sign=-1, weights=replace(consumer.weights, prefactor=0.37)
    )
    integral = replace(request.integral, contractions=(consumer,))
    stream = prepare_weighted_eri_stream(
        replace(request, integral=integral),
        fixture["primitives"],
        fixture["centers"],
        lambda descriptor, _: WeightTile(descriptor.layout, fixture["weights"].ravel()),
        projections=fixture["projections"],
    )
    scaled = compile_weighted_eri(integral, lib._weighted_compiler, lib._weighted_cache)
    assert scaled.native.metadata["key"] == artifact.native.metadata["key"]
    with PreparedWeightedEri(scaled, record_capacity=2) as plan:
        np.testing.assert_allclose(
            plan.contract(stream).values[0],
            -0.37 * fixture["reference"],
            atol=1e-11,
            rtol=1e-10,
        )


@pytest.mark.parametrize("name", ["dpsp", "fsss"])
@pytest.mark.parametrize("family", ["long_range", "short_range"])
def test_prepared_runtime_general_components_with_contracted_spherical_weights(
    runtime, name, family
):
    lib, _ = runtime
    fixture = make_fixture(
        name, "spherical", coulomb_kernel=CoulombKernel(family, 0.63)
    )
    stream = prepare_weighted_eri_stream(
        fixture["request"],
        fixture["primitives"],
        fixture["centers"],
        lambda descriptor, _: WeightTile(descriptor.layout, fixture["weights"].ravel()),
        projections=fixture["projections"],
    )
    artifact = compile_weighted_eri(
        fixture["request"].integral, lib._weighted_compiler, lib._weighted_cache
    )
    with PreparedWeightedEri(artifact, record_capacity=7) as plan:
        actual = plan.contract(stream)
        np.testing.assert_allclose(
            actual.values[0], fixture["reference"], atol=1e-11, rtol=1e-10
        )
        assert actual.diagnostics["chunks"] > 1


@pytest.mark.parametrize("name", ["psss", "dpsp", "fsss"])
@pytest.mark.parametrize("variant", ["cartesian", "spherical"])
@pytest.mark.parametrize("family", ["long_range", "short_range"])
def test_raw_public_components_match_independent_values_and_center_derivatives(
    runtime, name, variant, family
):
    """Raw unit cotangents use the same normalization/pullback as weighted calls."""
    lib, _ = runtime
    radial = CoulombKernel(family, 0.63)
    fixture = make_fixture(name, variant, coulomb_kernel=radial)
    artifact = compile_weighted_eri(
        fixture["request"].integral, lib._weighted_compiler, lib._weighted_cache
    )
    indices = (0, fixture["weights"].size // 2, fixture["weights"].size - 1)
    expected = [
        make_fixture(name, variant, unit_component=i, coulomb_kernel=radial)[
            "reference"
        ]
        for i in indices
    ]
    with PreparedWeightedEri(
        artifact, record_capacity=3, tile_capacity=len(indices)
    ) as plan:
        actual = plan.raw(
            fixture["primitives"],
            fixture["centers"],
            indices,
            projections=fixture["projections"],
            profile=True,
        )
        np.testing.assert_allclose(actual.values, expected, atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(
            actual.values[:, 1:].reshape(-1, 4, 3).sum(axis=1), 0, atol=1e-12
        )
        assert not actual.values.flags.writeable
        assert actual.diagnostics["raw_component_indices"] == list(indices)
        assert plan.raw(fixture["primitives"], fixture["centers"], ()).values.shape == (
            0,
            13,
        )


def test_raw_rejects_capacity_budget_and_incomplete_spherical_pullback_then_replays(
    runtime,
):
    lib, _ = runtime
    fixture = make_fixture("dsss", "spherical", coulomb_kernel=RADIAL)
    artifact = compile_weighted_eri(
        fixture["request"].integral,
        lib._weighted_compiler,
        lib._weighted_cache,
        component_indices=(0,),
    )
    with PreparedWeightedEri(artifact, record_capacity=2) as plan:
        args = (fixture["primitives"], fixture["centers"])
        with pytest.raises(ValueError, match="tile capacity"):
            plan.raw(*args, (0, 1), projections=fixture["projections"])
        with pytest.raises(ValueError, match="no supported plan fits"):
            plan.raw(
                *args,
                (0,),
                projections=fixture["projections"],
                adapter_budget_bytes=1 << 30,
            )
        with pytest.raises(ValueError, match="subset"):
            plan.raw(*args, (2,), projections=fixture["projections"])
        # A zero projection has no omitted nonzero Cartesian contribution.
        projection = tuple(np.zeros_like(p) for p in fixture["projections"])
        np.testing.assert_array_equal(
            plan.raw(*args, (0,), projections=projection).values, 0
        )


def test_preparation_rejects_forged_mathematical_and_component_metadata(
    runtime, monkeypatch
):
    artifact = runtime[0]._weighted_artifact

    def forbidden(*_, **__):
        raise AssertionError("library loaded before identity validation")

    monkeypatch.setattr(ct, "CDLL", forbidden)
    other = build_weighted_eri_ir(
        (1, 0, 0, 0),
        operator=four_center_eri_operator(CoulombKernel("long_range", 0.7)),
    )
    for forged in (
        replace(artifact, integral=other),
        replace(artifact, requested=other),
        replace(artifact, component_indices=(0,)),
        replace(artifact, component_quantums=()),
        replace(artifact, backend="cpu" if runtime[1] else "cuda"),
    ):
        with pytest.raises(ValueError, match="metadata|identity"):
            PreparedWeightedEri(forged)


@pytest.mark.parametrize("family", ["long_range", "short_range"])
@pytest.mark.parametrize("variant", ["orbit", "exchange"])
def test_range_weighted_orbits_and_exchange_cotangents(runtime, family, variant):
    """Fold ordered density/external cotangents with the existing orbit interface."""
    lib, _ = runtime
    fixture = make_fixture("dpsp", variant, coulomb_kernel=CoulombKernel(family, 0.63))
    artifact = compile_weighted_eri(
        fixture["request"].integral, lib._weighted_compiler, lib._weighted_cache
    )
    stream = prepare_weighted_eri_stream(
        fixture["request"],
        fixture["primitives"],
        fixture["centers"],
        lambda descriptor, _: WeightTile(descriptor.layout, fixture["weights"].ravel()),
    )
    with PreparedWeightedEri(artifact, record_capacity=7) as plan:
        np.testing.assert_allclose(
            plan.contract(stream).values[0],
            fixture["reference"],
            atol=1e-11,
            rtol=1e-10,
        )


@pytest.mark.parametrize("family", ["long_range", "short_range"])
def test_raw_f_shell_permutation_orbit_preserves_center_slots(runtime, family):
    """All eight ERI symmetries permute shell-center derivatives with their slots."""
    lib, _ = runtime
    radial = CoulombKernel(family, 0.63)
    fixture = make_fixture("fsss", "cartesian", coulomb_kernel=radial)
    indices = (0, 5, 9)
    expected = np.array(
        [
            make_fixture("fsss", "cartesian", unit_component=i, coulomb_kernel=radial)[
                "reference"
            ]
            for i in indices
        ]
    )
    for left in ((0, 1), (1, 0)):
        for right in ((2, 3), (3, 2)):
            for permutation in ((*left, *right), (*right, *left)):
                integral = build_weighted_eri_ir(
                    tuple((3, 0, 0, 0)[i] for i in permutation),
                    operator=four_center_eri_operator(radial),
                )
                artifact = compile_weighted_eri(
                    integral, lib._weighted_compiler, lib._weighted_cache
                )
                with PreparedWeightedEri(
                    artifact, record_capacity=3, tile_capacity=3
                ) as plan:
                    result = plan.raw(
                        tuple(fixture["primitives"][i] for i in permutation),
                        fixture["centers"][list(permutation)],
                        indices,
                    ).values
                restored = np.column_stack(
                    (
                        result[:, 0],
                        result[:, 1:]
                        .reshape(3, 4, 3)[:, np.argsort(permutation)]
                        .reshape(3, 12),
                    )
                )
                np.testing.assert_allclose(restored, expected, atol=1e-11, rtol=1e-10)


def test_range_capability_is_explicit_bounded_and_excludes_legacy_hf():
    integral = build_weighted_eri_ir(
        (3, 3, 3, 3), operator=four_center_eri_operator(RADIAL)
    )
    for backend in ("cpu_range_weighted_eri", "cuda_range_weighted_eri"):
        assert not query_integral_capability(integral, backend=backend).supported

        assert query_integral_capability(
            integral, backend=backend, component_indices=(0, 99)
        ).supported
        assert not query_integral_capability(
            integral, backend=backend, component_indices=(0, 0)
        ).supported
        assert not query_integral_capability(
            build_weighted_eri_ir((0, 0, 0, 0)), backend=backend
        ).supported
    for backend in ("cuda", "cuda_weighted_eri"):
        assert not query_integral_capability(integral, backend=backend).supported


@pytest.mark.parametrize("family", ["long_range", "short_range"])
def test_range_atomic_force_scatter_adds_coincident_shell_slots(runtime, family):
    """Two slots on one atom contribute separately before the existing scatter."""
    lib, _ = runtime
    fixture = make_fixture(
        "psss", "coincident", coulomb_kernel=CoulombKernel(family, 0.63)
    )
    request = fixture["request"]
    consumer = replace(
        request.consumer,
        output="atomic_force",
        output_sign=-1,
        output_layout=TensorLayout(("atom", "xyz"), (3, 3)),
    )
    integral = replace(request.integral, contractions=(consumer,))
    request = replace(request, integral=integral)
    stream = prepare_weighted_eri_stream(
        request,
        fixture["primitives"],
        fixture["centers"],
        lambda descriptor, _: WeightTile(descriptor.layout, fixture["weights"].ravel()),
    )
    artifact = compile_weighted_eri(
        integral, lib._weighted_compiler, lib._weighted_cache
    )
    with PreparedWeightedEri(artifact, record_capacity=3) as plan:
        result = plan.contract(stream)
    response = weighted_eri_response(request, result.values[0])
    gradient = fixture["reference"][1:].reshape(4, 3)
    expected = -np.array([gradient[0] + gradient[1], gradient[2], gradient[3]])
    np.testing.assert_allclose(
        np.array(response.values).reshape(3, 3), expected, atol=1e-11, rtol=1e-10
    )
