"""Correctness tests for build-time shell-class symbolic code generation."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tools.qce_codegen import (
    DDPS_SPEC,
    DPDS_SPEC,
    DPPP_SPEC,
    NvrtcCacheSpec,
    ShellClassSpec,
    build_dppp_component_kernel,
    build_dppp_contraction_kernel,
    build_dppp_fused_plan,
    build_fused_shell_plan,
    build_psss_kernel,
    build_shell_class_component_kernel,
    build_shell_class_contraction_kernel,
    cartesian_components,
    dppp_components,
    emit_dppp_fused_cuda,
    emit_shell_class_fused_cuda,
    evaluate_dppp_fused_component,
    evaluate_fused_shell_component,
    nvrtc_cache_key,
)
from tools.qce_codegen.batch_benchmark import (
    DEFAULT_CANDIDATES,
    emit_batch_driver,
    emit_candidate_translation_unit,
    rank_profiled_candidates,
)
from tools.qce_codegen.benchmark import (
    emit_dppp_benchmark_cuda,
    emit_shell_class_benchmark_cuda,
)
from tools.qce_codegen.production import (
    emit_registry_header,
    load_production_manifest,
    write_production_bundle,
)
from tools.qce_codegen.shell_class import (
    AXES,
    CENTERS,
    emit_dppp_component_cuda,
    emit_dppp_contraction_cuda,
    emit_psss_cuda,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


RTX5090_DPPP_RESOURCE_LIMITS = {
    "generated_dppp_shell_class_force_rhf_kernel": (158, 40, 2064),
    "generated_dppp_shell_class_force_uhf_kernel": (158, 40, 2064),
    "generated_dppp_shell_class_force_rhf_persistent_kernel": (159, 40, 2072),
    "generated_dppp_shell_class_force_uhf_persistent_kernel": (159, 40, 2072),
}
RTX5090_DPDS_RESOURCE_LIMITS = {
    "generated_dpds_shell_class_force_rhf_kernel": (154, 40, 1872),
    "generated_dpds_shell_class_force_uhf_kernel": (154, 40, 1872),
    "generated_dpds_shell_class_force_rhf_persistent_kernel": (156, 40, 1880),
    "generated_dpds_shell_class_force_uhf_persistent_kernel": (156, 40, 1880),
}
RTX5090_DDPS_RESOURCE_LIMITS = {
    "generated_ddps_shell_class_force_rhf_kernel": (160, 64, 1872),
    "generated_ddps_shell_class_force_uhf_kernel": (160, 64, 1872),
    "generated_ddps_shell_class_force_rhf_persistent_kernel": (160, 64, 1880),
    "generated_ddps_shell_class_force_uhf_persistent_kernel": (160, 64, 1880),
}


def assert_rtx5090_resources(
    ptxas_output: str,
    limits: dict[str, tuple[int, int, int]],
) -> None:
    """Reject CUDA 12.9 resource regressions before production integration."""

    for function, (register_limit, stack_limit, shared_limit) in (
        limits.items()
    ):
        match = re.search(
            rf"Function properties for {function}\n"
            r"\s+(\d+) bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads\n"
            r"ptxas info\s+: Used (\d+) registers,.*?, (\d+) bytes smem",
            ptxas_output,
        )
        assert match is not None, f"missing ptxas resources for {function}"
        stack, spill_stores, spill_loads, registers, shared = map(
            int, match.groups()
        )
        assert registers <= register_limit
        assert stack <= stack_limit
        assert spill_stores == 0
        assert spill_loads == 0
        assert shared <= shared_limit


def test_shell_spec_generates_cca_components_and_compile_time_bounds():
    """Derive shell schedules without handwritten component tables."""

    assert cartesian_components(0) == ("",)
    assert cartesian_components(1) == AXES
    assert cartesian_components(2) == ("xx", "xy", "xz", "yy", "yz", "zz")
    assert cartesian_components(3) == (
        "xxx",
        "xxy",
        "xxz",
        "xyy",
        "xyz",
        "xzz",
        "yyy",
        "yyz",
        "yzz",
        "zzz",
    )
    assert DPPP_SPEC.pair_orders == (3, 2)
    assert DPDS_SPEC.pair_orders == (3, 2)
    assert DDPS_SPEC.pair_orders == (4, 1)
    assert DPPP_SPEC.maximum_force_coulomb_order == 6
    assert DPPP_SPEC.component_count == 162
    assert DPPP_SPEC.component_strides == (27, 9, 3, 1)


def test_shell_spec_component_schedule_round_trips_without_manual_decoding():
    for index, component in enumerate(DPPP_SPEC.components):
        assert DPPP_SPEC.component_index(component) == index
        assert DPPP_SPEC.component_from_index(index) == component
    assert DPPP_SPEC.component_quantums(("xy", "z", "x", "y")) == (
        (0, 0),
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 1),
    )


def test_shell_spec_rejects_invalid_metadata_and_components():
    with pytest.raises(ValueError):
        ShellClassSpec("bad", (2, 1, 1))
    with pytest.raises(ValueError):
        ShellClassSpec("Bad", (2, 1, 1, 1))
    with pytest.raises(ValueError):
        DPPP_SPEC.validate_component(("xx", "x", "y", "xx"))
    with pytest.raises(IndexError):
        DPPP_SPEC.component_from_index(DPPP_SPEC.component_count)


@pytest.mark.parametrize(
    ("spec", "component"),
    (
        (DPDS_SPEC, ("xy", "z", "xz", "")),
        (DDPS_SPEC, ("xy", "xz", "z", "")),
    ),
)
def test_generic_shell_ad_matches_factored_lowering(spec, component):
    """Exercise pair orders 3+2 and 4+1 without handwritten builders."""

    full = build_shell_class_component_kernel(spec, component)
    factored = build_shell_class_contraction_kernel(spec, component)
    full_values = sample_variables()
    argument = full.graph.evaluate(full.boys_argument, full_values)
    for order, value in enumerate(
        boys_values(argument, spec.maximum_force_coulomb_order + 1)
    ):
        full_values[f"boys_{order}"] = value
    factored_values = factored_dppp_variables(full_values)

    full_value = full.graph.evaluate(full.value, full_values)
    factored_value = (
        factored_values["prefactor"]
        * factored.graph.evaluate(factored.value, factored_values)
    )
    assert factored_value == pytest.approx(full_value, rel=5.0e-13, abs=5.0e-13)
    for center in range(4):
        for axis in range(3):
            actual = factored.graph.evaluate(
                factored.gradients[center][axis], factored_values
            )
            expected = full.graph.evaluate(
                full.gradients[center][axis], full_values
            )
            assert actual == pytest.approx(
                expected, rel=3.0e-11, abs=3.0e-11
            )


@pytest.mark.parametrize("spec", (DPDS_SPEC, DDPS_SPEC))
def test_generic_fused_schedule_preserves_every_component_gradient(spec):
    """Audit generated lane schedules, including ddps double Wick matching."""

    plan = build_fused_shell_plan(spec)
    assert plan.components == spec.components
    assert plan.block_threads == 128
    assert plan.warp_count == 4
    assert len(plan.coulomb_states) == 84
    assert len(plan.coulomb_indices) == 7**3

    values = factored_dppp_variables(sample_variables())
    for component in plan.components:
        direct = build_shell_class_contraction_kernel(spec, component)
        fused = evaluate_fused_shell_component(spec, component, values)
        for center in range(4):
            for axis in range(3):
                expected = direct.graph.evaluate(
                    direct.gradients[center][axis], values
                )
                assert fused[center][axis] == pytest.approx(
                    expected, rel=8.0e-12, abs=8.0e-12
                )


def boys_values(argument: float, count: int = 3) -> list[float]:
    """Reference Boys sequence sufficient for the psss pilot."""

    if argument < 1.0e-8:
        return [
            sum(
                (-argument) ** term
                / (math.factorial(term) * (2 * order + 2 * term + 1))
                for term in range(18)
            )
            for order in range(count)
        ]
    root = math.sqrt(argument)
    values = [0.5 * math.sqrt(math.pi / argument) * math.erf(root)]
    exponential = math.exp(-argument)
    for order in range(count - 1):
        values.append(
            ((2 * order + 1) * values[-1] - exponential) / (2 * argument)
        )
    return values


def sample_variables() -> dict[str, float]:
    values = {
        "alpha": 1.3,
        "beta": 0.7,
        "gamma": 0.9,
        "delta": 0.5,
        "kPi": math.pi,
    }
    positions = {
        "first": (0.1, -0.3, 0.2),
        "second": (-0.4, 0.2, 0.5),
        "third": (0.6, -0.1, -0.2),
        "fourth": (-0.2, 0.4, -0.6),
    }
    for center, position in positions.items():
        for axis, coordinate in zip(AXES, position, strict=True):
            values[f"{center}_{axis}"] = coordinate
    return values


def factored_dppp_variables(values: dict[str, float]) -> dict[str, float]:
    """Construct the common primitive geometry consumed by factored lowering."""

    alpha = values["alpha"]
    beta = values["beta"]
    gamma = values["gamma"]
    delta = values["delta"]
    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    result = {
        "inverse_two_p": 0.5 / p,
        "inverse_two_q": 0.5 / q,
        "rho": p * q / (p + q),
        "first_product_scale": alpha / p,
        "second_product_scale": beta / p,
        "third_product_scale": gamma / q,
    }
    product_p = {}
    product_q = {}
    pair_distance_squared = 0.0
    for axis in AXES:
        first = values[f"first_{axis}"]
        second = values[f"second_{axis}"]
        third = values[f"third_{axis}"]
        fourth = values[f"fourth_{axis}"]
        product_p[axis] = (alpha * first + beta * second) / p
        product_q[axis] = (gamma * third + delta * fourth) / q
        result[f"pa_{axis}"] = product_p[axis] - first
        result[f"pb_{axis}"] = product_p[axis] - second
        result[f"qc_{axis}"] = product_q[axis] - third
        result[f"qd_{axis}"] = product_q[axis] - fourth
        result[f"difference_{axis}"] = product_p[axis] - product_q[axis]
        first_difference = first - second
        second_difference = third - fourth
        pair_distance_squared += (
            -mu * first_difference * first_difference
            - nu * second_difference * second_difference
        )
        result[f"decay_first_{axis}"] = -2.0 * mu * first_difference
        result[f"decay_second_{axis}"] = 2.0 * mu * first_difference
        result[f"decay_third_{axis}"] = -2.0 * nu * second_difference
    result["prefactor"] = (
        2.0
        * math.pi**2.5
        / (p * q * math.sqrt(p + q))
        * math.exp(pair_distance_squared)
    )
    argument = result["rho"] * sum(
        result[f"difference_{axis}"] ** 2 for axis in AXES
    )
    for order, value in enumerate(boys_values(argument, 7)):
        result[f"boys_{order}"] = value
    return result


def evaluate_value(kernel, values: dict[str, float], boys_count: int = 3) -> float:
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument, boys_count)):
        values[f"boys_{order}"] = value
    return kernel.graph.evaluate(kernel.value, values)


@pytest.mark.parametrize("p_axis", AXES)
def test_psss_symbolic_gradients_match_finite_difference(p_axis: str):
    kernel = build_psss_kernel(p_axis)
    values = sample_variables()
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument)):
        values[f"boys_{order}"] = value

    step = 2.0e-6
    for center_index, center in enumerate(CENTERS[:3]):
        for axis_index, axis in enumerate(AXES):
            variable = f"{center}_{axis}"
            plus = dict(values)
            minus = dict(values)
            plus[variable] += step
            minus[variable] -= step
            numerical = (
                evaluate_value(kernel, plus) - evaluate_value(kernel, minus)
            ) / (2.0 * step)
            analytic = kernel.graph.evaluate(
                kernel.gradients[center_index][axis_index], values
            )
            assert analytic == pytest.approx(numerical, rel=2.0e-8, abs=2.0e-9)


def test_psss_fourth_center_uses_exact_translation_recovery():
    kernel = build_psss_kernel("x")
    values = sample_variables()
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument)):
        values[f"boys_{order}"] = value
    for axis in range(3):
        total = sum(
            kernel.graph.evaluate(kernel.gradients[center][axis], values)
            for center in range(4)
        )
        assert total == pytest.approx(0.0, abs=2.0e-14)


@pytest.mark.parametrize(
    ("d_component", "p_components"),
    (("xx", "xxx"), ("xy", "xyz"), ("zz", "zyx")),
)
def test_dppp_symbolic_gradients_match_finite_difference(
    d_component: str, p_components: str
):
    kernel = build_dppp_component_kernel(d_component, tuple(p_components))
    values = sample_variables()
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument, 7)):
        values[f"boys_{order}"] = value

    step = 1.0e-6
    for center_index, center in enumerate(CENTERS[:3]):
        for axis_index, axis in enumerate(AXES):
            variable = f"{center}_{axis}"
            plus = dict(values)
            minus = dict(values)
            plus[variable] += step
            minus[variable] -= step
            numerical = (
                evaluate_value(kernel, plus, 7)
                - evaluate_value(kernel, minus, 7)
            ) / (2.0 * step)
            analytic = kernel.graph.evaluate(
                kernel.gradients[center_index][axis_index], values
            )
            assert analytic == pytest.approx(numerical, rel=3.0e-7, abs=3.0e-8)


def test_dppp_translation_and_ket_pair_permutation_invariants():
    kernel = build_dppp_component_kernel("xy", tuple("xyz"))
    values = sample_variables()
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument, 7)):
        values[f"boys_{order}"] = value
    for axis in range(3):
        total = sum(
            kernel.graph.evaluate(kernel.gradients[center][axis], values)
            for center in range(4)
        )
        assert total == pytest.approx(0.0, abs=2.0e-12)

    swapped_kernel = build_dppp_component_kernel("xy", tuple("xzy"))
    swapped_values = dict(values)
    swapped_values["gamma"], swapped_values["delta"] = (
        swapped_values["delta"],
        swapped_values["gamma"],
    )
    for axis in AXES:
        third = swapped_values[f"third_{axis}"]
        swapped_values[f"third_{axis}"] = swapped_values[f"fourth_{axis}"]
        swapped_values[f"fourth_{axis}"] = third
    assert evaluate_value(kernel, dict(values), 7) == pytest.approx(
        evaluate_value(swapped_kernel, swapped_values, 7),
        rel=2.0e-13,
        abs=2.0e-13,
    )


@pytest.mark.parametrize(
    ("d_component", "p_components"),
    (("xx", "xxx"), ("xy", "xyz"), ("zz", "zyx")),
)
def test_factored_dppp_lowering_matches_full_symbolic_kernel(
    d_component: str, p_components: str
):
    full = build_dppp_component_kernel(d_component, tuple(p_components))
    factored = build_dppp_contraction_kernel(d_component, tuple(p_components))
    full_values = sample_variables()
    argument = full.graph.evaluate(full.boys_argument, full_values)
    for order, value in enumerate(boys_values(argument, 7)):
        full_values[f"boys_{order}"] = value
    factored_values = factored_dppp_variables(full_values)

    full_value = full.graph.evaluate(full.value, full_values)
    factored_value = (
        factored_values["prefactor"]
        * factored.graph.evaluate(factored.value, factored_values)
    )
    assert factored_value == pytest.approx(full_value, rel=3.0e-13, abs=3.0e-13)
    for center in range(4):
        for axis in range(3):
            assert factored.graph.evaluate(
                factored.gradients[center][axis], factored_values
            ) == pytest.approx(
                full.graph.evaluate(full.gradients[center][axis], full_values),
                rel=2.0e-11,
                abs=2.0e-11,
            )


def test_dppp_fused_plan_covers_components_and_shared_coulomb_states():
    plan = build_dppp_fused_plan()
    components = dppp_components()
    assert plan.components == components
    assert len(components) == 162
    assert len(plan.coulomb_states) == 84
    assert len(plan.coulomb_indices) == 7**3
    assert plan.block_threads == 192
    assert plan.warp_count == 6
    for index, (x_order, y_order, z_order) in enumerate(plan.coulomb_states):
        dense_index = (x_order * 7 + y_order) * 7 + z_order
        assert plan.coulomb_indices[dense_index] == index
        assert x_order + y_order + z_order <= 6


def test_dppp_fused_schedule_preserves_all_component_gradients():
    values = factored_dppp_variables(sample_variables())
    for component in dppp_components():
        direct = build_dppp_contraction_kernel(component[0], component[1:])
        fused = evaluate_dppp_fused_component(component, values)
        for center in range(4):
            for axis in range(3):
                expected = direct.graph.evaluate(
                    direct.gradients[center][axis], values
                )
                actual = fused[center][axis]
                assert actual == pytest.approx(
                    expected, rel=4.0e-12, abs=4.0e-12
                )


def test_dppp_fused_cuda_emits_one_shared_shell_class_schedule():
    source = emit_dppp_fused_cuda()
    assert "kGeneratedDpppComponentCount = 162U" in source
    assert "kGeneratedDpppCoulombStateCount = 84U" in source
    assert "kGeneratedDpppBlockThreads = 192U" in source
    assert "__shared__ Shared shared" in source
    assert "generated_dppp_density_coefficient" in source
    assert "generated_dppp_component_gradient" in source
    assert "generated_dppp_shell_class_force_rhf_kernel" in source
    assert "generated_dppp_shell_class_force_uhf_kernel" in source
    assert "generated_dppp_shell_class_force_rhf_persistent_kernel" in source
    assert "generated_dppp_shell_class_force_uhf_persistent_kernel" in source
    assert "retained_by_schwarz" in source
    assert source.count("boys_values<6>") == 1
    assert "__noinline__" not in source
    assert "generated_dppp_orbit_" not in source
    assert "coordinate_gradient" not in source
    assert "Dual3" not in source


def test_dpds_fused_cuda_is_generated_from_shell_spec():
    source = emit_shell_class_fused_cuda(DPDS_SPEC)
    assert "kGeneratedDpdsComponentCount = 108U" in source
    assert "kGeneratedDpdsBlockThreads = 128U" in source
    assert "const unsigned third_d = component % 6U" in source
    assert "const unsigned fourth_s = 0U" in source
    assert "generated_dpds_d_axes[third_d][1]" in source
    assert "GeneratedDpdsPairTerm second_terms[4]" in source
    assert "generated_dpds_shell_class_force_rhf_kernel" in source
    assert "generated_dpds_shell_class_force_uhf_persistent_kernel" in source
    assert "generated_dppp" not in source
    assert "__noinline__" not in source
    assert "Dual3" not in source


def test_ddps_fused_cuda_generates_order_four_double_matchings():
    source = emit_shell_class_fused_cuda(DDPS_SPEC)
    assert "kGeneratedDdpsComponentCount = 108U" in source
    assert "kGeneratedDdpsBlockThreads = 128U" in source
    assert "PairOrder == 1U || PairOrder == 4U" in source
    assert "first_removed | second_removed, 2U" in source
    assert "first_d >= second_d" in source
    assert "GeneratedDdpsPairTerm second_terms[2]" in source
    assert "generated_ddps_shell_class_force_rhf_kernel" in source
    assert "generated_dppp" not in source
    assert "__noinline__" not in source
    assert "Dual3" not in source


def test_production_manifest_drives_generated_registry_and_shards(tmp_path: Path):
    """Keep machine CUDA out of Git while retaining deterministic builds."""

    manifest = (
        REPOSITORY_ROOT
        / "tools"
        / "qce_codegen"
        / "production_shell_classes.json"
    )
    specifications = load_production_manifest(manifest)
    assert tuple(spec.name for spec in specifications) == (
        "dppp",
        "dpds",
        "ppps",
        "dpps",
        "dsps",
        "dspp",
    )
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first = write_production_bundle(manifest, first_directory, shard_count=4)
    second = write_production_bundle(manifest, second_directory, shard_count=4)
    assert [path.name for path in first] == [path.name for path in second]
    for first_path, second_path in zip(first, second, strict=True):
        assert first_path.read_bytes() == second_path.read_bytes()
        assert b"\0" not in first_path.read_bytes()
    header = emit_registry_header(specifications)
    assert '{"ppps", 4U, 3U, 64U}' in header
    assert '{"dspp", 8U, 4U, 64U}' in header
    assert "QCE_AOT_SHELL_CLASSES" in header


def test_batch_screening_ranks_real_profile_and_emits_one_process_driver():
    payload = {
        "shell_classes": [
            {"class": "dppp"},
            {"class": "ppps"},
            {"class": "ddps"},
            {"class": "psss"},
        ]
    }
    ranked = rank_profiled_candidates(payload, limit=2)
    assert tuple(spec.name for spec in ranked) == ("ddps",)
    candidate = DEFAULT_CANDIDATES[0]
    source = emit_candidate_translation_unit(
        candidate,
        task_count=2,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
    )
    assert f'qce_run_shell_class_{candidate.name}' in source
    driver = emit_batch_driver((candidate,))
    assert "cudaFree(nullptr)" in driver
    assert f"qce_run_shell_class_{candidate.name}()" in driver


@pytest.mark.parametrize(
    ("spec", "resource_limits"),
    (
        (DPPP_SPEC, RTX5090_DPPP_RESOURCE_LIMITS),
        (DPDS_SPEC, RTX5090_DPDS_RESOURCE_LIMITS),
        (DDPS_SPEC, RTX5090_DDPS_RESOURCE_LIMITS),
    ),
)
def test_fused_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path, spec, resource_limits
):
    """Compile every generated shell class for explicit resource probes."""

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
    source = tmp_path / f"generated_{spec.name}_fused.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / f"generated_{spec.name}_fused.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if os.environ.get("QCE_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120" and resource_limits is not None:
        assert_rtx5090_resources(result.stdout + result.stderr, resource_limits)


def test_dppp_benchmark_compares_shared_and_recomputed_schedules():
    source = emit_dppp_benchmark_cuda(
        task_count=32,
        primitive_count=2,
        warmups=1,
        iterations=3,
        samples=5,
    )
    assert "constexpr unsigned kTaskCount = 32;" in source
    assert "constexpr unsigned kPrimitiveCount = 2;" in source
    assert "generated_dppp_component_gradient<true>" in source
    assert "generated_dppp_component_gradient<false>" in source
    assert "generated_dppp_component_recompute_rhf_kernel" in source
    assert '\\"speedup\\"' in source


@pytest.mark.parametrize(
    ("spec", "third_offset"),
    ((DPDS_SPEC, 9), (DDPS_SPEC, 12)),
)
def test_benchmark_is_generated_without_shell_specific_harness_code(
    spec, third_offset
):
    source = emit_shell_class_benchmark_cuda(
        spec,
        task_count=32,
        primitive_count=2,
        warmups=1,
        iterations=3,
        samples=5,
    )
    assert "constexpr std::size_t n = 16U" in source
    assert f"task.ao_begin[2] = {third_offset}U" in source
    assert "task.ao_begin[3] = 15U" in source
    assert f"generated_{spec.name}_component_gradient<true>" in source
    assert f"generated_{spec.name}_component_gradient<false>" in source
    assert f"generated_{spec.name}_component_recompute_rhf_kernel" in source
    assert "generated_dppp" not in source


def test_cuda_emission_is_deterministic_and_runtime_ad_free():
    first = emit_psss_cuda(build_psss_kernel("z"))
    second = emit_psss_cuda(build_psss_kernel("z"))
    assert first == second
    assert "boys_values<2>" in first
    assert "generated_psss_z_gradient" in first
    assert "Dual3" not in first


def test_dppp_cuda_emission_is_deterministic_and_runtime_ad_free():
    kernel = build_dppp_component_kernel("xy", tuple("xyz"))
    first = emit_dppp_component_cuda(kernel)
    second = emit_dppp_component_cuda(
        build_dppp_component_kernel("xy", tuple("xyz"))
    )
    assert first == second
    assert "boys_values<6>" in first
    assert "generated_dppp_xy_xyz_gradient" in first
    assert "Dual3" not in first

    factored = emit_dppp_contraction_cuda(
        build_dppp_contraction_kernel("xy", tuple("xyz"))
    )
    assert "GeneratedDpppGeometry" in factored
    assert "generated_dppp_xy_xyz_factored_gradient" in factored
    assert "boys_values" not in factored
    assert "Dual3" not in factored


def test_nvrtc_cache_key_covers_binary_compatibility_inputs():
    specification = NvrtcCacheSpec(
        generator_abi="1",
        shell_class=(2, 1, 2, 0),
        derivative_centers=(0, 1, 2),
        precision_policy="fp64",
        screening_policy="density-v1",
        source_digest="abc123",
        compute_capability="sm_120",
        nvrtc_version="12.9",
        driver_version="580.95.05",
    )
    key = nvrtc_cache_key(specification)
    assert len(key) == 64
    assert key == nvrtc_cache_key(specification)
    assert key != nvrtc_cache_key(
        replace(specification, precision_policy="mixed-fp32-control")
    )


def test_codegen_cli_writes_deterministic_aot_candidate(tmp_path: Path):
    output = tmp_path / "generated" / "psss_x.cuh"
    command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        "psss",
        "--axis",
        "x",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert "generated_psss_x_gradient" in first


def test_codegen_cli_writes_dppp_component_candidate(tmp_path: Path):
    output = tmp_path / "generated" / "dppp_xy_xyz.cuh"
    command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        "dppp",
        "--d-component",
        "xy",
        "--p-components",
        "xyz",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert "generated_dppp_xy_xyz_gradient" in first

    factored_output = tmp_path / "generated" / "dppp_xy_xyz_factored.cuh"
    factored_command = [
        *command[:-2],
        "--lowering",
        "factored",
        "--output",
        str(factored_output),
    ]
    subprocess.run(factored_command, check=True)
    factored = factored_output.read_text(encoding="utf-8")
    subprocess.run(factored_command, check=True)
    assert factored_output.read_text(encoding="utf-8") == factored
    assert "generated_dppp_xy_xyz_factored_gradient" in factored

    fused_output = tmp_path / "generated" / "dppp_fused.cuh"
    fused_command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        "dppp",
        "--lowering",
        "fused",
        "--output",
        str(fused_output),
    ]
    subprocess.run(fused_command, check=True)
    fused = fused_output.read_text(encoding="utf-8")
    subprocess.run(fused_command, check=True)
    assert fused_output.read_text(encoding="utf-8") == fused
    assert "generated_dppp_shell_class_force_rhf_kernel" in fused


@pytest.mark.parametrize("spec", (DPDS_SPEC, DDPS_SPEC))
def test_codegen_cli_writes_generated_fused_candidate(tmp_path: Path, spec):
    output = tmp_path / "generated" / f"{spec.name}_fused.cuh"
    command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        spec.name,
        "--lowering",
        "fused",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert first == emit_shell_class_fused_cuda(spec)
    assert f"generated_{spec.name}_shell_class_force_rhf_kernel" in first
