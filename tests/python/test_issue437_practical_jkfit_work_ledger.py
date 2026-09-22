"""Hardware-free contract tests for the issue #437 practical JKFIT work ledger."""

import copy
import typing

import pytest

from benchmarks.issue437_practical_jkfit_work_ledger import (
    WEIGHT_MAGNITUDE_BIN_COUNT,
    compare_cells,
    summarize_cell,
)


def _work_ledger(
    auxiliary_shells: typing.Any,
    classes: typing.Any,
    signatures: typing.Any,
) -> dict:
    return {
        "aos": 4,
        "host_reconstruction": {
            "shells": [
                [0, 2, 0, 1],
                [1, 1, 1, 3],
            ],
            "auxiliary_shells": auxiliary_shells,
            "signature_tasks": signatures,
        },
        "classes": classes,
    }


def _class(
    angular: typing.Any,
    *,
    tasks: typing.Any,
    active: typing.Any,
    primitives: typing.Any,
    loads: typing.Any,
    nonzero: typing.Any,
    gpu_ms: typing.Any,
    nsys: typing.Any = None,
) -> dict:
    return {
        "angular": angular,
        "work": {
            "shell_tasks": tasks,
            "active_shell_tasks": active,
            "primitive_products": primitives,
            "public_weight_loads": loads,
            "public_nonzero_weights": nonzero,
        },
        "lowering": "rys" if angular[-1] == 3 else "polynomial",
        "component_gpu_inclusive_ms": gpu_ms,
        **({"nsys": nsys} if nsys is not None else {}),
    }


def _trace(
    naux: typing.Any, *, root_ms: typing.Any = 4.0, pair_mode: typing.Any = 2
) -> dict:
    return {
        "operation": "force_response",
        "execution": "stream",
        "nbf": 4,
        "naux": naux,
        "counters": {"shell_work_pair_mode": pair_mode},
        "regions": [{"name": "force_response", "gpu_ms": root_ms}],
    }


def test_cell_reports_domains_work_weights_schedule_and_endpoint_fraction() -> None:
    work = _work_ledger(
        [[0, 1, 0, 1], [3, 2, 1, 7]],
        [
            _class(
                [0, 0, 0],
                tasks=3,
                active=2,
                primitives=10,
                loads=20,
                nonzero=5,
                gpu_ms=0.8,
                nsys={
                    "gpu_ms": 0.4,
                    "launches": 2,
                    "schedule": {
                        "variant": 2,
                        "component_lanes": 4,
                        "shell_tasks_per_block": 32,
                    },
                },
            ),
            _class(
                [1, 0, 3],
                tasks=2,
                active=1,
                primitives=4,
                loads=8,
                nonzero=2,
                gpu_ms=1.0,
                nsys={
                    "gpu_ms": 0.6,
                    "launches": 1,
                    "schedule": {
                        "variant": 3,
                        "component_lanes": 8,
                        "shell_tasks_per_block": 8,
                    },
                },
            ),
        ],
        [
            {"signature": [0, 0, 0, 2, 2, 1], "tasks": 3},
            {"signature": [1, 0, 3, 1, 2, 2], "tasks": 2},
        ],
    )
    cell = summarize_cell("practical-4", "practical", work, _trace(8))

    sss, psf = cell["classes"]
    assert (sss["orbital_shell_pairs"], sss["auxiliary_shells"]) == (1, 1)
    assert sss["primitive_products_considered"] == 12
    assert sss["primitive_products_executed"] == 10
    assert sss["public_nonzero_fraction"] == 0.25
    assert sss["logical_response_weight_bytes"] == 160
    assert sss["force_response_gpu_fraction"] == 0.1
    assert sss["launches"] == 2
    assert psf["orbital_shell_pairs"] == 1
    assert psf["auxiliary_shells"] == 1
    assert psf["primitive_products_considered"] == 8
    assert cell["classes_gpu_ms"] == 1.0
    assert cell["classes_force_response_gpu_fraction"] == 0.25
    assert cell["nsight_complete"] is True
    assert cell["nsight_launches"] == 3
    assert "distance_exponent_bins" in cell["missing_phase_a_fields"]


def test_cell_keeps_cuda_event_timing_explicit_when_nsight_is_absent() -> None:
    work = _work_ledger(
        [[0, 1, 0, 1]],
        [
            _class(
                [0, 0, 0],
                tasks=1,
                active=1,
                primitives=4,
                loads=2,
                nonzero=2,
                gpu_ms=0.5,
            )
        ],
        [{"signature": [0, 0, 0, 2, 2, 1], "tasks": 1}],
    )
    cell = summarize_cell("equal-4", "equal", work, _trace(1, root_ms=2.0))
    row = cell["classes"][0]
    assert row["gpu_timing_source"] == "cuda_event_inclusive"
    assert row["launches"] is None
    assert cell["nsight_complete"] is False
    assert cell["nsight_launches"] is None


def test_comparison_separates_naux_growth_and_new_auxiliary_classes() -> None:
    equal_work = _work_ledger(
        [[0, 1, 0, 1]],
        [
            _class(
                [0, 0, 0],
                tasks=2,
                active=2,
                primitives=8,
                loads=8,
                nonzero=4,
                gpu_ms=0.4,
            )
        ],
        [{"signature": [0, 0, 0, 2, 2, 1], "tasks": 2}],
    )
    practical_work = _work_ledger(
        [[0, 1, 0, 1], [3, 2, 1, 7]],
        [
            _class(
                [0, 0, 0],
                tasks=4,
                active=4,
                primitives=16,
                loads=16,
                nonzero=8,
                gpu_ms=0.8,
            ),
            _class(
                [0, 0, 3],
                tasks=1,
                active=1,
                primitives=8,
                loads=4,
                nonzero=1,
                gpu_ms=0.3,
            ),
        ],
        [
            {"signature": [0, 0, 0, 2, 2, 1], "tasks": 4},
            {"signature": [0, 0, 3, 2, 2, 2], "tasks": 1},
        ],
    )
    equal = summarize_cell("equal-4", "equal", equal_work, _trace(1, root_ms=2.0))
    practical = summarize_cell(
        "practical-4", "practical", practical_work, _trace(8, root_ms=5.0)
    )
    comparison = compare_cells(equal, practical)

    assert comparison["totals"]["naux"]["ratio"] == 8.0
    assert comparison["totals"]["primitive_products_considered"]["ratio"] == 3.0
    assert comparison["totals"]["force_response_gpu_ms"]["ratio"] == 2.5
    assert comparison["classes"][1] == {
        "angular": [0, 0, 3],
        "present_in_equal": False,
        "present_in_practical": True,
    }


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda work, trace: work.update(aos=5), "AO dimensions"),
        (
            lambda work, trace: work["classes"][0]["work"].update(active_shell_tasks=2),
            "exceeds",
        ),
        (
            lambda work, trace: work["classes"][0]["work"].update(primitive_products=5),
            "exceeds",
        ),
    ],
)
def test_cell_rejects_inconsistent_domains(
    mutation: typing.Any, message: typing.Any
) -> None:
    work = _work_ledger(
        [[0, 1, 0, 1]],
        [
            _class(
                [0, 0, 0],
                tasks=1,
                active=1,
                primitives=4,
                loads=1,
                nonzero=1,
                gpu_ms=0.2,
            )
        ],
        [{"signature": [0, 0, 0, 2, 2, 1], "tasks": 1}],
    )
    trace = _trace(1)
    mutation(work, trace)
    with pytest.raises(ValueError, match=message):
        summarize_cell("cell", "equal", work, trace)


def test_comparison_rejects_changed_orbital_shell_pair_domain() -> None:
    base_work = _work_ledger(
        [[0, 1, 0, 1]],
        [
            _class(
                [0, 0, 0],
                tasks=1,
                active=1,
                primitives=4,
                loads=1,
                nonzero=1,
                gpu_ms=0.2,
            )
        ],
        [{"signature": [0, 0, 0, 2, 2, 1], "tasks": 1}],
    )
    equal = summarize_cell("equal", "equal", base_work, _trace(1))
    changed = copy.deepcopy(base_work)
    changed["host_reconstruction"]["shells"].append([0, 2, 4, 1])
    practical = summarize_cell("practical", "practical", changed, _trace(1))
    with pytest.raises(ValueError, match="orbital shell-pair domain changed"):
        compare_cells(equal, practical)


def _minimal_cell(role: str) -> tuple[dict, dict]:
    work = _work_ledger(
        [[0, 1, 0, 1]],
        [
            _class(
                [0, 0, 0],
                tasks=1,
                active=1,
                primitives=4,
                loads=1,
                nonzero=1,
                gpu_ms=0.2,
            )
        ],
        [{"signature": [0, 0, 0, 2, 2, 1], "tasks": 1}],
    )
    return work, _trace(1)


@pytest.mark.parametrize("change", ["disjoint", "primitive_count", "pair_mode"])
def test_comparison_rejects_changed_complete_orbital_domain(change: str) -> None:
    work, trace = _minimal_cell("equal")
    equal = summarize_cell("equal", "equal", work, trace)
    other, other_trace = copy.deepcopy(work), copy.deepcopy(trace)
    if change == "disjoint":
        other["host_reconstruction"]["shells"] = [[1, 2, 0, 3], [0, 1, 3, 1]]
        other["classes"][0]["angular"] = [1, 1, 0]
        other["host_reconstruction"]["signature_tasks"][0]["signature"] = [
            1,
            1,
            0,
            2,
            2,
            1,
        ]
    elif change == "primitive_count":
        other["host_reconstruction"]["shells"][0][1] = 3
    else:
        other_trace["counters"]["shell_work_pair_mode"] = 0
    practical = summarize_cell("practical", "practical", other, other_trace)
    with pytest.raises(ValueError, match="orbital shell-pair domain"):
        compare_cells(equal, practical)


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_work_counters_are_not_silently_coerced(value: object) -> None:
    work, trace = _minimal_cell("equal")
    work["classes"][0]["work"]["active_shell_tasks"] = value
    with pytest.raises(ValueError, match="integer"):
        summarize_cell("cell", "equal", work, trace)


def test_weight_histogram_conserves_folded_samples_and_public_loads() -> None:
    work, trace = _minimal_cell("practical")
    counters = trace["counters"]
    counters.update(
        {
            "screening_weight_histogram_version": 1,
            "screening_weight_histogram_bin_count": WEIGHT_MAGNITUDE_BIN_COUNT,
            "screening_feature_weight_d2h_bytes": 8,
            "screening_feature_stream_drains": 1,
            "shell_000_weight_effective_samples": 1,
            "shell_000_weight_underlying_loads": 1,
        }
    )
    for index in range(WEIGHT_MAGNITUDE_BIN_COUNT):
        counters[f"shell_000_weight_magnitude_bin_{index:02d}"] = int(index == 5)

    cell = summarize_cell("practical", "practical", work, trace)
    histogram = cell["classes"][0]["response_weight_magnitude_distribution"]
    assert histogram["effective_samples"] == 1
    assert histogram["underlying_loads"] == 1
    assert histogram["zero"] == 0
    assert sum(histogram["positive_bins"]) == 1
    assert (
        "response_weight_magnitude_distribution" not in cell["missing_phase_a_fields"]
    )

    broken = copy.deepcopy(trace)
    broken["counters"]["shell_000_weight_underlying_loads"] = 2
    with pytest.raises(ValueError, match="conserve public loads"):
        summarize_cell("practical", "practical", work, broken)


@pytest.mark.parametrize(
    "corruption", ["wrong_sum", "negative", "bool", "missing_axis", "wrong_width"]
)
def test_distance_histogram_rejects_corrupt_bins(corruption: str) -> None:
    work, trace = _minimal_cell("practical")
    row = {
        "angular": [0, 0, 0],
        "primitive_products_considered": 4,
        "distance_bohr": {key: [4] + [0] * 7 for key in ("ab", "ac", "bc")},
        "exponents": {key: [4] + [0] * 10 for key in ("alpha", "beta", "gamma")},
    }
    work["distance_exponent_bins"] = {"classes": [row]}
    if corruption == "wrong_sum":
        row["distance_bohr"]["ab"][0] = 3
    elif corruption == "negative":
        row["exponents"]["alpha"][:2] = [-1, 5]
    elif corruption == "bool":
        row["distance_bohr"]["ab"][:2] = [True, 3]
    elif corruption == "missing_axis":
        del row["exponents"]["gamma"]
    else:
        row["distance_bohr"]["bc"].append(0)
    with pytest.raises((ValueError, TypeError), match="histogram|integer|feature"):
        summarize_cell("practical", "practical", work, trace)
