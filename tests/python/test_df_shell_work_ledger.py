"""Reject inconsistent scientific work evidence without requiring a GPU."""

import copy
import math
import sqlite3
import typing
from collections import Counter
from itertools import product

import pytest

from benchmarks.df_shell_work_ledger import (
    kernel_activity,
    reconstruct_domain,
    reduce_work,
)


@pytest.mark.parametrize("pair_mode", [0, 1, 2])
def test_reconstruction_matches_explicit_shell_visits(
    pair_mode: typing.Any,
) -> None:
    # The same angular class has different contraction lengths, and a p shell
    # crosses the panel boundary. Enumerate actual shell IDs independently of
    # the reducer's grouped combinatorics.
    shells = [(0, 2, 0, 1), (1, 1, 1, 3), (0, 1, 4, 1), (1, 1, 5, 3)]
    panels = [(0, 3, 2), (3, 5, 1)]
    expected = Counter()
    for ia, ib, ic in product(range(len(shells)), repeat=3):
        a, b, c = shells[ia], shells[ib], shells[ic]
        if pair_mode and (a[:2], ia) < (b[:2], ib):
            continue
        for begin, count, repeats in panels:
            if any(begin <= ao < begin + count for ao in range(c[2], c[2] + c[3])):
                expected[(a[0], b[0], c[0], a[1], b[1], c[1])] += repeats
    assert reconstruct_domain(shells, panels, pair_mode) == expected


@pytest.fixture
def sss_record() -> typing.Any:
    """One active coincident SSS task, two primitives per shell, one component.

    Each of its eight primitive products stores nine polynomial coefficients
    and executes six convolution iterations. All nine gradient atomics target
    the shared physical atom. Values are hand-counted, not emitted by the model
    whose evidence this reducer validates.
    """
    work = {
        "shell_tasks": 1,
        "active_shell_tasks": 1,
        "primitive_products": 8,
        "geometry_preparations": 8,
        "boys_evaluations": 8,
        "boys_order_sum": 8,
        "boys_series_iterations": 8,
        "boys_series": 8,
        "boys_small_argument": 8,
        "boys_large_argument": 0,
        "axis_polynomial_calls": 0,
        "specialized_prepare_axis_calls": 24,
        "cache_coefficient_values": 72,
        "convolution_iterations": 48,
        "active_component_products": 8,
        "public_weight_loads": 1,
        "public_nonzero_weights": 1,
        "expansion_term_products": 1,
        "folding_shared_atomics": 1,
        "folding_direct_stores": 0,
        "gradient_atomics_a": 3,
        "gradient_atomics_b": 3,
        "gradient_atomics_c": 3,
        "gradient_atomics_shared_atom": 9,
        "gradient_atomics_distinct_atom": 0,
        "subgroup_rendezvous": 26,
        "orbital_local_accumulations": 0,
    }
    counters = {
        "shell_work_diagnostics_enabled": 1,
        "shell_work_pair_mode": 2,
        "shell_work_panel_0_1": 1,
        "shell_triples_visited": 1,
        "shell_triples_nonzero": 1,
        "shell_primitive_products": 8,
        "shell_public_weights_consumed": 1,
        "shell_public_weights_nonzero": 1,
        "shell_cartesian_component_products": 8,
    }
    for prefix in ("shell_000_work_", "shell_000_p2_2_2_work_"):
        counters.update({prefix + name: value for name, value in work.items()})
    return {
        "counters": counters,
        "regions": [{"name": "shell_000_packet", "gpu_ms": 0.25}],
    }


def test_work_ledger_preserves_counts_groups_and_domain_hash(
    sss_record: typing.Any,
) -> None:
    result = reduce_work(sss_record, [(0, 2, 0, 1)])
    assert result["totals"]["primitive_products"] == 8
    assert result["totals"]["convolution_iterations"] == 48
    assert result["groups"]["pure_sp"] == result["groups"]["rys_prototype"]
    assert result["groups"]["d_containing"]["work"] == {}
    assert result["classes"][0]["component_gpu_inclusive_ms"] == 0.25
    assert len(result["host_reconstruction_sha256"]) == 64
    assert result == reduce_work(copy.deepcopy(sss_record), [(0, 2, 0, 1)])


def test_rys_work_conserves_primitive_domain_and_rejects_wrong_root_count(
    sss_record: typing.Any,
) -> None:
    """Changing lowering must conserve tasks and explicitly account for its roots."""
    for prefix in ("shell_000_work_", "shell_000_p2_2_2_work_"):
        for field in (
            "boys_evaluations",
            "boys_order_sum",
            "boys_series_iterations",
            "boys_series",
            "boys_small_argument",
            "specialized_prepare_axis_calls",
            "cache_coefficient_values",
            "convolution_iterations",
        ):
            sss_record["counters"][prefix + field] = 0
        for field, value in (
            ("rys_evaluations", 8),
            ("rys_roots", 8),
            ("recurrence_states", 48),
        ):
            sss_record["counters"][prefix + field] = value
    result = reduce_work(sss_record, [(0, 2, 0, 1)])
    assert result["classes"][0]["lowering"] == "rys"
    assert result["totals"]["primitive_products"] == 8
    sss_record["counters"]["shell_000_p2_2_2_work_rys_roots"] = 16
    with pytest.raises(ValueError, match="root/recurrence"):
        reduce_work(sss_record, [(0, 2, 0, 1)])


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("shell_tasks", 2, "shell count"),
        ("active_shell_tasks", 2, "active shell"),
        ("primitive_products", 7, "primitive count"),
        ("geometry_preparations", 7, "geometry/Boys"),
        ("boys_order_sum", 16, "Boys order"),
        ("boys_large_argument", 1, "branch counts"),
        ("boys_small_argument", 9, "subdomain"),
        ("boys_series_iterations", 0, "series iterations"),
        ("gradient_atomics_shared_atom", 6, "physical-atom"),
        ("axis_polynomial_calls", 24, "generated work model"),
        ("folding_shared_atomics", -1, "nonnegative"),
        ("folding_shared_atomics", 1.5, "nonnegative"),
    ],
)
def test_reject_tampered_scientific_counts(
    sss_record: typing.Any, field: typing.Any, value: typing.Any, message: typing.Any
) -> None:
    for prefix in ("shell_000_work_", "shell_000_p2_2_2_work_"):
        sss_record["counters"][prefix + field] = value
    with pytest.raises(ValueError, match=message):
        reduce_work(sss_record, [(0, 2, 0, 1)])


def test_reject_missing_or_disagreeing_diagnostic_rows(
    sss_record: typing.Any,
) -> None:
    incomplete = copy.deepcopy(sss_record)
    del incomplete["counters"]["shell_000_p2_2_2_work_folding_direct_stores"]
    with pytest.raises(ValueError, match="incomplete"):
        reduce_work(incomplete, [(0, 2, 0, 1)])
    sss_record["counters"]["shell_000_work_folding_shared_atomics"] += 1
    with pytest.raises(ValueError, match="signature/class"):
        reduce_work(sss_record, [(0, 2, 0, 1)])


def test_reject_diagnostic_domain_different_from_host(
    sss_record: typing.Any,
) -> None:
    with pytest.raises(ValueError, match="signature domain"):
        reduce_work(sss_record, [(0, 3, 0, 1)])
    sss_record["counters"]["shell_primitive_products"] = 7
    with pytest.raises(ValueError, match="detailed/existing"):
        reduce_work(sss_record, [(0, 2, 0, 1)])


def test_nsight_activity_must_match_class_launch_domain(
    tmp_path: typing.Any, sss_record: typing.Any
) -> None:
    path = tmp_path / "profile.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds(id INTEGER, value TEXT)")
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
            "demangledName INTEGER, start INTEGER, end INTEGER, registersPerThread INTEGER, "
            "staticSharedMemory INTEGER, dynamicSharedMemory INTEGER, "
            "localMemoryPerThread INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER)"
        )
        db.execute(
            "INSERT INTO StringIds VALUES(1, ?)",
            (
                (
                    "void vibeqc::scf::shell_packet<(unsigned int)0, (unsigned int)0, "
                    "(unsigned int)0, (unsigned int)2>(...)"
                ),
            ),
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1, 10, 100010, 64, 512, 0, 0, 128, 1, 1)"
        )
    observed = kernel_activity(path, sss_record)
    assert observed[(0, 0, 0)]["gpu_ms"] == 0.1
    assert observed[(0, 0, 0)]["resources"][0]["block_threads"] == 128
    assert observed[(0, 0, 0)]["schedule"] == {
        "variant": 2,
        "component_lanes": 4,
        "shell_tasks_per_block": 32,
    }
    assert observed[(0, 0, 0)]["gpu_ms"] != sss_record["regions"][0]["gpu_ms"]
    sss_record["regions"].append(copy.deepcopy(sss_record["regions"][0]))
    with pytest.raises(ValueError, match="launch counts"):
        kernel_activity(path, sss_record)


@pytest.mark.parametrize("pair_mode", (0, 1, 2))
def test_unequal_auxiliary_f_shell_domain_and_partial_panels(
    pair_mode: typing.Any,
) -> None:
    orbital = [(0, 2, 0, 1), (1, 1, 1, 3), (0, 1, 4, 1)]
    auxiliary = [(0, 1, 0, 1), (3, 2, 1, 7), (2, 1, 8, 5)]
    panels = [(0, 4, 1), (4, 9, 2)]
    expected = Counter()
    for ia, ib, ic in product(
        range(len(orbital)), range(len(orbital)), range(len(auxiliary))
    ):
        a, b, c = orbital[ia], orbital[ib], auxiliary[ic]
        if pair_mode and (a[:2], ia) < (b[:2], ib):
            continue
        for begin, count, repeats in panels:
            if any(begin <= ao < begin + count for ao in range(c[2], c[2] + c[3])):
                expected[(a[0], b[0], c[0], a[1], b[1], c[1])] += repeats
    assert reconstruct_domain(orbital, panels, pair_mode, auxiliary) == expected
    assert reconstruct_domain(orbital, panels, pair_mode) != expected


@pytest.mark.parametrize("states,valid", ((48, True), (47, False), (49, False)))
def test_shared_recurrence_work_is_counted_once_per_primitive(
    sss_record: typing.Any,
    monkeypatch: typing.Any,
    states: typing.Any,
    valid: typing.Any,
) -> None:
    from vibeqc_compiler.integral.df_rys_shell import shell_rys_work_model

    # A toy lowering moves the six SSS component states into a shared cache;
    # unchanged total work must not be compared with the now-zero component work.
    model = shell_rys_work_model((0, 0, 0))
    model.update(shared_recurrence_states=6, component_recurrence_states=[0])
    monkeypatch.setattr(
        "benchmarks.df_shell_work_ledger.shell_rys_work_model", lambda angular: model
    )
    for prefix in ("shell_000_work_", "shell_000_p2_2_2_work_"):
        for field in (
            "boys_evaluations",
            "boys_order_sum",
            "boys_series_iterations",
            "boys_series",
            "boys_small_argument",
            "specialized_prepare_axis_calls",
            "cache_coefficient_values",
            "convolution_iterations",
        ):
            sss_record["counters"][prefix + field] = 0
        for field, value in (
            ("rys_evaluations", 8),
            ("rys_roots", 8),
            ("recurrence_states", states),
        ):
            sss_record["counters"][prefix + field] = value
    if valid:
        assert (
            reduce_work(sss_record, [(0, 2, 0, 1)])["totals"]["recurrence_states"] == 48
        )
    else:
        with pytest.raises(ValueError, match="root/recurrence"):
            reduce_work(sss_record, [(0, 2, 0, 1)])


@pytest.mark.parametrize("primitive_delta", (-1, 0, 1))
def test_angular_group_zero_signature_is_a_sentinel(
    sss_record: typing.Any, primitive_delta: typing.Any
) -> None:
    counters = sss_record["counters"]
    counters["shell_primitive_signature_policy"] = 0
    for key in list(counters):
        if "_p2_2_2_" in key:
            counters[key.replace("_p2_2_2_", "_p0_0_0_")] = counters.pop(key)
    counters["shell_000_p0_0_0_work_primitive_products"] += primitive_delta
    if primitive_delta:
        with pytest.raises(ValueError, match="primitive count differs"):
            reduce_work(sss_record, [(0, 2, 0, 1)])
    else:
        assert (
            reduce_work(sss_record, [(0, 2, 0, 1)])["totals"]["primitive_products"] == 8
        )


@pytest.mark.parametrize("policy", (0, 1, 2))
def test_sparse_angular_work_bounds_use_host_primitive_costs(
    policy: typing.Any,
) -> None:
    from benchmarks.df_shell_work_ledger import (
        _active_primitive_bounds,
        _primitive_work_domains,
    )

    expected = Counter({(1, 0, 3, 1, 2, 3): 2, (1, 0, 3, 4, 2, 3): 3})
    domains = _primitive_work_domains(expected, policy)
    if policy == 0:
        costs = domains[(1, 0, 3, 0, 0, 0)]
        assert costs == {6: 2, 24: 3}
        assert _active_primitive_bounds(costs, 0) == (0, 0)
        assert _active_primitive_bounds(costs, 1) == (6, 24)
        assert _active_primitive_bounds(costs, 3) == (36, 72)
        assert _active_primitive_bounds(costs, 5) == (84, 84)
        with pytest.raises(ValueError, match="active shell count"):
            _active_primitive_bounds(costs, 6)
    else:
        assert set(domains) == set(expected)
        for key, costs in domains.items():
            assert _active_primitive_bounds(costs, 1) == (math.prod(key[3:]),) * 2
