"""Hardware-free integrity and accounting gates for component evidence."""

import copy
import json

import pytest

from benchmarks.df_component_ledger import (
    aggregate,
    force_attribution,
    read_trace,
    validate_record,
)


@pytest.fixture
def record():
    """One root with a nested generation region and a disjoint contraction."""
    return {
        "schema": "vibeqc.df_trace",
        "version": 1,
        "id": 0,
        "operation": "ri_j",
        "execution": "stream",
        "valid": True,
        "cuda_error": 0,
        "nvtx": True,
        "systems": 1,
        "system_offset": 0,
        "nbf": 3,
        "naux": 5,
        "source_backed": True,
        "streamed": True,
        "final_synchronization_ms": 7.0,
        "host_completion_ms": 17.0,
        "profiler_event_count": 8,
        "dropped_regions": 0,
        "dropped_tiles": 0,
        "regions": [
            {"name": "ri_j", "parent": -1, "host_ms": 10.0, "gpu_ms": 20.0},
            {"name": "charge", "parent": 0, "host_ms": 6.0, "gpu_ms": 12.0},
            {"name": "generation", "parent": 1, "host_ms": 4.0, "gpu_ms": 8.0},
            {"name": "output", "parent": 0, "host_ms": 3.0, "gpu_ms": 5.0},
        ],
        "counters": {"transformed_tile_productions": 2, "transformed_value_bytes": 96},
        "tiles": [
            {
                "system": 0,
                "pair_begin": 0,
                "pair_count": 3,
                "auxiliary_begin": 0,
                "auxiliary_count": 2,
                "derivative_coordinate": -1,
                "transformed": True,
                "productions": 2,
            }
        ],
    }


def test_exclusive_times_conserve_root_without_double_counting(record):
    summary = aggregate([record])["groups"][0]
    assert summary["gpu_exclusive_ms"] == {
        "ri_j": 3,
        "charge": 4,
        "generation": 8,
        "output": 5,
    }
    assert sum(summary["gpu_exclusive_ms"].values()) == summary["gpu_inclusive_ms"]
    assert (
        sum(summary["host_exclusive_ms"].values()) + summary["final_synchronization_ms"]
        == 17
    )
    assert summary["repeated_tile_productions"] == 1


def test_capture_counts_stay_separate_from_executed_work(record):
    capture = copy.deepcopy(record)
    capture.update(execution="graph_capture", profiler_event_count=0)
    for region in capture["regions"]:
        region["gpu_ms"] = None
    executed, constructed = aggregate([record, capture])["groups"]
    assert executed["calls"] == constructed["calls"] == 1
    assert constructed["gpu_inclusive_ms"] is None
    assert constructed["gpu_exclusive_ms"] == {}
    assert executed["counter_sums"]["transformed_tile_productions"] == 2
    assert constructed["counter_sums"]["transformed_tile_productions"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("valid", False),
        ("version", 8),
        ("cuda_error", 1),
        ("dropped_regions", 1),
        ("dropped_tiles", 1),
        ("profiler_event_count", 6),
        ("execution", "graph_replay"),
        ("host_completion_ms", 0),
        ("final_synchronization_ms", float("nan")),
        ("nbf", True),
        ("regions", []),
    ],
)
def test_reject_invalid_or_incomplete_records(record, field, value):
    record[field] = value
    with pytest.raises(ValueError):
        validate_record(record)


@pytest.mark.parametrize("parent", [-1, 2, 3, 99])
def test_reject_invalid_hierarchy(record, parent):
    record["regions"][2]["parent"] = parent
    with pytest.raises(ValueError, match="parent"):
        validate_record(record)


def test_reject_closed_parent_and_parent_timing_overrun(record):
    record["regions"].append({"name": "late", "parent": 1, "host_ms": 0, "gpu_ms": 0})
    record["profiler_event_count"] += 2
    with pytest.raises(ValueError, match="hierarchy"):
        validate_record(record)
    record["regions"].pop()
    record["profiler_event_count"] -= 2
    record["regions"][2]["gpu_ms"] = 15
    with pytest.raises(ValueError, match="children exceed"):
        validate_record(record)


@pytest.mark.parametrize(
    "field,value", [("gpu_ms", None), ("gpu_ms", float("inf")), ("host_ms", -1)]
)
def test_reject_missing_or_nonfinite_timing(record, field, value):
    record["regions"][1][field] = value
    with pytest.raises(ValueError):
        validate_record(record)


def test_reject_capture_with_execution_timings(record):
    record.update(execution="graph_capture", profiler_event_count=0)
    with pytest.raises(ValueError, match="capture"):
        validate_record(record)


def test_reject_counter_disagreement_and_duplicate_tiles(record):
    record["counters"]["transformed_value_bytes"] = 48
    with pytest.raises(ValueError, match="totals disagree"):
        validate_record(record)
    record["counters"]["transformed_value_bytes"] = 96
    record["tiles"].append(record["tiles"][0].copy())
    with pytest.raises(ValueError, match="duplicate logical tile"):
        validate_record(record)


@pytest.mark.parametrize(
    "field,value",
    [("system", 1), ("pair_count", 10), ("auxiliary_begin", 4), ("productions", 0)],
)
def test_reject_out_of_bounds_or_empty_tile(record, field, value):
    record["tiles"][0][field] = value
    with pytest.raises(ValueError):
        validate_record(record)


def test_raw_trace_requires_complete_records_and_unique_ids(record, tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="incomplete"):
        read_trace(path)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="incomplete"):
        read_trace(path)
    path.write_text(json.dumps(record) + "\n")
    assert read_trace(path) == [record]
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="duplicate operation ID"):
        read_trace(path)


def test_scratch_sum_is_not_peak_memory(record):
    record["counters"]["response_scratch_bytes"] = 80
    summary = aggregate([record, record])["groups"][0]
    assert summary["counter_sums"]["response_scratch_bytes"] == 160
    assert summary["counter_maxima"]["response_scratch_bytes"] == 80


def test_batch_member_uses_absolute_source_index(record):
    record["system_offset"] = 3
    record["tiles"][0]["system"] = 3
    validate_record(record)


def test_force_attribution_retains_unclassified_time(record):
    records = []
    for operation in ("force_response", "one_electron_response"):
        root = copy.deepcopy(record)
        root["operation"] = root["regions"][0]["name"] = operation
        records.append(root)
    force = {"seconds": 1.04, "components": aggregate(records)}
    result = force_attribution({"seconds": 1}, force)
    assert result["named_host_and_synchronization_ms"] == 32
    assert result["force_operations_host_completion_ms"] == 34
    assert result["unclassified_inside_operations_ms"] == 2
    assert result["outside_operations_delta_ms"] == pytest.approx(6)
    assert result["named_fraction_of_increment"] == pytest.approx(0.8)
