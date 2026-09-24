"""Hardware-free structural gates for the #206 resident response sentinel."""

from __future__ import annotations

import copy

import pytest

from benchmarks.issue206_resident_sentinel import validate_response_record


def resident_record() -> dict:
    """Return a small, complete resident-JK trace row for protocol tests."""

    nbf, naux = 4, 5
    return {
        "operation": "force_response",
        "nbf": nbf,
        "naux": naux,
        "source_backed": False,
        "streamed": False,
        "counters": {
            "response_borrowed_jk_bytes": 3 * nbf * nbf * naux * 8,
            "response_resident_auxiliary_tile": 2,
            "response_auxiliary_blocks": 3,
            "raw_value_upload_bytes": 0,
            "raw_value_bulk_uploads": 0,
            "raw_panel_host_gather_elements": 0,
            "response_streamed_fitting_raw_passes": 0,
            "host_to_device_bytes": 64,
            "device_to_host_bytes": 96,
        },
        "tiles": [],
    }


def test_resident_route_records_policy_and_dimension_bounds() -> None:
    result = validate_response_record(resident_record(), expected_policy="resident")
    assert result["policy"] == {
        "storage": "resident-jk-scratch",
        "residency": "resident",
        "exchange": "dense",
        "whitening": "unspecified",
        "value_provider": "host-raw",
        "response_streamed_raw_passes": 0,
        "source_backed": False,
        "nbf": 4,
        "naux": 5,
    }
    assert result["raw_value_upload_ceiling_bytes"] == 4 * 4 * 5 * 8
    assert result["response_auxiliary_blocks"] == 3
    assert result["response_auxiliary_blocks_lower_bound"] == 3
    assert result["response_auxiliary_blocks_upper_bound"] == 5


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("raw_panel_host_gather_elements", 16, "host raw-panel gather"),
        ("response_streamed_fitting_raw_passes", 1, "regenerated raw values"),
        ("raw_value_upload_bytes", 4 * 4 * 5 * 8 + 4 * 4 * 8, "one full"),
    ],
)
def test_resident_route_rejects_hidden_recomputation_or_gather(
    field: str, value: int, match: str
) -> None:
    record = resident_record()
    record["counters"][field] = value
    if field == "raw_value_upload_bytes":
        record["counters"]["raw_value_bulk_uploads"] = 2
    with pytest.raises(RuntimeError, match=match):
        validate_response_record(record, expected_policy="resident")


def test_resident_route_rejects_unbounded_panels_and_transformed_tiles() -> None:
    record = resident_record()
    record["counters"]["response_auxiliary_blocks"] = 2
    with pytest.raises(RuntimeError, match="panel count"):
        validate_response_record(record, expected_policy="resident")

    record = resident_record()
    # Shell-aligned consumers may exceed ceil(naux/tile), but not naux.
    record["counters"]["response_auxiliary_blocks"] = 4
    result = validate_response_record(record, expected_policy="resident")
    assert result["response_auxiliary_blocks_lower_bound"] == 3

    record = resident_record()
    record["counters"]["response_auxiliary_blocks"] = 6
    with pytest.raises(RuntimeError, match="panel count"):
        validate_response_record(record, expected_policy="resident")

    record = resident_record()
    record["tiles"] = [
        {
            "transformed": True,
            "productions": 2,
        }
    ]
    with pytest.raises(RuntimeError, match="transformed-tile"):
        validate_response_record(
            record,
            expected_policy="resident",
            max_transformed_tile_productions=1,
        )


def test_declared_transfer_ceilings_are_enforced_without_gpu() -> None:
    record = resident_record()
    with pytest.raises(RuntimeError, match="H2D bytes"):
        validate_response_record(record, max_h2d_bytes=63)
    with pytest.raises(RuntimeError, match="D2H bytes"):
        validate_response_record(record, max_d2h_bytes=95)


def test_streamed_and_fallback_routes_are_not_mislabeled_as_resident() -> None:
    streamed = resident_record()
    streamed["source_backed"] = True
    streamed["streamed"] = True
    streamed["counters"]["response_borrowed_jk_bytes"] = 0
    streamed["counters"]["response_streamed_fitting_raw_passes"] = 1
    assert validate_response_record(streamed)["policy"]["residency"] == "streamed"
    with pytest.raises(RuntimeError, match="non-resident route"):
        validate_response_record(streamed, expected_policy="resident")

    # The force bridge may publish streamed provider metadata while the
    # response itself borrows resident JK scratch; classify those dimensions
    # independently.
    resident = resident_record()
    resident["streamed"] = True
    resident_policy = validate_response_record(resident)["policy"]
    assert resident_policy["residency"] == "resident"
    assert resident_policy["value_provider"] == "host-raw"

    fallback = copy.deepcopy(resident_record())
    fallback["counters"]["response_borrowed_jk_bytes"] = 0
    fallback["counters"]["raw_value_upload_bytes"] = 4 * 4 * 5 * 8
    fallback["counters"]["raw_value_bulk_uploads"] = 1
    assert validate_response_record(fallback)["policy"]["residency"] == "fallback"
