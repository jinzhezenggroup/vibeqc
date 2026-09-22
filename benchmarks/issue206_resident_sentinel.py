"""Structural performance invariants for the #206 DF response matrix.

The endpoint timer cannot reveal that a supposedly resident response silently
fell back to one host gather and one ``N^2`` H2D upload per auxiliary panel.
This module validates the opt-in component trace that accompanies an endpoint,
records the selected route, and keeps the resource bounds tied to ``nbf`` and
``naux``.  It intentionally does not estimate wall time and never changes the
scientific execution path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

BYTES_PER_DOUBLE = 8
POLICIES = ("auto", "resident", "streamed", "fallback")


def _counter(counters: Mapping[str, Any], name: str) -> int:
    """Return a non-negative integer counter and reject malformed traces."""

    value = counters.get(name, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"trace counter {name!r} must be a non-negative integer")
    return value


def _shape(record: Mapping[str, Any]) -> tuple[int, int]:
    """Read the per-operation AO/auxiliary shape used by all byte bounds."""

    try:
        nbf, naux = record["nbf"], record["naux"]
    except KeyError as error:
        raise ValueError("response trace is missing its AO/auxiliary shape") from error
    if type(nbf) is not int or type(naux) is not int or min(nbf, naux) <= 0:
        raise ValueError("response trace shape must contain positive integer nbf/naux")
    return nbf, naux


def selected_response_policy(record: Mapping[str, Any]) -> dict[str, Any]:
    """Classify the actual response route from executed trace counters.

    Environment variables describe a request, not what ran.  Borrowed scratch,
    borrowed whitened factors, source regeneration, and fallback copies are
    therefore classified from completed counters and the trace shape itself.
    The force bridge's top-level ``streamed`` flag describes its provider
    protocol and is not, by itself, response storage evidence; the explicit
    response counters decide the residency label.  The result is deliberately
    descriptive; callers decide whether a route is acceptable for a particular
    benchmark row.
    """

    nbf, naux = _shape(record)
    counters = record.get("counters", {})
    if not isinstance(counters, Mapping):
        raise TypeError("response trace is missing its counters")
    borrowed_jk = _counter(counters, "response_borrowed_jk_bytes")
    borrowed_whitened = _counter(counters, "response_borrowed_whitened_bytes")
    raw_upload = _counter(counters, "raw_value_upload_bytes")
    raw_gather = _counter(counters, "raw_panel_host_gather_elements")
    streamed_fitting_passes = _counter(counters, "response_streamed_fitting_raw_passes")
    streamed_occupied_passes = _counter(
        counters, "response_streamed_occupied_raw_passes"
    )
    streamed_passes = streamed_fitting_passes + streamed_occupied_passes
    source_backed = record.get("source_backed") is True

    if borrowed_jk:
        storage = "resident-jk-scratch"
    elif borrowed_whitened:
        storage = "resident-whitened"
    elif source_backed or streamed_passes:
        storage = "streamed-source"
    elif raw_gather or raw_upload:
        storage = "host-panel-fallback"
    else:
        storage = "bounded-panel"

    occupied = any(
        _counter(counters, name)
        for name in (
            "response_occupied_rank",
            "response_occupied_projected_elements",
            "response_occupied_projection_products",
        )
    )
    if occupied:
        whitening = "occupied-factor"
    elif any(
        _counter(counters, name)
        for name in (
            "response_inverse_applied_factor_elements",
            "response_full_rank_factor_first",
            "response_full_rank_single_tensor",
            "response_full_rank_bounded_factor_first",
        )
    ):
        whitening = "inverse-first"
    elif streamed_passes:
        whitening = "streamed-inverse-first"
    else:
        whitening = "unspecified"

    residency = (
        "resident"
        if storage.startswith("resident-")
        else "streamed"
        if storage == "streamed-source"
        else "fallback"
    )
    return {
        "storage": storage,
        "residency": residency,
        "exchange": "occupied" if occupied else "dense",
        "whitening": whitening,
        "value_provider": "integral-source" if source_backed else "host-raw",
        "response_streamed_raw_passes": streamed_passes,
        "source_backed": source_backed,
        "nbf": nbf,
        "naux": naux,
    }


def _transformed_tile_productions(record: Mapping[str, Any]) -> int:
    """Count transformed logical tile productions retained by the trace."""

    tiles = record.get("tiles", [])
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
        raise TypeError("response trace tiles must be a sequence")
    total = 0
    for tile in tiles:
        if not isinstance(tile, Mapping):
            raise TypeError("response trace contains a malformed logical tile")
        if tile.get("transformed") is True:
            productions = tile.get("productions")
            if type(productions) is not int or productions < 1:
                raise ValueError(
                    "transformed tile productions must be positive integers"
                )
            total += productions
    return total


def validate_response_record(
    record: Mapping[str, Any],
    *,
    expected_policy: str = "auto",
    max_h2d_bytes: int | None = None,
    max_d2h_bytes: int | None = None,
    max_transformed_tile_productions: int | None = None,
) -> dict[str, Any]:
    """Validate one force-response trace and return its auditable invariants.

    ``expected_policy="resident"`` is the strict #206 closure-matrix mode.
    It accepts one whole-tensor upload when a resident plan must stage host raw
    values, but rejects repeated per-Q ``N^2`` uploads, host gathers, source
    rereads, and streamed fallback work.  Optional transfer/tile ceilings are
    explicit caller-owned acceptance controls, so a constrained-memory row can
    use a different declared ceiling without changing the validator.
    """

    if expected_policy not in POLICIES:
        raise ValueError(f"unknown expected response policy {expected_policy!r}")
    if record.get("operation") != "force_response":
        raise ValueError("resident sentinel requires a force_response trace record")
    nbf, naux = _shape(record)
    counters = record.get("counters", {})
    if not isinstance(counters, Mapping):
        raise TypeError("force_response trace is missing counters")
    policy = selected_response_policy(record)
    matrix_bytes = nbf * nbf * BYTES_PER_DOUBLE
    full_raw_bytes = matrix_bytes * naux
    raw_upload_bytes = _counter(counters, "raw_value_upload_bytes")
    raw_bulk_uploads = _counter(counters, "raw_value_bulk_uploads")
    raw_gather_elements = _counter(counters, "raw_panel_host_gather_elements")
    streamed_raw_passes = _counter(
        counters, "response_streamed_fitting_raw_passes"
    ) + _counter(counters, "response_streamed_occupied_raw_passes")
    if raw_upload_bytes % matrix_bytes:
        raise RuntimeError("raw value upload is not an integral number of N^2 slices")
    raw_slices = raw_upload_bytes // matrix_bytes

    if expected_policy == "resident" and policy["residency"] != "resident":
        raise RuntimeError(
            "resident response sentinel selected a non-resident route: "
            f"{policy['storage']}"
        )
    if expected_policy == "streamed" and policy["residency"] != "streamed":
        raise RuntimeError(
            "streamed response sentinel selected a different route: "
            f"{policy['storage']}"
        )
    if expected_policy == "fallback" and policy["residency"] != "fallback":
        raise RuntimeError(
            "fallback response sentinel selected a different route: "
            f"{policy['storage']}"
        )

    if expected_policy == "resident":
        if raw_gather_elements:
            raise RuntimeError("resident response performed a host raw-panel gather")
        if streamed_raw_passes:
            raise RuntimeError(
                "resident response regenerated raw values in streamed panels"
            )
        if raw_upload_bytes > full_raw_bytes:
            raise RuntimeError(
                "resident response exceeded one full N^2*Naux raw upload: "
                f"{raw_upload_bytes} > {full_raw_bytes}"
            )
        if raw_upload_bytes and raw_bulk_uploads != 1:
            raise RuntimeError(
                "resident response used slice uploads instead of one declared bulk upload"
            )
        if record.get("source_backed") is True and raw_upload_bytes:
            raise RuntimeError(
                "source-backed resident response uploaded host raw values"
            )

    panel_count = _counter(counters, "response_auxiliary_blocks")
    resident_tile = _counter(counters, "response_resident_auxiliary_tile")
    panel_lower_bound = 1
    panel_upper_bound = naux
    if resident_tile:
        if resident_tile > naux:
            raise RuntimeError("resident auxiliary tile exceeds naux")
        # A response consumer may shorten a panel to an auxiliary-shell
        # boundary.  Therefore ceil(naux/tile) is a lower bound, while one
        # panel per auxiliary function is the shape-only upper bound.
        panel_lower_bound = (naux + resident_tile - 1) // resident_tile
    if panel_count:
        if not panel_lower_bound <= panel_count <= panel_upper_bound:
            raise RuntimeError(
                "response panel count is outside its auxiliary-tile bounds: "
                f"{panel_count} not in [{panel_lower_bound}, {panel_upper_bound}]"
            )
    elif expected_policy == "resident":
        raise RuntimeError("resident response trace omitted its auxiliary panel count")

    transformed_tiles = _transformed_tile_productions(record)
    if (
        max_transformed_tile_productions is not None
        and transformed_tiles > max_transformed_tile_productions
    ):
        raise RuntimeError(
            "transformed-tile productions exceed the declared bound: "
            f"{transformed_tiles} > {max_transformed_tile_productions}"
        )

    h2d_bytes = _counter(counters, "host_to_device_bytes")
    d2h_bytes = _counter(counters, "device_to_host_bytes")
    if max_h2d_bytes is not None and h2d_bytes > max_h2d_bytes:
        raise RuntimeError(
            f"H2D bytes exceed the declared resident ceiling: {h2d_bytes} > {max_h2d_bytes}"
        )
    if max_d2h_bytes is not None and d2h_bytes > max_d2h_bytes:
        raise RuntimeError(
            f"D2H bytes exceed the declared resident ceiling: {d2h_bytes} > {max_d2h_bytes}"
        )

    return {
        "policy": policy,
        "raw_value_upload_bytes": raw_upload_bytes,
        "raw_value_upload_ceiling_bytes": full_raw_bytes,
        "raw_value_slices": raw_slices,
        "raw_value_bulk_uploads": raw_bulk_uploads,
        "raw_panel_host_gather_elements": raw_gather_elements,
        "response_auxiliary_blocks": panel_count,
        "response_auxiliary_blocks_lower_bound": panel_lower_bound,
        "response_auxiliary_blocks_upper_bound": panel_upper_bound,
        "response_resident_auxiliary_tile": resident_tile,
        "transformed_tile_productions": transformed_tiles,
        "host_to_device_bytes": h2d_bytes,
        "device_to_host_bytes": d2h_bytes,
        "declared_h2d_ceiling_bytes": max_h2d_bytes,
        "declared_d2h_ceiling_bytes": max_d2h_bytes,
    }


def validate_trace_file(
    path: Path,
    *,
    expected_policy: str = "auto",
    max_h2d_bytes: int | None = None,
    max_d2h_bytes: int | None = None,
    max_transformed_tile_productions: int | None = None,
) -> dict[str, Any]:
    """Validate every force-response record in a fresh component trace."""

    from benchmarks.df_component_ledger import read_trace

    records = read_trace(path)
    responses = [
        record for record in records if record["operation"] == "force_response"
    ]
    if not responses:
        raise ValueError("trace contains no force_response record")
    validated = [
        validate_response_record(
            response,
            expected_policy=expected_policy,
            max_h2d_bytes=max_h2d_bytes,
            max_d2h_bytes=max_d2h_bytes,
            max_transformed_tile_productions=max_transformed_tile_productions,
        )
        for response in responses
    ]
    return {
        "trace": str(path.resolve()),
        "expected_policy": expected_policy,
        "records": [
            {"id": response["id"], "invariants": result}
            for response, result in zip(responses, validated, strict=True)
        ],
    }


def main() -> None:
    """Validate one or more trace files without requiring a CUDA device."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="+")
    parser.add_argument("--expected-policy", choices=POLICIES, default="auto")
    parser.add_argument("--max-h2d-bytes", type=int)
    parser.add_argument("--max-d2h-bytes", type=int)
    parser.add_argument("--max-transformed-tile-productions", type=int)
    parser.add_argument("--output", type=raw_output_path)
    args = parser.parse_args()
    for name in (
        "max_h2d_bytes",
        "max_d2h_bytes",
        "max_transformed_tile_productions",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    payload = {
        "schema": "vibeqc.issue206.resident_sentinel",
        "version": 1,
        "expected_policy": args.expected_policy,
        "traces": [
            validate_trace_file(
                path,
                expected_policy=args.expected_policy,
                max_h2d_bytes=args.max_h2d_bytes,
                max_d2h_bytes=args.max_d2h_bytes,
                max_transformed_tile_productions=args.max_transformed_tile_productions,
            )
            for path in args.trace
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
