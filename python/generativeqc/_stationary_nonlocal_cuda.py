"""Compose resident semilocal and nonlocal geometry consumers on one stream.

This module owns scheduling only. AO/XC pullbacks and VV10 mathematics remain
in their existing generated/native owners. The final SCF rho/grad-rho are
seeded D2D into the resident VV10 owner, so both geometry consumers share one
bounded AO/grid collocation pass without retaining O(N*AO) jets. If that
optional binding is unavailable, two bounded passes collect device features
and consume device seeds without exporting either array to the host.
"""

from __future__ import annotations

import typing
from time import perf_counter

if typing.TYPE_CHECKING:
    import numpy as np


def resident_nonlocal_geometry(
    *,
    grid: typing.Any,
    sources: typing.Any,
    nonlocal_sources: typing.Any,
    nonlocal_owner: typing.Any,
    state: typing.Any,
    raw_weights: typing.Any,
    tile_points: int,
    ao_count: int,
    functional: int,
    ingredients: tuple[str, ...],
    ao_maps: typing.Any = None,
    ao_domain: typing.Any = None,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, typing.Any]]:
    """Prefer live SCF features, retaining the bounded device collection fallback.

    The caller has reset both stationary accumulators and enqueued the nuclear
    term only on the semilocal source. The first grid lease supplies the shared
    CUDA stream used to D2D-seed the full-grid resident VV10 owner from the exact
    current KS snapshot. Pair seeds are then available before that same lease
    reaches the nonlocal geometry consumer. A capability miss instead collects
    features alongside semilocal geometry and consumes pairs in a second pass.
    Token, device, allocation and numerical errors never select that fallback.
    No AO jet survives its lease; failed calls publish no component dictionary.
    """
    if type(tile_points) is not int or tile_points <= 0:
        raise ValueError("resident nonlocal tile_points must be a positive integer")
    if not callable(
        getattr(
            nonlocal_sources,
            "geometry_external_device_molecular_resident_weights",
            None,
        )
    ):
        raise TypeError("nonlocal stationary owner lacks the resident seed consumer")
    if not callable(getattr(nonlocal_owner, "seed_from_snapshot", None)):
        raise TypeError("resident nonlocal owner lacks the final-state feature handoff")
    points = state.grid.points
    count = len(points)
    if count == 0 or nonlocal_owner.point_count != count:
        raise ValueError("resident nonlocal owner/grid point count differs")
    resident_grid = state._source.cuda_resident_grid()
    if resident_grid is None:
        raise NotImplementedError(
            "resident nonlocal geometry requires the CUDA molecular-grid lease"
        )
    if resident_grid.device != grid.device_id or resident_grid.point_count != count:
        raise ValueError("resident molecular-grid lease differs from stationary grid")
    if count % sources.natom:
        raise ValueError("molecular grid point count is not atom-major uniform")
    points_per_atom = count // sources.natom
    if len(raw_weights) != count:
        raise ValueError("resident nonlocal raw quadrature size differs")
    if type(ao_count) is not int or ao_count <= 0:
        raise ValueError("resident nonlocal AO count must be a positive integer")

    if (ao_maps is None) != (ao_domain is None):
        raise ValueError("resident AO cache and domain must be supplied together")
    if ao_maps is not None:
        if (
            ao_domain.point_pointer != resident_grid.points
            or ao_domain.point_count != count
            or ao_domain.device != resident_grid.device
            or ao_domain.grid_identity != state.grid.identity
            or ao_domain.geometry_identity != state.identity.geometry_identity
        ):
            raise ValueError("resident AO domain differs from the current grid lease")
        ao_maps.reset_work()

    began = perf_counter()
    diagnostic = nonlocal_owner.diagnostic()
    if diagnostic.executed:
        nonlocal_owner.reset()
    elif diagnostic.collected_points:
        raise ValueError(
            "resident nonlocal owner contains an incomplete previous collection"
        )
    reset_seconds = perf_counter() - began

    pass_began = perf_counter()
    seed_seconds = 0.0
    pair_seconds = 0.0
    seeds = None
    local = None
    snapshot_seeded = True
    for phase in range(2):
        if phase == 1 and snapshot_seeded:
            break
        for begin in range(0, count, tile_points):
            end = min(begin + tile_points, count)
            point_pointer = resident_grid.points + 3 * begin * 8
            ao_ids = (
                None
                if ao_maps is None
                else ao_maps.select(grid, ao_domain, begin, end - begin)
            )
            with grid.feature_task_device_points(
                point_pointer,
                end - begin,
                ao_ids,
                ingredients,
            ) as task:
                weights = state.grid.weights[begin:end]
                device_weights = resident_grid.weights + begin * 8
                device_raw = resident_grid.atomic_weights + begin * 8
                host_raw = (
                    raw_weights[begin:end]
                    if getattr(sources, "profile_device", False)
                    or getattr(nonlocal_sources, "profile_device", False)
                    else None
                )
                if phase == 0:
                    sources.geometry_molecular_resident_weights(
                        task,
                        begin,
                        points_per_atom,
                        device_weights,
                        weights,
                        device_raw,
                        host_raw,
                        functional=functional,
                    )
                    if begin == 0:
                        began = perf_counter()
                        try:
                            nonlocal_owner.seed_from_snapshot(state._source, task)
                        except NotImplementedError:
                            # The native bridge checks the exact token before
                            # returning a capability miss, without seeding or
                            # enqueueing copies. Do not catch execution errors.
                            diagnostic = nonlocal_owner.diagnostic()
                            if diagnostic.executed or diagnostic.collected_points:
                                raise RuntimeError(
                                    "feature capability miss modified "
                                    "the nonlocal owner"
                                ) from None
                            snapshot_seeded = False
                        seed_seconds = perf_counter() - began
                    if not snapshot_seeded:
                        nonlocal_owner.collect(task, begin)
                        continue
                if seeds is None:
                    began = perf_counter()
                    seeds = nonlocal_owner.execute()
                    pair_seconds = perf_counter() - began
                    if (
                        not seeds.pointer
                        or seeds.stride != count
                        or not seeds.stream
                        or seeds.generation <= 0
                    ):
                        raise ValueError(
                            "resident nonlocal producer returned an invalid seed lease"
                        )
                if task.view.stream != seeds.stream:
                    raise ValueError(
                        "resident nonlocal seed and geometry streams differ"
                    )
                # Inactive MolecularV1 rows have all six seeds zeroed by the
                # native producer, so no host active mask is needed.
                nonlocal_sources.geometry_external_device_molecular_resident_weights(
                    task,
                    begin,
                    points_per_atom,
                    device_weights,
                    weights,
                    device_raw,
                    host_raw,
                    seeds.pointer,
                    seeds.stride,
                    begin,
                )
        if phase == 0:
            # Preserve individual source arrays/canonical final summation while
            # avoiding D2H publication of unused one-electron/Coulomb slots.
            local = sources.finish_span(
                ("xc_ao", "xc_grid", "xc_weight", "overlap_pulay", "nuclear")
            )

    if local is None or seeds is None:
        raise RuntimeError("resident nonlocal geometry did not complete")
    nonlocal_parts = nonlocal_sources.finish_span(("xc_ao", "xc_grid", "xc_weight"))
    components = {
        name: local[name] for name in ("xc_ao", "xc_grid", "xc_weight", "nuclear")
    }
    for suffix in ("ao", "grid", "weight"):
        components["nonlocal_" + suffix] = nonlocal_parts["xc_" + suffix]

    # These host-wall intervals are additive: the enclosing pass includes the
    # separately reported seed and pair submission intervals. Keep asynchronous
    # pair execution/drain in the shared tail, not in the enqueue measurements.
    geometry_and_drain_seconds = (
        perf_counter() - pass_began - seed_seconds - pair_seconds
    )
    geometry_passes = 1 if snapshot_seeded else 2
    geometry_timing = (
        "single_pass_geometry_and_pair_drain"
        if snapshot_seeded
        else "two_pass_geometry_and_pair_drain"
    )
    seconds = {
        "nonlocal_reset": reset_seconds,
        "resident_feature_seed_enqueue": seed_seconds,
        "vv10_pair_enqueue": pair_seconds,
        geometry_timing: geometry_and_drain_seconds,
    }
    work = {
        "nonlocal_execution": "resident-full-grid-device-seeds",
        "nonlocal_feature_source": (
            "exact-final-scf-device-binding"
            if snapshot_seeded
            else "bounded-grid-feature-collection"
        ),
        # D2D copy enqueues are distinct from the fallback collection kernel.
        "nonlocal_feature_d2d_bytes": 4 * count * 8 if snapshot_seeded else 0,
        "nonlocal_feature_collection_point_visits": 0 if snapshot_seeded else count,
        "nonlocal_feature_d2h_bytes": 0,
        "nonlocal_seed_h2d_bytes": 0,
        "grid_owner_source": "implicit-atom-major-index",
        "grid_owner_h2d_bytes": 0,
        "grid_point_source": "exact-native-resident-grid",
        "grid_point_h2d_bytes": 0,
        "grid_weight_source": "exact-native-resident-grid",
        "grid_weight_h2d_bytes": 0,
        "grid_atomic_measure_source": "exact-native-resident-grid",
        "grid_atomic_measure_h2d_bytes": 0,
        "nonlocal_dense_pair_capacity": count * count,
        "nonlocal_seed_generation": seeds.generation,
        "nonlocal_active_count_scope": "device-only; not measured by host scheduler",
        "ao_collocation_point_visits": geometry_passes * count,
        "geometry_point_visits": 2 * count,
        "stationary_source_d2h_policy": "contiguous-live-spans",
        "semilocal_stationary_source_d2h_bytes": 5 * 3 * sources.natom * 8,
        "nonlocal_stationary_source_d2h_bytes": 3 * 3 * sources.natom * 8,
        "stationary_source_full_arena_d2h_bytes_avoided": 6 * 3 * sources.natom * 8,
    }
    if ao_maps is not None:
        work["active_ao_maps"] = ao_maps.work
    # Detailed profiling deliberately uses the retained explicit-owner / host
    # weight route. Each geometry consumer visits the complete grid exactly once,
    # including the two-pass feature fallback; count transfers per consumer, not
    # per collocation pass. Points and nonlocal seeds remain resident either way.
    profiled = sum(
        bool(getattr(owner, "profile_device", False))
        for owner in (sources, nonlocal_sources)
    )
    if profiled:
        work["grid_owner_source"] = (
            "profile-host-explicit-owners"
            if profiled == 2
            else "mixed-implicit-and-profile-host-owners"
        )
        work["grid_weight_source"] = (
            "profile-host-partition-weights"
            if profiled == 2
            else "mixed-resident-and-profile-host-weights"
        )
        work["grid_owner_h2d_bytes"] = profiled * count * 8
        work["grid_weight_h2d_bytes"] = profiled * count * 8
        if "grid_atomic_measure_source" in work:
            work["grid_atomic_measure_source"] = (
                "profile-host-atomic-measures"
                if profiled == 2
                else "mixed-resident-and-profile-host-atomic-measures"
            )
            work["grid_atomic_measure_h2d_bytes"] = profiled * count * 8
    return components, seconds, work
