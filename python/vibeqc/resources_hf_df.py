"""Compose native CUDA DF storage and source-regeneration alternatives.

The current inventory is deliberately limited to small HF orbital spaces and
auxiliary dimensions up to 128. Runtime-dependent solver storage has an explicit
capacity allowance and is also charged by the native allocation ledger.
"""

import json
import os
import typing
from dataclasses import asdict

from vibeqc_compiler.common.resources import (
    ResourceCandidate,
    ResourceEstimate,
    checked_bytes,
)

from .resources_df import (
    density_fitting_diis_bytes,
    density_fitting_source_bytes,
    density_fitting_tile_plan,
)


def cuda_df_candidates(
    library: typing.Any,
    items: typing.Any,
    *,
    diis_history: typing.Any,
    device: typing.Any,
    first_phase: typing.Any,
    last_phase: typing.Any,
    requested_budget: typing.Any = 0,
) -> typing.Any:
    """Return actual resident/source choices for the native fleet's buckets.

    Buckets retain all their device plans. Temporary setup/force storage is
    shared across serialized buckets; the existing one-item cold retry can
    coexist with a warm bucket and has a separate reservation. Both provider
    modes first use CUDA SCF, including graph capture for generated tiles,
    with the existing CPU numerical recovery when the device solve fails.
    The source's forward tensor may be resident when the native allowance fits;
    response derivatives still use bounded regeneration. The source candidate's
    ``recomputed`` mode describes that complete policy, not every forward tile.
    """
    storage = os.environ.get("VIBEQC_DF_VALUE_STORAGE", "auto")
    if storage not in ("auto", "dense", "packed"):
        raise ValueError("VIBEQC_DF_VALUE_STORAGE must be auto, dense or packed")
    packed_values = storage == "packed"
    pair_storage = "packed" if packed_values else "dense"
    occupied_exchange = os.environ.get("VIBEQC_DF_EXCHANGE") == "occupied"
    buckets = {}
    for item in items:
        orbital, auxiliary = item["orbital"], item["auxiliary"]
        if orbital["nbf"] > 16 or auxiliary["nbf"] > 128:
            raise NotImplementedError(
                "CUDA DF inventory v1 requires <=16 orbital AOs and <=128 auxiliary AOs"
            )
        state = item["electrons"]
        key = orbital["nbf"], state["nalpha"], state["nbeta"], orbital["primitives"]
        buckets.setdefault(key, []).append(item)
    groups = [buckets[key] for key in sorted(buckets)]
    rows = []
    for group in groups:
        first = group[0]
        o, a = first["orbital"], first["auxiliary"]
        n, c, aux, ac = o["nbf"], o["cartesian_nbf"], a["nbf"], a["cartesian_nbf"]
        b, d = len(group), 3 * first["atoms"]
        if any(
            (
                item["orbital"]["cartesian_nbf"],
                item["auxiliary"]["nbf"],
                item["auxiliary"]["cartesian_nbf"],
                item["atoms"],
            )
            != (c, aux, ac, first["atoms"])
            for item in group
        ):
            raise NotImplementedError(
                "CUDA DF bucket has incompatible Cartesian/auxiliary/atom dimensions"
            )
        source_bytes = density_fitting_source_bytes(
            library,
            batch=b,
            atoms=sum(item["atoms"] for item in group),
            shells=sum(
                item["orbital"]["shells"] + item["auxiliary"]["shells"] + 1
                for item in group
            ),
            cartesian_aos=b * (c + ac + 1),
            primitives=sum(
                item["orbital"]["primitives"] + item["auxiliary"]["primitives"] + 1
                for item in group
            ),
            transforms=b * (n * c + aux * ac),
        )
        occupied = max(first["electrons"]["nalpha"], first["electrons"]["nbeta"], 1)
        rhf_occupied = first["electrons"]["nalpha"] if first["spins"] == 1 else None
        diis_bytes = density_fitting_diis_bytes(library, b, n, diis_history)
        default_tile = density_fitting_tile_plan(
            library,
            b,
            n,
            aux,
            occupied,
            budget_bytes=0,
            fixed_device_bytes=source_bytes + diis_bytes,
            generated_source=True,
            pair_storage=pair_storage,
            rhf_occupied=rhf_occupied,
        )
        # Match the native matrix-only one-electron chunk preflight. Direct-ERI
        # task tables are omitted by this exporter, leaving quadratic metadata.
        preparation_metadata = 0
        preparation_retained = 0
        preparation_temporary = 0
        metadata = 0
        for item in group:
            orbital, auxiliary = item["orbital"], item["auxiliary"]
            pairs = orbital["shells"] * (orbital["shells"] + 1) // 2
            packed = 512 * (
                1
                + item["atoms"]
                + orbital["shells"]
                + orbital["primitives"]
                + c
                + n
                + pairs
                + auxiliary["shells"]
                + auxiliary["primitives"]
            )
            # Native preparation holds Cartesian/public outputs together and
            # retains all preceding items while preparing the next singleton.
            # This global request permits both properties, so use force data
            # for the selected provider even when a replay requests only E.
            copies = 1
            retained = packed + 8 * (2 * n * n * copies + d)
            temporary = packed + 8 * (
                2 * c * c * copies
                + d
                + 4 * c * c
                + 1
                + (0)
            )
            preparation_metadata += packed
            preparation_retained += retained
            preparation_temporary = max(preparation_temporary, temporary)
            combined_shells = orbital["shells"] + auxiliary["shells"] + 1
            combined_pairs = combined_shells * (combined_shells + 1) // 2
            metadata += 2048 * (
                1
                + item["atoms"]
                + combined_shells
                + orbital["primitives"]
                + auxiliary["primitives"]
                + c
                + ac
                + combined_pairs
                + combined_pairs**2
            )
        rows.append(
            {
                "batch": b,
                "nbf": n,
                "naux": aux,
                "cartesian_nbf": c,
                "cartesian_naux": ac,
                "coordinates": d,
                "source_bytes": source_bytes,
                "diis_device_bytes": diis_bytes,
                "occupied": occupied,
                "rhf_occupied": rhf_occupied,
                "host_metadata": metadata,
                "preparation_minimum": preparation_metadata
                + preparation_retained
                + preparation_temporary,
                "preparation_retained": preparation_retained,
                "preparation_temporary": preparation_temporary,
                "default_tile": default_tile,
            }
        )
        # Source response owns packed AO metadata, four device metric matrices,
        # density matrices and bounded auxiliary weights. The compatibility
        # resident adapter retains its independent host spectral allowance.
        # This LP64 capacity allowance covers vector growth for public s/p/d/f
        # expansions (at most three Cartesian terms) without a CUDA context.
        response_metadata = max(
            256
            * (
                1
                + item["atoms"]
                + n
                + aux
                + item["orbital"]["primitives"]
                + item["auxiliary"]["primitives"]
            )
            for item in group
        )
        response_spectral = 8 * (12 * aux * aux + 10 * aux + n * n)
        # Three density terms cover UHF; its total-density matrix is included.
        response_fixed = 8 * (3 * aux * aux + 6 * aux + 4 * n * n)
        rows[-1]["response_minimum"] = response_metadata + 8 * (
            4 * aux * aux + 9 * n * n + 6 * aux
        )
        rows[-1]["resident_response_capacity"] = response_metadata + max(
            response_spectral, response_fixed + 16 * n * n * aux
        )
        # Include the caller's UHF total-density staging along with three
        # uploaded terms. Device tiles are capped by the supplied half-budget.
        rows[-1]["response_capacity"] = response_metadata + 8 * (
            4 * aux * aux + (7 + 2 * aux) * n * n + 6 * aux
        )
        rows[-1]["response_host_capacity"] = response_metadata + 8 * n * n
    source_budget = requested_budget or max(
        max(
            row["preparation_minimum"],
            2 * row["default_tile"].peak_workspace_bytes,
            2 * row["response_minimum"],
        )
        for row in rows
    )
    choices = [("cuda-df-resident", 0, "resident", 0)] if requested_budget == 0 else []
    choices.append(("cuda-df-source", source_budget, "recomputed", 1))
    if packed_values:
        # An explicit packed request must not be estimated as a dense owner or
        # silently fall back to a different representation when its owners fail
        # the budget. The native planner chooses its bounded Q panel first.
        choices = [("cuda-df-packed", source_budget, "resident", 0)]
    candidates = []
    for name, sub_budget, mode, cost in choices:
        source = sub_budget > 0
        host_resident, device_resident, host_work, device_work, inventories = (
            [],
            [],
            [],
            [],
            [],
        )
        for row in rows:
            b, n, aux, c, ac, d = (
                row[key]
                for key in (
                    "batch",
                    "nbf",
                    "naux",
                    "cartesian_nbf",
                    "cartesian_naux",
                    "coordinates",
                )
            )
            tile = density_fitting_tile_plan(
                library,
                b,
                n,
                aux,
                row["occupied"],
                # Prepared requests admit energy and forces. Energy can use
                # the full value allowance; this larger live set conservatively
                # bounds the force plan's half-allowance scratch as well.
                budget_bytes=sub_budget if source else 0,
                fixed_device_bytes=(row["source_bytes"] if source else 0)
                + row["diis_device_bytes"],
                generated_source=source,
                pair_storage=pair_storage,
                rhf_occupied=row["rhf_occupied"],
            )
            force_tile = tile
            if source:
                force_tile = density_fitting_tile_plan(
                    library,
                    b,
                    n,
                    aux,
                    row["occupied"],
                    budget_bytes=max(1, sub_budget // 2),
                    fixed_device_bytes=row["source_bytes"] + row["diis_device_bytes"],
                    generated_source=True,
                    pair_storage=pair_storage,
                    rhf_occupied=row["rhf_occupied"],
                )
            pairs = (
                min(n * n, max(n, (tile.ao_pair_tile // n) * n)) if source else n * n
            )
            q = tile.auxiliary_tile if source else aux
            matrix, metric, tensor = (
                8 * b * n * n,
                8 * b * aux * aux,
                8 * b * n * n * aux,
            )
            tile_bytes = 8 * pairs * q
            # Actual queried metric/SCF workspaces are numeric allocations in
            # the ledger. These explicit conservative allowances are shared
            # with neither caller inputs nor opaque provider allocations.
            solver = (64 << 20) + 16 * matrix + 128 * aux * aux
            # Dense keeps its original capacity. Occupied mode reserves both
            # spin factors at full rank; actual factors use nbf*max_occupied.
            # Generation controls fit in the existing 1024-byte item allowance.
            factors_reserved = occupied_exchange or bool(tile.automatic_rhf_rank)
            persistent_device = (
                (34 if factors_reserved else 32) * matrix
                + 16 * b * aux
                + solver
                + 1024 * b
            )
            # One ordinary AO eigensystem serves the bucket serially. The
            # native adapter checks queried device/host workspace against this
            # allowance before allocation; neither uses the opaque allowance.
            ordinary_eigen_workspace = (1 << 20) + 128 * n * n
            ordinary_eigen_device = ordinary_eigen_workspace + 8 * (3 * n * n + n) + 5
            persistent_device += ordinary_eigen_device
            # Both full spin frames survive inactive launches; the independent
            # single-item retry below receives its own complete snapshot share.
            final_snapshot_device = b * (16 * (n * n + n) + 24)
            persistent_device += final_snapshot_device
            persistent_device += row["diis_device_bytes"]
            # All value plans retain the original device metric eigensystem
            # and inverse root for force response, transferred out of setup.
            persistent_device += 2 * metric + 8 * b * aux
            persistent_device += (
                tile.stored_factor_bytes
                + tile.raw_factor_bytes
                + tile.contraction_scratch_bytes
                + row["source_bytes"]
                if packed_values
                else (
                    (
                        tensor + 3 * tile_bytes
                        if tile.stores_full_three_center
                        else 4 * tile_bytes
                    )
                    + row["source_bytes"]
                    if source
                    else tensor + 3 * tile_bytes
                )
            )
            setup = metric + 8 * b * aux + solver + 4 * b
            # The promoted force consumer streams bounded weights. It owns no
            # raw coordinate derivative tensors or coordinate-wise CUDA scratch.
            # Retained values admit both the device path and explicit host
            # diagnostic ablation under one conservative resource reservation.
            force = (
                min(row["response_capacity"], sub_budget // 2)
                if source
                else max(
                    row["resident_response_capacity"],
                    min(row["response_capacity"], 128 << 20),
                )
            )
            generation = row["source_bytes"] + 8 * b * (
                (1) * 2 * c * c
                + (0 if source else ac * ac + c * c * ac)
            )
            persistent_host = row["host_metadata"] + 8 * b * (32 * n * n + 16 * d)
            # One verified X and its exact S/geometry key per source survive
            # value-plan rebuilds. Device SCF already reserves d_orthogonalizer;
            # these are additional host copies only, retained through teardown.
            overlap_cache_host = 2 * matrix + 8 * b * d
            persistent_host += overlap_cache_host + ordinary_eigen_workspace + 64 * b
            one_electron = (
                8 * b * ((1) * 2 * n * n + d)
            )
            raw = 8 * b * (aux * aux + n * n * aux)
            persistent_host += one_electron
            if not source:
                persistent_host += raw + tensor
            scf = (
                8
                * b
                * (
                    # The DIIS-enabled bucket uploads S alongside its H/X/D
                    # staging. It stays live until compact dispatch returns.
                    129 * n * n
                    + 4 * (diis_history + 1) * n * n
                    + 4 * (diis_history + 1) ** 2
                )
            )
            host_temporary = (
                (64 << 20) + scf + one_electron + 64 * (metric // b) + 8 * (tensor // b)
            )
            host_temporary += 2 * row["host_metadata"] + (
                2 * tile_bytes if source else 2 * raw + 8 * b * (ac * ac + c * c * ac)
            )
            if source and sub_budget < row["preparation_minimum"]:
                raise ValueError(
                    "DF sub-budget cannot hold the native one-electron preparation minimum"
                )
            if source and sub_budget // 2 < row["response_minimum"]:
                raise ValueError(
                    "DF sub-budget cannot hold the generated response minimum"
                )
            # Strict selection serializes one item. Reserve detached candidates,
            # accepted outputs and validation/correction products explicitly;
            # Two-spin candidate/correction/output frames may coexist (six
            # epsilon vectors); eight vectors also cover the provider result.
            # Matrix scratch includes canonical export products and both W.
            final_selection_host = 8 * (32 * n * n + 8 * n)
            host_temporary += final_selection_host
            host_temporary = max(host_temporary, row["preparation_temporary"])
            host_temporary += (
                row["response_host_capacity"]
                if source
                else row["resident_response_capacity"]
            )
            host_resident.append(checked_bytes(persistent_host))
            device_resident.append(checked_bytes(persistent_device))
            host_work.append(checked_bytes(host_temporary))
            # Reserve one additional complete item for existing cold numerical
            # recovery. This is independent of allocation-failure retries.
            # A batched RHF solve can retry a singleton. Query that method-aware
            # plan separately instead of guessing eligibility from the batch.
            retry_tile = density_fitting_tile_plan(
                library,
                1,
                n,
                aux,
                row["occupied"],
                budget_bytes=sub_budget if source else 0,
                fixed_device_bytes=(row["source_bytes"] // b if source else 0)
                + density_fitting_diis_bytes(library, 1, n, diis_history),
                generated_source=source,
                pair_storage=pair_storage,
                rhf_occupied=row["rhf_occupied"],
            )
            retry_factor_extra = (
                2 * matrix // b
                if retry_tile.automatic_rhf_rank and not factors_reserved
                else 0
            )
            device_work.append(
                checked_bytes(
                    max(setup, force, generation)
                    + (persistent_device - ordinary_eigen_device) // b
                    # A separate single-item retry needs its own whole ordinary
                    # workspace; this bucket-serialized capacity does not scale
                    # down with the original batch's item count.
                    + ordinary_eigen_device
                    + solver
                    + retry_factor_extra
                )
            )
            inventories.append(
                {
                    **{k: v for k, v in row.items() if k != "default_tile"},
                    # Preserve the historical all-properties (force) route,
                    # and expose the larger energy route used for the bound.
                    "tiles": asdict(force_tile),
                    "energy_tiles": asdict(tile),
                    "force_tiles": asdict(force_tile),
                    "resident_host_bytes": persistent_host,
                    "overlap_cache_host_bytes": overlap_cache_host,
                    "ordinary_eigen_device_bytes": ordinary_eigen_device,
                    "final_snapshot_device_bytes": final_snapshot_device,
                    "final_selection_host_bytes": final_selection_host,
                    "ordinary_eigen_host_workspace_bytes": ordinary_eigen_workspace,
                    "resident_device_bytes": persistent_device,
                }
            )
        estimates = (
            ResourceEstimate(
                "all DF bucket caches and native solver workspaces",
                checked_bytes(sum(device_resident)),
                f"device:{device}",
                first_phase,
                last_phase,
                kind="persistent",
                recomputable=source,
            ),
            ResourceEstimate(
                "largest serialized DF setup/force/retry workspace",
                max(device_work),
                f"device:{device}",
                first_phase,
                last_phase,
            ),
            ResourceEstimate(
                "all retained DF host state",
                checked_bytes(sum(host_resident)),
                "pageable",
                first_phase,
                last_phase,
                kind="persistent",
                cacheable=not source,
            ),
            ResourceEstimate(
                "largest serialized DF host workspace",
                max(host_work),
                "pageable",
                first_phase,
                last_phase,
            ),
            ResourceEstimate(
                "opaque CUDA library retention allowance",
                (256 << 20) * (len(rows) + 1),
                f"device:{device}",
                first_phase,
                last_phase,
                kind="library",
                accounting="runtime_allowance",
            ),
        )
        layout_identities = sorted(
            {
                inventory[route]["value_layout_identity"]
                for inventory in inventories
                for route in ("energy_tiles", "force_tiles")
            }
        )
        candidates.append(
            ResourceCandidate(
                name,
                mode,
                estimates,
                relative_cost=cost,
                decisions=(
                    ("df_pair_storage", pair_storage),
                    ("df_value_layout_identities", json.dumps(layout_identities)),
                    ("density_fitting_memory_budget_bytes", str(sub_budget)),
                    (
                        "batch_execution",
                        "serialized buckets; all retained caches summed",
                    ),
                    (
                        "scf_driver",
                        "CUDA SCF with existing CPU numerical recovery; budget-selected CUDA forward storage, generated response and J/K"
                        if source
                        else "CUDA SCF with existing CPU numerical recovery; CUDA J/K",
                    ),
                    ("bucket_inventory", json.dumps(inventories, sort_keys=True)),
                ),
            )
        )
    return tuple(candidates)
