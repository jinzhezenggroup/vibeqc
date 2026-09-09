"""Compose native CUDA DF storage and source-regeneration alternatives.

The current inventory is deliberately limited to small HF orbital spaces and
auxiliary dimensions up to 128. Runtime-dependent solver storage has an explicit
capacity allowance and is also charged by the native allocation ledger.
"""

import json
from dataclasses import asdict

from .resources import ResourceCandidate, ResourceEstimate, checked_bytes
from .resources_df import density_fitting_source_bytes, density_fitting_tile_plan


def cuda_df_candidates(
    library, items, *, diis_history, device, first_phase, last_phase, requested_budget=0
):
    """Return actual resident/source choices for the native fleet's buckets.

    Buckets retain all their device plans. Temporary setup/force storage is
    shared across serialized buckets; the existing one-item cold retry can
    coexist with a warm bucket and has a separate reservation. Source mode
    explicitly uses CPU SCF/DIIS/eigensolvers with CUDA J/K and integral tiles,
    as required by the existing streamed provider's capability contract.
    """
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
        default_tile = density_fitting_tile_plan(
            library,
            b,
            n,
            aux,
            occupied,
            budget_bytes=0,
            fixed_device_bytes=source_bytes,
        )
        # Match the existing native one-electron chunk preflight, whose
        # conservative packed-topology allowance includes possible PSSS queues.
        preparation_minimum = 0
        metadata = 0
        for item in group:
            orbital, auxiliary = item["orbital"], item["auxiliary"]
            pairs = orbital["shells"] * (orbital["shells"] + 1) // 2
            quartets = pairs * (pairs + 1) // 2
            packed = 512 * (
                1
                + item["atoms"]
                + orbital["shells"]
                + orbital["primitives"]
                + c
                + n
                + pairs
                + quartets
            )
            preparation_minimum = max(
                preparation_minimum,
                packed + 8 * (2 * c * c * (d + 1) + d + 4 * c * c + 1),
            )
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
                "occupied": occupied,
                "host_metadata": metadata,
                "preparation_minimum": preparation_minimum,
                "default_tile": default_tile,
            }
        )
    source_budget = requested_budget or max(
        max(row["preparation_minimum"], row["default_tile"].peak_workspace_bytes)
        for row in rows
    )
    choices = [("cuda-df-resident", 0, "resident", 0)] if requested_budget == 0 else []
    choices.append(("cuda-df-source", source_budget, "recomputed", 1))
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
                budget_bytes=sub_budget if source else 0,
                fixed_device_bytes=row["source_bytes"] if source else 0,
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
            tile_bytes, metric_tile = 8 * pairs * q, 8 * q * aux
            # Actual queried metric/SCF workspaces are numeric allocations in
            # the ledger. These explicit conservative allowances are shared
            # with neither caller inputs nor opaque provider allocations.
            solver = (64 << 20) + 16 * matrix + 128 * aux * aux
            persistent_device = 32 * matrix + 16 * b * aux + solver + 1024 * b
            persistent_device += (
                4 * tile_bytes + metric_tile + metric + row["source_bytes"]
                if source
                else tensor + 3 * tile_bytes
            )
            setup = 3 * metric + 16 * b * aux + solver + 4 * b
            force = (
                2 * tile_bytes + metric_tile
                if source
                else 8 * (tensor // b)
                + 4 * (metric // b)
                + 2 * (matrix // b)
                + 16 * aux
                + 8
            )
            # Default raw-value/derivative generation is a separate temporary
            # phase; charge its complete chunk ceiling conservatively.
            generation = row["source_bytes"] + 8 * b * (d + 1) * (
                2 * c * c + (0 if source else ac * ac + c * c * ac)
            )
            persistent_host = row["host_metadata"] + 8 * b * (32 * n * n + 16 * d)
            one_electron = 8 * b * ((d + 1) * 2 * n * n + d)
            raw = 8 * b * (d + 1) * (aux * aux + n * n * aux)
            if source:
                persistent_host += 2 * metric
            else:
                persistent_host += one_electron + raw + tensor
            scf = (
                8
                * b
                * (
                    128 * n * n
                    + 4 * (diis_history + 1) * n * n
                    + 4 * (diis_history + 1) ** 2
                )
            )
            host_temporary = (
                (64 << 20) + scf + one_electron + 64 * (metric // b) + 8 * (tensor // b)
            )
            host_temporary += 2 * row["host_metadata"] + (
                2 * tile_bytes
                if source
                else 2 * raw + 8 * b * (d + 1) * (ac * ac + c * c * ac)
            )
            if source and sub_budget < row["preparation_minimum"]:
                raise ValueError(
                    "DF sub-budget cannot hold the native one-electron preparation minimum"
                )
            host_resident.append(checked_bytes(persistent_host))
            device_resident.append(checked_bytes(persistent_device))
            host_work.append(checked_bytes(host_temporary))
            # Reserve one additional complete item for existing cold numerical
            # recovery. This is independent of allocation-failure retries.
            device_work.append(
                checked_bytes(
                    max(setup, force, generation) + persistent_device // b + solver
                )
            )
            inventories.append(
                {
                    **{k: v for k, v in row.items() if k != "default_tile"},
                    "tiles": asdict(tile),
                    "resident_host_bytes": persistent_host,
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
        candidates.append(
            ResourceCandidate(
                name,
                mode,
                estimates,
                relative_cost=cost,
                decisions=(
                    ("density_fitting_memory_budget_bytes", str(sub_budget)),
                    (
                        "batch_execution",
                        "serialized buckets; all retained caches summed",
                    ),
                    (
                        "scf_driver",
                        "CPU DIIS/eigensolvers; CUDA integral regeneration and J/K"
                        if source
                        else "CUDA SCF with existing CPU numerical recovery; CUDA J/K",
                    ),
                    ("bucket_inventory", json.dumps(inventories, sort_keys=True)),
                ),
            )
        )
    return tuple(candidates)
