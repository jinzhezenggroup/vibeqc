"""Project integral operator routes into the shared schedule contract.

Domain owners retain scientific legality and route construction. This module
only removes repeated measured-route selection glue so HF and KS consumers can
reuse one compiler promotion/fallback contract.
"""

from __future__ import annotations

import collections.abc
import typing

from vibeqc_compiler.common.gpu_profitability import (
    ENDPOINT_NOISE_FRACTION,
    GpuProfitability,
)
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.schedule import (
    ScheduleContract,
    ScheduleResources,
    ScheduleTopology,
    select_measured_schedule_contract,
)

OPERATOR_ROUTE_SCHEMA = "vibeqc.integral.operator-route.v1"


def _nonempty_text(value: typing.Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def operator_route_contract(
    route: str,
    *,
    consumer: str,
    profitability: GpuProfitability,
    workload_hash: str,
    profile_key: str,
    target_hash: str,
    precision_schedule_hash: str,
    topology: ScheduleTopology,
    fallback: bool = False,
    legal: bool = True,
    reasons: tuple[str, ...] = (),
    resources: ScheduleResources | None = None,
    provenance: tuple[tuple[str, str], ...] = (),
    schema: str = OPERATOR_ROUTE_SCHEMA,
    identity_fields: tuple[tuple[str, typing.Any], ...] = (),
) -> ScheduleContract:
    """Build one operator implementation route with a stable domain identity.

    identity_fields are owned by the domain adapter. They are hashed with
    schema and route but otherwise remain opaque to the shared compiler. This
    lets an existing adapter preserve its historical schedule identity while
    another consumer introduces a different route vocabulary.
    """

    route = _nonempty_text(route, "operator route")
    consumer = _nonempty_text(consumer, "operator route consumer")
    schema = _nonempty_text(schema, "operator route schema")
    if not isinstance(profitability, GpuProfitability):
        raise TypeError("operator route requires GpuProfitability")
    if not isinstance(topology, ScheduleTopology):
        raise TypeError("operator route requires ScheduleTopology")
    if type(fallback) is not bool or type(legal) is not bool:
        raise TypeError("operator route fallback/legal flags must be boolean")
    if resources is None:
        resources = ScheduleResources()
    elif not isinstance(resources, ScheduleResources):
        raise TypeError("operator route resources must be ScheduleResources or None")

    identity: dict[str, typing.Any] = {"schema": schema, "route": route}
    for pair in tuple(identity_fields):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("operator route identity fields must be name/value pairs")
        name, value = pair
        name = _nonempty_text(name, "operator route identity name")
        if name in identity:
            raise ValueError(f"duplicate operator route identity field {name!r}")
        identity[name] = value

    return ScheduleContract(
        consumer=consumer,
        schedule_hash=canonical_hash(identity),
        workload_hash=workload_hash,
        profile_key=profile_key,
        target_hash=target_hash,
        precision_schedule_hash=precision_schedule_hash,
        fallback=fallback,
        legal=legal,
        reasons=reasons,
        topology=topology,
        resources=resources,
        profitability=profitability,
        provenance=provenance,
    )


def select_operator_route(
    routes: collections.abc.Mapping[str, ScheduleContract],
    *,
    minimum_speedup: float,
    endpoint_noise_fraction: float = ENDPOINT_NOISE_FRACTION,
) -> str:
    """Choose one measured operator route through the shared promotion policy."""

    if not isinstance(routes, collections.abc.Mapping):
        raise TypeError("operator route selection requires a mapping")
    materialized = tuple(routes.items())
    if not materialized:
        raise ValueError("operator route selection requires candidates")
    for name, contract in materialized:
        _nonempty_text(name, "operator route name")
        if not isinstance(contract, ScheduleContract):
            raise TypeError("operator route selection requires ScheduleContract records")

    selected = select_measured_schedule_contract(
        tuple(contract for _, contract in materialized),
        minimum_speedup=minimum_speedup,
        endpoint_noise_fraction=endpoint_noise_fraction,
    )
    for name, contract in materialized:
        if contract is selected:
            return name
    raise AssertionError("shared measured selector returned an unknown operator route")
