"""Project Direct-J/K Fock materialization into the shared schedule contract."""

from __future__ import annotations

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

DIRECT_FOCK_ROUTE_SCHEMA = "vibeqc.integral.direct-fock-route.v1"


def direct_fock_route_contract(
    route: str,
    *,
    shell_class: int,
    profitability: GpuProfitability,
    workload_hash: str,
    profile_key: str,
    target_hash: str,
    precision_schedule_hash: str,
    page_size: int | None = None,
    legal: bool = True,
    reasons: tuple[str, ...] = (),
) -> ScheduleContract:
    """Project paged/streaming execution into compiler-common schedule metadata."""

    if route not in ("paged", "streaming"):
        raise ValueError("Direct Fock route must be paged or streaming")
    if type(shell_class) is not int or not 0 <= shell_class < 64:
        raise ValueError("Direct Fock shell class must fit the 64-bit registry mask")
    if not isinstance(profitability, GpuProfitability):
        raise TypeError("Direct Fock route requires GpuProfitability")
    if route == "streaming" and page_size is not None:
        raise ValueError("streaming Direct Fock route cannot carry a page size")
    if page_size is not None and (type(page_size) is not int or page_size < 1):
        raise ValueError("Direct Fock page size must be positive or None")
    if type(legal) is not bool:
        raise TypeError("Direct Fock route legality must be boolean")

    return ScheduleContract(
        consumer="integral.direct-fock",
        schedule_hash=canonical_hash(
            {
                "schema": DIRECT_FOCK_ROUTE_SCHEMA,
                "route": route,
                "shell_class": shell_class,
                "page_size": page_size,
            }
        ),
        workload_hash=workload_hash,
        profile_key=profile_key,
        target_hash=target_hash,
        precision_schedule_hash=precision_schedule_hash,
        fallback=route == "paged",
        legal=legal,
        reasons=reasons,
        topology=ScheduleTopology(
            materialization="materialize" if route == "paged" else "stream",
            residency="paged-device" if route == "paged" else "direct-device",
            page_size=page_size if route == "paged" else None,
            bucket=f"shell-class-{shell_class}",
        ),
        resources=ScheduleResources(),
        profitability=profitability,
        provenance=(
            ("fock_route", route),
            ("shell_class", str(shell_class)),
        ),
    )


def select_direct_fock_route(
    *,
    shell_class: int,
    paged: GpuProfitability,
    streaming: GpuProfitability,
    workload_hash: str,
    profile_key: str,
    target_hash: str,
    precision_schedule_hash: str,
    page_size: int | None = None,
    streaming_legal: bool = True,
    streaming_reasons: tuple[str, ...] = (),
    minimum_speedup: float = 1.02,
    endpoint_noise_fraction: float = ENDPOINT_NOISE_FRACTION,
) -> str:
    """Choose materialization through the shared measured-schedule selector."""

    baseline = direct_fock_route_contract(
        "paged",
        shell_class=shell_class,
        profitability=paged,
        workload_hash=workload_hash,
        profile_key=profile_key,
        target_hash=target_hash,
        precision_schedule_hash=precision_schedule_hash,
        page_size=page_size,
    )
    candidate = direct_fock_route_contract(
        "streaming",
        shell_class=shell_class,
        profitability=streaming,
        workload_hash=workload_hash,
        profile_key=profile_key,
        target_hash=target_hash,
        precision_schedule_hash=precision_schedule_hash,
        legal=streaming_legal,
        reasons=streaming_reasons,
    )
    selected = select_measured_schedule_contract(
        (baseline, candidate),
        minimum_speedup=minimum_speedup,
        endpoint_noise_fraction=endpoint_noise_fraction,
    )
    return (
        "streaming"
        if selected.topology.materialization == "stream"
        else "paged"
    )
