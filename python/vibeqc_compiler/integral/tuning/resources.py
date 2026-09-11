"""Occupancy estimates and explicit resource-limit rejection for compiled candidates."""

from __future__ import annotations

from ..batch_benchmark import KernelResources
from ..cuda_target import (
    CudaTargetInfo,
)
from ..ir import KernelConsumer
from .policy import ScheduleTrial


def estimate_occupancy(
    resources: tuple[KernelResources, ...],
    trial: ScheduleTrial,
    target: CudaTargetInfo,
) -> dict[str, object]:
    """Estimate resource-limited occupancy for one compiled schedule.

    The estimate is deliberately conservative about what it claims: it uses
    the integer limits visible in the target record and PTXAS resources, but
    does not model register/shared-memory allocation granularity or compiler
    partitioning.  It is therefore an auditable upper bound, not a substitute
    for a device occupancy API.  A row is still emitted when no PTXAS record
    exists so rejected candidates retain a complete diagnostic trail.
    """

    block_threads = trial.schedule.block_threads
    if not resources:
        return {
            "available": False,
            "method": "resource_upper_bound",
            "block_threads": block_threads,
            "kernels": [],
        }

    kernels = []
    occupancies = []
    for resource in resources:
        thread_limit = target.maximum_threads_per_sm // block_threads
        register_limit = (
            target.registers_per_sm // (resource.registers * block_threads)
            if resource.registers > 0
            else target.maximum_blocks_per_sm
        )
        shared_limit = (
            target.shared_memory_per_sm // resource.shared_bytes
            if resource.shared_bytes > 0
            else target.maximum_blocks_per_sm
        )
        resident_blocks = max(
            0,
            min(
                target.maximum_blocks_per_sm,
                thread_limit,
                register_limit,
                shared_limit,
            ),
        )
        active_threads = resident_blocks * block_threads
        occupancy = (
            active_threads / target.maximum_threads_per_sm
            if target.maximum_threads_per_sm > 0
            else 0.0
        )
        occupancies.append(occupancy)
        kernels.append(
            {
                "function": resource.function,
                "registers_per_thread": resource.registers,
                "stack_bytes": resource.stack_bytes,
                "shared_bytes": resource.shared_bytes,
                "resident_blocks_per_sm": resident_blocks,
                "active_threads_per_sm": active_threads,
                "estimated_occupancy": occupancy,
                "limits": {
                    "threads": thread_limit,
                    "registers": register_limit,
                    "shared_memory": shared_limit,
                    "blocks": target.maximum_blocks_per_sm,
                },
            }
        )
    return {
        "available": True,
        "method": "resource_upper_bound",
        "block_threads": block_threads,
        "kernels": kernels,
        "minimum_estimated_occupancy": min(occupancies),
        "maximum_estimated_occupancy": max(occupancies),
    }


def _resource_rejections(
    resources: tuple[KernelResources, ...],
    *,
    consumer: KernelConsumer,
    maximum_registers: int,
    maximum_stack_bytes: int,
    maximum_shared_bytes: int,
    expected_kernel_records: int = 1,
) -> list[str]:
    """Return deterministic resource-gate failures for one schedule."""

    reasons = []
    if len(resources) != expected_kernel_records:
        reasons.append(
            f"expected {expected_kernel_records} {consumer.value} kernel records, "
            f"found {len(resources)}"
        )
    if any(item.spill_store_bytes or item.spill_load_bytes for item in resources):
        reasons.append("ptxas reported local-memory spills")
    if any(item.registers > maximum_registers for item in resources):
        reasons.append(f"register use exceeds {maximum_registers}")
    if any(item.stack_bytes > maximum_stack_bytes for item in resources):
        reasons.append(f"stack frame exceeds {maximum_stack_bytes} bytes")
    if any(item.shared_bytes > maximum_shared_bytes for item in resources):
        reasons.append(f"static shared memory exceeds {maximum_shared_bytes} bytes")
    return reasons
