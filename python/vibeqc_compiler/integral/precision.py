"""Precision schedules for generated direct-Fock execution.

The existing mixed Fock kernel evaluates the ERI recurrence in FP32 while
retaining density/Fock storage and accumulation in FP64.  This module records
that execution choice in the same common contract used by TensorIR and DFT;
it does not enable the route or decide its scientific promotion domain.
"""

from __future__ import annotations

from vibeqc_compiler.common.precision import (
    ExecutionPrecisionSchedule,
    PrecisionDirective,
    uniform_precision_schedule,
)


def generated_fock_precision_schedule(
    *,
    mixed_eri: bool = False,
    qualification: str | None = None,
) -> ExecutionPrecisionSchedule:
    """Describe strict or qualified mixed generated-Fock arithmetic."""

    if not mixed_eri:
        if qualification is not None:
            raise ValueError("strict FP64 Fock does not require a qualification")
        return uniform_precision_schedule("integral.direct_fock")

    if not isinstance(qualification, str) or not qualification.strip():
        raise ValueError("mixed direct-Fock precision requires a qualification")
    return ExecutionPrecisionSchedule(
        (
            (
                "eri_recurrence",
                PrecisionDirective(
                    storage_dtype="float32",
                    compute_dtype="float32",
                    accumulation_dtype="float32",
                    qualification=qualification,
                ),
            ),
            (
                "fock_accumulation",
                PrecisionDirective(
                    storage_dtype="float64",
                    compute_dtype="float64",
                    accumulation_dtype="float64",
                ),
            ),
        ),
        strict_audit_dtype="float64",
        audit_owner="method-controller",
    )
