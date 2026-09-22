"""Effect-aware post-transform cleanup for serial ProgramIR graphs.

ProgramIR provider calls are opaque by default. This adapter only value-numbers
providers that an owning subsystem explicitly proves PURE under the shared
#673 effect contract. Buffer names and call names remain debug/provenance
identity; semantic equivalence is provider identity + remapped inputs + output
contracts. Exported writes are never coalesced, preserving the public boundary
ABI even when their computations would otherwise be equivalent.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass

from .liveness import EffectKind
from .pass_manager import PassManager, PassRecord, PassStage
from .program import PlanCall, ProgramBuffer, ProgramIR
from .provenance import canonical_hash
from .value_numbering import ValueNumberingDiagnostics, ValueNumberTable


@dataclass(frozen=True)
class ProgramCleanupResult:
    """Optimized ProgramIR plus compact deterministic cleanup evidence."""

    program: ProgramIR
    pipeline_identity: str
    records: tuple[PassRecord, ...]
    value_numbering: ValueNumberingDiagnostics
    eliminated_calls: tuple[str, ...]
    eliminated_buffers: tuple[str, ...]
    calls_before: int
    buffers_before: int
    before_identity: str
    effect_policy: tuple[tuple[str, str], ...]

    def to_payload(self) -> dict[str, typing.Any]:
        """Return diagnostics suitable for CI and benchmark evidence."""

        return {
            "schema": "vibeqc.compiler.program-cleanup.v1",
            "pipeline_identity": self.pipeline_identity,
            "before_identity": self.before_identity,
            "after_identity": self.program.identity,
            "effect_policy": dict(self.effect_policy),
            "passes": [
                {
                    "name": record.name,
                    "version": record.version,
                    "changed": record.changed,
                }
                for record in self.records
            ],
            "value_numbering": self.value_numbering.to_payload(),
            "eliminated_calls": list(self.eliminated_calls),
            "eliminated_buffers": list(self.eliminated_buffers),
            "calls_before": self.calls_before,
            "calls_after": len(self.program.calls),
            "buffers_before": self.buffers_before,
            "buffers_after": len(self.program.buffers),
        }


@dataclass(frozen=True)
class _CallValue:
    """One semantic provider-call value and its positional output owners."""

    semantic: tuple[typing.Any, ...]
    call: PlanCall


def _buffer_contract(buffer: ProgramBuffer) -> str:
    """Hash physical result requirements while deliberately ignoring its name."""

    return canonical_hash(
        {
            "bytes": buffer.bytes,
            "space": buffer.space,
            "layout": None if buffer.layout is None else buffer.layout.to_payload(),
            "itemsize": buffer.itemsize,
        }
    )


def _normalize_effects(effects: typing.Any) -> dict[str, EffectKind]:
    if not isinstance(effects, Mapping):
        raise TypeError("ProgramIR effects must be a provider-to-EffectKind mapping")
    normalized: dict[str, EffectKind] = {}
    for provider, effect in effects.items():
        if not isinstance(provider, str) or not provider:
            raise ValueError("ProgramIR effect providers must be nonempty strings")
        if not isinstance(effect, EffectKind):
            raise TypeError("ProgramIR effects must use the shared EffectKind")
        normalized[provider] = effect
    return normalized


def _value_number_program(
    program: ProgramIR,
    effects: Mapping[str, EffectKind],
) -> tuple[
    ProgramIR,
    ValueNumberingDiagnostics,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Eliminate equivalent proven-pure internal provider calls.

    A provider absent from the effects mapping is OPAQUE. Calls with exported
    writes are also treated as barriers: ProgramIR has disjoint named output
    ownership and this pass must not turn that ABI into an alias.
    """

    buffers = {buffer.name: buffer for buffer in program.buffers}
    replacements: dict[str, str] = {}
    eliminated_calls: list[str] = []
    eliminated_buffers: list[str] = []
    retained_calls: list[PlanCall] = []
    table = ValueNumberTable[_CallValue](
        equivalent=lambda left, right: left.semantic == right.semantic
    )

    def representative(name: str) -> str:
        while name in replacements:
            name = replacements[name]
        return name

    exported = set(program.outputs)
    for call in program.calls:
        reads = tuple(representative(name) for name in call.reads)
        updated = PlanCall(
            call.name,
            call.provider,
            call.identity,
            reads,
            call.writes,
        )
        output_contracts = tuple(
            _buffer_contract(buffers[name]) for name in updated.writes
        )
        semantic = (
            updated.provider,
            updated.identity,
            updated.reads,
            output_contracts,
        )
        bucket = canonical_hash(
            {
                "provider": updated.provider,
                "identity": updated.identity,
                "reads": list(updated.reads),
                "outputs": list(output_contracts),
            }
        )
        effect = effects.get(updated.provider, EffectKind.OPAQUE)
        if exported.intersection(updated.writes):
            effect = EffectKind.OPAQUE
        decision = table.number(
            bucket,
            _CallValue(semantic, updated),
            effect=effect,
        )
        if not decision.reused:
            retained_calls.append(updated)
            continue

        representative_writes = decision.representative.call.writes
        if len(representative_writes) != len(updated.writes):
            raise AssertionError("equivalent ProgramIR calls changed output arity")
        for removed, retained in zip(
            updated.writes, representative_writes, strict=True
        ):
            replacements[removed] = representative(retained)
            eliminated_buffers.append(removed)
        eliminated_calls.append(updated.name)

    if not eliminated_calls and all(
        call.reads == original.reads
        for call, original in zip(retained_calls, program.calls, strict=True)
    ):
        result = program
    else:
        removed = set(eliminated_buffers)
        result = ProgramIR(
            program.name,
            tuple(buffer for buffer in program.buffers if buffer.name not in removed),
            program.inputs,
            tuple(retained_calls),
            program.outputs,
            schema_version=program.schema_version,
        )
    return (
        result,
        table.diagnostics,
        tuple(eliminated_calls),
        tuple(eliminated_buffers),
    )


def cleanup_program_ir(
    program: ProgramIR,
    *,
    effects: Mapping[str, EffectKind],
) -> ProgramCleanupResult:
    """Run the lightweight post-transform ProgramIR redundancy cleanup.

    The pass is intentionally opt-in at the provider-effect boundary. It is
    suitable after scheduling/composition stages that may clone pure calls, and
    it does not perform DCE, reordering, alias inference, or buffer allocation.
    """

    if not isinstance(program, ProgramIR):
        raise TypeError("ProgramIR cleanup requires ProgramIR")
    normalized = _normalize_effects(effects)
    evidence: list[
        tuple[
            ValueNumberingDiagnostics,
            tuple[str, ...],
            tuple[str, ...],
        ]
    ] = []

    def gvn(value: ProgramIR) -> ProgramIR:
        result, diagnostics, calls, buffers = _value_number_program(value, normalized)
        evidence.append((diagnostics, calls, buffers))
        return result

    run = PassManager(
        name="program.cleanup",
        version=1,
        stages=(
            PassStage(
                "effect_aware_gvn",
                1,
                gvn,
                invalidates=("liveness", "storage"),
            ),
        ),
        fingerprint=lambda value: value.identity,
    ).run(program)
    diagnostics, eliminated_calls, eliminated_buffers = evidence[0]
    return ProgramCleanupResult(
        run.value,
        run.pipeline_identity,
        run.records,
        diagnostics,
        eliminated_calls,
        eliminated_buffers,
        len(program.calls),
        len(program.buffers),
        program.identity,
        tuple(
            sorted((provider, effect.value) for provider, effect in normalized.items())
        ),
    )
