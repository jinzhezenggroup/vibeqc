"""Versioned physical-storage overlay for immutable ProgramIR graphs.

ProgramIR remains the scientific/provider dependency graph. This module binds
explicit physical alias and memory-effect facts to that immutable identity, then
reuses the shared #831 storage analysis without turning ProgramIR into an allocator.
"""

from __future__ import annotations

import json
import typing
from dataclasses import asdict, dataclass

from .program import ProgramIR
from .provenance import canonical_hash
from .storage import (
    AliasKind,
    BufferOp,
    BufferValue,
    MemoryEffect,
    StorageAnalysis,
    analyze_storage,
)


@dataclass(frozen=True)
class BufferAliasBinding:
    """Physical alias fact for one logical ProgramIR buffer."""

    buffer: str
    alias: AliasKind
    alias_of: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.buffer, str) or not self.buffer:
            raise ValueError("alias buffer must be a nonempty string")
        if not isinstance(self.alias, AliasKind):
            raise TypeError("alias binding requires AliasKind")
        if self.alias is AliasKind.VIEW:
            if not isinstance(self.alias_of, str) or not self.alias_of:
                raise ValueError("view alias requires an owner")
            if self.alias_of == self.buffer:
                raise ValueError("view alias owner must be distinct")
        elif self.alias_of is not None:
            raise ValueError("only view aliases may name an owner")


@dataclass(frozen=True)
class CallEffectBinding:
    """Memory-effect fact for one opaque ProgramIR provider boundary."""

    call: str
    effect: MemoryEffect

    def __post_init__(self) -> None:
        if not isinstance(self.call, str) or not self.call:
            raise ValueError("effect call must be a nonempty string")
        if not isinstance(self.effect, MemoryEffect):
            raise TypeError("effect binding requires MemoryEffect")


@dataclass(frozen=True)
class ProgramStoragePlan:
    """Explicit physical-storage facts bound to one immutable ProgramIR."""

    program: ProgramIR
    aliases: tuple[BufferAliasBinding, ...] = ()
    effects: tuple[CallEffectBinding, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.program, ProgramIR):
            raise TypeError("storage plan requires ProgramIR")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported ProgramStoragePlan schema")
        aliases = tuple(self.aliases)
        effects = tuple(self.effects)
        if not all(isinstance(item, BufferAliasBinding) for item in aliases):
            raise TypeError("storage plan requires structured alias bindings")
        if not all(isinstance(item, CallEffectBinding) for item in effects):
            raise TypeError("storage plan requires structured effect bindings")
        if len({item.buffer for item in aliases}) != len(aliases):
            raise ValueError("duplicate alias binding")
        if len({item.call for item in effects}) != len(effects):
            raise ValueError("duplicate effect binding")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "effects", effects)
        self._validate_bindings()

    def _validate_bindings(self) -> None:
        buffers = {item.name: item for item in self.program.buffers}
        calls = {item.name: item for item in self.program.calls}
        phases = dict.fromkeys(self.program.inputs, 0)
        for phase, call in enumerate(self.program.calls, 1):
            phases.update({name: phase for name in call.writes})
        alias_map = {item.buffer: item for item in self.aliases}
        for item in self.aliases:
            if item.buffer not in buffers:
                raise ValueError("alias binding references unknown ProgramIR buffer")
            if item.alias is AliasKind.UNKNOWN:
                if item.buffer not in self.program.inputs:
                    raise ValueError("unknown alias is only valid for borrowed inputs")
                continue
            if item.alias is not AliasKind.VIEW:
                continue
            if item.alias_of not in buffers:
                raise ValueError("view alias owner is unknown")
            owner = buffers[typing.cast("str", item.alias_of)]
            view = buffers[item.buffer]
            if owner.space != view.space or view.bytes > owner.bytes:
                raise ValueError("view alias is incompatible with owner storage")
            if phases[item.alias_of] > phases[item.buffer]:
                raise ValueError("view alias owner is unavailable at view production")
        for item in self.effects:
            if item.call not in calls:
                raise ValueError("effect binding references unknown ProgramIR call")

        def root(name: str, stack: tuple[str, ...] = ()) -> str:
            binding = alias_map.get(name)
            if binding is None or binding.alias is not AliasKind.VIEW:
                return name
            owner = typing.cast("str", binding.alias_of)
            if owner in stack:
                raise ValueError("cyclic ProgramIR storage alias")
            return root(owner, (*stack, name))

        for name in buffers:
            root(name)

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        return json.loads(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "program_identity": self.program.identity,
                    "aliases": [
                        {**asdict(item), "alias": item.alias.value}
                        for item in self.aliases
                    ],
                    "effects": [
                        {**asdict(item), "effect": item.effect.value}
                        for item in self.effects
                    ],
                }
            )
        )

    @classmethod
    def from_payload(
        cls, program: ProgramIR, payload: typing.Any
    ) -> ProgramStoragePlan:
        expected = {"schema_version", "program_identity", "aliases", "effects"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("invalid ProgramStoragePlan fields")
        if payload["program_identity"] != program.identity:
            raise ValueError("storage plan does not match the supplied ProgramIR")
        if not isinstance(payload["aliases"], list) or not isinstance(
            payload["effects"], list
        ):
            raise TypeError("storage plan bindings must be lists")
        aliases = []
        for row in payload["aliases"]:
            if not isinstance(row, dict) or set(row) != {"buffer", "alias", "alias_of"}:
                raise ValueError("invalid alias binding fields")
            try:
                alias = AliasKind(row["alias"])
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid alias binding kind") from exc
            aliases.append(BufferAliasBinding(row["buffer"], alias, row["alias_of"]))
        effects = []
        for row in payload["effects"]:
            if not isinstance(row, dict) or set(row) != {"call", "effect"}:
                raise ValueError("invalid effect binding fields")
            try:
                effect = MemoryEffect(row["effect"])
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid effect binding kind") from exc
            effects.append(CallEffectBinding(row["call"], effect))
        return cls(program, tuple(aliases), tuple(effects), payload["schema_version"])

    def storage_analysis(self) -> StorageAnalysis:
        """Analyze the bound physical contract with fail-closed defaults."""
        alias_map = {item.buffer: item for item in self.aliases}
        effect_map = {item.call: item.effect for item in self.effects}
        values = []
        for buffer in self.program.buffers:
            binding = alias_map.get(buffer.name)
            alias = AliasKind.OWNER if binding is None else binding.alias
            alias_of = None if binding is None else binding.alias_of
            values.append(
                BufferValue(
                    buffer.name,
                    buffer.bytes,
                    buffer.space,
                    buffer.layout,
                    alias=alias,
                    alias_of=alias_of,
                    compiler_owned=(
                        alias is AliasKind.OWNER
                        and buffer.name not in self.program.inputs
                    ),
                )
            )
        operations = tuple(
            BufferOp(
                call.name,
                call.reads,
                call.writes,
                effect_map.get(call.name, MemoryEffect.EXPLICIT),
            )
            for call in self.program.calls
        )
        return analyze_storage(
            tuple(values),
            operations,
            inputs=self.program.inputs,
            outputs=self.program.outputs,
        )

    def diagnostics(self) -> dict[str, typing.Any]:
        """Detached live-range/reuse diagnostics for provenance and review."""
        analysis = self.storage_analysis()
        return {
            "schema": "vibeqc.program-storage-plan.diagnostics.v1",
            "identity": self.identity,
            "ranges": [asdict(item) for item in analysis.ranges],
            "assignments": list(analysis.assignments),
            "slots": [asdict(item) for item in analysis.slots],
            "interference": list(analysis.interference),
            "peak_live_bytes": list(analysis.peak_live_bytes),
            "blocked_spaces": list(analysis.blocked_spaces),
            "elided_alias_allocations": [
                item.buffer for item in self.aliases if item.alias is AliasKind.VIEW
            ],
        }
