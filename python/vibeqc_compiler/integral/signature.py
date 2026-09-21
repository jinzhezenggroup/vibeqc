"""Deterministic manifests for compiler-owned generated kernel signatures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class GeneratedKernelArgument:
    """One compiler-owned internal kernel argument."""

    c_type: str
    name: str
    wrapper_expression: str | None = None

    def __post_init__(self) -> None:
        if not self.c_type.strip():
            raise ValueError("generated kernel argument type must be non-empty")
        if not self.name.isidentifier():
            raise ValueError(
                f"generated kernel argument name must be an identifier: {self.name!r}"
            )

    @property
    def declaration(self) -> str:
        """Return the C/CUDA declaration for this argument."""

        return f"{self.c_type} {self.name}"

    def forwarded_expression(self) -> str:
        """Return the stable-wrapper expression forwarded to the internal ABI."""

        return self.wrapper_expression or self.name


@dataclass(frozen=True, slots=True)
class GeneratedKernelSignature:
    """Ordered internal ABI manifest shared by declaration and launch packing."""

    arguments: tuple[GeneratedKernelArgument, ...]

    def __post_init__(self) -> None:
        names = self.names
        if len(names) != len(set(names)):
            raise ValueError("generated kernel signature contains duplicate arguments")

    @property
    def names(self) -> tuple[str, ...]:
        """Return the ordered internal ABI argument names."""

        return tuple(argument.name for argument in self.arguments)

    @property
    def identity(self) -> tuple[tuple[str, str], ...]:
        """Return a deterministic type/name identity for artifact provenance."""

        return tuple((argument.c_type, argument.name) for argument in self.arguments)

    def without(self, *dead_names: str) -> GeneratedKernelSignature:
        """Remove proven-dead arguments, rejecting stale pruning requests."""

        requested = frozenset(dead_names)
        unknown = requested.difference(self.names)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"cannot prune unknown generated arguments: {names}")
        return GeneratedKernelSignature(
            tuple(
                argument
                for argument in self.arguments
                if argument.name not in requested
            )
        )

    def pruning_diagnostics(
        self,
        original: GeneratedKernelSignature,
    ) -> dict[str, object]:
        """Describe compiler-owned ABI pruning without changing either signature."""

        if not isinstance(original, GeneratedKernelSignature):
            raise TypeError("signature diagnostics require a GeneratedKernelSignature")
        original_names = original.names
        retained = set(self.names)
        if not retained <= set(original_names):
            raise ValueError(
                "pruned signature contains arguments absent from the source"
            )
        ordered_retained = tuple(name for name in original_names if name in retained)
        if ordered_retained != self.names:
            raise ValueError("pruned signature must preserve source argument order")
        removed = tuple(name for name in original_names if name not in retained)
        return {
            "schema": "vibeqc.compiler.signature-pruning.v1",
            "parameters_before": list(original_names),
            "parameters_after": list(self.names),
            "removed_parameters": list(removed),
            "parameter_count_before": len(original_names),
            "parameter_count_after": len(self.names),
            "reason": "compile-time liveness",
        }

    def parameter_list(self, *, indent: str = "    ") -> str:
        """Render the internal function/kernel declaration from the manifest."""

        return (",\n").join(
            f"{indent}{argument.declaration}" for argument in self.arguments
        )

    def argument_list(
        self,
        *,
        wrapper: bool = False,
        replacements: Mapping[str, str] | None = None,
    ) -> str:
        """Render a launch/call list from the same ordered manifest."""

        replacements = {} if replacements is None else replacements
        unknown = set(replacements).difference(self.names)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown generated argument replacements: {names}")
        values = []
        for argument in self.arguments:
            value = argument.forwarded_expression() if wrapper else argument.name
            values.append(replacements.get(argument.name, value))
        return ", ".join(values)
