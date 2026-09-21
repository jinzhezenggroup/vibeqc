"""Deterministic compiler pass pipelines with compact diagnostics.

The manager is deliberately backend- and IR-neutral.  Scientific subsystems own
pass implementations and analyses; this module owns only ordered execution,
versioned pipeline identity, dependency validation, bisection controls, and a
small invalidation trace.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from .provenance import canonical_hash

T = typing.TypeVar("T")
_SCHEMA = "vibeqc.compiler.pass-manager.v1"


def _names(values: typing.Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(values)
    if any(type(value) is not str or not value.strip() for value in result):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicates")
    return result


@dataclass(frozen=True)
class PassStage(typing.Generic[T]):
    """One named, versioned transform in an optimizer pipeline."""

    name: str
    version: int
    transform: typing.Callable[[T], T]
    requires: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("pass name must be a nonempty string")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("pass version must be a positive integer")
        if not callable(self.transform):
            raise TypeError("pass transform must be callable")
        object.__setattr__(self, "requires", _names(self.requires, "pass requirements"))
        object.__setattr__(
            self, "invalidates", _names(self.invalidates, "invalidated analyses")
        )


@dataclass(frozen=True)
class PassRecord:
    """Compact deterministic evidence for one executed pass."""

    name: str
    version: int
    changed: bool
    before: str
    after: str
    invalidated_analyses: tuple[str, ...]


@dataclass(frozen=True)
class PassRun(typing.Generic[T]):
    """Result plus the exact active pipeline and pass-by-pass fingerprints."""

    value: T
    pipeline_identity: str
    records: tuple[PassRecord, ...]
    disabled: tuple[str, ...]
    stopped_after: str | None


@dataclass(frozen=True)
class PassManager(typing.Generic[T]):
    """Validate and run an ordered compiler pass pipeline.

    ``requires`` names earlier pass stages whose execution is mandatory.  The
    manager does not own analysis caches yet; ``invalidates`` is surfaced in the
    trace so IR adapters can make invalidation explicit without inventing a
    second analysis framework.
    """

    name: str
    version: int
    stages: tuple[PassStage[T], ...]
    fingerprint: typing.Callable[[T], str]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("pass manager name must be a nonempty string")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("pass manager version must be a positive integer")
        if not isinstance(self.stages, (tuple, list)):
            raise TypeError("pass stages must be a sequence")
        stages = tuple(self.stages)
        if any(not isinstance(stage, PassStage) for stage in stages):
            raise TypeError("pass manager requires PassStage records")
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise ValueError("pass stage names must be unique")
        prior: set[str] = set()
        for stage in stages:
            missing = set(stage.requires) - prior
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(
                    f"pass {stage.name!r} requires non-prior stage(s): {missing_list}"
                )
            prior.add(stage.name)
        if not callable(self.fingerprint):
            raise TypeError("pass manager fingerprint must be callable")
        object.__setattr__(self, "stages", stages)

    def _fingerprint(self, value: T) -> str:
        digest = self.fingerprint(value)
        if type(digest) is not str or not digest:
            raise ValueError("pass fingerprint must be a nonempty string")
        return digest

    def run(
        self,
        value: T,
        *,
        disabled: typing.Any = (),
        stop_after: str | None = None,
    ) -> PassRun[T]:
        """Execute selected stages in declaration order.

        ``disabled`` supports pass bisection without changing the pipeline
        declaration. ``stop_after`` executes through the named stage and then
        stops, providing a deterministic prefix for diagnosis.
        """

        disabled = _names(disabled, "disabled passes")
        known = {stage.name for stage in self.stages}
        unknown = set(disabled) - known
        if unknown:
            raise ValueError(f"unknown disabled pass(es): {', '.join(sorted(unknown))}")
        if stop_after is not None and stop_after not in known:
            raise ValueError(f"unknown stop_after pass: {stop_after!r}")
        if stop_after in disabled:
            raise ValueError("stop_after pass cannot also be disabled")

        disabled_set = set(disabled)
        completed: set[str] = set()
        active: list[PassStage[T]] = []
        records: list[PassRecord] = []
        result = value
        for stage in self.stages:
            if stage.name in disabled_set:
                continue
            missing = set(stage.requires) - completed
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(
                    f"pass {stage.name!r} requires disabled/unrun stage(s): {missing_list}"
                )
            before = self._fingerprint(result)
            result = stage.transform(result)
            after = self._fingerprint(result)
            changed = before != after
            records.append(
                PassRecord(
                    name=stage.name,
                    version=stage.version,
                    changed=changed,
                    before=before,
                    after=after,
                    invalidated_analyses=stage.invalidates if changed else (),
                )
            )
            active.append(stage)
            completed.add(stage.name)
            if stage.name == stop_after:
                break

        pipeline_identity = canonical_hash(
            {
                "schema": _SCHEMA,
                "manager": self.name,
                "manager_version": self.version,
                "passes": [
                    {"name": stage.name, "version": stage.version} for stage in active
                ],
            }
        )
        return PassRun(
            value=result,
            pipeline_identity=pipeline_identity,
            records=tuple(records),
            disabled=disabled,
            stopped_after=stop_after,
        )
