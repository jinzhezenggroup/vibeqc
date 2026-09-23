"""Fail-closed persistent-cache identity for bulk Libxc AOT artifacts (#1123).

The census recipe in :mod:`vibeqc_compiler.xc.bulk_aot` intentionally is not a
cross-run cache key because it does not fingerprint the downstream dependency
closure.  This module defines the missing identity/invalidation contract without
reading, writing, or promoting a persistent cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from vibeqc_compiler.common.provenance import canonical_hash

CACHE_SCHEMA = "vibeqc.libxc-aot-cache/v1"
BACKENDS = ("cpu", "cuda")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMON_REQUIRED_ROLES = frozenset(
    {"compiler-executable", "compiler-version", "system-header-manifest"}
)
_CUDA_REQUIRED_ROLES = frozenset({"cuda-device-toolchain-manifest"})


def _sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, order=True)
class CacheDependency:
    """One exact input in the transitive object-compilation dependency closure."""

    role: str
    identity: str
    content_sha256: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.role, self.identity)
        ):
            raise ValueError("cache dependency role and identity must be nonempty strings")
        _sha256(self.content_sha256, "content_sha256")

    def to_payload(self) -> dict[str, str]:
        return {
            "role": self.role,
            "identity": self.identity,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class CacheClosure:
    """Exact executable-build inputs required before a reusable key may exist.

    ``complete`` is an explicit producer assertion that every transitive input
    relevant to object generation has been included in ``dependencies``.  The
    role gate below is only a minimum structural check; it does not infer that a
    partial header/toolchain census is complete.
    """

    emission_identity: str
    translation_unit_sha256: str
    backend: str
    target: str
    flags: tuple[str, ...]
    dependencies: tuple[CacheDependency, ...]
    complete: bool = False

    def __post_init__(self) -> None:
        _sha256(self.emission_identity, "emission_identity")
        _sha256(self.translation_unit_sha256, "translation_unit_sha256")
        if self.backend not in BACKENDS:
            raise ValueError("backend must be cpu or cuda")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("target must be a nonempty string")
        if not isinstance(self.flags, tuple) or any(
            not isinstance(flag, str) or not flag for flag in self.flags
        ):
            raise ValueError("flags must be a tuple of nonempty strings")
        if type(self.complete) is not bool:
            raise ValueError("complete must be bool")
        if not isinstance(self.dependencies, tuple) or any(
            not isinstance(dependency, CacheDependency)
            for dependency in self.dependencies
        ):
            raise ValueError("dependencies must be a tuple of CacheDependency records")
        keys = [
            (dependency.role, dependency.identity) for dependency in self.dependencies
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate cache dependency role/identity")

    @property
    def required_roles(self) -> frozenset[str]:
        roles = set(_COMMON_REQUIRED_ROLES)
        if self.backend == "cuda":
            roles.update(_CUDA_REQUIRED_ROLES)
        return frozenset(roles)

    @property
    def missing_roles(self) -> tuple[str, ...]:
        present = {dependency.role for dependency in self.dependencies}
        return tuple(sorted(self.required_roles - present))

    @property
    def reusable(self) -> bool:
        return self.complete and not self.missing_roles

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": CACHE_SCHEMA,
            "emission_identity": self.emission_identity,
            "translation_unit_sha256": self.translation_unit_sha256,
            "backend": self.backend,
            "target": self.target,
            # Flag ordering is command-line semantics and must not be normalized.
            "flags": list(self.flags),
            "dependencies": [
                dependency.to_payload() for dependency in sorted(self.dependencies)
            ],
            "closure_complete": self.complete,
        }

    @property
    def cache_key(self) -> str | None:
        """Return a reusable key only for an explicitly complete closure."""
        if not self.reusable:
            return None
        return canonical_hash(self.to_payload())

    @property
    def blocker(self) -> dict[str, Any] | None:
        if self.reusable:
            return None
        return {
            "reason": "incomplete-dependency-closure",
            "closure_complete": self.complete,
            "missing_roles": list(self.missing_roles),
        }


def closure_from_probe_recipe(
    recipe: dict[str, Any],
    *,
    backend: str,
    target: str,
    dependencies: tuple[CacheDependency, ...],
    complete: bool,
) -> CacheClosure:
    """Upgrade an offline census recipe only when extra closure evidence exists.

    The compiler executable and version recorded by ``compile_probe`` are folded
    into the dependency set automatically.  System-header and, for CUDA, device
    toolchain manifests remain explicit caller-supplied evidence.
    """
    try:
        compiler = recipe["compiler"]
        executable_sha256 = compiler["executable_sha256"]
        version = compiler["version"]
        emission_identity = recipe["emission_identity"]
        translation_unit_sha256 = recipe["translation_unit_sha256"]
        flags = recipe["flags"]
    except (KeyError, TypeError) as error:
        raise ValueError("invalid census probe recipe") from error
    if not isinstance(version, str) or not version:
        raise ValueError("compiler version must be nonempty")
    if not isinstance(flags, list) or any(
        not isinstance(flag, str) or not flag for flag in flags
    ):
        raise ValueError("probe flags must be a list of nonempty strings")
    automatic = (
        CacheDependency("compiler-executable", "compiler", executable_sha256),
        CacheDependency(
            "compiler-version", "compiler", canonical_hash({"version": version})
        ),
    )
    return CacheClosure(
        emission_identity=emission_identity,
        translation_unit_sha256=translation_unit_sha256,
        backend=backend,
        target=target,
        flags=tuple(flags),
        dependencies=automatic + dependencies,
        complete=complete,
    )


def invalidation_reasons(before: CacheClosure, after: CacheClosure) -> tuple[str, ...]:
    """Explain why a persistent object cannot be reused across two closures."""
    reasons: list[str] = []
    scalar_fields = (
        ("emission_identity", "emission-identity"),
        ("translation_unit_sha256", "translation-unit"),
        ("backend", "backend"),
        ("target", "target"),
        ("flags", "compiler-flags"),
        ("complete", "closure-completeness"),
    )
    for attribute, reason in scalar_fields:
        if getattr(before, attribute) != getattr(after, attribute):
            reasons.append(reason)
    before_dependencies = {
        (dependency.role, dependency.identity): dependency.content_sha256
        for dependency in before.dependencies
    }
    after_dependencies = {
        (dependency.role, dependency.identity): dependency.content_sha256
        for dependency in after.dependencies
    }
    if before_dependencies.keys() != after_dependencies.keys():
        reasons.append("dependency-set")
    if any(
        before_dependencies[key] != after_dependencies[key]
        for key in before_dependencies.keys() & after_dependencies.keys()
    ):
        reasons.append("dependency-content")
    return tuple(reasons)
