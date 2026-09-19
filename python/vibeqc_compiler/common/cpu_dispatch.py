"""Portable CPU feature detection and generated-kernel target selection."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .cpu_target import CpuTargetInfo


def normalize_cpu_architecture(value: str) -> str:
    name = value.strip().lower().replace("-", "_")
    if name in ("x86_64", "amd64"):
        return "x86_64"
    if name in ("aarch64", "arm64"):
        return "aarch64"
    return name or "unknown"


def _binary_abi(value: str) -> str:
    triple = value.strip().lower()
    if "darwin" in triple or "apple" in triple:
        return "darwin"
    if "linux" in triple:
        return "linux"
    if any(token in triple for token in ("mingw", "windows", "msvc")):
        return "windows"
    return "unknown"


def _runtime_binary_abi() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform in ("win32", "cygwin"):
        return "windows"
    return "unknown"


def cpu_binary_target_supported(
    compiler_target: str,
    runtime: CpuRuntimeFeatures,
) -> bool:
    """Check the compiled shared object's architecture/ABI before dlopen."""

    parts = compiler_target.strip().split("-", 1)
    architecture = normalize_cpu_architecture(parts[0] if parts else "")
    abi = _binary_abi(compiler_target)
    runtime_abi = _runtime_binary_abi()
    return (
        architecture == runtime.architecture
        and abi != "unknown"
        and runtime_abi != "unknown"
        and abi == runtime_abi
    )


@dataclass(frozen=True, slots=True)
class CpuRuntimeFeatures:
    """Runtime ISA facts; CPU model/brand strings are deliberately excluded."""

    architecture: str
    features: tuple[str, ...]
    source: str = "explicit"

    def __post_init__(self) -> None:
        architecture = normalize_cpu_architecture(self.architecture)
        features = tuple(sorted(set(self.features)))
        if not self.source:
            raise ValueError("CPU feature source must be named")
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "features", features)

    def to_payload(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "features": self.features,
            "source": self.source,
        }


def _linux_cpu_features() -> set[str]:
    try:
        records = Path("/proc/cpuinfo").read_text().lower().split("\n\n")
    except OSError:
        return set()
    common: set[str] | None = None
    for record in records:
        fields = {}
        for line in record.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        if not any(key in fields for key in ("processor", "flags", "features")):
            continue
        # Threads may migrate: a flag advertised by only one CPU is not safe.
        # A processor without feature information conservatively admits none.
        flags = set(fields.get("flags", fields.get("features", "")).split())
        common = flags if common is None else common & flags
    return common or set()


def _darwin_cpu_features() -> set[str]:
    flags: set[str] = set()
    for name in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", name],
                text=True,
                timeout=2,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        flags.update(item.lower() for item in output.split())
    return flags


def detect_cpu_features() -> CpuRuntimeFeatures:
    """Conservatively detect only ISA facts used by generated CPU candidates."""

    architecture = normalize_cpu_architecture(platform.machine())
    if architecture != "x86_64":
        return CpuRuntimeFeatures(architecture, (), "conservative-non-x86")
    if sys.platform.startswith("linux"):
        raw = _linux_cpu_features()
        source = "linux-proc-cpuinfo"
    elif sys.platform == "darwin":
        raw = _darwin_cpu_features()
        source = "darwin-sysctl"
    else:
        raw = set()
        source = "conservative-unknown-os"
    admitted = tuple(
        feature for feature in ("avx2", "avx512f", "fma") if feature in raw
    )
    return CpuRuntimeFeatures(architecture, admitted, source)


def cpu_target_supported(
    target: CpuTargetInfo,
    runtime: CpuRuntimeFeatures,
) -> bool:
    if target.architecture != "portable" and (
        normalize_cpu_architecture(target.architecture) != runtime.architecture
    ):
        return False
    available = set(runtime.features)
    return all(feature in available for feature in target.features)


@dataclass(frozen=True, slots=True)
class CpuDispatchDecision:
    selected_target: str
    available_targets: tuple[str, ...]
    runtime: CpuRuntimeFeatures
    forced_target: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "vibeqc.cpu-dispatch.v1",
            "selected_target": self.selected_target,
            "available_targets": self.available_targets,
            "runtime": self.runtime.to_payload(),
            "forced_target": self.forced_target,
        }


def select_cpu_target(
    targets: Iterable[CpuTargetInfo],
    runtime: CpuRuntimeFeatures,
    *,
    forced_target: str | None = None,
) -> CpuDispatchDecision:
    """Select the widest supported candidate without loading its binary first."""

    candidates = tuple(targets)
    if not candidates:
        raise ValueError("CPU dispatch requires at least one candidate")
    names = tuple(target.name for target in candidates)
    if len(set(names)) != len(names):
        raise ValueError("CPU dispatch candidate target names must be unique")
    forced = forced_target
    if forced is None:
        forced = os.environ.get("VIBEQC_CPU_TARGET") or None
    if forced is not None:
        matches = tuple(target for target in candidates if target.name == forced)
        if not matches:
            raise ValueError(f"forced CPU target {forced!r} is not in this bundle")
        target = matches[0]
        if not cpu_target_supported(target, runtime):
            raise ValueError(
                f"forced CPU target {forced!r} is unsupported by the runtime ISA"
            )
        return CpuDispatchDecision(target.name, names, runtime, forced)

    supported = tuple(
        target for target in candidates if cpu_target_supported(target, runtime)
    )
    if not supported:
        raise ValueError("CPU bundle has no candidate compatible with this runtime")
    target = max(supported, key=lambda item: (item.vector_lanes, item.name))
    return CpuDispatchDecision(target.name, names, runtime, None)
