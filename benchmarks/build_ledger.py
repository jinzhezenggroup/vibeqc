"""Collect reproducible build timing and rebuild-scope provenance.

The ledger intentionally treats the build tool as the source of truth.  Ninja's
``.ninja_log`` contains one row per output and records compile durations even
when a compiler cache turns a rebuild into a hit.  The script does not infer a
critical path from source size alone; missing measurements are represented as
``null`` so a later benchmark cannot accidentally publish fabricated numbers.

Use ``--build`` to time one additional ``cmake --build`` invocation, or import
``build_ledger`` from a benchmark harness that runs the clean/cache-hit and
touch scenarios under controlled conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NinjaLogEntry:
    """One output timing row from Ninja's append-only build log."""

    output: str
    duration_seconds: float
    start_milliseconds: int
    end_milliseconds: int
    restat: int
    hash: str

    @property
    def is_cuda(self) -> bool:
        """Whether this output is a CUDA source/object or CUDA link artifact."""

        lowered = self.output.lower()
        return lowered.endswith(
            (".cu", ".cuh", ".o", ".obj", ".so", ".ptx", ".cubin", ".fatbin")
        ) and (
            "cuda" in lowered
            or "generated" in lowered
            or lowered.endswith((".cu", ".cuh", ".ptx", ".cubin", ".fatbin"))
        )


def _parse_ninja_log(path: Path) -> tuple[NinjaLogEntry, ...]:
    """Parse Ninja log v5--v7 rows, ignoring malformed/header lines."""

    if not path.exists():
        return ()
    entries: list[NinjaLogEntry] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 5:
            continue
        try:
            start, end, restat = (int(fields[index]) for index in range(3))
        except ValueError:
            continue
        output, digest = fields[3], fields[4]
        if end < start:
            continue
        entries.append(
            NinjaLogEntry(
                output=output,
                duration_seconds=(end - start) / 1000.0,
                start_milliseconds=start,
                end_milliseconds=end,
                restat=restat,
                hash=digest,
            )
        )
    return tuple(entries)


def _command_output(arguments: Sequence[str]) -> str:
    """Return a version string without making a missing tool fatal."""

    try:
        completed = subprocess.run(
            tuple(arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (completed.stdout or completed.stderr).strip()


def _tool_version(name: str, *arguments: str) -> dict[str, str]:
    path = shutil.which(name)
    if path is None:
        return {"path": "", "version": ""}
    return {"path": path, "version": _command_output((path, *arguments))}


def _git_metadata(root: Path) -> dict[str, object]:
    commit = _command_output(("git", "-C", str(root), "rev-parse", "HEAD"))
    status = _command_output(("git", "-C", str(root), "status", "--porcelain"))
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _cpu_metadata() -> dict[str, object]:
    logical = os.cpu_count() or 1
    physical_text = _command_output(("lscpu", "-p=CPU,CORE,SOCKET"))
    physical = None
    if physical_text:
        cores = set()
        for line in physical_text.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split(",")
            if len(fields) >= 3:
                # CPU is the logical-thread field; (core, socket) identifies
                # one physical core even when SMT exposes multiple CPUs.
                cores.add((fields[1], fields[2]))
        physical = len(cores)
    memory_bytes = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError):
        pass
    model = platform.processor() or _command_output(("uname", "-p"))
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "model": model,
        "logical_cores": logical,
        "physical_cores": physical,
        "memory_bytes": memory_bytes,
    }


def _cmake_cache(build_directory: Path) -> dict[str, str]:
    """Read relevant CMake cache values without depending on CMake APIs."""

    path = build_directory / "CMakeCache.txt"
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("//", "#")):
            continue
        match = re.match(r"([^:=]+)(?::[^=]+)?=(.*)$", line)
        if match and (
            "CUDA" in match.group(1)
            or match.group(1).startswith("VIBEQC_")
            or match.group(1) == "CMAKE_GENERATOR"
        ):
            values[match.group(1)] = match.group(2)
    return dict(sorted(values.items()))


def _cache_metadata() -> dict[str, object]:
    for name in ("sccache", "ccache"):
        path = shutil.which(name)
        if path is not None:
            return {
                "name": name,
                "path": path,
                "version": _command_output((path, "--version")),
                "stats": _command_output((path, "--show-stats" if name == "sccache" else "-s")),
            }
    return {"name": "", "path": "", "version": "", "stats": ""}


def _artifact_sizes(
    output: str,
    build_directory: Path | None,
    repository_root: Path | None = None,
) -> tuple[int | None, int | None]:
    """Return source/object sizes for a Ninja output when files still exist."""

    if build_directory is None:
        return None, None
    object_path = Path(output)
    if not object_path.is_absolute():
        object_path = build_directory / object_path
    object_bytes = object_path.stat().st_size if object_path.is_file() else None
    source_bytes = None
    if object_path.suffix in (".o", ".obj"):
        source_path = Path(str(object_path)[:-len(object_path.suffix)])
        candidates = [source_path]
        # CMake/Ninja object paths retain the source-relative suffix after a
        # ``CMakeFiles/<target>.dir/`` component. Resolve that suffix against
        # both the build and repository roots for handwritten and generated TUs.
        marker = ".dir/"
        relative = str(source_path).split(marker, 1)[-1] if marker in str(source_path) else ""
        if relative:
            candidates.append(build_directory / relative)
            if repository_root is not None:
                candidates.append(repository_root / relative)
        for candidate in candidates:
            if candidate.suffix == ".cu" and candidate.is_file():
                source_bytes = candidate.stat().st_size
                break
    elif object_path.suffix in (".cu", ".cuh") and object_path.is_file():
        source_bytes = object_path.stat().st_size
    return source_bytes, object_bytes


def _ninja_summary(
    entries: Iterable[NinjaLogEntry],
    build_directory: Path | None = None,
    repository_root: Path | None = None,
) -> dict[str, object]:
    rows = tuple(entries)
    if not rows:
        return {
            "entry_count": 0,
            "cuda_entry_count": 0,
            "total_recorded_seconds": 0.0,
            "critical_path_seconds": 0.0,
            "critical_path_output": None,
            "critical_path_translation_units": [],
            "final_host_link_seconds": None,
            "cuda_device_link_seconds": None,
            "maximum_concurrent_compiler_memory_bytes": None,
            "maximum_compiler_memory_bytes": None,
            "entries": [],
        }
    critical = max(rows, key=lambda row: row.duration_seconds)
    host_link_rows = tuple(
        row
        for row in rows
        if row.output.endswith((".so", ".a", ".dll", ".dylib"))
        or row.output.rsplit("/", 1)[-1] in ("vibeqc", "libvibeqc")
    )
    device_link_rows = tuple(
        row for row in rows if "cuda_device_link" in row.output or "cmake_device_link" in row.output
    )

    def entry_payload(row: NinjaLogEntry) -> dict[str, object]:
        source_bytes, object_bytes = _artifact_sizes(
            row.output, build_directory, repository_root
        )
        return {
            **asdict(row),
            "is_cuda": row.is_cuda,
            "source_bytes": source_bytes,
            "object_bytes": object_bytes,
            # Cache hit/miss is populated by a launcher wrapper when one is
            # configured; Ninja's native log has no cache-status column.
            "cache_status": "unknown",
        }

    return {
        "entry_count": len(rows),
        "cuda_entry_count": sum(row.is_cuda for row in rows),
        "total_recorded_seconds": sum(row.duration_seconds for row in rows),
        "critical_path_seconds": critical.duration_seconds,
        "critical_path_output": critical.output,
        "critical_path_translation_units": [critical.output],
        "final_host_link_seconds": max(
            (row.duration_seconds for row in host_link_rows), default=None
        ),
        "cuda_device_link_seconds": max(
            (row.duration_seconds for row in device_link_rows), default=None
        ),
        # Ninja does not persist RSS.  Keep this explicit rather than
        # pretending the longest compiler row is a memory measurement.
        "maximum_concurrent_compiler_memory_bytes": None,
        "maximum_compiler_memory_bytes": None,
        "entries": [entry_payload(row) for row in rows],
    }


def run_build(build_directory: Path, *, target: str | None = None) -> dict[str, object]:
    """Time a finite build invocation and return its process provenance."""

    command = ["cmake", "--build", str(build_directory)]
    if target:
        command.extend(("--target", target))
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def build_ledger(
    repository_root: Path,
    build_directory: Path,
    *,
    build_result: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable ledger from source, toolchain, and Ninja logs."""

    entries = _parse_ninja_log(build_directory / ".ninja_log")
    cmake_values = _cmake_cache(build_directory)
    cuda_architectures = {
        key: value
        for key, value in cmake_values.items()
        if "ARCHITECTURE" in key or "VIBEQC_CUDA" in key
    }
    cuda_compiler = cmake_values.get("CMAKE_CUDA_COMPILER", "nvcc")
    host_compiler = cmake_values.get(
        "CMAKE_CXX_COMPILER", os.environ.get("CXX", "c++")
    )
    scenarios = {
        name: {"status": "not_run", "wall_seconds": None}
        for name in (
            "clean_single_architecture_release",
            "no_op",
            "exact_cache_hit_rebuild",
            "touch_host_rhf_graph_plan",
            "touch_generic_cuda_kernel_family",
            "touch_generated_shell_class",
            "touch_generated_consumer",
            "final_host_link",
            "cuda_device_link",
        )
    }
    if build_result is not None:
        scenarios["recorded_build"] = build_result
    return {
        "schema_version": 1,
        "benchmark": "cuda_build_ledger",
        "repository": str(repository_root),
        "git": _git_metadata(repository_root),
        "host": _cpu_metadata(),
        "toolchain": {
            "cuda": _tool_version(cuda_compiler, "--version"),
            "cmake": _tool_version("cmake", "--version"),
            "ninja": _tool_version("ninja", "--version"),
            "host_cxx": _tool_version(host_compiler, "--version"),
            "cache": _cache_metadata(),
        },
        "cmake": {
            "build_directory": str(build_directory),
            "cache": cmake_values,
            "cuda_architectures": cuda_architectures,
        },
        "scenarios": scenarios,
        "ninja": _ninja_summary(entries, build_directory, repository_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true", help="time one cmake --build invocation")
    parser.add_argument("--target")
    arguments = parser.parse_args()
    result = (
        run_build(arguments.build_dir, target=arguments.target)
        if arguments.build
        else None
    )
    payload = build_ledger(
        arguments.repository.resolve(), arguments.build_dir.resolve(), build_result=result
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"JSON result: {arguments.output}")


if __name__ == "__main__":
    main()
