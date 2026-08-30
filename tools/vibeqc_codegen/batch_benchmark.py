"""Batch-compile and screen fused shell-class CUDA candidates.

Every candidate is emitted into its own CUDA translation unit, while one
linked executable runs all correctness and timing checks in a single process.
The Python driver launches that executable through exactly one ``srun`` GPU
allocation and combines runtime measurements with per-TU ``ptxas`` resources.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .benchmark import emit_shell_class_benchmark_cuda
from .production import load_production_manifest
from .shell_spec import (
    FUSED_SHELL_SPEC_BY_NAME,
    FUSED_SHELL_SPECS,
    ShellClassSpec,
)

PRODUCTION_SHELL_CLASSES = frozenset(
    spec.name
    for spec in load_production_manifest(
        Path(__file__).with_name("production_shell_classes.json")
    )
)
DEFAULT_CANDIDATES = tuple(
    spec for spec in FUSED_SHELL_SPECS if spec.name not in PRODUCTION_SHELL_CLASSES
)


@dataclass(frozen=True, slots=True)
class KernelResources:
    """Static resources reported by CUDA 12.9 ``ptxas`` for one kernel."""

    function: str
    registers: int
    stack_bytes: int
    spill_store_bytes: int
    spill_load_bytes: int
    shared_bytes: int


def candidate_specs(
    names: Iterable[str] | None = None,
) -> tuple[ShellClassSpec, ...]:
    """Resolve an explicit sparse list of uncovered s/p/d/f candidates."""

    if names is None:
        raise ValueError(
            "candidate screening requires --profile or an explicit --shell-class"
        )
    specifications = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        try:
            specification = FUSED_SHELL_SPEC_BY_NAME[name]
        except KeyError as error:
            choices = ", ".join(FUSED_SHELL_SPEC_BY_NAME)
            raise ValueError(f"unknown shell class {name!r}; choose from {choices}") from error
        if name in PRODUCTION_SHELL_CLASSES:
            raise ValueError(f"{name} is already covered by production AOT")
        specifications.append(specification)
        seen.add(name)
    if not specifications:
        raise ValueError("at least one candidate shell class is required")
    return tuple(specifications)


def rank_profiled_candidates(
    payload: dict[str, object], limit: int
) -> tuple[ShellClassSpec, ...]:
    """Select the highest active primitive-work classes from a real profile."""

    if limit < 1:
        raise ValueError("candidate limit must be positive")
    rows = payload.get("shell_classes")
    if not isinstance(rows, list):
        raise TypeError("profile JSON does not contain a shell_classes list")
    ranked = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("class")
        if (
            isinstance(name, str)
            and name not in PRODUCTION_SHELL_CLASSES
            and name in FUSED_SHELL_SPEC_BY_NAME
        ):
            ranked.append(FUSED_SHELL_SPEC_BY_NAME[name])
    if not ranked:
        raise ValueError("profile contains no uncovered compilable shell classes")
    return tuple(ranked[:limit])


def emit_candidate_translation_unit(
    spec: ShellClassSpec,
    *,
    task_count: int,
    primitive_count: int,
    warmups: int,
    iterations: int,
    samples: int,
) -> str:
    """Emit one independently compilable candidate benchmark translation unit."""

    source = emit_shell_class_benchmark_cuda(
        spec,
        task_count=task_count,
        primitive_count=primitive_count,
        warmups=warmups,
        iterations=iterations,
        samples=samples,
    )
    entry = f'vibeqc_run_shell_class_{spec.name}'
    source = source.replace("int main() {", f'extern "C" int {entry}() {{', 1)
    marker = r'{\"task_count\":%u'
    replacement = rf'{{\"shell_class\":\"{spec.name}\",\"task_count\":%u'
    if marker not in source:
        raise RuntimeError("benchmark JSON marker changed unexpectedly")
    return source.replace(marker, replacement, 1)


def emit_batch_driver(specifications: Iterable[ShellClassSpec]) -> str:
    """Emit the one-process driver that invokes every compiled candidate."""

    specs = tuple(specifications)
    declarations = "\n".join(
        f'extern "C" int vibeqc_run_shell_class_{spec.name}();' for spec in specs
    )
    calls = "\n".join(
        f"  failures += vibeqc_run_shell_class_{spec.name}() != 0;" for spec in specs
    )
    return f"""#include <cuda_runtime.h>
#include <cstdio>

{declarations}

int main() {{
  const cudaError_t initialization = cudaFree(nullptr);
  if (initialization != cudaSuccess) {{
    std::fprintf(stderr, "CUDA initialization failed: %s\\n",
                 cudaGetErrorString(initialization));
    return 2;
  }}
  int failures = 0;
{calls}
  return failures == 0 ? 0 : 3;
}}
"""


_RESOURCE_PATTERN = re.compile(
    r"Function properties for (?P<function>[^\n]+)\n"
    r"\s+(?P<stack>\d+) bytes stack frame, "
    r"(?P<spill_stores>\d+) bytes spill stores, "
    r"(?P<spill_loads>\d+) bytes spill loads\n"
    r"ptxas info\s+: Used (?P<registers>\d+) registers,.*?, "
    r"(?P<shared>\d+) bytes smem",
)


def parse_ptxas_resources(
    output: str,
    shell_class: str,
    *,
    symbol_prefix: str | None = None,
) -> tuple[KernelResources, ...]:
    """Extract the four force kernels from verbose ``ptxas`` output.

    Autotuning translation units suffix every generated symbol so several
    schedules for the same shell class can coexist in one executable.
    ``symbol_prefix`` lets that driver reuse the production resource parser.
    """

    marker = symbol_prefix or f"generated_{shell_class}_shell_class_force_"
    resources = []
    for match in _RESOURCE_PATTERN.finditer(output):
        function = match.group("function")
        if marker not in function:
            continue
        resources.append(
            KernelResources(
                function=function,
                registers=int(match.group("registers")),
                stack_bytes=int(match.group("stack")),
                spill_store_bytes=int(match.group("spill_stores")),
                spill_load_bytes=int(match.group("spill_loads")),
                shared_bytes=int(match.group("shared")),
            )
        )
    return tuple(resources)


def _compile_candidate(
    nvcc: Path,
    architecture: str,
    directory: Path,
    spec: ShellClassSpec,
) -> dict[str, object]:
    """Compile one candidate TU and retain diagnostics for automatic gates."""

    source = directory / f"{spec.name}_candidate.cu"
    obj = directory / f"{spec.name}_candidate.o"
    started = time.monotonic()
    result = subprocess.run(
        [
            str(nvcc),
            "-std=c++17",
            f"-arch={architecture}",
            "-O3",
            "-Xptxas=-v",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    duration_seconds = time.monotonic() - started
    diagnostics = result.stdout + result.stderr
    return {
        "name": spec.name,
        "object": obj,
        "returncode": result.returncode,
        "compile_seconds": duration_seconds,
        "source_bytes": _artifact_size(source),
        "object_bytes": _artifact_size(obj),
        "diagnostics": diagnostics,
        "resources": parse_ptxas_resources(diagnostics, spec.name),
    }


def _runtime_environment(nvcc: Path) -> dict[str, str]:
    """Expose the selected toolkit runtime without overriding GPU allocation."""

    environment = dict(os.environ)
    if environment.get("CUDA_VISIBLE_DEVICES") == "":
        environment.pop("CUDA_VISIBLE_DEVICES")
    library = nvcc.parent.parent / "lib64"
    previous = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        str(library) if not previous else f"{library}:{previous}"
    )
    return environment


def _artifact_size(path: Path) -> int | None:
    """Return a generated artifact's byte size, if compilation produced it."""

    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        pass
    return None


def _resource_gate(
    resources: tuple[KernelResources, ...],
    *,
    maximum_registers: int,
    maximum_stack_bytes: int,
    maximum_shared_bytes: int,
) -> tuple[bool, list[str]]:
    """Apply architecture-specific static resource limits."""

    reasons = []
    if len(resources) != 4:
        reasons.append(f"expected 4 fused kernel resource records, found {len(resources)}")
    if any(item.spill_store_bytes or item.spill_load_bytes for item in resources):
        reasons.append("ptxas reported local-memory spills")
    if any(item.registers > maximum_registers for item in resources):
        reasons.append(f"register use exceeds {maximum_registers}")
    if any(item.stack_bytes > maximum_stack_bytes for item in resources):
        reasons.append(f"stack frame exceeds {maximum_stack_bytes} bytes")
    if any(item.shared_bytes > maximum_shared_bytes for item in resources):
        reasons.append(f"static shared memory exceeds {maximum_shared_bytes} bytes")
    return not reasons, reasons


def _run_batch(arguments: argparse.Namespace) -> dict[str, object]:
    """Generate, compile, run, and gate one candidate batch."""

    if arguments.profile is not None:
        profile = json.loads(arguments.profile.read_text(encoding="utf-8"))
        specifications = rank_profiled_candidates(profile, arguments.limit)
    else:
        specifications = candidate_specs(arguments.shell_class or None)

    work_directory_owner = None
    if arguments.work_directory is None:
        work_directory_owner = tempfile.TemporaryDirectory(
            prefix="vibeqc-shell-batch-"
        )
        directory = Path(work_directory_owner.name)
    else:
        directory = arguments.work_directory
        directory.mkdir(parents=True, exist_ok=True)

    try:
        for spec in specifications:
            source = emit_candidate_translation_unit(
                spec,
                task_count=arguments.tasks,
                primitive_count=arguments.primitives,
                warmups=arguments.warmups,
                iterations=arguments.iterations,
                samples=arguments.samples,
            )
            (directory / f"{spec.name}_candidate.cu").write_text(
                source, encoding="utf-8"
            )

        with ThreadPoolExecutor(max_workers=arguments.compile_jobs) as executor:
            compile_rows = list(
                executor.map(
                    lambda spec: _compile_candidate(
                        arguments.nvcc, arguments.architecture, directory, spec
                    ),
                    specifications,
                )
            )

        runnable_specs = tuple(
            spec
            for spec, row in zip(specifications, compile_rows, strict=True)
            if row["returncode"] == 0
        )
        runtime_rows: dict[str, dict[str, object]] = {}
        run_stderr = ""
        run_returncode = None
        link_seconds: float | None = None
        linked_binary_bytes: int | None = None
        if runnable_specs:
            driver = directory / "batch_driver.cu"
            driver.write_text(emit_batch_driver(runnable_specs), encoding="utf-8")
            executable = directory / "shell_batch_benchmark"
            objects = [
                str(row["object"])
                for row in compile_rows
                if row["returncode"] == 0
            ]
            link_started = time.monotonic()
            link = subprocess.run(
                [
                    str(arguments.nvcc),
                    "-std=c++17",
                    f"-arch={arguments.architecture}",
                    "-O3",
                    str(driver),
                    *objects,
                    "-o",
                    str(executable),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            link_seconds = time.monotonic() - link_started
            if link.returncode != 0:
                raise RuntimeError(link.stdout + link.stderr)
            linked_binary_bytes = _artifact_size(executable)
            run = subprocess.run(
                [
                    arguments.srun,
                    f"--partition={arguments.partition}",
                    f"--gres={arguments.gres}",
                    "--nodes=1",
                    "--ntasks=1",
                    str(executable),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=arguments.timeout,
                env=_runtime_environment(arguments.nvcc),
            )
            run_returncode = run.returncode
            run_stderr = run.stderr
            for line in run.stdout.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = row.get("shell_class")
                if isinstance(name, str):
                    runtime_rows[name] = row

        candidates = []
        accepted = []
        for spec, compile_row in zip(specifications, compile_rows, strict=True):
            resources = compile_row["resources"]
            resource_ok, reasons = _resource_gate(
                resources,
                maximum_registers=arguments.max_registers,
                maximum_stack_bytes=arguments.max_stack_bytes,
                maximum_shared_bytes=arguments.max_shared_bytes,
            )
            if compile_row["returncode"] != 0:
                reasons.append("NVCC compilation failed")
            runtime = runtime_rows.get(spec.name)
            if runtime is None:
                reasons.append("candidate did not produce a runtime result")
            else:
                maximum_force = float(runtime["maximum_force"])
                maximum_error = float(runtime["maximum_force_error"])
                tolerance = arguments.absolute_tolerance + (
                    arguments.relative_tolerance * maximum_force
                )
                if maximum_error > tolerance:
                    reasons.append(
                        f"force error {maximum_error:.3e} exceeds {tolerance:.3e}"
                    )
                if float(runtime["speedup"]) < arguments.minimum_speedup:
                    reasons.append(
                        f"speedup is below {arguments.minimum_speedup:.3f}x"
                    )
            passed = resource_ok and not reasons
            if passed:
                accepted.append(spec.name)
            candidates.append(
                {
                    "shell_class": spec.name,
                    "shell_angular": list(spec.angular),
                    "component_count": spec.component_count,
                    "pair_orders": list(spec.pair_orders),
                    "compile_succeeded": compile_row["returncode"] == 0,
                    "compile_seconds": compile_row.get("compile_seconds"),
                    "source_bytes": compile_row.get("source_bytes"),
                    "object_bytes": compile_row.get("object_bytes"),
                    "resources": [asdict(item) for item in resources],
                    "runtime": runtime,
                    "accepted": passed,
                    "rejection_reasons": reasons,
                }
            )
            if arguments.verbose and compile_row["diagnostics"]:
                print(compile_row["diagnostics"], file=sys.stderr, end="")

        return {
            "schema_version": 1,
            "architecture": arguments.architecture,
            "nvcc": str(arguments.nvcc),
            "single_gpu_process": True,
            "srun": {
                "partition": arguments.partition,
                "gres": arguments.gres,
                "returncode": run_returncode,
                "stderr": run_stderr,
            },
            "artifacts": {
                "linked_executable_bytes": linked_binary_bytes,
                "link_seconds": link_seconds,
                "candidate_objects": {
                    row["name"]: row.get("object_bytes") for row in compile_rows
                },
            },
            "gates": {
                "minimum_speedup": arguments.minimum_speedup,
                "absolute_tolerance": arguments.absolute_tolerance,
                "relative_tolerance": arguments.relative_tolerance,
                "maximum_registers": arguments.max_registers,
                "maximum_stack_bytes": arguments.max_stack_bytes,
                "maximum_shared_bytes": arguments.max_shared_bytes,
                "spills_allowed": False,
            },
            "accepted_shell_classes": accepted,
            "candidates": candidates,
        }
    finally:
        if work_directory_owner is not None:
            work_directory_owner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nvcc",
        type=Path,
        default=Path("/group/software/cuda-12.9.1/bin/nvcc"),
    )
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--srun", default="srun")
    parser.add_argument("--partition", default="main")
    parser.add_argument("--gres", default="gpu:5090:1")
    parser.add_argument("--shell-class", action="append")
    parser.add_argument(
        "--profile",
        type=Path,
        help="rank candidates by an --all-orders active profile JSON",
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--tasks", type=int, default=512)
    parser.add_argument("--primitives", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--compile-jobs", type=int, default=2)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-10)
    parser.add_argument("--relative-tolerance", type=float, default=2.0e-10)
    parser.add_argument("--max-registers", type=int, default=192)
    parser.add_argument("--max-stack-bytes", type=int, default=128)
    parser.add_argument("--max-shared-bytes", type=int, default=49152)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    if arguments.compile_jobs < 1:
        parser.error("--compile-jobs must be positive")
    if arguments.profile is not None and arguments.shell_class:
        parser.error("pass either --profile or --shell-class, not both")
    if arguments.profile is None and not arguments.shell_class:
        parser.error("candidate screening requires --profile or --shell-class")
    report = _run_batch(arguments)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(output, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output, encoding="utf-8")
    if not report["accepted_shell_classes"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
