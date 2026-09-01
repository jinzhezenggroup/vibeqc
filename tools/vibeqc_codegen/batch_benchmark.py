"""Batch-compile and screen fused shell-class CUDA candidates.

Every candidate is emitted into its own CUDA translation unit, while one
linked executable runs all correctness and timing checks in a single process.
The Python driver launches that executable through exactly one ``srun`` GPU
allocation and combines runtime measurements with per-TU ``ptxas`` resources.
"""

from __future__ import annotations

import argparse
import json
import math
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
from .ir import KernelConsumer
from .production import load_production_fock_manifest, load_production_manifest
from .shell_spec import (
    FUSED_SHELL_SPEC_BY_NAME,
    FUSED_SHELL_SPECS,
    ShellClassSpec,
)

_PRODUCTION_MANIFEST_PATH = Path(__file__).with_name("production_shell_classes.json")
PRODUCTION_SHELL_CLASSES = frozenset(
    spec.name for spec in load_production_manifest(_PRODUCTION_MANIFEST_PATH)
)
PRODUCTION_FOCK_SHELL_CLASSES = frozenset(
    spec.name for spec in load_production_fock_manifest(_PRODUCTION_MANIFEST_PATH)
)
_PRODUCTION_SHELL_CLASSES_BY_CONSUMER = {
    KernelConsumer.FORCE: PRODUCTION_SHELL_CLASSES,
    KernelConsumer.FOCK: PRODUCTION_FOCK_SHELL_CLASSES,
}
DEFAULT_CANDIDATES = tuple(
    spec for spec in FUSED_SHELL_SPECS if spec.name not in PRODUCTION_SHELL_CLASSES
)


def _production_shell_classes(consumer: KernelConsumer | str) -> frozenset[str]:
    """Return the manifest classes already generated for one consumer."""

    selected_consumer = KernelConsumer(consumer)
    return _PRODUCTION_SHELL_CLASSES_BY_CONSUMER[selected_consumer]


def _row_matches_consumer(row: dict[str, object], consumer: KernelConsumer) -> bool:
    """Keep profile rows scoped to the consumer being promoted.

    Older workload histograms omit ``consumer`` and are intentionally treated
    as compatible with either ranking path.  New batch/autotune artifacts may
    use either one string or a list of consumers, both of which are accepted.
    """

    value = row.get("consumer")
    if value is None:
        return True
    if isinstance(value, str):
        return value == consumer.value
    if isinstance(value, list):
        return consumer.value in value
    return False


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
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
) -> tuple[ShellClassSpec, ...]:
    """Resolve an explicit sparse list of uncovered s/p/d/f candidates.

    Production coverage is consumer-specific: a force-only class remains a
    valid Fock candidate, and vice versa.  Keep the explicit and profile-driven
    paths on the same exclusion policy by resolving it here.
    """

    if names is None:
        raise ValueError(
            "candidate screening requires --profile or an explicit --shell-class"
        )
    selected_consumer = KernelConsumer(consumer)
    production_shell_classes = _production_shell_classes(selected_consumer)
    specifications = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        try:
            specification = FUSED_SHELL_SPEC_BY_NAME[name]
        except KeyError as error:
            choices = ", ".join(FUSED_SHELL_SPEC_BY_NAME)
            raise ValueError(
                f"unknown shell class {name!r}; choose from {choices}"
            ) from error
        if name in production_shell_classes:
            raise ValueError(
                f"{name} is already covered by {selected_consumer.value} production AOT"
            )
        specifications.append(specification)
        seen.add(name)
    if not specifications:
        raise ValueError("at least one candidate shell class is required")
    return tuple(specifications)


def discover_candidate_specs(
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
    *,
    limit: int | None = None,
) -> tuple[ShellClassSpec, ...]:
    """Discover uncovered classes using only structural work estimates.

    This is intentionally a discovery pass, not a promotion decision.  It
    ranks the manifest gap by Cartesian component count and angular work so a
    caller can screen a bounded prefix without hand-maintaining a shell-name
    list.  Real molecular endpoint gates remain outside this synthetic batch
    screener and are still required before production manifest edits.
    """

    selected_consumer = KernelConsumer(consumer)
    excluded = _production_shell_classes(selected_consumer)
    uncovered = [spec for spec in FUSED_SHELL_SPECS if spec.name not in excluded]
    uncovered.sort(
        key=lambda spec: (
            -spec.component_count,
            -sum(spec.angular),
            spec.name,
        )
    )
    if limit is None:
        return tuple(uncovered)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("candidate limit must be positive")
    return tuple(uncovered[:limit])


def rank_profiled_candidates(
    payload: dict[str, object],
    limit: int,
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
) -> tuple[ShellClassSpec, ...]:
    """Select uncovered classes by descending measured primitive work.

    Runtime shell profiles are not required to be sorted: some producers keep
    canonical catalog order while others preserve discovery order.  Prefer
    ``primitive_work`` when available, fall back to ``primitive_quartets`` for
    older artifacts, and use the original row order as a deterministic tie
    breaker.  Duplicate class rows are aggregated because profile producers
    may emit one row per orientation for a canonical shell class.
    """

    selected_consumer = KernelConsumer(consumer)
    production_shell_classes = _production_shell_classes(selected_consumer)
    if limit < 1:
        raise ValueError("candidate limit must be positive")
    rows = payload.get("shell_classes")
    if not isinstance(rows, list):
        raise TypeError("profile JSON does not contain a shell_classes list")
    aggregated: dict[str, tuple[float, int, ShellClassSpec]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        name = row.get("class")
        if (
            _row_matches_consumer(row, selected_consumer)
            and isinstance(name, str)
            and name not in production_shell_classes
            and name in FUSED_SHELL_SPEC_BY_NAME
        ):
            work = 0.0
            for field in (
                "primitive_work",
                "primitive_quartets",
                "primitive_work_fraction",
            ):
                value = row.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric = float(value)
                    if math.isfinite(numeric) and numeric >= 0.0:
                        work = numeric
                        break
            previous = aggregated.get(name)
            if previous is None:
                aggregated[name] = (work, index, FUSED_SHELL_SPEC_BY_NAME[name])
            else:
                aggregated[name] = (
                    previous[0] + work,
                    previous[1],
                    previous[2],
                )
    if not aggregated:
        raise ValueError("profile contains no uncovered compilable shell classes")
    ranked = list(aggregated.values())
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(spec for _, _, spec in ranked[:limit])


def pareto_front(
    candidates: Iterable[dict[str, object]],
    *,
    runtime_key: str = "runtime_seconds",
    compile_key: str = "compile_seconds",
    source_key: str = "source_bytes",
    object_key: str = "object_bytes",
    timing_noise: float = 0.01,
) -> tuple[dict[str, object], ...]:
    """Return candidates not dominated by runtime, compile cost, or size.

    Runtime is the scientific objective and is therefore allowed to improve
    by ``timing_noise`` before a smaller/faster-compiling candidate wins.  A
    candidate with no provenance remains eligible (old benchmark artifacts are
    still valid); missing dimensions simply do not participate in dominance.
    The function is pure so autotune and batch screening can share the same
    promotion policy without changing their compiler drivers.
    """

    if (
        isinstance(timing_noise, bool)
        or not isinstance(timing_noise, (int, float))
        or not 0.0 <= timing_noise < 1.0
    ):
        raise ValueError("timing_noise must be in [0, 1)")
    rows = tuple(row for row in candidates if isinstance(row, dict))

    def numeric(row: dict[str, object], key: str) -> float | None:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) and value >= 0.0 else None

    def dominates(first: dict[str, object], second: dict[str, object]) -> bool:
        first_runtime, second_runtime = (
            numeric(first, runtime_key),
            numeric(second, runtime_key),
        )
        if first_runtime is None and second_runtime is not None:
            # An unmeasured candidate cannot dominate a measured scientific
            # result merely because its compiler metadata is smaller.
            return False
        if first_runtime is not None and second_runtime is not None:
            # Faster by less than measurement noise is treated as a tie. This
            # permits compile/size dimensions to decide without promoting a
            # scientifically slower candidate by accident.
            runtime_better = first_runtime <= second_runtime * (1.0 + timing_noise)
            runtime_strict = first_runtime < second_runtime * (1.0 - timing_noise)
        else:
            runtime_better, runtime_strict = True, False
        no_worse = runtime_better
        strict = runtime_strict
        for key in (compile_key, source_key, object_key):
            first_value, second_value = numeric(first, key), numeric(second, key)
            if first_value is None or second_value is None:
                continue
            if first_value > second_value:
                no_worse = False
                break
            strict |= first_value < second_value
        return no_worse and strict

    return tuple(
        row
        for row in rows
        if not any(dominates(other, row) for other in rows if other is not row)
    )


def rank_compile_aware_candidates(
    payload: dict[str, object],
    limit: int,
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
    *,
    timing_noise: float = 0.01,
) -> tuple[ShellClassSpec, ...]:
    """Rank profile rows using endpoint runtime and compile-cost provenance.

    This is an opt-in companion to :func:`rank_profiled_candidates`.  It keeps
    the historical primitive-work ranking when an artifact predates runtime or
    compiler measurements, and applies the Pareto policy only when those
    dimensions are present.
    """

    if limit < 1:
        raise ValueError("candidate limit must be positive")
    selected_consumer = KernelConsumer(consumer)
    excluded = _production_shell_classes(selected_consumer)
    rows = payload.get("shell_classes")
    if not isinstance(rows, list):
        raise TypeError("profile JSON does not contain a shell_classes list")
    aggregated: dict[str, dict[str, object]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        name = row.get("class")
        if (
            not _row_matches_consumer(row, selected_consumer)
            or not isinstance(name, str)
            or name in excluded
            or name not in FUSED_SHELL_SPEC_BY_NAME
        ):
            continue
        item = aggregated.setdefault(
            name,
            {"class": name, "first_index": index, "primitive_work": 0.0},
        )
        for key in ("primitive_work", "primitive_quartets", "primitive_work_fraction"):
            value = row.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                item["primitive_work"] = float(item["primitive_work"]) + float(value)
                break
        for key in (
            "runtime_seconds",
            "compile_seconds",
            "source_bytes",
            "object_bytes",
        ):
            value = row.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and value >= 0
            ):
                # Runtime and compile/size rows are conservatively aggregated
                # across duplicate orientations: runtime is best observed,
                # compiler cost is the largest unit that must be rebuilt.
                if key == "runtime_seconds":
                    previous = item.get(key)
                    item[key] = (
                        float(value)
                        if previous is None
                        else min(float(previous), float(value))
                    )
                else:
                    previous = item.get(key)
                    item[key] = (
                        float(value)
                        if previous is None
                        else max(float(previous), float(value))
                    )
    if not aggregated:
        raise ValueError("profile contains no uncovered compilable shell classes")
    materialized = tuple(aggregated.values())
    measured = any(row.get("runtime_seconds") is not None for row in materialized)
    if measured:
        selected = pareto_front(materialized, timing_noise=timing_noise)
    else:
        selected = materialized
    selected = sorted(
        selected,
        key=lambda row: (
            float(row.get("runtime_seconds", math.inf)),
            float(row.get("compile_seconds", math.inf)),
            float(row.get("source_bytes", math.inf)),
            -float(row.get("primitive_work", 0.0)),
            int(row["first_index"]),
        ),
    )
    return tuple(
        FUSED_SHELL_SPEC_BY_NAME[str(row["class"])] for row in selected[:limit]
    )


def emit_candidate_translation_unit(
    spec: ShellClassSpec,
    *,
    task_count: int,
    primitive_count: int,
    warmups: int,
    iterations: int,
    samples: int,
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
) -> str:
    """Emit one independently compilable candidate benchmark translation unit.

    The consumer is explicit so the same sparse/profile-driven screener can
    measure coefficient-only Fock candidates as well as first-force kernels.
    Both paths retain the shared task ABI and independent recompute oracle.
    """

    selected_consumer = KernelConsumer(consumer)
    source = emit_shell_class_benchmark_cuda(
        spec,
        task_count=task_count,
        primitive_count=primitive_count,
        warmups=warmups,
        iterations=iterations,
        samples=samples,
        consumer=selected_consumer,
    )
    entry = f"vibeqc_run_shell_class_{spec.name}"
    source = source.replace("int main() {", f'extern "C" int {entry}() {{', 1)
    marker = r"{\"task_count\":%u"
    replacement = (
        rf"{{\"shell_class\":\"{spec.name}\","
        rf"\"consumer\":\"{selected_consumer.value}\",\"task_count\":%u"
    )
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
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
) -> tuple[KernelResources, ...]:
    """Extract the four selected-consumer kernels from verbose ``ptxas`` output.

    Autotuning translation units suffix every generated symbol so several
    schedules for the same shell class can coexist in one executable.
    ``symbol_prefix`` lets that driver reuse the production resource parser;
    ``consumer`` selects the force or coefficient-only Fock symbol family.
    """

    selected_consumer = KernelConsumer(consumer)
    marker = symbol_prefix or (
        f"generated_{shell_class}_shell_class_{selected_consumer.value}_"
    )
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
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
) -> dict[str, object]:
    """Compile one candidate TU and retain diagnostics for automatic gates."""

    selected_consumer = KernelConsumer(consumer)
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
        "resources": parse_ptxas_resources(
            diagnostics,
            spec.name,
            consumer=selected_consumer,
        ),
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


def benchmark_command(
    executable: Path,
    *,
    srun: str = "srun",
    partition: str = "main",
    gres: str = "gpu:5090:1",
    slurm_time: str = "00:10:00",
) -> list[str]:
    """Build the finite Slurm command used for one GPU benchmark process."""

    if not slurm_time or not slurm_time.strip():
        raise ValueError("Slurm benchmark time must be non-empty")
    return [
        srun,
        f"--partition={partition}",
        f"--gres={gres}",
        "--nodes=1",
        "--ntasks=1",
        f"--time={slurm_time}",
        str(executable),
    ]


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
        reasons.append(
            f"expected 4 fused kernel resource records, found {len(resources)}"
        )
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

    # Keep direct Python callers compatible with the pre-consumer namespace;
    # the CLI always supplies the explicit option.
    selected_consumer = KernelConsumer(
        getattr(arguments, "consumer", KernelConsumer.FORCE.value)
    )
    discovery = bool(getattr(arguments, "discover", False))
    if discovery:
        specifications = discover_candidate_specs(
            consumer=selected_consumer,
            limit=arguments.limit,
        )
        selection_mode = "manifest_gap"
    elif arguments.profile is not None:
        profile = json.loads(arguments.profile.read_text(encoding="utf-8"))
        ranker = (
            rank_compile_aware_candidates
            if getattr(arguments, "compile_aware", False)
            else rank_profiled_candidates
        )
        selection_mode = "profile"
        specifications = ranker(profile, arguments.limit, consumer=selected_consumer)
    else:
        specifications = candidate_specs(
            arguments.shell_class or None,
            consumer=selected_consumer,
        )
        selection_mode = "explicit"

    work_directory_owner = None
    if arguments.work_directory is None:
        work_directory_owner = tempfile.TemporaryDirectory(prefix="vibeqc-shell-batch-")
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
                consumer=selected_consumer,
            )
            (directory / f"{spec.name}_candidate.cu").write_text(
                source, encoding="utf-8"
            )

        with ThreadPoolExecutor(max_workers=arguments.compile_jobs) as executor:
            compile_rows = list(
                executor.map(
                    lambda spec: _compile_candidate(
                        arguments.nvcc,
                        arguments.architecture,
                        directory,
                        spec,
                        selected_consumer,
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
                str(row["object"]) for row in compile_rows if row["returncode"] == 0
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
                benchmark_command(
                    executable,
                    srun=arguments.srun,
                    partition=arguments.partition,
                    gres=arguments.gres,
                    slurm_time=arguments.slurm_time,
                ),
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
                observable = selected_consumer.value
                maximum_value = float(runtime[f"maximum_{observable}"])
                maximum_error = float(runtime[f"maximum_{observable}_error"])
                tolerance = arguments.absolute_tolerance + (
                    arguments.relative_tolerance * maximum_value
                )
                if maximum_error > tolerance:
                    reasons.append(
                        f"{observable} error {maximum_error:.3e} exceeds "
                        f"{tolerance:.3e}"
                    )
                if float(runtime["speedup"]) < arguments.minimum_speedup:
                    reasons.append(f"speedup is below {arguments.minimum_speedup:.3f}x")
            passed = resource_ok and not reasons
            if passed:
                accepted.append(spec.name)
            candidates.append(
                {
                    "shell_class": spec.name,
                    "consumer": selected_consumer.value,
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
            "consumer": selected_consumer.value,
            "selection": {
                "mode": selection_mode,
                "discovered_count": len(specifications) if discovery else None,
            },
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
    parser.add_argument(
        "--slurm-time",
        default="00:10:00",
        help="finite Slurm allocation time used for the benchmark process",
    )
    parser.add_argument(
        "--consumer",
        choices=tuple(item.value for item in KernelConsumer),
        default=KernelConsumer.FORCE.value,
        help="screen force or coefficient-only Fock candidates",
    )
    parser.add_argument("--shell-class", action="append")
    parser.add_argument(
        "--profile",
        type=Path,
        help="rank candidates by an --all-orders active profile JSON",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help=(
            "automatically select the highest-work classes missing from the "
            "consumer-specific production manifest"
        ),
    )
    parser.add_argument(
        "--compile-aware",
        action="store_true",
        help=(
            "include endpoint runtime, compile seconds, and artifact sizes in "
            "Pareto candidate promotion"
        ),
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
    selected_modes = (
        int(arguments.profile is not None)
        + int(arguments.discover)
        + int(bool(arguments.shell_class))
    )
    if selected_modes > 1:
        parser.error("pass only one of --profile, --discover, or --shell-class")
    if (
        arguments.profile is None
        and not arguments.shell_class
        and not arguments.discover
    ):
        parser.error(
            "candidate screening requires --profile, --discover, or --shell-class"
        )
    if arguments.limit < 1:
        parser.error("--limit must be positive")
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
