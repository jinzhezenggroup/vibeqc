"""Coordinate bounded compilation, measured execution and promotion reporting.

The driver composes policy, emission, resource and manifest helpers. Compiler
workers remain distinct from the benchmark executor, preserving finite Slurm
allocation and the existing accuracy/performance acceptance boundaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from ..cuda_adapter import CudaBenchmarkExecutor, CudaCompilerAdapter
from ..cuda_schedule import (
    ScheduleIR,
    ScheduleKind,
)
from ..cuda_target import (
    CudaTargetInfo,
    cuda_target_info,
    normalize_cuda_architecture,
)
from ..ir import KernelConsumer
from .emission import (
    _oracle_schedule_trial,
    _oracle_symbol_prefix,
    emit_schedule_driver,
    emit_schedule_oracle_translation_unit,
    emit_schedule_resource_translation_unit,
    emit_schedule_translation_unit,
)
from .inputs import (
    _requested_schedule_kinds,
    _requested_shell_class_names,
    _resolve_specifications,
)
from .manifest import write_tuned_manifest
from .policy import (
    ScheduleTrial,
    _production_fock_schedule_index,
    schedule_payload,
    supported_schedule_trials,
)
from .process import _artifact_size, _compile_trial, _runtime_environment, _tool_version
from .resources import _resource_rejections, estimate_occupancy


def _run_autotune(
    arguments: argparse.Namespace, *, runtime_target: CudaTargetInfo | None = None
) -> dict[str, object]:
    """Generate, compile, run, rank, and optionally persist schedule winners."""

    arguments.architecture = normalize_cuda_architecture(arguments.architecture)
    target = runtime_target or cuda_target_info(arguments.architecture)
    if target.architecture != arguments.architecture:
        raise ValueError("probed and requested CUDA targets differ")
    compiler = CudaCompilerAdapter(
        nvcc=arguments.nvcc,
        target=target,
        compile_timeout=arguments.compile_timeout,
    )
    benchmark_executor = CudaBenchmarkExecutor(
        timeout=arguments.timeout,
        local=arguments.local,
        srun=arguments.srun,
        partition=arguments.partition,
        gres=arguments.gres,
        slurm_time=arguments.slurm_time,
    )
    if arguments.max_registers is None:
        arguments.max_registers = target.tuning_maximum_registers
    if arguments.max_packed_registers is None:
        arguments.max_packed_registers = target.tuning_maximum_packed_registers
    if arguments.max_stack_bytes is None:
        arguments.max_stack_bytes = target.tuning_maximum_stack_bytes
    if arguments.max_shared_bytes is None:
        arguments.max_shared_bytes = min(
            target.tuning_maximum_shared_bytes,
            target.shared_memory_per_block,
        )
    requested_names = _requested_shell_class_names(arguments)
    specifications = _resolve_specifications(requested_names)
    selected_consumer = KernelConsumer(arguments.consumer)
    selected_schedule_kinds = _requested_schedule_kinds(arguments)
    # Keep the measured production mapping in the same candidate set as new
    # proposals.  This gives every Fock proposal an explicit replacement
    # target and prevents the independent recompute oracle from becoming an
    # accidentally weaker baseline for an already-promoted class.
    production_baselines = (
        dict(_production_fock_schedule_index(arguments.architecture))
        if selected_consumer == KernelConsumer.FOCK
        else {}
    )
    trials = tuple(
        trial
        for spec in specifications
        for trial in supported_schedule_trials(spec, selected_consumer, target)
        if (
            not selected_schedule_kinds
            or trial.schedule.kind in selected_schedule_kinds
            or (
                selected_consumer == KernelConsumer.FOCK
                and production_baselines.get(spec.name) == trial.schedule
            )
        )
    )
    # User-local quick/full modes bound work per class while retaining any
    # official Fock baseline needed by the existing comparative gate.
    limit = getattr(arguments, "max_candidates", None)
    if limit is not None:
        if limit < 1:
            raise ValueError("candidate limit must be positive")
        bounded = []
        for spec in specifications:
            candidates = [t for t in trials if t.spec.name == spec.name]
            chosen = candidates[:limit]
            baseline = production_baselines.get(spec.name)
            for candidate in candidates:
                if candidate.schedule == baseline and candidate not in chosen:
                    chosen.append(candidate)
            bounded.extend(chosen)
        trials = tuple(bounded)
    if not trials:
        requested = ", ".join(spec.name for spec in specifications)
        selected = ", ".join(kind.value for kind in selected_schedule_kinds)
        raise ValueError(
            f"no schedule trials remain for shell classes {requested} "
            f"after filtering to {selected}"
        )
    production_baseline_keys = {
        name: next(
            (
                trial.key
                for trial in trials
                if trial.spec.name == name and trial.schedule == schedule
            ),
            None,
        )
        for name, schedule in production_baselines.items()
        if any(trial.spec.name == name for trial in trials)
    }
    work_directory_owner = None
    if arguments.work_directory is None:
        work_directory_owner = tempfile.TemporaryDirectory(
            prefix="vibeqc-shell-autotune-"
        )
        directory = Path(work_directory_owner.name)
    else:
        directory = arguments.work_directory
        directory.mkdir(parents=True, exist_ok=True)

    try:
        oracle_by_trial: dict[str, ScheduleTrial] = {}
        oracle_trials: dict[str, ScheduleTrial] = {}
        for trial in trials:
            oracle_trial = _oracle_schedule_trial(trial)
            oracle_prefix = _oracle_symbol_prefix(oracle_trial)
            oracle_by_trial[trial.key] = oracle_trial
            oracle_trials.setdefault(oracle_prefix, oracle_trial)
        for oracle_trial in oracle_trials.values():
            oracle_source = emit_schedule_oracle_translation_unit(oracle_trial)
            oracle_path = directory / (
                f"{oracle_trial.spec.name}_{oracle_trial.schedule_id}"
                f"{oracle_trial.integral_suffix}_oracle.cu"
            )
            oracle_path.write_text(oracle_source, encoding="utf-8")
        with ThreadPoolExecutor(max_workers=arguments.compile_jobs) as compile_pool:
            oracle_compile_rows = list(
                compile_pool.map(
                    lambda trial: _compile_trial(
                        arguments.nvcc,
                        arguments.architecture,
                        directory,
                        trial,
                        arguments.compile_timeout,
                        "_oracle",
                    ),
                    oracle_trials.values(),
                )
            )
        oracle_compile_by_prefix = {
            prefix: row
            for prefix, row in zip(
                oracle_trials,
                oracle_compile_rows,
                strict=True,
            )
        }

        for trial in trials:
            source = emit_schedule_translation_unit(
                trial,
                task_count=arguments.tasks,
                primitive_count=arguments.primitives,
                warmups=arguments.warmups,
                iterations=arguments.iterations,
                samples=arguments.samples,
                oracle_trial=oracle_by_trial[trial.key],
            )
            source_path = directory / (
                f"{trial.spec.name}_{trial.schedule_id}{trial.integral_suffix}.cu"
            )
            source_path.write_text(source, encoding="utf-8")

        with ThreadPoolExecutor(max_workers=arguments.compile_jobs) as compile_pool:
            compile_rows = list(
                compile_pool.map(
                    lambda trial: _compile_trial(
                        arguments.nvcc,
                        arguments.architecture,
                        directory,
                        trial,
                        arguments.compile_timeout,
                    ),
                    trials,
                )
            )

        runnable_trials = tuple(
            trial
            for trial, row in zip(trials, compile_rows, strict=True)
            if row["returncode"] == 0
            and oracle_compile_by_prefix[
                _oracle_symbol_prefix(oracle_by_trial[trial.key])
            ]["returncode"]
            == 0
        )
        runtime_rows: dict[str, dict[str, object]] = {}
        run_returncode = None
        run_stderr = ""
        link_seconds: float | None = None
        linked_binary_bytes: int | None = None
        if runnable_trials:
            driver = directory / "autotune_driver.cu"
            driver.write_text(
                emit_schedule_driver(runnable_trials, arguments.architecture),
                encoding="utf-8",
            )
            executable = directory / "shell_schedule_autotune"
            compile_by_key = {
                trial.key: row for trial, row in zip(trials, compile_rows, strict=True)
            }
            objects = [
                Path(compile_by_key[trial.key]["object"]) for trial in runnable_trials
            ]
            used_oracle_prefixes = {
                _oracle_symbol_prefix(oracle_by_trial[trial.key])
                for trial in runnable_trials
            }
            objects.extend(
                Path(oracle_compile_by_prefix[prefix]["object"])
                for prefix in sorted(used_oracle_prefixes)
            )
            link_started = time.monotonic()
            link = compiler.link(driver, objects, executable)
            link_seconds = time.monotonic() - link_started
            if link.returncode != 0:
                raise RuntimeError(link.stdout + link.stderr)
            linked_binary_bytes = _artifact_size(executable)
            run = benchmark_executor.run(
                executable,
                _runtime_environment(arguments.nvcc),
            )
            run_returncode = run.returncode
            run_stderr = run.stderr
            runtime_probe = None
            for line in run.stdout.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("target_probe") is True:
                    runtime_probe = row
                    continue
                shell_class = row.get("shell_class")
                consumer = row.get("consumer")
                schedule_id = row.get("schedule_id")
                if (
                    isinstance(shell_class, str)
                    and isinstance(consumer, str)
                    and isinstance(schedule_id, str)
                ):
                    trial_key = row.get("trial_key")
                    if not isinstance(trial_key, str):
                        # Accept benchmark executables emitted before explicit
                        # IR identity was added; new sources always include
                        # the full key so distinct IRs cannot overwrite rows.
                        trial_key = f"{shell_class}:{consumer}:{schedule_id}"
                    runtime_rows[trial_key] = row

        else:
            runtime_probe = None

        candidates = []
        passing_by_class: dict[
            str,
            list[tuple[ScheduleTrial, dict[str, object], dict[str, object]]],
        ] = {}
        for trial, compile_row in zip(trials, compile_rows, strict=True):
            resources = compile_row["resources"]
            oracle_compile = oracle_compile_by_prefix[
                _oracle_symbol_prefix(oracle_by_trial[trial.key])
            ]
            is_production_baseline = (
                selected_consumer == KernelConsumer.FOCK
                and production_baselines.get(trial.spec.name) == trial.schedule
            )
            maximum_registers = (
                arguments.max_packed_registers
                if trial.schedule.kind == ScheduleKind.PACKED_TASKS
                else arguments.max_registers
            )
            if is_production_baseline:
                # Existing production rows may intentionally use the full
                # target register envelope (for example DDDS on sm_120).
                # Keep spills and compile failures hard errors, but do not
                # reject the baseline solely because the exploratory tuning
                # cap is narrower than the code shape already shipped.
                maximum_registers = max(
                    maximum_registers,
                    target.maximum_registers_per_thread,
                )
                maximum_shared_bytes = max(
                    arguments.max_shared_bytes,
                    target.shared_memory_per_block,
                )
            else:
                maximum_shared_bytes = arguments.max_shared_bytes
            reasons = _resource_rejections(
                resources,
                consumer=trial.consumer,
                maximum_registers=maximum_registers,
                maximum_stack_bytes=arguments.max_stack_bytes,
                maximum_shared_bytes=maximum_shared_bytes,
            )
            if (
                trial.schedule.kind == ScheduleKind.SUBGROUP_TASKS
                and not arguments.allow_experimental_subgroup_winner
            ):
                reasons.append(
                    "subgroup schedules require explicit end-to-end "
                    "production acceptance"
                )
            if compile_row["timed_out"]:
                reasons.append(
                    f"NVCC compilation exceeded {arguments.compile_timeout:g} seconds"
                )
            elif compile_row["returncode"] != 0:
                reasons.append("NVCC compilation failed")
            if oracle_compile["timed_out"]:
                reasons.append(
                    "oracle NVCC compilation exceeded "
                    f"{arguments.compile_timeout:g} seconds"
                )
            elif oracle_compile["returncode"] != 0:
                reasons.append("oracle NVCC compilation failed")
            runtime = runtime_rows.get(trial.key)
            speedup_vs_baseline = None
            baseline_key = production_baseline_keys.get(trial.spec.name)
            baseline_runtime = (
                runtime_rows.get(baseline_key) if baseline_key is not None else None
            )
            if (
                selected_consumer == KernelConsumer.FOCK
                and baseline_key is not None
                and not is_production_baseline
                and baseline_runtime is None
            ):
                # A proposal is only production-comparable when the shipped
                # mapping ran in the same executable.  Without this guard a
                # missing baseline row silently turns the independent oracle
                # into the acceptance baseline and can promote an unsafe or
                # slower mapping after a partial runtime failure.
                reasons.append("production baseline did not produce a runtime result")
            if run_returncode == 5:
                reasons.append("compile and runtime CUDA targets differ")
            if runtime is None:
                reasons.append("schedule did not produce a runtime result")
            else:
                maximum_value = float(runtime[f"maximum_{trial.consumer.value}"])
                maximum_error = float(runtime[f"maximum_{trial.consumer.value}_error"])
                tolerance = arguments.absolute_tolerance + (
                    arguments.relative_tolerance * maximum_value
                )
                if maximum_error > tolerance:
                    reasons.append(
                        f"{trial.consumer.value} error {maximum_error:.3e} "
                        f"exceeds {tolerance:.3e}"
                    )
                oracle_speedup = float(runtime["speedup"])
                if (
                    not is_production_baseline
                    and oracle_speedup < arguments.minimum_speedup
                ):
                    reasons.append(f"speedup is below {arguments.minimum_speedup:.3f}x")
                if baseline_runtime is not None:
                    baseline_ms = float(baseline_runtime["fused_ms"])
                    if float(runtime["fused_ms"]) > 0.0:
                        speedup_vs_baseline = baseline_ms / float(runtime["fused_ms"])
                    if (
                        not is_production_baseline
                        and speedup_vs_baseline < arguments.minimum_speedup
                    ):
                        reasons.append(
                            "speedup versus production baseline is below "
                            f"{arguments.minimum_speedup:.3f}x"
                        )
                else:
                    speedup_vs_baseline = 1.0 if is_production_baseline else None
            accepted = not reasons
            row = {
                "shell_class": trial.spec.name,
                "consumer": trial.consumer.value,
                "schedule_id": trial.schedule_id,
                "trial_key": trial.key,
                "schedule": schedule_payload(trial.schedule),
                "static_model": trial.static_model.to_payload(),
                "compile_succeeded": compile_row["returncode"] == 0,
                "compile_timed_out": compile_row["timed_out"],
                "compile_seconds": compile_row["duration_seconds"],
                "source_bytes": compile_row.get("source_bytes"),
                "object_bytes": compile_row.get("object_bytes"),
                "resources": [asdict(item) for item in resources],
                "occupancy": estimate_occupancy(resources, trial, target),
                "runtime": runtime,
                "production_baseline": is_production_baseline,
                "speedup_vs_production_baseline": speedup_vs_baseline,
                "accepted": accepted,
                "rejection_reasons": reasons,
                "production_validation": None,
            }
            candidates.append(row)
            if accepted and runtime is not None:
                passing_by_class.setdefault(trial.spec.name, []).append(
                    (trial, runtime, row)
                )
            if arguments.verbose and compile_row["diagnostics"]:
                print(compile_row["diagnostics"], file=sys.stderr, end="")

        winners: dict[str, ScheduleIR] = {}
        winner_rows = []
        winner_provenance: dict[str, dict[str, object]] = {}
        for spec in specifications:
            passing = passing_by_class.get(spec.name, [])
            if not passing:
                continue
            fastest_ms = min(float(item[1]["fused_ms"]) for item in passing)

            def winner_key(
                item: tuple[ScheduleTrial, dict[str, object], dict[str, object]],
                *,
                fastest: float = fastest_ms,
            ):
                trial, runtime, candidate = item
                elapsed_ms = float(runtime["fused_ms"])

                def metric_or_inf(name: str) -> float:
                    value = candidate.get(name)
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and value >= 0
                    ):
                        return float(value)
                    return math.inf

                # Within one percent of the fastest endpoint, compile time and
                # binary footprint decide the winner. Once a candidate falls
                # outside that noise band, endpoint runtime is the primary
                # key again; otherwise a much slower but tiny artifact could
                # displace a scientifically faster schedule.
                near_fastest = elapsed_ms <= fastest * 1.01
                if near_fastest:
                    return (
                        0,
                        metric_or_inf("compile_seconds"),
                        metric_or_inf("source_bytes"),
                        metric_or_inf("object_bytes"),
                        elapsed_ms,
                        trial.schedule_id,
                    )
                return (
                    1,
                    elapsed_ms,
                    metric_or_inf("compile_seconds"),
                    metric_or_inf("source_bytes"),
                    metric_or_inf("object_bytes"),
                    trial.schedule_id,
                )

            ranked = sorted(passing, key=winner_key)
            for trial, runtime, candidate_row in ranked:
                resource_source = emit_schedule_resource_translation_unit(trial)
                resource_suffix = "_production_resources"
                resource_path = directory / (
                    f"{trial.spec.name}_{trial.schedule_id}"
                    f"{trial.integral_suffix}{resource_suffix}.cu"
                )
                resource_path.write_text(resource_source, encoding="utf-8")
                resource_compile = _compile_trial(
                    arguments.nvcc,
                    arguments.architecture,
                    directory,
                    trial,
                    arguments.compile_timeout,
                    resource_suffix,
                )
                maximum_registers = (
                    arguments.max_packed_registers
                    if trial.schedule.kind == ScheduleKind.PACKED_TASKS
                    else arguments.max_registers
                )
                production_max_shared_bytes = arguments.max_shared_bytes
                if (
                    selected_consumer == KernelConsumer.FOCK
                    and production_baselines.get(trial.spec.name) == trial.schedule
                ):
                    maximum_registers = max(
                        maximum_registers,
                        target.maximum_registers_per_thread,
                    )
                    production_max_shared_bytes = max(
                        production_max_shared_bytes,
                        target.shared_memory_per_block,
                    )
                production_reasons = _resource_rejections(
                    resource_compile["resources"],
                    consumer=trial.consumer,
                    maximum_registers=maximum_registers,
                    maximum_stack_bytes=arguments.max_stack_bytes,
                    maximum_shared_bytes=production_max_shared_bytes,
                    expected_kernel_records=4,
                )
                if resource_compile["timed_out"]:
                    production_reasons.append(
                        "NVCC compilation exceeded "
                        f"{arguments.compile_timeout:g} seconds"
                    )
                elif resource_compile["returncode"] != 0:
                    production_reasons.append("NVCC compilation failed")
                production_validation = {
                    "compile_succeeded": resource_compile["returncode"] == 0,
                    "compile_timed_out": resource_compile["timed_out"],
                    "compile_seconds": resource_compile["duration_seconds"],
                    "source_bytes": resource_compile.get("source_bytes"),
                    "object_bytes": resource_compile.get("object_bytes"),
                    "resources": [
                        asdict(item) for item in resource_compile["resources"]
                    ],
                    "occupancy": estimate_occupancy(
                        resource_compile["resources"], trial, target
                    ),
                    "accepted": not production_reasons,
                    "rejection_reasons": production_reasons,
                }
                candidate_row["production_validation"] = production_validation
                if arguments.verbose and resource_compile["diagnostics"]:
                    print(
                        resource_compile["diagnostics"],
                        file=sys.stderr,
                        end="",
                    )
                if production_reasons:
                    candidate_row["accepted"] = False
                    candidate_row["rejection_reasons"].extend(
                        f"production validation: {reason}"
                        for reason in production_reasons
                    )
                    continue
                winners[spec.name] = trial.schedule
                winner_rows.append(
                    {
                        "shell_class": spec.name,
                        "consumer": selected_consumer.value,
                        "schedule_id": trial.schedule_id,
                        "trial_key": trial.key,
                        "schedule": schedule_payload(trial.schedule),
                        "static_model": trial.static_model.to_payload(),
                        "runtime": runtime,
                        "source_bytes": candidate_row.get("source_bytes"),
                        "object_bytes": candidate_row.get("object_bytes"),
                        "occupancy": candidate_row.get("occupancy"),
                        "production_validation": production_validation,
                    }
                )
                winner_provenance[spec.name] = {
                    "runtime_seconds": float(runtime["fused_ms"]) / 1000.0,
                    "compile_seconds": candidate_row.get("compile_seconds"),
                    "source_bytes": candidate_row.get("source_bytes"),
                    "object_bytes": candidate_row.get("object_bytes"),
                }
                break

        requested_set = {spec.name for spec in specifications}
        missing_winners = sorted(requested_set - set(winners))
        require_all_winners = bool(getattr(arguments, "require_all_winners", False))
        manifest_written = False
        manifest_write_skipped = False
        if arguments.manifest_output is not None:
            if require_all_winners and missing_winners:
                # Do not leave a partially tuned production manifest behind
                # when one class fails a compile/resource/correctness gate.
                # The report still preserves diagnostics for a focused rerun.
                manifest_write_skipped = True
            elif winners:
                write_tuned_manifest(
                    arguments.manifest,
                    arguments.manifest_output,
                    arguments.architecture,
                    winners,
                    selected_consumer,
                    winner_provenance,
                )
                manifest_written = True

        return {
            "schema_version": 1,
            "architecture": arguments.architecture,
            "consumer": selected_consumer.value,
            "nvcc": str(arguments.nvcc),
            "target": target.to_payload(),
            "toolchain": {
                "nvcc": _tool_version(arguments.nvcc),
                "ptxas": _tool_version(arguments.nvcc.with_name("ptxas")),
                "generator_abi": target.generator_abi,
            },
            "single_gpu_process": True,
            "runtime": {
                "local": arguments.local,
                "srun": None if arguments.local else arguments.srun,
                "partition": None if arguments.local else arguments.partition,
                "gres": None if arguments.local else arguments.gres,
                "returncode": run_returncode,
                "stderr": run_stderr,
                "device": runtime_probe,
            },
            "artifacts": {
                # The linked executable contains every runnable candidate and
                # shared oracle.  Per-candidate object sizes below preserve a
                # useful code-size comparison without retaining temp paths.
                "linked_executable_bytes": linked_binary_bytes,
                "link_seconds": link_seconds,
                "schedule_objects": {
                    row["key"]: row.get("object_bytes") for row in compile_rows
                },
                "oracle_objects": {
                    prefix: row.get("object_bytes")
                    for prefix, row in oracle_compile_by_prefix.items()
                },
            },
            "gates": {
                "minimum_speedup": arguments.minimum_speedup,
                "absolute_tolerance": arguments.absolute_tolerance,
                "relative_tolerance": arguments.relative_tolerance,
                "maximum_registers": arguments.max_registers,
                "maximum_packed_registers": arguments.max_packed_registers,
                "maximum_stack_bytes": arguments.max_stack_bytes,
                "maximum_shared_bytes": arguments.max_shared_bytes,
                "compile_timeout_seconds": arguments.compile_timeout,
                "spills_allowed": False,
                "experimental_subgroup_winners_allowed": (
                    arguments.allow_experimental_subgroup_winner
                ),
                "require_all_winners": require_all_winners,
            },
            "search": {
                "schedule_kinds": [kind.value for kind in selected_schedule_kinds],
                "trial_count": len(trials),
            },
            "requested_shell_classes": [spec.name for spec in specifications],
            "missing_winners": missing_winners,
            "manifest": {
                "output": (
                    str(arguments.manifest_output)
                    if arguments.manifest_output is not None
                    else None
                ),
                "written": manifest_written,
                "write_skipped": manifest_write_skipped,
            },
            "winners": winner_rows,
            "candidates": candidates,
            "oracles": [
                {
                    "symbol_prefix": prefix,
                    "compile_succeeded": row["returncode"] == 0,
                    "compile_timed_out": row["timed_out"],
                    "compile_seconds": row["duration_seconds"],
                    "source_bytes": row.get("source_bytes"),
                    "object_bytes": row.get("object_bytes"),
                }
                for prefix, row in oracle_compile_by_prefix.items()
            ],
        }
    finally:
        if work_directory_owner is not None:
            work_directory_owner.cleanup()
