"""Tiered f-shell evidence, separate from production selection.

Source/IR validity, compilation, resources, GPU numerics, and molecular
promotion are independent states. In particular, an existing manifest entry
does not retroactively establish numerical or endpoint acceptance.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.vibeqc_codegen.autotune import schedule_payload
from tools.vibeqc_codegen.batch_benchmark import parse_ptxas_resources
from tools.vibeqc_codegen.benchmark import emit_shell_class_resource_cuda
from tools.vibeqc_codegen.capabilities import build_capability_report
from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_codegen.fused_schedule import build_fused_shell_plan
from tools.vibeqc_codegen.ir import KernelConsumer
from tools.vibeqc_codegen.ir_serialization import integral_to_payload
from tools.vibeqc_codegen.production import shell_class_index
from tools.vibeqc_codegen.shell_spec import FUSED_SHELL_SPEC_BY_NAME, FUSED_SHELL_SPECS

from .schema import canonical_hash, file_hash, outcome

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tools/vibeqc_codegen/production_shell_classes.json"
F_SHELL_CLASSES = tuple(spec.name for spec in FUSED_SHELL_SPECS if 3 in spec.angular)
SMOKE_CLASSES = ("fsss", "fsps", "fpps", "fdfd", "ffff")
CONSUMERS = (KernelConsumer.FOCK, KernelConsumer.FORCE)
SCHEMA_VERSION = 1


def f_shell_plan(name: str, architecture: str = "sm_120"):
    """Use the common complete-component lowering for one f-containing class."""
    if name not in F_SHELL_CLASSES:
        raise ValueError(f"not a canonical s/p/d/f class containing f: {name}")
    return build_fused_shell_plan(
        FUSED_SHELL_SPEC_BY_NAME[name],
        consumers=CONSUMERS,
        target=cuda_target_info(architecture),
    )


def generated_symbols(name: str) -> tuple[str, ...]:
    """Production ABI wrappers that must be present for the two consumers."""
    return tuple(
        f"generated_{name}_shell_class_{consumer}_{spin}{persistent}_kernel"
        for consumer in ("fock", "force")
        for spin in ("rhf", "uhf")
        for persistent in ("", "_persistent")
    )


def source_audit(name: str, architecture: str = "sm_120") -> tuple[dict, str]:
    """Gate 1: emit twice and check components, centers, symbols, and metadata."""
    plan = f_shell_plan(name, architecture)
    source = emit_shell_class_resource_cuda(plan.spec, plan)
    repeated = emit_shell_class_resource_cuda(
        plan.spec, f_shell_plan(name, architecture)
    )
    if source != repeated:
        raise ValueError(f"nondeterministic generated source for {name}")
    spec, integral = plan.spec, plan.kernel.integral
    if (
        len(plan.components) != spec.component_count
        or len(set(plan.components)) != spec.component_count
    ):
        raise ValueError(f"incomplete Cartesian component coverage for {name}")
    for index, component in enumerate(plan.components):
        if (
            spec.component_from_index(index) != component
            or spec.component_index(component) != index
        ):
            raise ValueError(f"inconsistent component ordering for {name}")
    if (
        set(integral.independent_derivative_centers)
        | set(integral.recovered_derivative_centers)
    ) != set(range(4)):
        raise ValueError(f"incomplete derivative-center coverage for {name}")
    if (
        len(integral.independent_derivative_centers) != 3
        or len(integral.recovered_derivative_centers) != 1
    ):
        raise ValueError(f"unexpected translation recovery for {name}")
    symbols = generated_symbols(name)
    if any(
        len(re.findall(r"\bvoid " + symbol + r"\(", source)) != 1 for symbol in symbols
    ):
        raise ValueError(f"missing or duplicate generated wrapper for {name}")
    return {
        "status": "pass",
        "reason": None,
        "source_hash": hashlib.sha256(source.encode()).hexdigest(),
        "source_bytes": len(source.encode()),
        "deterministic_generations": 2,
        "ir_hash": canonical_hash(integral_to_payload(integral)),
        "component_count": len(plan.components),
        "component_order_hash": canonical_hash(plan.components),
        "independent_derivative_centers": list(integral.independent_derivative_centers),
        "recovered_derivative_centers": list(integral.recovered_derivative_centers),
        "symbols": list(symbols),
        "registry_class_index": shell_class_index(spec),
        "schedule": schedule_payload(plan.schedule),
    }, source


def catalog(*, architecture: str = "sm_120", names=F_SHELL_CLASSES) -> dict:
    """Gate 0/1 report with recurrence/schedule legality and provisional selection."""
    names = tuple(names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in F_SHELL_CLASSES for n in names)
    ):
        raise ValueError("catalog requires distinct canonical f-containing classes")
    manifest = json.loads(MANIFEST.read_text())
    capabilities = build_capability_report(
        architecture=architecture,
        manifest=MANIFEST,
        specifications=tuple(FUSED_SHELL_SPEC_BY_NAME[n] for n in names),
    )
    capability_by_name = {
        row["shell_class"]: row for row in capabilities["shell_classes"]
    }
    rows = []
    for name in names:
        source, _ = source_audit(name, architecture)
        capability = capability_by_name[name]
        selected = capability["production"]
        rows.append(
            {
                "shell_class": name,
                "angular": list(FUSED_SHELL_SPEC_BY_NAME[name].angular),
                "consumers": [c.value for c in CONSUMERS],
                "capabilities": capability,
                "source": source,
                "manifest": {
                    **selected,
                    "acceptance": "provisional"
                    if selected.get("status") == "manifest_selected"
                    else "unselected",
                    "reason": "manifest selection alone is not compile/numerical/endpoint evidence",
                },
                "prior_evidence": {
                    "status": "historical" if name == "fpps" else "not-run",
                    "reason": "historical selection does not establish all current-source acceptance gates",
                    "records": [
                        {
                            "commit": "c0683c5b0a66b6330d16117ab8a4dd812956843b",
                            "document": "docs/shell_codegen.md",
                            "basis": "water/def2-TZVP",
                            "isolated_speedup": 2.27,
                            "endpoint_speedups": {
                                "1": 1.0022,
                                "4": 1.0054,
                                "8": 1.0060,
                            },
                            "maximum_force_difference": 7.17e-13,
                            "scope": "documented historical RHF force promotion; incomplete current all-consumer/oracle matrix",
                        }
                    ]
                    if name == "fpps"
                    else [],
                },
                "compilation": outcome("not-run", "compile tier not run"),
                "resources": outcome("not-run", "compile tier not run"),
                "numerical": outcome("not-run", "GPU numerical tier not run"),
                "endpoint": outcome(
                    "not-run", "f-containing molecular endpoint tier not run"
                ),
                "promotion": outcome(
                    "not-run", "requires isolated and real-workload evidence"
                ),
            }
        )
    return {
        "schema": "vibeqc.f_shell_validation",
        "schema_version": SCHEMA_VERSION,
        "architecture": architecture,
        "generator_abi": manifest["architectures"][manifest["default_architecture"]][
            "generator_abi"
        ],
        "manifest_hash": file_hash(MANIFEST),
        "rows": rows,
    }


def _tool_version(tool: Path) -> str:
    result = subprocess.run(
        [str(tool), "--version"], check=True, capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr).strip()


def cuobjdump_resources(output: str) -> dict[str, dict[str, int]]:
    """Decode per-entry cubin resources without confusing PTX with device code."""
    rows = {}
    for match in re.finditer(r" Function (\w+):\n([^\n]+)", output):
        fields = {
            key: int(value)
            for key, value in re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)", match[2])
        }
        if set(fields) != {"REG", "STACK", "SHARED", "LOCAL"} or match[1] in rows:
            raise ValueError("incomplete or duplicate CUOBJDump kernel resources")
        rows[match[1]] = fields
    return rows


def write_archive(report: dict, directory: Path) -> None:
    """Preserve complete per-class measurements with a hashed, compact index.

    Combined numerical evidence can exceed the repository's per-file limit.
    Every class file retains all measurements; the index exposes each stage.
    """
    directory.mkdir(parents=True, exist_ok=True)
    index = {key: value for key, value in report.items() if key != "rows"}
    index["rows"] = []
    for row in report["rows"]:
        name = row["shell_class"]
        if name not in F_SHELL_CLASSES:
            raise ValueError("archive contains an unknown f-shell class")
        path = directory / f"{name}.json"
        path.write_text(
            json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        index["rows"].append(
            {
                "shell_class": name,
                "file": path.name,
                "sha256": file_hash(path),
                "source_hash": row["source"]["source_hash"],
                "stages": {
                    stage: row[stage]["status"]
                    for stage in (
                        "source",
                        "compilation",
                        "resources",
                        "numerical",
                        "endpoint",
                        "promotion",
                    )
                },
                "manifest": row["manifest"],
            }
        )
    (directory / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def compile_matrix(
    report: dict,
    *,
    nvcc: Path,
    cache: Path,
    jobs: int = 2,
    timeout: float = 600,
    progress=None,
) -> dict:
    """Gate 2/3: bounded independent release compilations with verified cache hits.

    No GPU is needed here. The key includes generated bytes, generator ABI,
    target, schedule, and NVCC/PTXAS versions; failed compiles are retriable and
    successful cache hits require matching object bytes and resource logs.
    """
    if jobs < 1 or timeout <= 0:
        raise ValueError("compile concurrency and timeout must be positive")
    report = json.loads(json.dumps(report))
    nvcc = nvcc.resolve()
    tools = {
        name: _tool_version(nvcc.with_name(name))
        for name in ("nvcc", "ptxas", "cuobjdump")
    }
    report["toolchain"] = tools
    report["compile_policy"] = {
        "optimization": "-O3",
        "fast_compile": False,
        "jobs": jobs,
        "timeout_seconds": timeout,
    }
    target = cuda_target_info(report["architecture"])
    compiler = CudaCompilerAdapter(nvcc, target, compile_timeout=timeout)
    cache.mkdir(parents=True, exist_ok=True)

    def compile_one(row):
        audit, source = source_audit(row["shell_class"], report["architecture"])
        if audit != row["source"]:
            raise ValueError("catalog source changed before compile")
        identity = {
            "source": audit["source_hash"],
            "generator_abi": report["generator_abi"],
            "architecture": report["architecture"],
            "schedule": audit["schedule"],
            "toolchain": tools,
            "flags": ["-std=c++17", "-O3", "-Xptxas=-v"],
        }
        key = canonical_hash(identity)
        directory = cache / key
        directory.mkdir(exist_ok=True)
        source_path, object_path = directory / "kernel.cu", directory / "kernel.o"
        metadata_path, log_path = directory / "compile.json", directory / "ptxas.txt"
        metadata = None
        if metadata_path.exists():
            candidate = json.loads(metadata_path.read_text())
            if (
                candidate.get("status") == "pass"
                and candidate.get("identity") == identity
                and object_path.is_file()
                and log_path.is_file()
                and candidate.get("object_hash") == file_hash(object_path)
                and candidate.get("log_hash") == file_hash(log_path)
            ):
                metadata = {**candidate, "cache_hit": True}
        if metadata is None:
            source_path.write_text(source)
            compiled = compiler.compile(source_path, object_path)
            log_path.write_text(compiled.stdout + compiled.stderr)
            metadata = {
                **outcome(
                    "pass" if compiled.returncode == 0 else "fail",
                    None
                    if compiled.returncode == 0
                    else "NVCC timeout"
                    if compiled.timed_out
                    else "NVCC compilation failed",
                ),
                "identity": identity,
                "cache_key": key,
                "cache_hit": False,
                "seconds": compiled.duration_seconds,
                "returncode": compiled.returncode,
                "object_bytes": object_path.stat().st_size
                if compiled.returncode == 0
                else None,
                "object_hash": file_hash(object_path)
                if compiled.returncode == 0
                else None,
                "log_hash": file_hash(log_path),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
        row["compilation"] = metadata
        if metadata["status"] == "pass":
            resources = parse_ptxas_resources(
                log_path.read_text(),
                row["shell_class"],
                symbol_prefix=f"generated_{row['shell_class']}_shell_class_",
            )
            selected = [r for r in resources if r.function in audit["symbols"]]
            missing = set(audit["symbols"]) - {r.function for r in selected}
            dumped = subprocess.run(
                [
                    str(nvcc.with_name("cuobjdump")),
                    "--dump-resource-usage",
                    str(object_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            dump_path = directory / "cuobjdump.txt"
            previous_dump = dump_path.read_text() if dump_path.exists() else None
            dump_path.write_text(dumped.stdout)
            cubin_resources = cuobjdump_resources(dumped.stdout)
            failures = [f"missing PTXAS rows: {sorted(missing)}"] if missing else []
            # Report the real device image cost separately from the host object
            # (which also embeds PTX and registration code). A fresh directory
            # prevents an old extracted image from masquerading as this object.
            with TemporaryDirectory(dir=directory) as extracted:
                subprocess.run(
                    [
                        str(nvcc.with_name("cuobjdump")),
                        "--extract-elf",
                        "all",
                        str(object_path.resolve()),
                    ],
                    cwd=extracted,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                images = list(Path(extracted).glob("*.cubin"))
                cubins = [
                    {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in images
                ]
            if len(cubins) != 1:
                failures.append("expected exactly one release cubin")
            if set(audit["symbols"]) != set(cubin_resources):
                failures.append("CUOBJDump does not cover all eight device entries")
            if (
                metadata["cache_hit"]
                and previous_dump is not None
                and previous_dump != dumped.stdout
            ):
                failures.append(
                    "resource reporting changed for the same cached object/toolchain"
                )
            for resource in selected:
                actual = cubin_resources.get(resource.function)
                if actual and (actual["REG"], actual["STACK"]) != (
                    resource.registers,
                    resource.stack_bytes,
                ):
                    failures.append(f"PTXAS/CUOBJDump disagree: {resource.function}")
                if actual:
                    # CUDA 12.9 sm_120 cubins include 1024 extra shared bytes
                    # beyond PTXAS/cudaFuncGetAttributes' user allocation.
                    # Preserve both measurements instead of equating them.
                    actual["EXTRA_SHARED"] = actual["SHARED"] - resource.shared_bytes
                    if actual["EXTRA_SHARED"] < 0:
                        failures.append(
                            f"cubin shared allocation is smaller than PTXAS: {resource.function}"
                        )
                if (
                    not 0 < resource.registers <= target.maximum_registers_per_thread
                    or resource.shared_bytes > target.shared_memory_per_block
                ):
                    failures.append(
                        f"device register/shared-memory limit: {resource.function}"
                    )
            row["resources"] = {
                **outcome(
                    "fail" if failures else "pass",
                    "; ".join(failures) if failures else None,
                ),
                "kernels": [asdict(r) for r in selected],
                "cubin_kernels": cubin_resources,
                "cubins": cubins,
                "spill_free": all(
                    r.spill_load_bytes == r.spill_store_bytes == 0 for r in selected
                ),
                "cuobjdump_hash": file_hash(dump_path),
                "cuobjdump": dumped.stdout,
                "occupancy": outcome(
                    "not-run", "actual occupancy is measured in the GPU tier"
                ),
            }
        if progress is not None:
            progress(row)
        return row

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        report["rows"] = list(executor.map(compile_one, report["rows"]))
    return report
