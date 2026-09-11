"""Independent all-spin numerical gates for a proposed local CUDA schedule.

Reuse the f-shell matrix's libcint fixtures, explicit ERI contractions, and
native host driver. The scientific oracle is independent of the schedule tuner.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from vibeqc_compiler.integral.autotune import schedule_payload
from vibeqc_compiler.integral.benchmark import _CUDA_PRELUDE
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_lowering import emit_shell_class_fused_cuda
from vibeqc_compiler.integral.fused_schedule import build_fused_shell_plan
from vibeqc_compiler.integral.ir import KernelConsumer
from vibeqc_compiler.integral.shell_spec import FUSED_SHELL_SPEC_BY_NAME

from .f_shell import cuobjdump_resources
from .f_shell_cuda import emit_numerical_driver
from .f_shell_numerics import (
    class_fixtures,
    decoded_outputs,
    numerical_error,
    write_fixture,
)
from .schema import canonical_hash, file_hash


def validate_schedule(
    name,
    consumer,
    schedule,
    target,
    nvcc: Path,
    directory: Path,
    *,
    timeout=600,
    production_source: Path | None = None,
    production_object: Path | None = None,
) -> dict:
    """Run all-spin wrappers, optionally from the exact candidate native object.

    The production-object mode preserves the native build's actual code and
    compiler resources. Only the host fixture ABI and external C kernel names
    are declared by the separately linked numerical driver.
    """
    consumers = (
        (KernelConsumer.FOCK, KernelConsumer.FORCE)
        if consumer == "fock"
        else (KernelConsumer.FORCE,)
    )
    plan = build_fused_shell_plan(
        FUSED_SHELL_SPEC_BY_NAME[name],
        consumers=consumers,
        schedule=schedule,
        target=target,
    )
    # A manifest's explicit Fock schedule honors its block geometry. Match
    # that production lowering, including when a force companion would have
    # chosen a different implicit value block size in the timing harness.
    source = _CUDA_PRELUDE + emit_shell_class_fused_cuda(
        plan.spec, plan, fock_schedule=schedule if consumer == "fock" else None
    )
    directory.mkdir(parents=True, exist_ok=True)
    path, obj, driver, executable = [
        directory / n for n in ("kernel.cu", "kernel.o", "driver.cu", "numerical")
    ]
    driver_source = emit_numerical_driver(
        name, target.architecture, plan=plan, source=source, consumer=consumer
    )
    compiler = CudaCompilerAdapter(nvcc, target, compile_timeout=timeout)
    compiled = None
    if production_object is not None:
        if production_source is None:
            raise ValueError("a production object requires its generated source")
        path.write_bytes(production_source.read_bytes())
        shutil.copyfile(production_object, obj)
        prefix = target.architecture.replace("_", "")
        driver_source = driver_source.replace(
            f"generated_{name}_", f"generated_{prefix}_{name}_"
        )
    else:
        if production_source is not None:
            raise ValueError("a production source requires its compiled object")
        path.write_text(source)
        compiled = compiler.compile(path, obj)
        (directory / "ptxas.txt").write_text(compiled.stdout + compiled.stderr)
        if compiled.returncode:
            raise ValueError("isolated production-wrapper compilation failed")
    driver.write_text(driver_source)
    linked = compiler.link(driver, [obj], executable, timeout=timeout)
    if linked.returncode:
        (directory / "link.txt").write_text(linked.stdout + linked.stderr)
        raise ValueError("isolated production-wrapper link failed")
    fixtures, _ = class_fixtures(name)
    paths = [directory / f"fixture-{i}.bin" for i in range(len(fixtures))]
    for fixture, fixture_path in zip(fixtures, paths):
        write_fixture(fixture, fixture_path)
    # The user-facing workflow is itself run in the allocated GPU job. Child
    # processes inherit its device visibility; nested allocations are avoided.
    run = subprocess.run(
        [str(executable), *map(str, paths)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    (directory / "numerical.jsonl").write_text(run.stdout)
    (directory / "numerical.stderr").write_text(run.stderr)
    if run.returncode:
        raise ValueError("isolated numerical CUDA execution failed")
    records = [json.loads(line) for line in run.stdout.splitlines()]
    if len(records) != len(fixtures) + 1 or records[0].get("kind") != "device":
        raise ValueError("incomplete isolated CUDA results")
    expected = {
        f"{spin}_{consumer}{suffix}"
        for spin in ("rhf", "uhf")
        for suffix in ("", "_persistent")
    }
    errors = {}
    for i, (fixture, row) in enumerate(zip(fixtures, records[1:])):
        if row.get("ordinal") != i or set(row["outputs"]) != expected:
            raise ValueError("missing or reordered consumer results")
        for key, actual in decoded_outputs(fixture, row).items():
            errors[f"{i}/{key}"] = numerical_error(
                actual,
                fixture.reference[key.removesuffix("_persistent")],
                atol=2e-10,
                rtol=2e-10,
            )
            if consumer == "force":
                errors[f"{i}/{key}/translation"] = numerical_error(
                    actual.sum(axis=0), np.zeros(3), atol=2e-10, rtol=0
                )
    dump = subprocess.run(
        [str(nvcc.with_name("cuobjdump")), "--dump-resource-usage", str(obj)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    cubins = cuobjdump_resources(dump.stdout)
    if not cubins:
        raise ValueError("CUOBJDump resource evidence is missing")
    with TemporaryDirectory(dir=directory) as extracted:
        subprocess.run(
            [
                str(nvcc.with_name("cuobjdump")),
                "--extract-elf",
                "all",
                str(obj.resolve()),
            ],
            cwd=extracted,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        images = list(Path(extracted).glob("*.cubin"))
        if len(images) != 1:
            raise ValueError("expected exactly one target cubin")
        cubin = {"bytes": images[0].stat().st_size, "sha256": file_hash(images[0])}
    return {
        "passed": all(e["passed"] for e in errors.values()),
        "source_hash": file_hash(path),
        "schedule_hash": canonical_hash(schedule_payload(schedule)),
        "fixture_hashes": [fixture.inputs_hash for fixture in fixtures],
        "errors": errors,
        "device": records[0],
        "cubin_resources": cubins,
        "cubin": cubin,
        "source_bytes": path.stat().st_size,
        "object_bytes": obj.stat().st_size,
        "object_hash": file_hash(obj),
        "compile_seconds": compiled.duration_seconds if compiled is not None else None,
        "object_origin": "candidate native build"
        if production_object is not None
        else "isolated release compile",
        "driver_hash": file_hash(driver),
    }
