"""Finite parallel candidate compilation through the shared direct CUDA adapter."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from ...common.cuda_resources import parse_resources
from ..tuning.resources import _resource_rejections, estimate_kernel_occupancy


def compile_batch(
    trials,
    emit,
    *,
    directory,
    compiler,
    includes,
    generator_sha256,
    toolchain,
    consumer,
    kernel_name,
    jobs=2,
    maximum_stack_bytes=0,
):
    """Retain every compile/resource rejection without cancelling other candidates.

    Mathematical trial identity and emission are supplied by the consumer. The
    process adapter bounds each compiler tree; this pool bounds host concurrency.
    No CUDA execution or architecture probe occurs during compilation.
    """
    if type(jobs) is not int or not 1 <= jobs <= 64:
        raise ValueError("compile jobs must be in [1,64]")

    def compile_trial(trial):
        source = directory / f"{trial.symbol}.cu"
        obj = directory / f"{trial.symbol}.o"
        source.write_text(emit(trial))
        compiled = compiler.compile(source, obj, includes=includes, standard="c++20")
        diagnostics = compiled.stdout + compiled.stderr
        (directory / f"{trial.symbol}.log").write_text(diagnostics)
        resources = tuple(
            r for r in parse_resources(diagnostics) if kernel_name in r.function
        )
        reasons = _resource_rejections(
            resources,
            consumer=consumer,
            maximum_registers=255,
            maximum_stack_bytes=maximum_stack_bytes,
            maximum_shared_bytes=48 * 1024,
        )
        if compiled.returncode:
            reasons.append(
                "compiler timeout" if compiled.timed_out else "compilation failed"
            )
        return {
            "key": trial.key,
            "artifact_key": trial.artifact_key(
                generator_sha256=generator_sha256,
                architecture=compiler.target.architecture,
                toolchain=toolchain,
            ),
            "compile": asdict(compiled),
            "resources": [asdict(r) for r in resources],
            "occupancy": estimate_kernel_occupancy(
                resources, trial.block_threads, compiler.target
            ),
            "source_bytes": source.stat().st_size,
            "object_bytes": obj.stat().st_size if obj.exists() else None,
            "rejections": reasons,
            "eligible": not reasons,
        }

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(compile_trial, trials))
