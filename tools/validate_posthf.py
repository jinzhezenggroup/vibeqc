"""Run reproducible same-C post-HF endpoints and native HF snapshot checks.

CPU mode needs the native CPU library. CUDA mode additionally compiles the
explicit cuBLAS transform runtime and requires the caller's finite GPU job.
This records initial interface costs, not an automatic performance promotion.
"""

from __future__ import annotations

# Source-tree CLI bootstrap; importing the compiler needs no native runtime.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import numpy as np
from vibeqc.profiles import probe_device
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info

from tools.vibeqc_posthf.conventions import MOBlock
from tools.vibeqc_posthf.cuda import compile_cuda
from tools.vibeqc_posthf.df import DFProvider, MetricFactor
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.mp2 import restricted_mp2
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import CudaDFSource, NativeSource
from tools.vibeqc_validation.schema import (
    block_error,
    canonical_hash,
    file_hash,
    new_evidence,
    outcome,
    validate_evidence,
)


def source_hash():
    files = [
        *ROOT.glob("src/posthf/*"),
        *ROOT.glob("tools/vibeqc_posthf/*.py"),
        ROOT / "src/integrals/s_integrals.cpp",
        ROOT / "src/tensor/cuda_runtime.cuh",
        ROOT / "src/tensor/metrics.hpp",
    ]
    return canonical_hash({str(p.relative_to(ROOT)): file_hash(p) for p in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--generated-df", action="store_true")
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/group/software/cuda-12.9.1/bin/nvcc")
    )
    parser.add_argument("--cache", type=Path, default=Path("/tmp/posthf147-cuda-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["h2", "water", "lih"],
        choices=["h2", "water", "lih"],
    )
    args = parser.parse_args()
    if args.samples < 5:
        parser.error("at least five interleaved samples are required")
    if (args.cuda or args.generated_df) and not os.environ.get("SLURM_JOB_ID"):
        parser.error("GPU execution requires a finite Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        )
    )
    artifact = (
        compile_cuda(
            CudaCompilerAdapter(args.nvcc, cuda_target_info("sm_120")), args.cache
        )
        if args.cuda
        else None
    )
    records = []
    for name in args.cases:
        meta, arrays = load_fixture(name)
        snapshot = fixture_snapshot(meta, arrays)
        with NativeSource(**source_arguments(meta)) as source:
            device = (
                probe_device(source._library, 0)
                if args.cuda or args.generated_df
                else {"name": platform.processor() or platform.machine(), "kind": "cpu"}
            )
            backends = ["cpu", "cuda"] if args.cuda else ["cpu"]
            by_backend = {b: [] for b in backends}
            for trial in range(args.samples):
                for backend in backends if trial % 2 == 0 else list(reversed(backends)):
                    start = time.perf_counter()
                    errors = {}
                    diagnostics = []
                    with ConventionalProvider(
                        snapshot,
                        source,
                        backend=backend,
                        axis_tile=2,
                        cuda_artifact=artifact,
                        budget_bytes=512 << 20,
                    ) as provider:
                        for spaces in ("ovov", "oovv", "ovvv"):
                            block = MOBlock.from_spaces(snapshot, spaces)
                            result = provider.get(block)
                            host = result.to_host()
                            errors[spaces] = block_error(
                                host,
                                arrays["conventional_mo"][np.ix_(*block.slots)],
                                atol=1e-11,
                                rtol=1e-10,
                            )
                            diagnostic = dict(result.diagnostics)
                            if backend == "cuda":
                                diagnostic["cuda"] = result.values.metrics()
                            diagnostics.append(diagnostic)
                        mp = restricted_mp2(snapshot, provider)
                        errors["amplitudes"] = block_error(
                            mp.amplitudes,
                            arrays["conventional_t2"],
                            atol=1e-11,
                            rtol=1e-10,
                        )
                        errors["energy"] = block_error(
                            [mp.correlation_energy],
                            [meta["records"]["conventional"]["correlation_energy"]],
                            atol=1e-9,
                            rtol=0,
                        )
                        cold = time.perf_counter() - start
                        warm = []
                        for _ in range(5):
                            tick = time.perf_counter()
                            restricted_mp2(snapshot, provider)
                            warm.append(time.perf_counter() - tick)
                        stats = dict(provider.statistics)
                    assert all(e["passed"] for e in errors.values()), errors
                    by_backend[backend].append(
                        {
                            "trial": trial,
                            "cold_seconds": cold,
                            "warm_seconds": warm,
                            "errors": errors,
                            "blocks": diagnostics,
                            "statistics": stats,
                        }
                    )
            for backend, rows in by_backend.items():
                digest = canonical_hash(
                    {
                        "arrays": meta["array_hash"],
                        "reference": snapshot.identity,
                        "blocks": ["ovov", "oovv", "ovvv"],
                    }
                )
                record = new_evidence(
                    tier="endpoint",
                    subject=f"posthf/{name}/conventional/{backend}",
                    inputs_hash=digest,
                )
                record.update(
                    revision=revision,
                    device=device
                    if backend == "cuda"
                    else {
                        "name": platform.processor() or platform.machine(),
                        "kind": "cpu",
                    },
                    backend_selected=backend,
                    settings={
                        "device": backend,
                        "dirty": dirty,
                        "fast_compile": False,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "placement": "CPU raw AO tiles; "
                        + (
                            "CUDA cuBLAS transforms and retained MO outputs"
                            if backend == "cuda"
                            else "CPU transforms and retained outputs"
                        ),
                        "reference_version": "PySCF 2.14.0",
                        "reference_id": snapshot.identity,
                        "hamiltonian_id": snapshot.hamiltonian_id,
                        "fixed_state_hash": digest,
                        "scope": "initial provider and CPU MP2 bridge; no registered method or performance promotion",
                    },
                    toolchain={
                        "python": sys.version,
                        "numpy": np.__version__,
                        **(
                            artifact.metadata["identity"]["toolchain"]
                            if artifact
                            else {}
                        ),
                    },
                )
                record["hashes"].update(
                    equation=canonical_hash(
                        "g[pqrs]=C[up]C[vq]C[wr]C[xs](uv|wx); E=sum t(2g-g_exchange)"
                    ),
                    ir=canonical_hash(
                        {"conventions": "chemists/spatial/ijab", "version": 1}
                    ),
                    source=source_hash(),
                    schedule=canonical_hash({"axis_tile": 2, "backend": backend}),
                )
                record["hardware"] = outcome("pass")
                for stage in (
                    "representation",
                    "source",
                    "compilation",
                    "numerical",
                    "endpoint",
                ):
                    record["stages"][stage] = outcome("pass")
                record["performance"] = outcome(
                    "not-run",
                    "initial interface evidence; no performance replacement promoted",
                )
                record["compilation"] = {
                    "seconds": artifact.metadata["compile_seconds"]
                    if backend == "cuda"
                    else None,
                    "reason": None
                    if backend == "cuda"
                    else "native CPU build recorded separately",
                    "resources": artifact.metadata["resources"]
                    if backend == "cuda"
                    else [],
                }
                record["block_errors"] = rows[0]["errors"]
                record["residuals"] = dict(snapshot.diagnostics)
                record["memory"] = {
                    "allocated_bytes": max(
                        sum(
                            d.get("cuda", {}).get("owned_device_bytes", 0)
                            for d in row["blocks"]
                        )
                        for row in rows
                    ),
                    "peak_bytes": max(row["statistics"]["peak_bytes"] for row in rows),
                    "reason": None,
                    "scope": "numeric buffers plus provider allowance; see docs/posthf.md",
                }
                for row in rows:
                    record["timings"].append(
                        {
                            "selection": "baseline"
                            if backend == "cpu"
                            else "candidate",
                            "seconds": row["cold_seconds"],
                            "workload": "cold-start",
                            "inputs_hash": digest,
                            "trial": row["trial"],
                        }
                    )
                    for value in row["warm_seconds"]:
                        record["timings"].append(
                            {
                                "selection": "baseline"
                                if backend == "cpu"
                                else "candidate",
                                "seconds": value,
                                "workload": "unchanged-geometry",
                                "inputs_hash": digest,
                                "trial": row["trial"],
                            }
                        )
                record["details"] = rows
                validate_evidence(record)
                records.append(record)
                print(record["subject"], "pass", flush=True)
            # Matched DF Hamiltonian comparison, with distinct placement records.
            for generated in [False, True] if args.generated_df else [False]:
                df_source = (
                    CudaDFSource(**source_arguments(meta), tile_capacity=64)
                    if generated
                    else source
                )
                try:
                    metric = MetricFactor.from_source(df_source)
                    df_snapshot = fixture_snapshot(
                        meta, arrays, label="df", metric=metric
                    )
                    with DFProvider(
                        df_snapshot, df_source, metric, auxiliary_tile=3
                    ) as provider:
                        begin = time.perf_counter()
                        result = restricted_mp2(df_snapshot, provider)
                        cold = time.perf_counter() - begin
                        block = MOBlock.from_spaces(df_snapshot, "ovov")
                        mo = provider.get(block)
                        errors = {
                            "mo": block_error(
                                mo.to_host(),
                                arrays["df_mo"][np.ix_(*block.slots)],
                                atol=1e-11,
                                rtol=1e-10,
                            ),
                            "amplitudes": block_error(
                                result.amplitudes,
                                arrays["df_t2"],
                                atol=1e-11,
                                rtol=1e-10,
                            ),
                            "energy": block_error(
                                [result.correlation_energy],
                                [meta["records"]["df"]["correlation_energy"]],
                                atol=1e-9,
                                rtol=0,
                            ),
                        }
                        assert all(e["passed"] for e in errors.values())
                        data = {
                            "subject": f"posthf/{name}/DF/" + df_source.backend,
                            "same_hamiltonian_errors": errors,
                            "diagnostics": mo.diagnostics,
                            "metric": {
                                k: v
                                for k, v in asdict(metric).items()
                                if k != "inverse_square_root"
                            },
                            "cold_seconds": cold,
                            "statistics": dict(provider.statistics),
                            "conventional_vs_df_ao_difference": meta[
                                "conventional_df_ao_max_difference"
                            ],
                        }
                        (args.output / f"{name}-df-{int(generated)}.json").write_text(
                            json.dumps(data, indent=2) + "\n"
                        )
                        print(data["subject"], "pass", flush=True)
                finally:
                    if generated:
                        df_source.close()
            # Complete VibeQC HF -> owned snapshot -> provider -> MP2.
            for backend in backends:
                own, export = export_rhf(
                    source, backend=backend, generation_id=f"{name}-{backend}-native"
                )
                with ConventionalProvider(
                    own, source, backend=backend, cuda_artifact=artifact
                ) as provider:
                    mp = restricted_mp2(own, provider)
                error = abs(
                    mp.correlation_energy
                    - meta["records"]["conventional"]["correlation_energy"]
                )
                assert error <= 1e-9, (name, backend, error)
                (args.output / f"{name}-native-{backend}.json").write_text(
                    json.dumps(
                        {
                            "export": export,
                            "mp2_energy": mp.correlation_energy,
                            "mp2_absolute_error": error,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(f"posthf/{name}/native-HF-{backend}->MP2 pass", flush=True)
        (args.output / "evidence.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


if __name__ == "__main__":
    main()
