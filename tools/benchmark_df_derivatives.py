"""Batch-compile exact DF derivative kernels and rank real signature workloads.

Compilation runs on the host. One finite Slurm job runs all runnable candidates,
independent native CPU oracles and CUDA timings. This first-stage report never
writes a production manifest or treats a class timing as an endpoint speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc import Atom
from vibeqc.calculator import _named_basis_shells
from vibeqc_compiler.common.cuda_adapter import (
    CudaBenchmarkExecutor,
    CudaCompilerAdapter,
)
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.integral.df_tuning.batch import compile_batch
from vibeqc_compiler.integral.df_tuning.emission import API, emit_candidate, emit_driver
from vibeqc_compiler.integral.df_tuning.policy import (
    enumerate_trials,
    rank_profiles,
    read_profile,
)
from vibeqc_compiler.integral.ir import KernelConsumer
from vibeqc_compiler.integral.tuning.process import _runtime_environment, _tool_version

from benchmarks._cases import benchmark_cases
from benchmarks.df_policy_endpoint import CASES


def source_identity(generated):
    """Conservatively bind both compiler and transitive native template inputs."""
    digest = hashlib.sha256()
    for base in (ROOT / "python/vibeqc_compiler", ROOT / "src", generated):
        for path in sorted(base.rglob("*")):
            if path.suffix in (".py", ".json", ".hpp", ".cuh", ".cpp", ".cu"):
                digest.update(str(path.relative_to(base)).encode() + b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()


def write_workloads(path, profiles):
    """Serialize real geometries/bases and the ledger's precise panel domain.

    Only ordinary basis metadata is imported from the runtime package. This is
    the benchmark/reference boundary, separate from GPU-free compiler emission.
    Deterministic dense weights exercise the measured work; their counts must
    match the retained physical ledger or ranking refuses the candidate.
    """
    lines = [f"VQDF4041 {len(profiles)}"]
    for name, payload in profiles.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("profile name must be a simple identifier")
        case = benchmark_cases()[CASES[payload["aos"]]]
        atoms = [Atom.from_value(atom) for atom in case.atoms]
        shells = _named_basis_shells(case.vibeqc_basis, atoms)
        reconstruction = payload["host_reconstruction"]
        domain = read_profile(payload)
        panels = reconstruction["panels"]
        lines.append(
            f"{name} {len(atoms)} {len(shells)} {int(case.basis_representation == 'spherical')} "
            f"{payload['benchmark_pair_mode']} {len(panels)} {len(domain)}"
        )
        for atom in atoms:
            lines.append(" ".join(map(str, (atom.atomic_number, *atom.position))))
        for shell in shells:
            lines.append(
                f"{shell.atom_index} {shell.angular_momentum} {len(shell.primitives)}"
            )
            lines.extend(
                f"{p.exponent:.17g} {p.coefficient:.17g}" for p in shell.primitives
            )
        lines.extend(" ".join(map(str, p)) for p in panels)
        for (angular, primitives), work in domain.items():
            lines.append(" ".join(map(str, (*angular, *primitives, *work.values()))))
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument(
        "--pair-mode",
        choices=("full", "symmetric", "packed"),
        action="append",
        required=True,
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--oracle-library", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--compile-jobs", type=int, default=2)
    parser.add_argument("--compile-timeout", type=float, default=600)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.compile_jobs < 1 or not 0 < args.compile_timeout <= 3600:
        parser.error("bounded positive compilation limits required")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    generated = args.generated.resolve()
    if len(args.profile) != len(args.pair_mode):
        parser.error("each profile requires its explicit pair-mode in the same order")
    profiles = {}
    for path, mode in zip(args.profile, args.pair_mode):
        payload = json.loads(path.read_text())
        payload["benchmark_pair_mode"] = ("full", "symmetric", "packed").index(mode)
        profiles[str(payload["aos"])] = payload
    if len(profiles) != len(args.profile):
        parser.error("duplicate AO workload profile")
    trials = enumerate_trials()
    target = cuda_target_info(args.architecture)
    compiler = CudaCompilerAdapter(args.nvcc.resolve(), target, args.compile_timeout)
    identity = source_identity(generated)
    toolchain = _tool_version(args.nvcc)
    includes = (
        directory,
        generated,
        ROOT / "include",
        ROOT / "src",
        ROOT / "tools/vibeqc_validation",
    )
    (directory / "df_benchmark_api.hpp").write_text(API)
    write_workloads(directory / "workload.txt", profiles)

    compiled = compile_batch(
        trials,
        emit_candidate,
        directory=directory,
        compiler=compiler,
        includes=includes,
        generator_sha256=identity,
        toolchain=toolchain,
        consumer=KernelConsumer.FORCE,
        kernel_name="shell_packet",
        jobs=args.compile_jobs,
    )
    runnable = [t for t, row in zip(trials, compiled) if row["eligible"]]
    report = {
        "schema": "vibeqc.df_derivative_tuning",
        "version": 1,
        "generator_sha256": identity,
        "architecture": args.architecture,
        "toolchain": toolchain,
        "profiles": {
            k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
            for k, v in profiles.items()
        },
        "compiled": compiled,
        "production_promoted": False,
    }
    report_path = directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.compile_only:
        return
    if not runnable:
        raise RuntimeError("no runnable candidates; see report.json")
    driver, executable = directory / "driver.cu", directory / "benchmark"
    driver.write_text(emit_driver(runnable))
    library = args.oracle_library.resolve()
    linked = compiler.link(
        driver,
        [directory / f"{t.symbol}.o" for t in runnable],
        executable,
        includes=includes,
        standard="c++20",
        timeout=600,
        options=(f"-Xlinker={library}", f"-Xlinker=-rpath={library.parent}"),
    )
    (directory / "link.log").write_text(linked.stdout + linked.stderr)
    if linked.returncode:
        raise RuntimeError("benchmark link failed; see link.log")
    executor = CudaBenchmarkExecutor(
        timeout=1800, partition="main", gres="gpu:5090:1", slurm_time="00:30:00"
    )
    # Exactly one srun; preserve even an empty inherited CUDA visibility mask.
    command = [*executor.command(executable), str(directory / "workload.txt")]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=1900,
        check=False,
        env=_runtime_environment(args.nvcc.resolve()),
    )
    (directory / "runtime.jsonl").write_text(result.stdout)
    (directory / "runtime.log").write_text(result.stderr)
    report["execution"] = {"command": command, "returncode": result.returncode}
    if result.returncode:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError("batch execution failed; see runtime.log")
    records = [json.loads(line) for line in result.stdout.splitlines()]
    devices = [r for r in records if r.get("kind") == "device"]
    if len(devices) != 1 or devices[0]["architecture"] != args.architecture:
        raise RuntimeError("benchmark device does not match compiled target")
    report["device"] = devices[0]
    report["oracle_library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    results = [r for r in records if r.get("kind") != "device"]
    for row in results:
        row["eligible"] = True
    baselines = {
        t.class_name: t.key
        for t in trials
        if t.lowering == "polynomial" and t.variant == 2
    }
    report["ranking"] = rank_profiles(profiles, results, baselines=baselines)
    report["results"] = results
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["ranking"], indent=2))


if __name__ == "__main__":
    main()
