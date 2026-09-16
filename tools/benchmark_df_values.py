"""Batch value lowerings on stratified real shell/signature distributions.

This first-stage experiment measures bounded samples of the exact Cartesian raw
value domain. Frequencies come from the actual 384/768 basis populations; CUDA
replication removes launch-dominated tiny samples. Weighted class estimates are
not cold/source-backed endpoint timings and never automatically promote policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
import subprocess
import sys
from collections import defaultdict
from importlib.metadata import version
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

import numpy as np
from vibeqc import Atom
from vibeqc.calculator import _named_basis_shells
from vibeqc_compiler.common.cuda_adapter import (
    CudaBenchmarkExecutor,
    CudaCompilerAdapter,
)
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.integral.df_tuning.batch import compile_batch
from vibeqc_compiler.integral.df_tuning.values import (
    VALUE_API,
    emit_value_candidate,
    emit_value_driver,
    enumerate_value_trials,
)
from vibeqc_compiler.integral.df_value_candidates import VALUE_CLASSES
from vibeqc_compiler.integral.ir import KernelConsumer
from vibeqc_compiler.integral.shell_spec import cartesian_components
from vibeqc_compiler.integral.tuning.process import _runtime_environment, _tool_version

from benchmarks._cases import benchmark_cases
from benchmarks.df_policy_endpoint import CASES
from tools.benchmark_df_derivatives import source_identity
from tools.generate_validation_references import pyscf_molecule
from tools.vibeqc_validation.df_values import PRIMITIVE_DTYPE
from tools.vibeqc_validation.f_shell_numerics import _normalized_primitives


def write_workloads(path, *, aos=(384, 768), samples=8):
    """Sample actual shell triples, retaining exact complete signature frequency.

    Reference contraction uses libcint's independent three-center API. Records
    use the same public Cartesian normalization as native tensor export. Every
    real primitive signature is represented; ordered pairs include diagonals.
    """
    groups = []
    with path.open("wb") as output:
        output.write(b"VQDV4051" + struct.pack("<I", 0))
        for size in aos:
            case = benchmark_cases()[CASES[size]]
            atoms = [Atom.from_value(atom) for atom in case.atoms]
            shells = [
                {
                    "atom_index": s.atom_index,
                    "angular_momentum": s.angular_momentum,
                    "primitives": [[p.exponent, p.coefficient] for p in s.primitives],
                }
                for s in _named_basis_shells(case.vibeqc_basis, atoms)
            ]
            coordinates = np.array([a.position for a in atoms])
            inputs = {
                "name": CASES[size],
                "atomic_numbers": [a.atomic_number for a in atoms],
                "coordinates": coordinates.tolist(),
                "shells": shells,
                "basis_representation": "cartesian",
                "charge": 0,
                "multiplicity": 1,
            }
            mol, scales, _ = pyscf_molecule(inputs)
            offsets = mol.ao_loc_nr()
            signatures = defaultdict(list)
            for i, shell in enumerate(shells):
                signatures[
                    (shell["angular_momentum"], len(shell["primitives"]))
                ].append(i)
            normalized = [_normalized_primitives(s) for s in shells]
            for angular in VALUE_CLASSES:
                lengths = [
                    sorted({p for a, p in signatures if a == l}) for l in angular
                ]
                components = tuple(product(*(cartesian_components(l) for l in angular)))
                factors = np.ones(len(components))
                powers = []
                for slot in range(3):
                    power = np.array(
                        [[c[slot].count(axis) for axis in "xyz"] for c in components],
                        dtype=np.uint32,
                    )
                    powers.append(power)
                    factors /= np.sqrt(
                        [
                            math.prod(math.prod(range(1, 2 * int(p), 2)) for p in row)
                            for row in power
                        ]
                    )
                for primitives in product(*lengths):
                    populations = [
                        signatures[(l, p)] for l, p in zip(angular, primitives)
                    ]
                    frequency = math.prod(map(len, populations))
                    chosen = np.unique(
                        np.linspace(
                            0, frequency - 1, min(samples, frequency), dtype=int
                        )
                    )
                    records, references, shell_ids = [], [], []
                    for index in chosen:
                        selected = tuple(
                            populations[k][j]
                            for k, j in enumerate(
                                np.unravel_index(index, tuple(map(len, populations)))
                            )
                        )
                        shell_ids.append(selected)
                        radial = list(product(*(normalized[s] for s in selected)))
                        row = np.zeros(
                            (len(components), len(radial)), dtype=PRIMITIVE_DTYPE
                        )
                        row["centers_count"] = 3
                        for k, shell in enumerate(selected):
                            row["angular"][:, :, k, :] = powers[k][:, None, :]
                            row["centers"][:, :, k, :] = coordinates[
                                shells[shell]["atom_index"]
                            ]
                            row["exponents"][:, :, k] = [p[k][0] for p in radial]
                        row["weight"] = factors[:, None] * [
                            math.prod(c for _, c in p) for p in radial
                        ]
                        reference = mol.intor_by_shell("int3c2e_cart", selected)
                        for k, shell in enumerate(selected):
                            shape = [1] * 3
                            shape[k] = reference.shape[k]
                            reference *= scales[
                                offsets[shell] : offsets[shell + 1]
                            ].reshape(shape)
                        records.append(row.reshape(-1))
                        references.append(reference.reshape(-1))
                    records, reference = (
                        np.concatenate(records),
                        np.concatenate(references),
                    )
                    tasks = len(reference)
                    replicas = max(1, (8192 + tasks - 1) // tasks)
                    output.write(
                        struct.pack(
                            "<11Q",
                            size,
                            *angular,
                            *primitives,
                            len(chosen),
                            frequency,
                            tasks,
                            replicas,
                        )
                    )
                    output.write(records.tobytes())
                    output.write(reference.astype("<f8").tobytes())
                    groups.append(
                        {
                            "profile": str(size),
                            "angular": angular,
                            "primitives": primitives,
                            "shell_frequency": frequency,
                            "sampled_shell_blocks": len(chosen),
                            "sampled_shell_ids": shell_ids,
                            "sampled_components": tasks,
                            "replicas": replicas,
                            "measured_primitive_components": tasks
                            * replicas
                            * math.prod(primitives),
                            "workload_cartesian_primitive_components": frequency
                            * len(components)
                            * math.prod(primitives),
                            "input_sha256": hashlib.sha256(
                                records.tobytes()
                            ).hexdigest(),
                        }
                    )
        output.seek(8)
        output.write(struct.pack("<I", len(groups)))
    return groups


def rank_values(groups, records, compiled, *, minimum_gain=0.03):
    """Fail closed on absent/failed signatures and retain profile disagreements."""
    if not groups or not math.isfinite(minimum_gain) or not 0 <= minimum_gain < 1:
        raise ValueError("nonempty workloads and minimum_gain in [0,1) required")
    eligible = {r["key"] for r in compiled if r["eligible"]}
    indexed = {}
    for row in records:
        key = (row["group"], row["candidate"])
        if key in indexed:
            raise ValueError("duplicate value signature result")
        indexed[key] = row
    trials = enumerate_value_trials()
    profiles = {}
    for profile in sorted({g["profile"] for g in groups}):
        classes = {}
        for angular in VALUE_CLASSES:
            relevant = [
                (i, g)
                for i, g in enumerate(groups)
                if g["profile"] == profile and tuple(g["angular"]) == angular
            ]
            if not relevant:
                continue
            scores, rejections = {}, {}
            candidates = [t for t in trials if t.angular == angular]
            baseline = next(
                t.key for t in candidates if t.lowering == "generic" and t.lanes == 1
            )
            for trial in candidates:
                reasons, score = [], 0.0
                if trial.key not in eligible:
                    reasons.append("compile/resource gate failed")
                for i, g in relevant:
                    row = indexed.get((i, trial.key))
                    if row is None:
                        reasons.append("incomplete signature coverage")
                        continue
                    if row.get("numerical_passed") is not True:
                        reasons.append("independent value gate failed")
                    if any(
                        row.get(k) != g[k]
                        for k in (
                            "profile",
                            "shell_frequency",
                            "sampled_shell_blocks",
                            "sampled_components",
                            "replicas",
                            "measured_primitive_components",
                        )
                    ):
                        reasons.append("changed measured work")
                    times = row.get("milliseconds", [])
                    if len(times) < 5 or any(
                        type(t) not in (int, float) or not math.isfinite(t) or t <= 0
                        for t in times
                    ):
                        reasons.append("invalid timing samples")
                    else:
                        score += (
                            statistics.median(times)
                            * g["shell_frequency"]
                            / (g["sampled_shell_blocks"] * g["replicas"])
                        )
                if reasons:
                    rejections[trial.key] = sorted(set(reasons))
                else:
                    scores[trial.key] = score
            winner = baseline
            if baseline in scores:
                best = min(scores, key=lambda k: (scores[k], k))
                if scores[best] < scores[baseline] * (1 - minimum_gain):
                    winner = best
            classes["".join(map(str, angular))] = {
                "baseline": baseline,
                "baseline_valid": baseline in scores,
                "winner": winner,
                "estimated_class_ms": scores,
                "rejections": rejections,
            }
        profiles[profile] = classes
    combined, conflicts = {}, {}
    for cls in sorted({c for p in profiles.values() for c in p}):
        choices = {
            p: rows[cls]["winner"] for p, rows in profiles.items() if cls in rows
        }
        if (
            len(choices) == len(profiles)
            and len(set(choices.values())) == 1
            and all(rows[cls]["baseline_valid"] for rows in profiles.values())
        ):
            combined[cls] = next(iter(choices.values()))
        else:
            conflicts[cls] = choices
    return {
        "profiles": profiles,
        "proposed_mapping": combined,
        "conflicts": conflicts,
        "production_promoted": False,
        "minimum_gain": minimum_gain,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--compile-jobs", type=int, default=2)
    args = parser.parse_args()
    directory, generated = args.directory.resolve(), args.generated.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    groups = write_workloads(directory / "workload.bin")
    (directory / "workloads.json").write_text(json.dumps(groups, indent=2) + "\n")
    (directory / "df_value_benchmark_api.hpp").write_text(VALUE_API)
    trials = enumerate_value_trials()
    identity, toolchain = source_identity(generated), _tool_version(args.nvcc)
    compiler = CudaCompilerAdapter(
        args.nvcc.resolve(), cuda_target_info(args.architecture), 600
    )
    includes = (directory, generated, ROOT / "tools/vibeqc_validation")
    compiled = compile_batch(
        trials,
        emit_value_candidate,
        directory=directory,
        compiler=compiler,
        includes=includes,
        generator_sha256=identity,
        toolchain=toolchain,
        consumer=KernelConsumer.FOCK,
        kernel_name="value_candidate",
        jobs=args.compile_jobs,
        maximum_stack_bytes=4096,
    )
    report = {
        "schema": "vibeqc.df_value_tuning",
        "version": 1,
        "architecture": args.architecture,
        "generator_sha256": identity,
        "toolchain": toolchain,
        "reference_versions": {"pyscf": version("pyscf"), "numpy": np.__version__},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiled": compiled,
        "workload_sha256": hashlib.sha256(
            (directory / "workload.bin").read_bytes()
        ).hexdigest(),
        "ranking_scope": "stratified raw Cartesian class estimates; endpoints required",
        "groups": groups,
    }
    report_path = directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    runnable = [t for t, r in zip(trials, compiled) if r["eligible"]]
    if not runnable:
        raise RuntimeError("no eligible value candidates")
    driver, executable = directory / "driver.cu", directory / "benchmark"
    driver.write_text(emit_value_driver(runnable))
    linked = compiler.link(
        driver,
        [directory / f"{t.symbol}.o" for t in runnable],
        executable,
        includes=includes,
        standard="c++20",
        timeout=600,
    )
    (directory / "link.log").write_text(linked.stdout + linked.stderr)
    if linked.returncode:
        raise RuntimeError("value batch link failed")
    executor = CudaBenchmarkExecutor(
        timeout=1800, partition="main", gres="gpu:5090:1", slurm_time="00:30:00"
    )
    command = [*executor.command(executable), str(directory / "workload.bin")]
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
        raise RuntimeError("value batch execution failed")
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    devices = [r for r in rows if r.get("kind") == "device"]
    if len(devices) != 1 or devices[0]["architecture"] != args.architecture:
        raise RuntimeError("device and compiled target differ")
    report["device"] = devices[0]
    report["results"] = [r for r in rows if r.get("kind") != "device"]
    report["ranking"] = rank_values(groups, report["results"], compiled)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["ranking"], indent=2))


if __name__ == "__main__":
    main()
