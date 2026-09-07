"""Supported workload-first user-local autotuning over the existing code generator.

Tuning requires a source checkout and CUDA compiler; ordinary calculations and
reuse of a validated binary do not. Every proposed consumer must improve the
current complete endpoint before it joins the local AOT manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from . import _native
from .profiles import (
    POLICY,
    PROFILE_SCHEMA,
    atomic_json,
    cache_root,
    canonical_hash,
    compatibility_identity,
    export_bundle,
    file_hash,
    find_nvcc,
    install_bundle,
    probe_device,
    select_library,
    toolchain_identity,
)


def source_identity(source: Path) -> str:
    """Mirror the CMake compatibility inventory before spending a tuning budget."""
    paths = {source / "CMakeLists.txt", source / "tools/generate_shell_kernels.py"}
    for directory in ("src", "include"):
        paths.update(p for p in (source / directory).rglob("*") if p.is_file())
    paths.update((source / "python/vibeqc").rglob("*.py"))
    for pattern in ("*.py", "*.json"):
        paths.update((source / "tools/vibeqc_codegen").rglob(pattern))
    text = "".join(
        f"{p.relative_to(source).as_posix()}:{file_hash(p)}\n"
        for p in sorted(paths, key=lambda p: p.relative_to(source).as_posix())
    )
    return hashlib.sha256(text.encode()).hexdigest()


def read_xyz(path: Path, *, units="angstrom") -> list:
    """Read one ordinary XYZ geometry and convert coordinates to public-API Bohr."""
    lines = path.read_text().splitlines()
    count = int(lines[0])
    if (
        count < 1
        or len(lines) < count + 2
        or any(line.strip() for line in lines[count + 2 :])
    ):
        raise ValueError("expected one complete XYZ geometry")
    factor = 1 / 0.529177210903 if units == "angstrom" else 1.0
    atoms = []
    for line in lines[2 : count + 2]:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("XYZ atom rows require an element and three coordinates")
        xyz = [float(value) * factor for value in fields[1:]]
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("XYZ coordinates must be finite")
        atoms.append([fields[0], xyz])
    return atoms


def rank_hotspots(rows: list[dict], *, coverage=0.97, maximum_classes=8) -> list[dict]:
    """Use exact active primitive work; never compile absent shell classes by default."""
    if not 0 < coverage <= 1 or maximum_classes < 1:
        raise ValueError("coverage must be in (0,1] and class limit positive")
    active = []
    for row in rows:
        work = row["primitive_quartets"]
        if not isinstance(work, int) or work < 0:
            raise ValueError("primitive work must be a nonnegative integer")
        if work:
            active.append(
                {**row, "name": "".join("spdf"[l] for l in row["shell_angular"])}
            )
    total = sum(row["primitive_quartets"] for row in active)
    active.sort(key=lambda r: (-r["primitive_quartets"], r["name"]))
    selected, accumulated = [], 0
    for row in active[:maximum_classes]:
        accumulated += row["primitive_quartets"]
        selected.append(
            {
                **row,
                "fraction": row["primitive_quartets"] / total,
                "cumulative_fraction": accumulated / total,
            }
        )
        if accumulated >= total * coverage:
            break
    return selected


def endpoint_gate(
    baseline: list[dict], candidate: list[dict], *, minimum_speedup=1.02
) -> dict:
    """Reject noisy/slower proposals and changed SCF branches without dropping samples."""
    if len(baseline) != len(candidate) or len(baseline) < 4:
        raise ValueError("endpoint gate requires at least four paired repeats")
    failures = []
    for sample in baseline + candidate:
        if not sample["converged"] or any(b != "cuda" for b in sample["backend"]):
            failures.append("nonconverged or non-CUDA endpoint")
        if not math.isfinite(sample["seconds"]) or sample["seconds"] <= 0:
            raise ValueError("endpoint times must be positive and finite")
    iterations = {tuple(sample["iterations"]) for sample in baseline + candidate}
    if len(iterations) != 1:
        failures.append("SCF iteration branches differ")
    energy_error, force_error, translation_error = 0.0, 0.0, 0.0
    for left, right in zip(baseline, candidate):
        energies = np.asarray(left["energies"]) - np.asarray(right["energies"])
        forces = np.asarray(left["forces"]) - np.asarray(right["forces"])
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(forces)):
            raise ValueError("endpoint results must be finite")
        energy_error = max(energy_error, float(np.abs(energies).max()))
        force_error = max(force_error, float(np.abs(forces).max()))
        translation_error = max(
            translation_error,
            float(np.abs(np.asarray(right["forces"]).sum(axis=1)).max()),
        )
    if energy_error > 1e-9 or force_error > 1e-7 or translation_error > 1e-7:
        failures.append("energy/analytic-force/translation parity")
    left = np.array([sample["seconds"] for sample in baseline])
    right = np.array([sample["seconds"] for sample in candidate])
    speedup = float(np.median(left) / np.median(right))
    rng = np.random.default_rng(136)
    resamples = rng.integers(len(left), size=(4096, len(left)))
    lower = float(
        np.quantile(
            np.median(left[resamples], axis=1) / np.median(right[resamples], axis=1),
            0.05,
        )
    )
    if speedup < minimum_speedup or lower <= 1.0:
        failures.append("endpoint improvement is below the threshold or within noise")
    return {
        "passed": not failures,
        "failures": failures,
        "speedup": speedup,
        "bootstrap_speedup_lower_95": lower,
        "minimum_speedup": minimum_speedup,
        "maximum_energy_error": energy_error,
        "maximum_force_error": force_error,
        "maximum_translation_error": translation_error,
        "baseline": baseline,
        "candidate": candidate,
    }


def _worker(
    library: Path,
    workload: Path,
    output: Path,
    *,
    generic=False,
    profile=False,
    timeout=600,
) -> dict:
    env = {**os.environ, "VIBEQC_LIBRARY": str(library), "VIBEQC_PROFILE": "off"}
    # Ambient debugging masks would otherwise turn an A/B into a comparison of
    # arbitrary subsets. Only the initial portable baseline intentionally clears
    # generated consumers; later accepted builds use their complete manifest.
    for name in (
        "VIBEQC_AOT_SHELL_CLASSES",
        "VIBEQC_AOT_FOCK_SHELL_CLASSES",
        "VIBEQC_AOT_MIXED_FOCK_SHELL_CLASSES",
    ):
        if generic:
            env[name] = ""
        else:
            env.pop(name, None)
    command = [
        sys.executable,
        "-m",
        "vibeqc._autotune_worker",
        str(workload),
        str(output),
    ]
    if profile:
        command.append("--profile")
    run = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=timeout, check=False
    )
    output.with_suffix(".log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(
            f"workload execution failed; see {output.with_suffix('.log')}"
        )
    return json.loads(output.read_text())


def _compare(base, candidate, workload, directory, *, repeats, generic, timeout):
    samples = {"baseline": [], "candidate": []}
    # Alternate fresh processes in balanced ABBA blocks. Startup, cold SCF,
    # and warmup are outside each sample; all raw repeats enter the gate.
    for i in range(repeats):
        order = ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
        for side in order:
            samples[side].append(
                _worker(
                    base if side == "baseline" else candidate,
                    workload,
                    directory / f"endpoint-{i}-{side}.json",
                    generic=generic and side == "baseline",
                    timeout=timeout,
                )
            )
    return endpoint_gate(samples["baseline"], samples["candidate"])


def _build(source, build, manifest, target, nvcc, *, timeout, jobs):
    command = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DVIBEQC_BUILD_TESTS=OFF",
        "-DVIBEQC_ENABLE_CUDA=ON",
        "-DVIBEQC_CUDA_FAST_COMPILE=OFF",
        "-DVIBEQC_AOT_UNIT_MODE=class",
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DVIBEQC_CUDA_COMPILE_ARCHITECTURES={target.architecture[3:]}-real",
        f"-DVIBEQC_AOT_SHELL_MANIFEST={manifest}",
        f"-DVIBEQC_CUDA_COMPILE_JOBS={jobs}",
    ]
    build.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (build / "configure.log").open("w") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout
        )
    with (build / "compile.log").open("w") as log:
        subprocess.run(
            ["cmake", "--build", str(build), "--parallel", str(jobs)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    library = build / "libvibeqc.so"
    if not library.exists():
        raise RuntimeError("local CUDA build did not produce libvibeqc.so")
    return library.resolve(), time.monotonic() - started


def _native_kernel_paths(build, architecture, name):
    """Locate the generated source and exact object in a class-mode native build."""
    relative = (
        Path("generated/production_shell_kernels")
        / architecture
        / f"vibeqc_generated_shell_{architecture.replace('_', '')}_{name}.cu"
    )
    return (
        build / relative,
        build
        / "CMakeFiles"
        / f"vibeqc_aot_{architecture}_{name}.dir"
        / (str(relative) + ".o"),
    )


def run(args) -> dict:
    """Tune measured hotspots; publish only complete accepted endpoint replacements."""
    source = args.source_dir.resolve()
    if not (source / "tools/vibeqc_codegen/autotune.py").is_file():
        raise ValueError(
            "autotuning needs the matching VibeQC source checkout; pass --source-dir"
        )
    sys.path.insert(0, str(source))
    # Fail early with an actionable missing-dependency message before compiling.
    import pyscf

    from tools.vibeqc_codegen.autotune import (
        _run_autotune,
        argument_parser,
        supported_schedule_trials,
    )
    from tools.vibeqc_codegen.cuda_target import cuda_target_info
    from tools.vibeqc_codegen.shell_spec import FUSED_SHELL_SPEC_BY_NAME
    from tools.vibeqc_validation.local_tuning import validate_schedule

    nvcc = args.nvcc or find_nvcc()
    if nvcc is None:
        raise ValueError("autotuning requires NVCC/PTXAS; set --nvcc or CUDA_PATH")
    nvcc = nvcc.resolve()
    base_library = _native.load_library()
    probe = probe_device(base_library, args.device_id)
    if source_identity(source) != probe["source_identity"]:
        raise ValueError(
            "source checkout differs from the baseline native build; rebuild it before tuning"
        )
    tools = toolchain_identity(nvcc)
    device = probe["device"]
    if device["release_build"] != 1 or device["fast_compile"] != 0:
        raise ValueError(
            "autotuning requires a Release baseline with VIBEQC_CUDA_FAST_COMPILE=OFF"
        )
    architecture = f"sm_{device['major']}{device['minor']}"
    official = json.loads(
        (source / "tools/vibeqc_codegen/production_shell_classes.json").read_text()
    )
    abi_profile = (
        official["architectures"].get(architecture)
        or official["architectures"][official["default_architecture"]]
    )
    target = cuda_target_info(architecture)
    values = {
        name: value
        for name, value in device.items()
        if name in target.__dataclass_fields__
    }
    target = target.with_runtime_probe(
        **values,
        nvcc_version=tools["nvcc"],
        ptxas_version=tools["ptxas"],
        generator_abi=int(abi_profile["generator_abi"]),
    )
    workload = {
        "atoms": read_xyz(args.input, units=args.units),
        "basis": args.basis,
        "method": args.method,
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "representation": args.representation,
        "device_id": args.device_id,
        "batch": args.batch,
    }
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=root)).resolve()
    workload_path = directory / "workload.json"
    atomic_json(workload_path, workload)
    started = time.monotonic()

    def remaining():
        seconds = args.budget_seconds - (time.monotonic() - started)
        if seconds <= 0:
            raise TimeoutError("local autotuning budget exhausted")
        return max(1, int(seconds))

    report = {
        "schema": "vibeqc.local_autotune",
        "schema_version": PROFILE_SCHEMA,
        "identity": compatibility_identity(probe),
        "toolchain": tools,
        "workload": workload,
        "mode": "full" if args.full else "quick",
        "directory": str(directory),
        "candidates": [],
        "installed": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "endpoint": {"passed": False},
        "reference_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
        },
    }
    report.update(revision=None, dirty=None)
    if shutil.which("git") is not None:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if revision.returncode == 0:
            report["revision"] = revision.stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=source,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            report["dirty"] = bool(status.stdout.strip())
    base_path = Path(base_library._name).resolve()
    incumbent, incumbent_diagnostics = select_library(base_library, args.device_id)
    report["incumbent_profile"] = incumbent_diagnostics
    generic = args.portable_baseline or bool(device["portable"])
    current_path = base_path
    try:
        measured = _worker(
            base_path,
            workload_path,
            directory / "work.json",
            generic=generic,
            profile=True,
            timeout=remaining(),
        )
        hotspots = rank_hotspots(
            measured["work"], coverage=args.coverage, maximum_classes=args.max_classes
        )
        report.update(work_profile=measured, hotspots=hotspots)
        if not hotspots:
            report["reason"] = (
                measured.get("profiling_reason")
                or "no active direct shell-class work was measured"
            )
        manifest = copy.deepcopy(official)
        if generic or architecture not in manifest["architectures"]:
            manifest["architectures"][architecture] = {
                "kind": "tuned",
                "generator_abi": target.generator_abi,
                "target": target.to_payload(),
                "kernels": [],
            }
        manifest["default_architecture"] = architecture
        accepted = []
        accepted_schedules = {}
        build = directory / "build"
        for hotspot in hotspots:
            name = hotspot["name"]
            for consumer in ("fock", "force"):
                remaining()
                existing = {
                    row["shell_class"]: row
                    for row in manifest["architectures"][architecture]["kernels"]
                }
                if not args.retune and consumer in existing.get(name, {}).get(
                    "consumers", []
                ):
                    continue
                if (
                    consumer == "force"
                    and "fock" in existing.get(name, {}).get("consumers", [])
                    and "fock_schedule" not in existing[name]
                ):
                    # Older official rows derive the Fock mapping from the
                    # force schedule. A force-only update must not silently
                    # change that companion's execution geometry. A preceding
                    # accepted explicit Fock tune removes this restriction.
                    report["candidates"].append(
                        {
                            "shell_class": name,
                            "consumer": consumer,
                            "accepted": False,
                            "reason": "implicit official Fock companion requires an accepted explicit Fock schedule first",
                        }
                    )
                    continue
                trial_directory = directory / f"{name}-{consumer}"
                trial_directory.mkdir()
                input_manifest = trial_directory / "input-manifest.json"
                atomic_json(input_manifest, manifest)
                output_manifest = trial_directory / "manifest.json"
                parsed = argument_parser().parse_args(
                    [
                        "--nvcc",
                        str(nvcc),
                        "--architecture",
                        architecture,
                        "--local",
                        # This workflow supplies the independent native-object
                        # and endpoint gates required for subgroup promotion.
                        "--allow-experimental-subgroup-winner",
                        "--shell-class",
                        name,
                        "--consumer",
                        consumer,
                        "--compile-jobs",
                        str(args.compile_jobs),
                        "--max-candidates",
                        str(args.max_candidates),
                        "--samples",
                        "7" if args.full else "5",
                        "--minimum-speedup",
                        "1.02",
                        "--compile-timeout",
                        str(min(600, remaining())),
                        "--timeout",
                        str(min(900, remaining())),
                        "--work-directory",
                        str(trial_directory / "schedules"),
                        "--manifest",
                        str(input_manifest),
                        "--manifest-output",
                        str(output_manifest),
                    ]
                )
                record = {"shell_class": name, "consumer": consumer, "accepted": False}
                report["candidates"].append(record)
                print(
                    f"Tuning {name}/{consumer} ({hotspot['fraction']:.1%} measured primitive work)",
                    flush=True,
                )
                try:
                    tuned = _run_autotune(parsed, runtime_target=target)
                    atomic_json(trial_directory / "tuning.json", tuned)
                    record["tuning"] = tuned
                    record["tuning_report"] = str(trial_directory / "tuning.json")
                    if not tuned["winners"]:
                        record["reason"] = (
                            "no schedule passed compile/resource/numerical/timing gates"
                        )
                        continue
                    winner = tuned["winners"][0]
                    record["selected_trial"] = winner["trial_key"]
                    trial = next(
                        t
                        for t in supported_schedule_trials(
                            FUSED_SHELL_SPEC_BY_NAME[name], consumer, target
                        )
                        if t.key == winner["trial_key"]
                    )
                    isolated = validate_schedule(
                        name,
                        consumer,
                        trial.schedule,
                        target,
                        nvcc,
                        trial_directory / "isolated",
                        timeout=min(900, remaining()),
                    )
                    atomic_json(trial_directory / "isolated.json", isolated)
                    record["isolated"] = isolated
                    if not isolated["passed"]:
                        record["reason"] = "independent all-spin numerical gate failed"
                        continue
                    candidate_manifest = json.loads(output_manifest.read_text())
                    rows = candidate_manifest["architectures"][architecture]["kernels"]
                    row = next(r for r in rows if r["shell_class"] == name)
                    # The developer tuner emits a Fock force companion for ABI
                    # sharing. A local profile enables only independently accepted
                    # consumers and preserves any already accepted companion.
                    row["consumers"] = sorted(
                        set(existing.get(name, {}).get("consumers", [])) | {consumer}
                    )
                    if consumer == "force":
                        row["recurrence"] = "subset_wick"
                        row.pop("resident_force_recurrence", None)
                        row["capabilities"] = [
                            c
                            for c in row.get("capabilities", [])
                            if c != "resident_force"
                        ]
                    if consumer == "fock":
                        row["capabilities"] = [
                            c for c in row.get("capabilities", []) if c != "mixed_fock"
                        ]
                    atomic_json(output_manifest, candidate_manifest)
                    build_manifest = directory / "build-manifest.json"
                    atomic_json(build_manifest, candidate_manifest)
                    compiled, build_seconds = _build(
                        source,
                        build,
                        build_manifest,
                        target,
                        nvcc,
                        timeout=remaining(),
                        jobs=args.compile_jobs,
                    )
                    # Recheck every spin/launch wrapper from the object that
                    # enters the native library. Manifest companions and the
                    # native compiler flags cannot hide behind a standalone
                    # source-emission or synthetic benchmark result.
                    record["prebuild_isolated"] = isolated
                    production_source, production_object = _native_kernel_paths(
                        build, architecture, name
                    )
                    isolated = validate_schedule(
                        name,
                        consumer,
                        trial.schedule,
                        target,
                        nvcc,
                        trial_directory / "native-isolated",
                        timeout=min(900, remaining()),
                        production_source=production_source,
                        production_object=production_object,
                    )
                    record["isolated"] = isolated
                    atomic_json(trial_directory / "native-isolated.json", isolated)
                    if not isolated["passed"]:
                        record["reason"] = "exact native-object numerical gate failed"
                        continue
                    # Snapshot each build: the next incremental relink must not
                    # overwrite the accepted baseline used by fresh A/B processes.

                    candidate_library = trial_directory / "libvibeqc.so"
                    shutil.copyfile(compiled, candidate_library)
                    endpoint = _compare(
                        current_path,
                        candidate_library,
                        workload_path,
                        trial_directory,
                        repeats=args.repeats,
                        generic=generic,
                        timeout=remaining(),
                    )
                    record.update(endpoint=endpoint, build_seconds=build_seconds)
                    if not endpoint["passed"]:
                        record["reason"] = "complete endpoint rejected the candidate"
                        continue
                    kernel = {
                        "shell_class": name,
                        "consumer": consumer,
                        "schedule": winner["schedule"],
                        "schedule_hash": canonical_hash(winner["schedule"]),
                        "source_hash": isolated["source_hash"],
                        "gates": {
                            k: "pass"
                            for k in (
                                "target",
                                "resources",
                                "numerical",
                                "performance",
                                "endpoint",
                            )
                        },
                    }
                    accepted.append(kernel)
                    accepted_schedules[(name, consumer)] = trial.schedule
                    manifest, current_path, generic = (
                        candidate_manifest,
                        candidate_library,
                        False,
                    )
                    record["accepted"] = True
                    report["endpoint"] = endpoint
                except (
                    ValueError,
                    RuntimeError,
                    OSError,
                    subprocess.SubprocessError,
                ) as error:
                    record["reason"] = str(error)
                    (trial_directory / "failure.txt").write_text(traceback.format_exc())
                finally:
                    # Persist rejections too: continue statements must not
                    # hide the completed gates until the whole search ends.
                    atomic_json(directory / "report.json", report)
        if accepted:
            # A later companion changes a class's shared helpers, while a
            # rejected trial can leave the incremental build on another
            # manifest. Rebuild the accepted set and validate every enabled
            # consumer from those final objects before publishing one library.
            final_manifest = directory / "build-manifest.json"
            atomic_json(final_manifest, manifest)
            final_library, report["final_build_seconds"] = _build(
                source,
                build,
                final_manifest,
                target,
                nvcc,
                timeout=remaining(),
                jobs=args.compile_jobs,
            )
            for kernel in accepted:
                name, consumer = kernel["shell_class"], kernel["consumer"]
                production_source, production_object = _native_kernel_paths(
                    build, architecture, name
                )
                isolated = validate_schedule(
                    name,
                    consumer,
                    accepted_schedules[(name, consumer)],
                    target,
                    nvcc,
                    directory / f"final-{name}-{consumer}",
                    timeout=min(900, remaining()),
                    production_source=production_source,
                    production_object=production_object,
                )
                if not isolated["passed"]:
                    raise ValueError(
                        f"final native-object numerical gate failed: {name}/{consumer}"
                    )
                record = next(
                    row
                    for row in report["candidates"]
                    if row.get("accepted")
                    and row["shell_class"] == name
                    and row["consumer"] == consumer
                )
                record["promotion_isolated"] = record["isolated"]
                record["isolated"] = isolated
                kernel["source_hash"] = isolated["source_hash"]
            # Rebuilding an identical accepted manifest must reproduce the
            # endpoint-tested executable; a difference needs another A/B run.
            if file_hash(final_library) != file_hash(current_path):
                raise ValueError(
                    "final native library differs from the endpoint-tested build"
                )
            if incumbent_diagnostics["source"] == "local":
                # A new workload search starts from official/generic code.
                # It may replace a previous local build only after beating
                # that actual incumbent as well as its incremental baselines.
                comparison_directory = directory / "incumbent"
                comparison_directory.mkdir()
                comparison = _compare(
                    Path(incumbent._name).resolve(),
                    current_path,
                    workload_path,
                    comparison_directory,
                    repeats=args.repeats,
                    generic=False,
                    timeout=remaining(),
                )
                report["incumbent_endpoint"] = comparison
                if not comparison["passed"]:
                    report["reason"] = (
                        "current local profile remains faster or within noise"
                    )
                    return report
                report["endpoint"] = comparison
            bundle = directory / "accepted"
            bundle.mkdir()
            shutil.copyfile(current_path, bundle / "libvibeqc.so")
            atomic_json(bundle / "manifest.json", manifest)
            atomic_json(bundle / "evidence.json", report)
            profile = {
                "schema": "vibeqc.local_profile",
                "schema_version": PROFILE_SCHEMA,
                "identity": compatibility_identity(probe),
                "toolchain": tools,
                "generator_abi": target.generator_abi,
                "policy": POLICY,
                "kernels": accepted,
                "artifacts": {
                    name: file_hash(bundle / name)
                    for name in ("libvibeqc.so", "manifest.json", "evidence.json")
                },
            }
            atomic_json(bundle / "profile.json", profile)
            installed = install_bundle(bundle, probe, tools)
            report["installed"] = str(installed)
            if args.export:
                export_bundle(installed, args.export)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report["failure"] = str(error)
    finally:
        atomic_json(directory / "report.json", report)
    return report
