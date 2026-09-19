"""Persistent multi-ISA CPU bundles and safe runtime dispatch for integral kernels."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cpu_dispatch import (
    CpuDispatchDecision,
    CpuRuntimeFeatures,
    cpu_binary_target_supported,
    detect_cpu_features,
    normalize_cpu_architecture,
    select_cpu_target,
)
from vibeqc_compiler.common.cpu_target import (
    AVX2_FMA_TARGET,
    AVX512F_FMA_TARGET,
    GENERIC_CPU_TARGET,
    CpuTargetInfo,
)
from vibeqc_compiler.common.cuda_runtime import CudaArtifact
from vibeqc_compiler.common.provenance import atomic_json, canonical_hash, file_hash

from .cpu_lane_execute import (
    CompiledFirstDerivativeCpuLane,
    FirstDerivativeCpuLaneEvaluator,
    compile_first_derivative_cpu_lane,
)
from .cpu_schedule import (
    CpuAlgebraPlacement,
    CpuLanePacking,
    CpuScheduleIR,
    CpuTailPolicy,
    default_cpu_schedule,
)
from .expr import AlgebraForm, AlgebraFusion, AlgebraOrdering, PowerLowering
from .ir_serialization import integral_from_payload, integral_to_payload


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise TypeError(f"CPU manifest field {key!r} must be an integer")
    return value


def _target_from_payload(payload: Mapping[str, object]) -> CpuTargetInfo:
    if payload.get("schema") != "vibeqc.cpu-target.v1":
        raise ValueError("unsupported CPU target manifest schema")
    features = payload.get("features")
    options = payload.get("compiler_options")
    if not isinstance(features, (list, tuple)) or not isinstance(
        options, (list, tuple)
    ):
        raise TypeError("CPU target manifest features/options must be arrays")
    return CpuTargetInfo(
        name=str(payload["name"]),
        architecture=str(payload["architecture"]),
        features=tuple(str(value) for value in features),
        vector_lanes=_required_int(payload, "vector_lanes"),
        compiler_options=tuple(str(value) for value in options),
    )


def _schedule_from_payload(payload: Mapping[str, object]) -> CpuScheduleIR:
    if payload.get("schema") != "vibeqc.cpu-schedule.v1":
        raise ValueError("unsupported CPU schedule manifest schema")
    return CpuScheduleIR(
        vector_lanes=_required_int(payload, "vector_lanes"),
        lane_packing=CpuLanePacking(str(payload["lane_packing"])),
        tail_policy=CpuTailPolicy(str(payload["tail_policy"])),
        algebra_placement=CpuAlgebraPlacement(str(payload["algebra_placement"])),
        algebra_ordering=AlgebraOrdering(str(payload["algebra_ordering"])),
        algebra_fusion=AlgebraFusion(str(payload["algebra_fusion"])),
        algebra_form=AlgebraForm(str(payload["algebra_form"])),
        power_lowering=PowerLowering(str(payload["power_lowering"])),
    )


def _candidate_binary_target(artifact: CompiledFirstDerivativeCpuLane) -> str:
    identity = artifact.native.metadata.get("identity")
    target = identity.get("target") if isinstance(identity, dict) else None
    if not isinstance(target, dict):
        raise TypeError("CPU bundle candidate lacks a concrete compiler target")
    value = target.get("architecture")
    if (
        not isinstance(value, str)
        or not value.strip()
        or target.get("backend") != "cpu"
    ):
        raise ValueError("CPU bundle candidate lacks a concrete compiler target")
    return value


def _candidate_identity_record(
    artifact: CompiledFirstDerivativeCpuLane,
) -> dict[str, object]:
    return {
        "target": artifact.target.to_payload(),
        "schedule": artifact.schedule.to_payload(),
        "program_identity": artifact.program_identity,
        "artifact_key": artifact.native.metadata["key"],
        "binary_sha256": artifact.native.metadata["binary_sha256"],
        "binary_target": _candidate_binary_target(artifact),
    }


def cpu_lane_bundle_identity(
    candidates: Sequence[CompiledFirstDerivativeCpuLane],
) -> str:
    normalized = tuple(candidates)
    if not normalized:
        raise ValueError("CPU lane bundle requires at least one candidate")
    integral = normalized[0].integral
    components = normalized[0].component_indices
    if any(
        candidate.integral != integral or candidate.component_indices != components
        for candidate in normalized
    ):
        raise ValueError("CPU lane bundle candidates must share integral/components")
    return canonical_hash(
        {
            "schema": "vibeqc.cpu-lane-bundle.identity.v1",
            "integral": integral_to_payload(integral),
            "components": components,
            "candidates": tuple(
                _candidate_identity_record(candidate) for candidate in normalized
            ),
        }
    )


@dataclass(frozen=True)
class CompiledFirstDerivativeCpuBundle:
    """Portable folder containing baseline and optional ISA-specific binaries."""

    directory: Path
    candidates: tuple[CompiledFirstDerivativeCpuLane, ...]
    program_identity: str

    def validate(self) -> None:
        if not self.candidates:
            raise ValueError("CPU derivative bundle has no candidates")
        names = tuple(candidate.target.name for candidate in self.candidates)
        if len(set(names)) != len(names):
            raise ValueError("CPU derivative bundle has duplicate targets")
        if "generic" not in names:
            raise ValueError("CPU derivative bundle requires a generic fallback")
        if self.program_identity != cpu_lane_bundle_identity(self.candidates):
            raise ValueError("CPU derivative bundle identity mismatch")
        directory = self.directory.resolve()
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            raise ValueError("CPU derivative bundle manifest is missing")
        for candidate in self.candidates:
            candidate.validate()
            try:
                candidate.native.library.resolve().relative_to(directory)
            except ValueError as exc:
                raise ValueError(
                    "CPU bundle library escaped its bundle directory"
                ) from exc

    @property
    def integral(self):
        return self.candidates[0].integral

    @property
    def component_indices(self) -> tuple[int, ...]:
        return self.candidates[0].component_indices

    @property
    def targets(self) -> tuple[CpuTargetInfo, ...]:
        return tuple(candidate.target for candidate in self.candidates)


def _manifest_payload(
    identity: str,
    candidates: Sequence[CompiledFirstDerivativeCpuLane],
) -> dict[str, object]:
    first = candidates[0]
    rows = []
    for candidate in candidates:
        rows.append(
            {
                **_candidate_identity_record(candidate),
                "library": f"libraries/{candidate.target.name}.so",
                "native_metadata": candidate.native.metadata,
            }
        )
    return {
        "schema": "vibeqc.cpu-lane-bundle.v1",
        "bundle_identity": identity,
        "integral": integral_to_payload(first.integral),
        "components": first.component_indices,
        "candidates": rows,
    }


def _materialize_bundle(
    cache: Path,
    candidates: Sequence[CompiledFirstDerivativeCpuLane],
) -> CompiledFirstDerivativeCpuBundle:
    candidates = tuple(candidates)
    identity = cpu_lane_bundle_identity(candidates)
    root = cache.resolve() / "cpu-bundles"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / identity
    if not destination.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".cpu-bundle-", dir=root))
        try:
            libraries = temporary / "libraries"
            libraries.mkdir()
            for candidate in candidates:
                shutil.copy2(
                    candidate.native.library,
                    libraries / f"{candidate.target.name}.so",
                )
            atomic_json(
                temporary / "manifest.json",
                _manifest_payload(identity, candidates),
            )
            try:
                os.rename(temporary, destination)
            except OSError:
                if not destination.is_dir():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return load_first_derivative_cpu_bundle(destination)


def load_first_derivative_cpu_bundle(
    directory: str | Path,
) -> CompiledFirstDerivativeCpuBundle:
    """Load a persisted bundle without loading any candidate shared library."""

    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid CPU derivative bundle manifest") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "vibeqc.cpu-lane-bundle.v1"
    ):
        raise ValueError("unsupported CPU derivative bundle schema")
    integral_payload = payload.get("integral")
    component_payload = payload.get("components")
    candidate_payload = payload.get("candidates")
    if not isinstance(integral_payload, dict):
        raise TypeError("CPU bundle integral payload is invalid")
    if not isinstance(component_payload, list):
        raise TypeError("CPU bundle component list is invalid")
    if not isinstance(candidate_payload, list) or not candidate_payload:
        raise ValueError("CPU bundle candidate list is invalid")
    integral = integral_from_payload(integral_payload)
    if any(type(value) is not int for value in component_payload):
        raise TypeError("CPU bundle components must be integers")
    components = tuple(component_payload)
    candidates: list[CompiledFirstDerivativeCpuLane] = []
    for record in candidate_payload:
        if not isinstance(record, dict):
            raise TypeError("CPU bundle candidate entry is invalid")
        target_payload = record.get("target")
        schedule_payload = record.get("schedule")
        metadata = record.get("native_metadata")
        relative = record.get("library")
        if (
            not isinstance(target_payload, dict)
            or not isinstance(schedule_payload, dict)
            or not isinstance(metadata, dict)
            or not isinstance(relative, str)
        ):
            raise TypeError("CPU bundle candidate metadata is invalid")
        target = _target_from_payload(target_payload)
        schedule = _schedule_from_payload(schedule_payload)
        expected_relative = Path("libraries") / f"{target.name}.so"
        if Path(relative) != expected_relative:
            raise ValueError("CPU bundle library path does not match its target")
        library = (directory / relative).resolve()
        try:
            library.relative_to(directory)
        except ValueError as exc:
            raise ValueError("CPU bundle library path escapes its directory") from exc
        if not library.is_file():
            raise ValueError("CPU bundle candidate library is missing")
        if metadata.get("binary_sha256") != file_hash(library):
            raise ValueError("CPU bundle candidate binary hash mismatch")
        candidate = CompiledFirstDerivativeCpuLane(
            native=CudaArtifact(library, metadata),
            integral=integral,
            component_indices=components,
            target=target,
            schedule=schedule,
            program_identity=str(record.get("program_identity", "")),
        )
        if record.get("artifact_key") != metadata.get("key"):
            raise ValueError("CPU bundle candidate artifact key mismatch")
        if record.get("binary_target") != _candidate_binary_target(candidate):
            raise ValueError("CPU bundle candidate binary target mismatch")
        candidate.validate()
        candidates.append(candidate)
    bundle = CompiledFirstDerivativeCpuBundle(
        directory=directory,
        candidates=tuple(candidates),
        program_identity=str(payload.get("bundle_identity", "")),
    )
    bundle.validate()
    return bundle


def _default_bundle_targets(compiler: CppCompilerAdapter) -> tuple[CpuTargetInfo, ...]:
    architecture = normalize_cpu_architecture(
        compiler.target.architecture.split("-", 1)[0]
    )
    if architecture == "x86_64":
        return (GENERIC_CPU_TARGET, AVX2_FMA_TARGET, AVX512F_FMA_TARGET)
    return (GENERIC_CPU_TARGET,)


def compile_first_derivative_cpu_bundle(
    integral,
    compiler,
    cache,
    *,
    component_indices,
    targets: Sequence[CpuTargetInfo] | None = None,
    schedules: Mapping[str, CpuScheduleIR] | None = None,
) -> CompiledFirstDerivativeCpuBundle:
    """Compile and materialize all requested CPU ISA candidates."""

    if not isinstance(compiler, CppCompilerAdapter):
        raise TypeError("CPU bundle compilation requires an explicit C++ compiler")
    selected_targets = tuple(targets or _default_bundle_targets(compiler))
    if not selected_targets:
        raise ValueError("CPU bundle target set must not be empty")
    if len({target.name for target in selected_targets}) != len(selected_targets):
        raise ValueError("CPU bundle targets must be unique")
    schedule_map = dict(schedules or {})
    candidates = []
    cache = Path(cache).resolve()
    for target in selected_targets:
        schedule = schedule_map.get(target.name, default_cpu_schedule(target))
        schedule.validate_for(target)
        candidates.append(
            compile_first_derivative_cpu_lane(
                integral,
                compiler,
                cache / "candidate-cache" / target.name,
                component_indices=component_indices,
                target=target,
                schedule=schedule,
            )
        )
    return _materialize_bundle(cache, candidates)


class FirstDerivativeCpuDispatchEvaluator:
    """Select one compatible bundle binary before loading it into the process."""

    def __init__(
        self,
        bundle: CompiledFirstDerivativeCpuBundle,
        *,
        runtime: CpuRuntimeFeatures | None = None,
        forced_target: str | None = None,
        record_capacity: int = 128,
        budget_bytes: int = 1 << 20,
    ):
        if not isinstance(bundle, CompiledFirstDerivativeCpuBundle):
            raise TypeError("CPU dispatch evaluator requires a compiled bundle")
        bundle.validate()
        runtime = runtime or detect_cpu_features()
        compatible = tuple(
            candidate
            for candidate in bundle.candidates
            if cpu_binary_target_supported(_candidate_binary_target(candidate), runtime)
        )
        if not compatible:
            raise ValueError(
                "CPU bundle has no compiled binary compatible with this runtime architecture/ABI"
            )
        decision = select_cpu_target(
            (candidate.target for candidate in compatible),
            runtime,
            forced_target=forced_target,
        )
        selected = next(
            candidate
            for candidate in compatible
            if candidate.target.name == decision.selected_target
        )
        self.bundle = bundle
        self.decision: CpuDispatchDecision = decision
        self.selected = FirstDerivativeCpuLaneEvaluator(
            selected,
            record_capacity=record_capacity,
            budget_bytes=budget_bytes,
        )
        self.numeric_bytes = self.selected.numeric_bytes

    @property
    def selected_target(self) -> str:
        return self.decision.selected_target

    def diagnostics(self) -> dict[str, object]:
        return {
            **self.decision.to_payload(),
            "bundle_identity": self.bundle.program_identity,
            "candidate_artifact_keys": {
                candidate.target.name: candidate.native.metadata["key"]
                for candidate in self.bundle.candidates
            },
        }

    def contract(self, primitives, centers):
        return self.selected.contract(primitives, centers)
