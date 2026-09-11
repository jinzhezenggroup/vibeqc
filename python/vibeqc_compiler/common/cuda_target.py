"""CUDA device capabilities used by scheduling, tuning, and manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .backend import TargetInfo


@dataclass(frozen=True, slots=True)
class CudaArchitecture:
    """Keep profile identity separate from NVCC code-generation intent.

    CMake accepts architecture values such as ``120-real`` and
    ``120-virtual``.  The production manifest, runtime registry, and cache
    keys intentionally use the numeric ``sm_120`` identity, while NVCC must
    retain the suffix because it controls whether a real cubin or PTX image is
    emitted.  Passing a single normalized string through both layers used to
    silently discard that distinction.
    """

    profile: str
    compile: str

    @property
    def profile_architecture(self) -> str:
        """Descriptive alias for callers that serialize build metadata."""

        return self.profile

    @property
    def compile_architecture(self) -> str:
        """Descriptive alias for the value passed to NVCC/CMake."""

        return self.compile


def cuda_architecture(architecture: str | int) -> CudaArchitecture:
    """Parse one CUDA architecture into profile and compile representations."""

    value = str(architecture).strip().lower()
    if value.startswith("compute_"):
        value = value.removeprefix("compute_")
    elif value.startswith("sm_"):
        value = value.removeprefix("sm_")
    suffix = ""
    if value.endswith("-real"):
        value, suffix = value[:-5], "-real"
    elif value.endswith("-virtual"):
        value, suffix = value[:-8], "-virtual"
    if not value.isdigit() or len(value) < 2:
        raise ValueError(f"invalid CUDA architecture {architecture!r}")
    numeric = str(int(value))
    return CudaArchitecture(profile=f"sm_{numeric}", compile=f"{numeric}{suffix}")


def normalize_cuda_architecture(architecture: str | int) -> str:
    """Return a canonical ``sm_XX`` architecture name."""

    return cuda_architecture(architecture).profile


def normalize_cuda_compile_architecture(architecture: str | int) -> str:
    """Return NVCC's numeric architecture while preserving ``real``/``virtual``."""

    return cuda_architecture(architecture).compile


def compute_capability_from_architecture(architecture: str | int) -> tuple[int, int]:
    """Decode a canonical CUDA architecture into major/minor components."""

    digits = normalize_cuda_architecture(architecture).removeprefix("sm_")
    if len(digits) == 2:
        return int(digits[0]), int(digits[1])
    return int(digits[:-1]), int(digits[-1])


@dataclass(frozen=True, slots=True)
class CudaTargetInfo:
    """CUDA limits and toolchain identity relevant to generated kernels.

    Static catalog entries intentionally leave ``sm_count`` and runtime
    version strings unset.  The autotuner fills those fields from the device
    it was actually allocated before it benchmarks a schedule.
    """

    architecture: str
    compute_capability_major: int
    compute_capability_minor: int
    warp_size: int
    maximum_threads_per_block: int
    maximum_threads_per_sm: int
    maximum_blocks_per_sm: int
    registers_per_sm: int
    maximum_registers_per_thread: int
    shared_memory_per_block: int
    shared_memory_per_block_optin: int
    shared_memory_per_sm: int
    sm_count: int = 0
    required_cuda_features: tuple[str, ...] = ("fp64", "fp64_atomic_add")
    minimum_cuda_toolkit: str = "12.0"
    cuda_toolkit_version: str = ""
    nvcc_version: str = ""
    ptxas_version: str = ""
    driver_version: str = ""
    generator_abi: int = 1
    tuning_maximum_registers: int = 192
    tuning_maximum_packed_registers: int = 224
    tuning_maximum_stack_bytes: int = 128
    tuning_maximum_shared_bytes: int = 49152
    packed_register_caps: tuple[int, ...] = (192, 208)

    def __post_init__(self) -> None:
        canonical = normalize_cuda_architecture(self.architecture)
        if canonical != self.architecture:
            raise ValueError("CUDA architecture must use canonical sm_XX notation")
        if compute_capability_from_architecture(canonical) != (
            self.compute_capability_major,
            self.compute_capability_minor,
        ):
            raise ValueError("CUDA architecture and compute capability disagree")
        if self.warp_size < 1:
            raise ValueError("CUDA warp size must be positive")
        if self.maximum_threads_per_block < self.warp_size:
            raise ValueError("CUDA block limit must contain one warp")
        if self.maximum_threads_per_sm < self.maximum_threads_per_block:
            raise ValueError("CUDA SM thread limit cannot be below block limit")
        if self.maximum_blocks_per_sm < 1:
            raise ValueError("CUDA resident block limit must be positive")
        if self.maximum_registers_per_thread < 1 or self.registers_per_sm < 1:
            raise ValueError("CUDA register limits must be positive")
        if self.shared_memory_per_block < 1 or self.shared_memory_per_sm < 1:
            raise ValueError("CUDA shared-memory limits must be positive")
        if self.shared_memory_per_block_optin < self.shared_memory_per_block:
            raise ValueError("opt-in shared memory cannot be below the base limit")
        if self.generator_abi < 1:
            raise ValueError("generator ABI must be positive")

    @property
    def target_info(self) -> TargetInfo:
        """Return the backend-neutral schedule-validation view."""

        return TargetInfo(
            backend="cuda",
            architecture=self.architecture,
            subgroup_size=self.warp_size,
            maximum_workgroup_threads=self.maximum_threads_per_block,
            maximum_resident_workgroups=self.maximum_blocks_per_sm,
        )

    @property
    def compute_capability(self) -> tuple[int, int]:
        """Return the runtime major/minor pair."""

        return self.compute_capability_major, self.compute_capability_minor

    def with_runtime_probe(self, **values: object) -> CudaTargetInfo:
        """Return a catalog target enriched with measured device/tool data."""

        return replace(self, **values)

    def to_payload(self) -> dict[str, object]:
        """Serialize all resource and provenance fields for tuning artifacts."""

        payload = asdict(self)
        payload["compute_capability"] = (
            f"{self.compute_capability_major}.{self.compute_capability_minor}"
        )
        return payload


def _target(
    architecture: str,
    *,
    maximum_threads_per_sm: int,
    maximum_blocks_per_sm: int,
    shared_memory_per_block_optin: int,
    shared_memory_per_sm: int,
    minimum_cuda_toolkit: str,
    sm_count: int = 0,
) -> CudaTargetInfo:
    major, minor = compute_capability_from_architecture(architecture)
    return CudaTargetInfo(
        architecture=architecture,
        compute_capability_major=major,
        compute_capability_minor=minor,
        warp_size=32,
        maximum_threads_per_block=1024,
        maximum_threads_per_sm=maximum_threads_per_sm,
        maximum_blocks_per_sm=maximum_blocks_per_sm,
        registers_per_sm=65536,
        maximum_registers_per_thread=255,
        shared_memory_per_block=49152,
        shared_memory_per_block_optin=shared_memory_per_block_optin,
        shared_memory_per_sm=shared_memory_per_sm,
        sm_count=sm_count,
        minimum_cuda_toolkit=minimum_cuda_toolkit,
    )


CUDA_TARGETS: dict[str, CudaTargetInfo] = {
    "sm_80": _target(
        "sm_80",
        maximum_threads_per_sm=2048,
        maximum_blocks_per_sm=32,
        shared_memory_per_block_optin=163840,
        shared_memory_per_sm=163840,
        minimum_cuda_toolkit="11.0",
    ),
    "sm_86": _target(
        "sm_86",
        maximum_threads_per_sm=1536,
        maximum_blocks_per_sm=16,
        shared_memory_per_block_optin=101376,
        shared_memory_per_sm=102400,
        minimum_cuda_toolkit="11.1",
    ),
    "sm_89": _target(
        "sm_89",
        maximum_threads_per_sm=1536,
        maximum_blocks_per_sm=24,
        shared_memory_per_block_optin=101376,
        shared_memory_per_sm=102400,
        minimum_cuda_toolkit="11.8",
    ),
    "sm_90": _target(
        "sm_90",
        maximum_threads_per_sm=2048,
        maximum_blocks_per_sm=32,
        shared_memory_per_block_optin=232448,
        shared_memory_per_sm=233472,
        minimum_cuda_toolkit="12.0",
    ),
    "sm_120": _target(
        "sm_120",
        maximum_threads_per_sm=1536,
        maximum_blocks_per_sm=24,
        shared_memory_per_block_optin=101376,
        shared_memory_per_sm=102400,
        minimum_cuda_toolkit="12.8",
        sm_count=170,
    ),
}


def cuda_target_info(architecture: str | int) -> CudaTargetInfo:
    """Return a measured catalog target or a conservative portable target."""

    canonical = normalize_cuda_architecture(architecture)
    known = CUDA_TARGETS.get(canonical)
    if known is not None:
        return known
    major, minor = compute_capability_from_architecture(canonical)
    return CudaTargetInfo(
        architecture=canonical,
        compute_capability_major=major,
        compute_capability_minor=minor,
        warp_size=32,
        maximum_threads_per_block=1024,
        maximum_threads_per_sm=1024,
        maximum_blocks_per_sm=16,
        registers_per_sm=65536,
        maximum_registers_per_thread=255,
        shared_memory_per_block=49152,
        shared_memory_per_block_optin=49152,
        shared_memory_per_sm=65536,
        minimum_cuda_toolkit="12.0",
        tuning_maximum_registers=160,
        tuning_maximum_packed_registers=192,
        tuning_maximum_stack_bytes=64,
        packed_register_caps=(160, 176),
    )


DEFAULT_CUDA_TARGET = CUDA_TARGETS["sm_120"]
