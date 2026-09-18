"""Reusable CUDA hardware profiles for benchmark eligibility.

Benchmark evidence should retain exact device identity, but eligibility should be
expressed in terms of the capability/profile the executable actually requires.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaBenchmarkProfile:
    """Hardware contract for a CUDA benchmark or reproduction run."""

    name: str
    target_architecture: str
    compute_capability: tuple[int, int]
    exact_device_name: str | None = None


CUDA_BENCHMARK_PROFILES = {
    "sm120": CudaBenchmarkProfile(
        name="sm120",
        target_architecture="sm_120",
        compute_capability=(12, 0),
    ),
    "rtx5090-reproduction": CudaBenchmarkProfile(
        name="rtx5090-reproduction",
        target_architecture="sm_120",
        compute_capability=(12, 0),
        exact_device_name="NVIDIA GeForce RTX 5090",
    ),
}


def qualify_cuda_device(
    device: dict[str, object], profile: CudaBenchmarkProfile
) -> dict[str, object]:
    """Validate only the hardware properties declared by *profile*."""

    raw_capability = device.get("compute_capability")
    if not isinstance(raw_capability, (tuple, list)) or len(raw_capability) != 2:
        raise RuntimeError("CUDA benchmark device is missing compute capability")
    capability = tuple(int(value) for value in raw_capability)
    if capability != profile.compute_capability:
        expected = ".".join(map(str, profile.compute_capability))
        actual = ".".join(map(str, capability))
        raise RuntimeError(
            f"hardware profile {profile.name} requires CUDA compute capability "
            f"{expected}; assigned device reports {actual}"
        )

    name = str(device.get("name", ""))
    if profile.exact_device_name is not None and name != profile.exact_device_name:
        raise RuntimeError(
            f"hardware profile {profile.name} requires device name "
            f"{profile.exact_device_name!r}; assigned device reports {name!r}"
        )

    return {
        "profile": profile.name,
        "target_architecture": profile.target_architecture,
        "compute_capability": list(capability),
        "exact_device_name_required": profile.exact_device_name is not None,
    }
