"""Reject incorrect hardware evidence without loading CUDA in ordinary CI."""

import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import benchmark_density_sources as benchmark
from tools.vibeqc_validation.hardware import CUDA_BENCHMARK_PROFILES


def _runtime(
    pci_status: typing.Any = 0,
    bus: typing.Any = b"0000:03:00.0",
    runtime_version: typing.Any = 12090,
) -> typing.Any:
    def pci_bus(
        buffer: typing.Any, length: typing.Any, ordinal: typing.Any
    ) -> typing.Any:
        assert ordinal == 0 and length >= 16
        buffer.value = bus
        return pci_status

    def version(output: typing.Any) -> typing.Any:
        output._obj.value = runtime_version
        return 0

    return SimpleNamespace(
        cudaDeviceGetPCIBusId=pci_bus,
        cudaRuntimeGetVersion=version,
    )


@pytest.mark.parametrize(
    "row,profile,accepted",
    [
        (
            "NVIDIA GeForce RTX 5090, GPU-assigned, 580.95.05, 32607 MiB, 12.0",
            "sm120",
            True,
        ),
        (
            "Different SM120 GPU, GPU-assigned, 580.95.05, 32607 MiB, 12.0",
            "sm120",
            True,
        ),
        (
            "Different SM120 GPU, GPU-assigned, 580.95.05, 32607 MiB, 11.0",
            "sm120",
            False,
        ),
        (
            "NVIDIA GeForce RTX 5090, GPU-assigned, 580.95.05, 32607 MiB, 12.0",
            "rtx5090-reproduction",
            True,
        ),
        (
            "Different SM120 GPU, GPU-assigned, 580.95.05, 32607 MiB, 12.0",
            "rtx5090-reproduction",
            False,
        ),
        ("NVIDIA GeForce RTX 5090", "sm120", False),
        (
            (
                "Different GPU, GPU-assigned, 580, 16 MiB, 12.0\n"
                "NVIDIA GeForce RTX 5090, GPU-other, 580, 32 MiB, 12.0"
            ),
            "sm120",
            False,
        ),
    ],
)
def test_probe_binds_assigned_device_then_qualifies_requested_profile(
    monkeypatch: typing.Any, row: typing.Any, profile: typing.Any, accepted: typing.Any
) -> None:
    monkeypatch.setattr(benchmark.ctypes, "CDLL", lambda _: _runtime())

    def query(arguments: typing.Any) -> typing.Any:
        assert "--id=0000:03:00.0" in arguments
        assert "compute_cap" in arguments[2]
        return row

    monkeypatch.setattr(benchmark, "capture", query)
    selected = CUDA_BENCHMARK_PROFILES[profile]
    if accepted:
        result = benchmark.probe_gpu(Path("/toolkit/bin/nvcc"), selected)
        assert result["gpu"] == row
        assert result["pci_bus_id"] == "0000:03:00.0"
        assert result["visible_device_ordinal"] == 0
        assert result["qualification"]["profile"] == profile
        assert result["qualification"]["compute_capability"] == [12, 0]
        assert result["cuda_runtime_version"] == 12090
    else:
        with pytest.raises(RuntimeError):
            benchmark.probe_gpu(Path("/toolkit/bin/nvcc"), selected)


@pytest.mark.parametrize("status,bus", [(100, b""), (0, b"")])
def test_failed_runtime_probe_cannot_publish_hardware_success(
    monkeypatch: typing.Any, status: typing.Any, bus: typing.Any
) -> None:
    monkeypatch.setattr(
        benchmark.ctypes,
        "CDLL",
        lambda _: _runtime(pci_status=status, bus=bus),
    )

    def forbidden(_: typing.Any) -> typing.Any:
        raise AssertionError(
            "NVML must not substitute another GPU after a CUDA probe failure"
        )

    monkeypatch.setattr(benchmark, "capture", forbidden)
    with pytest.raises(RuntimeError, match="cannot identify the assigned CUDA device"):
        benchmark.probe_gpu(Path("/toolkit/bin/nvcc"), CUDA_BENCHMARK_PROFILES["sm120"])
