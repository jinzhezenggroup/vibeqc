"""Reject incorrect hardware evidence without loading CUDA in ordinary CI."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import benchmark_density_sources as benchmark


@pytest.mark.parametrize(
    "row,accepted",
    [
        ("NVIDIA GeForce RTX 5090, GPU-assigned, 580.95.05, 32607 MiB", True),
        ("NVIDIA GeForce RTX 5080, GPU-assigned, 580.95.05, 16303 MiB", False),
        ("NVIDIA GeForce RTX 5090", False),
        (
            "NVIDIA GeForce RTX 5080, GPU-assigned, 580, 16 MiB\nNVIDIA GeForce RTX 5090, GPU-other, 580, 32 MiB",
            False,
        ),
    ],
)
def test_probe_binds_the_assigned_cuda_device_before_validating_its_model(
    monkeypatch, row, accepted
):
    def pci_bus(buffer, length, ordinal):
        assert ordinal == 0 and length >= 16
        buffer.value = b"0000:03:00.0"
        return 0

    monkeypatch.setattr(
        benchmark.ctypes,
        "CDLL",
        lambda _: SimpleNamespace(cudaDeviceGetPCIBusId=pci_bus),
    )

    def query(arguments):
        assert "--id=0000:03:00.0" in arguments
        return row

    monkeypatch.setattr(benchmark, "capture", query)
    if accepted:
        result = benchmark.probe_gpu(Path("/toolkit/bin/nvcc"))
        assert result == {
            "gpu": row,
            "pci_bus_id": "0000:03:00.0",
            "visible_device_ordinal": 0,
        }
    else:
        with pytest.raises(RuntimeError, match="assigned device.*RTX 5090"):
            benchmark.probe_gpu(Path("/toolkit/bin/nvcc"))


@pytest.mark.parametrize("status,bus", [(100, b""), (0, b"")])
def test_failed_runtime_probe_cannot_publish_hardware_success(monkeypatch, status, bus):
    def pci_bus(buffer, *_):
        buffer.value = bus
        return status

    monkeypatch.setattr(
        benchmark.ctypes,
        "CDLL",
        lambda _: SimpleNamespace(cudaDeviceGetPCIBusId=pci_bus),
    )

    def forbidden(_):
        raise AssertionError(
            "NVML must not substitute another GPU after a CUDA probe failure"
        )

    monkeypatch.setattr(benchmark, "capture", forbidden)
    with pytest.raises(RuntimeError, match="cannot identify the assigned CUDA device"):
        benchmark.probe_gpu(Path("/toolkit/bin/nvcc"))
