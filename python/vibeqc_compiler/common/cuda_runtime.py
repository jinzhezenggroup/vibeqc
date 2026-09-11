"""Native artifact/metrics ABI and allocation-snapshot synchronization.

The metrics layout matches src/tensor/cuda_runtime.cuh and its DFT consumers.
All providers share one lock so creation/destruction cannot corrupt another
provider's allocation delta. Execution on independent streams remains parallel.
"""

import ctypes
import threading
from dataclasses import dataclass
from pathlib import Path

_PREPARATION_LOCK = threading.RLock()


class _Metrics(ctypes.Structure):
    _fields_ = [
        ("owned_device_bytes", ctypes.c_uint64),
        ("provider_retained_bytes", ctypes.c_uint64),
        ("prepare_device_delta", ctypes.c_uint64),
        ("observed_device_delta", ctypes.c_uint64),
        *[
            (name, ctypes.c_double)
            for name in (
                "device_ms",
                "input_ms",
                "output_ms",
                "packing_ms",
                "library_ms",
                "kernel_ms",
            )
        ],
    ]


@dataclass(frozen=True)
class CudaArtifact:
    """Content-verified native artifact; compilation alone is not validation."""

    library: Path
    metadata: dict
