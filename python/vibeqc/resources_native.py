"""Private native accounting for accepted prepared resource requests."""

import ctypes

from .resources import ResourceAllocationError, _account


def check_resource_status(library, status, diagnostics):
    """Keep resource evidence on failed native calls without guessing OOM space.

    CPU allocation failures are host failures. For CUDA, only an actual ledger
    rejection identifies a device failure; opaque library/host allocation
    errors retain unknown placement and cannot trigger a space-specific retry.
    """
    from . import _native

    try:
        _native.check(library, status)
    except RuntimeError as error:
        failure = error
        if status == _native.STATUS_OUT_OF_MEMORY:
            observation = diagnostics.get("observation", {})
            ledger = observation.get("device_ledger")
            backend = next(
                r["identity"]["backend"]
                for r in diagnostics["plan"]["requests"]
                if r["name"] == "hf"
            )
            space = (
                "host"
                if backend == "cpu"
                else (
                    f"device:{ledger['device']}"
                    if ledger and ledger["rejected_allocations"]
                    else None
                )
            )
            failure = (
                ResourceAllocationError(space, str(error))
                if space
                else MemoryError(str(error))
            )
        failure.resource_diagnostics = diagnostics
        if failure is error:
            raise
        raise failure from error


class NativeDeviceLedger:
    """Persist charges across warm calls and release them with native buffers.

    Driver, graph, pool and library-internal allocations are outside this
    numeric-buffer ledger; the common plan reports their allowances/exclusions.
    Each ledger belongs to one prepared request on one visible CUDA device.
    """

    def __init__(self, library, plan, *, owner="hf"):
        plan.require_feasible()
        request = next(r for r in plan.requests if r.name == owner)
        if request.identity.backend != "cuda":
            raise ValueError("native device ledger requires a CUDA request")
        selected = dict(plan.selections)[owner]
        candidate = next(c for c in request.candidates if c.name == selected)
        devices = {
            e.space for e in candidate.estimates if e.space.startswith("device:")
        }
        if len(devices) != 1:
            raise NotImplementedError(
                "one native HF owner must use one visible CUDA device"
            )
        self.device = int(devices.pop().split(":")[1])
        numeric = tuple(
            e for e in candidate.estimates if e.accounting != "runtime_allowance"
        )
        self.limit = _account(numeric)[0][f"device:{self.device}"]
        self.library = library
        self.handle = None
        for name in ("create", "destroy", "bind", "read"):
            if not hasattr(library, f"vibeqc_resource_ledger_{name}_v1"):
                raise NotImplementedError(
                    "native library has no persistent allocation ledger v1"
                )
        library.vibeqc_resource_ledger_create_v1.argtypes = [
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        library.vibeqc_resource_ledger_create_v1.restype = ctypes.c_void_p
        library.vibeqc_resource_ledger_destroy_v1.argtypes = [ctypes.c_void_p]
        library.vibeqc_resource_ledger_destroy_v1.restype = None
        library.vibeqc_resource_ledger_bind_v1.argtypes = [ctypes.c_void_p]
        library.vibeqc_resource_ledger_bind_v1.restype = ctypes.c_int
        library.vibeqc_resource_ledger_read_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        library.vibeqc_resource_ledger_read_v1.restype = ctypes.c_int
        self.handle = library.vibeqc_resource_ledger_create_v1(self.limit, self.device)
        if not self.handle:
            raise MemoryError("could not allocate native resource ledger metadata")

    def to_dict(self):
        """Read owned capacities without inferring physical GPU free memory."""
        if not self.handle:
            raise RuntimeError("native resource ledger is closed")
        values = (ctypes.c_uint64 * 4)()
        if self.library.vibeqc_resource_ledger_read_v1(self.handle, values):
            raise RuntimeError("native resource ledger is unavailable")
        return {
            "device": self.device,
            "limit_bytes": self.limit,
            "live_bytes": values[0],
            "peak_bytes": values[1],
            "allocations": values[2],
            "rejected_allocations": values[3],
            "scope": "owned HF CUDA buffer capacities; excludes driver/graph/pool and library-internal allocations",
        }

    def close(self):
        """Release the observation handle; any live native buffers keep charges."""
        if self.handle:
            self.library.vibeqc_resource_ledger_destroy_v1(self.handle)
            self.handle = None

    def __del__(self):
        if getattr(self, "handle", None):
            self.close()
