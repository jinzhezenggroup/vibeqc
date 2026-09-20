"""Private native accounting for accepted prepared resource requests."""

import ctypes
import typing

from .resources import ResourceAllocationError, _account


def observe_method_call(
    library: typing.Any,
    plan: typing.Any,
    ledger: typing.Any,
    callback: typing.Any,
    *,
    owner: typing.Any,
    phase: typing.Any = "observation",
    previous: typing.Any = None,
) -> typing.Any:
    """Bind the same prepared owner around setup and each synchronous replay.

    Preparation can allocate persistent scientific buffers. Its evidence must
    survive subsequent execution scopes and failures, while each replay resets
    the ledger peak to the buffers still owned by the prepared calculation.
    """
    from .resources import CpuResourceObservation

    observed = CpuResourceObservation(library, cpu_workers=1, ledger=ledger)
    diagnostics = dict(previous or {})
    diagnostics.update(plan=plan.to_dict(), owner=owner, phase=phase)

    def evidence() -> typing.Any:
        record = observed.to_dict()
        if owner == "ks":
            record["cuda_scope"] = (
                "common direct-J provider arena samples; complete explicit KS device capacities are in device_ledger"
            )
            record["scope"] = (
                "explicit KS grid/basis/provider/SCF capacities with retained fleet and warm buffers"
            )
            record["excludes"] = [
                "unsampled setup/XC/recurrence/eigensolver temporaries",
                "object metadata and runtime overhead",
            ]
        return record

    try:
        with observed:
            status = callback()
        diagnostics[phase] = evidence()
        observed.verify(plan)
    except Exception as error:
        diagnostics[phase] = evidence()
        error.resource_diagnostics = diagnostics
        raise
    return status, diagnostics


def check_resource_status(
    library: typing.Any, status: typing.Any, diagnostics: typing.Any
) -> None:
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
            observation = diagnostics.get(diagnostics.get("phase", "observation"), {})
            ledger = observation.get("device_ledger")
            backend = next(
                r["identity"]["backend"]
                for r in diagnostics["plan"]["requests"]
                if r["name"] == diagnostics.get("owner", "hf")
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

    def __init__(
        self, library: typing.Any, plan: typing.Any, *, owner: typing.Any = "hf"
    ) -> None:
        plan.require_feasible()
        self.owner = owner
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
                "one native prepared owner must use one visible CUDA device"
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

    def to_dict(self) -> typing.Any:
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
            "owner": self.owner,
            "scope": "owned CUDA buffer capacities; excludes driver/graph/pool and library-internal allocations",
        }

    def close(self) -> None:
        """Release the observation handle; any live native buffers keep charges."""
        if self.handle:
            self.library.vibeqc_resource_ledger_destroy_v1(self.handle)
            self.handle = None

    def __del__(self) -> None:
        if getattr(self, "handle", None):
            self.close()
