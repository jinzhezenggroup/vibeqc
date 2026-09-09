"""Portable resource requests and deterministic composition of buffer lifetimes.

Planning uses integers and metadata only. It does not initialize a backend,
allocate scientific tensors, query free memory, or modify solver state. A
phase is a logical time slot: providers that can overlap must use overlapping
phase intervals, even when their individual streams have different names.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass, field

from .profiles import canonical_hash

MAX_BYTES = 2**63 - 1
KINDS = frozenset(("persistent", "workspace", "cache", "library", "runtime", "output"))
RETAINED = KINDS - {"workspace"}


def checked_bytes(value: int, name: str = "bytes") -> int:
    """Reject bools, negative sizes and products outside a portable int64 ABI."""
    if type(value) is not int or not 0 <= value <= MAX_BYTES:
        raise ValueError(f"{name} must be an integer in [0, {MAX_BYTES}]")
    return value


def byte_product(*values: int) -> int:
    """Multiply dimensions without truncation, including on a 32-bit consumer."""
    for value in values:
        checked_bytes(value, "dimension")
    return checked_bytes(math.prod(values), "byte product")


def _sum(values) -> int:
    return checked_bytes(sum(values), "concurrently live bytes")


@dataclass(frozen=True)
class ResourceBudget:
    """Limits for the declared allocation scope, with explicit runtime reserve.

    ``host_bytes`` bounds pageable plus pinned host memory. The optional pinned
    cap applies in addition to that total. ``device_bytes`` is a total across
    devices; per-device caps apply in addition, so neither can hide the other.
    ``None`` means no user limit, whereas zero permits no allocation. Fixed
    reserves and fractional headroom are withheld before provider selection.
    Driver/context allocations outside an estimate's scope are never implied
    to be measured or exactly bounded by these numeric-buffer limits.
    """

    host_bytes: int | None = None
    device_bytes: int | None = None
    pinned_host_bytes: int | None = None
    per_device_bytes: tuple[tuple[int, int], ...] = ()
    host_reserve_bytes: int = 0
    device_reserve_bytes: int = 0
    headroom_fraction: float = 0.0
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported resource budget schema")
        for name in ("host_bytes", "device_bytes", "pinned_host_bytes"):
            value = getattr(self, name)
            if value is not None:
                checked_bytes(value, name)
        for name in ("host_reserve_bytes", "device_reserve_bytes"):
            checked_bytes(getattr(self, name), name)
        if (
            type(self.headroom_fraction) not in (int, float)
            or not math.isfinite(self.headroom_fraction)
            or not 0 <= self.headroom_fraction < 1
        ):
            raise ValueError("headroom_fraction must be finite in [0, 1)")
        devices = tuple(sorted(tuple(row) for row in self.per_device_bytes))
        if len({row[0] for row in devices}) != len(devices):
            raise ValueError("duplicate resource device")
        for device, limit in devices:
            checked_bytes(device, "device ordinal")
            checked_bytes(limit, "per-device budget")
        object.__setattr__(self, "per_device_bytes", devices)

    def limits(self) -> dict[str, int]:
        """Effective caps; reserve uses exact integer arithmetic for large sizes."""
        numerator, denominator = float(self.headroom_fraction).as_integer_ratio()

        def remaining(value, reserve=0):
            headroom = (value * numerator + denominator - 1) // denominator
            return max(0, value - headroom - reserve)

        limits = {}
        for name, value, reserve in (
            ("host", self.host_bytes, self.host_reserve_bytes),
            ("device", self.device_bytes, self.device_reserve_bytes),
            ("pinned", self.pinned_host_bytes, 0),
        ):
            if value is not None:
                limits[name] = remaining(value, reserve)
        limits.update({f"device:{d}": remaining(n) for d, n in self.per_device_bytes})
        return limits

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ResourceIdentity:
    """Scientific/topological identity required to reuse a provider's plan.

    Geometry-only updates can reuse topology planning when basis dimensions,
    primitives, atom ordering, spin, precision and schedule remain compatible.
    ``topology`` therefore excludes coordinates but includes all dimensions
    and shapes. Provider adapters must still validate scientific identity on
    execution; this record does not make a warm density reusable.
    """

    method: str
    provider: str
    backend: str
    precision: str
    topology: str
    observables: tuple[str, ...]
    schedule: str
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported resource identity schema")
        for name in ("method", "provider", "backend", "precision", "schedule"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"resource identity requires {name}")
        topology = json.loads(self.topology)
        if not isinstance(topology, dict) or not topology:
            raise ValueError("resource topology must be a nonempty JSON object")
        # Canonical JSON detaches caller dictionaries without retaining mutable
        # object graphs in an otherwise frozen identity.
        object.__setattr__(
            self, "topology", json.dumps(topology, sort_keys=True, allow_nan=False)
        )
        observables = tuple(sorted(set(self.observables)))
        if not observables or any(not isinstance(x, str) or not x for x in observables):
            raise ValueError("resource identity requires requested observables")
        object.__setattr__(self, "observables", observables)

    @property
    def identity(self):
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class ResourceEstimate:
    """One owned allocation capacity over an inclusive phase interval.

    Shared buffers appear once, under their owner, rather than once per user.
    ``kind`` determines residency; cacheability/recomputation/stream traffic
    describe the same allocation and are not additional bytes. Output
    lifetimes must include the caller's retention interval. Unknown allocator
    overhead belongs in ``scope_exclusions`` on its request, not a false zero
    measured peak. ``runtime`` and ``library`` can hold conservative allowances.
    """

    name: str
    bytes: int
    space: str
    first_phase: int
    last_phase: int
    kind: str = "workspace"
    cacheable: bool = False
    recomputable: bool = False
    streamed_bytes: int = 0
    accounting: str = "capacity_bound"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("resource allocation needs an owner name")
        checked_bytes(self.bytes)
        checked_bytes(self.streamed_bytes, "streamed bytes")
        checked_bytes(self.first_phase, "first phase")
        checked_bytes(self.last_phase, "last phase")
        if self.last_phase < self.first_phase:
            raise ValueError("resource lifetime ends before its first use")
        if self.space not in ("pageable", "pinned"):
            if not self.space.startswith("device:"):
                raise ValueError("unknown resource memory space")
            ordinal = self.space.removeprefix("device:")
            if not ordinal.isdecimal() or str(int(ordinal)) != ordinal:
                raise ValueError("invalid device memory space")
            checked_bytes(int(ordinal), "device ordinal")
        if self.kind not in KINDS:
            raise ValueError("unknown resource allocation kind")
        if self.accounting not in ("capacity_bound", "runtime_allowance"):
            raise ValueError("unknown resource accounting scope")
        if type(self.cacheable) is not bool or type(self.recomputable) is not bool:
            raise ValueError("cache/recompute flags must be booleans")


@dataclass(frozen=True)
class ResourceCandidate:
    """A provider-supported alternative, ordered by declared relative cost."""

    name: str
    mode: str
    estimates: tuple[ResourceEstimate, ...]
    relative_cost: int = 0
    decisions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if not self.name or self.mode not in (
            "resident",
            "streamed",
            "recomputed",
            "tiled",
        ):
            raise ValueError("candidate requires a name and supported memory mode")
        checked_bytes(self.relative_cost, "relative cost")
        object.__setattr__(self, "estimates", tuple(self.estimates))
        object.__setattr__(self, "decisions", tuple(tuple(x) for x in self.decisions))
        if not self.estimates or not all(
            isinstance(e, ResourceEstimate) for e in self.estimates
        ):
            raise ValueError("candidate requires structured allocation estimates")
        if len({e.name for e in self.estimates}) != len(self.estimates):
            raise ValueError("duplicate allocation owner within a candidate")


@dataclass(frozen=True)
class ResourceRequest:
    """Provider alternatives with explicit phase, ownership and scope contracts."""

    name: str
    identity: ResourceIdentity
    candidates: tuple[ResourceCandidate, ...]
    scope_exclusions: tuple[str, ...] = ()
    unsupported_reason: str | None = None
    infeasible_reason: str | None = None

    def __post_init__(self):
        if not self.name or not isinstance(self.identity, ResourceIdentity):
            raise ValueError("request requires an owner and scientific identity")
        candidates = tuple(
            sorted(self.candidates, key=lambda c: (c.relative_cost, c.name))
        )
        if len({c.name for c in candidates}) != len(candidates):
            raise ValueError("duplicate provider candidate")
        if self.unsupported_reason and self.infeasible_reason:
            raise ValueError("request cannot be both unsupported and infeasible")
        if not candidates and not (self.unsupported_reason or self.infeasible_reason):
            raise ValueError(
                "request has neither a provider nor an unsupported diagnostic"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "scope_exclusions", tuple(self.scope_exclusions))


def _account(estimates, *, resident=False):
    """Sweep lifetime endpoints; complexity depends on resources, not phase IDs."""
    resources = [e for e in estimates if not resident or e.kind in RETAINED]
    phases = sorted(
        {point for e in resources for point in (e.first_phase, e.last_phase + 1)}
    )
    peaks = {"host": 0, "device": 0, "pinned": 0, "pageable": 0}
    phase_peaks = {}
    for phase in phases:
        live = [e for e in resources if e.first_phase <= phase <= e.last_phase]
        values = {
            space: _sum(e.bytes for e in live if e.space == space)
            for space in {e.space for e in resources}
        }
        values["host"] = _sum((values.get("pageable", 0), values.get("pinned", 0)))
        values["device"] = _sum(
            n for space, n in values.items() if space.startswith("device:")
        )
        for space, value in values.items():
            if value > peaks.get(space, 0):
                peaks[space] = value
                phase_peaks[space] = phase
    return peaks, phase_peaks


@dataclass(frozen=True)
class ResourcePlan:
    """Resolved composition; no native pointer, allocator or mutable solver state."""

    budget: ResourceBudget
    requests: tuple[ResourceRequest, ...]
    selections: tuple[tuple[str, str], ...]
    status: str
    diagnostic: str | None = None
    schema_version: int = field(default=1, init=False)

    def __post_init__(self):
        if not isinstance(self.budget, ResourceBudget):
            raise TypeError("plan requires a ResourceBudget")
        object.__setattr__(self, "requests", tuple(self.requests))
        object.__setattr__(
            self, "selections", tuple(tuple(row) for row in self.selections)
        )
        if self.status not in ("feasible", "infeasible", "unsupported"):
            raise ValueError("unknown resource plan status")
        names = {r.name for r in self.requests}
        if len(names) != len(self.requests) or len(dict(self.selections)) != len(
            self.selections
        ):
            raise ValueError("duplicate resource plan owner")
        provider_failure = self.status == "infeasible" and any(
            r.infeasible_reason for r in self.requests
        )
        if (
            self.status != "unsupported"
            and not provider_failure
            and set(dict(self.selections)) != names
        ):
            raise ValueError("resource plan must select every provider")
        selected = dict(self.selections)
        for request in self.requests:
            if request.name in selected and selected[request.name] not in {
                c.name for c in request.candidates
            }:
                raise ValueError(
                    "resource plan selects an unknown provider alternative"
                )
        if self.status == "feasible":
            if not self.requests or any(
                r.unsupported_reason or r.infeasible_reason for r in self.requests
            ):
                raise ValueError("feasible resource plan requires supported providers")
            peaks = self.peak_bytes
            if any(
                peaks.get(space, 0) > limit
                for space, limit in self.budget.limits().items()
            ):
                raise ValueError("feasible resource plan exceeds its budget")

    @property
    def estimates(self):
        selected = dict(self.selections)
        return tuple(
            e
            for r in self.requests
            for c in r.candidates
            if selected.get(r.name) == c.name
            for e in c.estimates
        )

    @property
    def peak_bytes(self):
        return _account(self.estimates)[0]

    @property
    def resident_bytes(self):
        return _account(self.estimates, resident=True)[0]

    def require_feasible(self):
        if self.status == "unsupported":
            raise NotImplementedError(self.diagnostic or self.status)
        if self.status != "feasible":
            raise MemoryError(self.diagnostic or self.status)
        return self

    def to_dict(self):
        payload = asdict(self)
        payload.update(
            peak_bytes=self.peak_bytes,
            resident_bytes=self.resident_bytes,
            peak_phases=_account(self.estimates)[1],
            limits=self.budget.limits(),
        )
        payload["identity"] = canonical_hash(payload)
        return payload

    @property
    def identity(self):
        return self.to_dict()["identity"]

    @classmethod
    def from_dict(cls, value):
        """Load a data-only plan and verify both identity and derived accounting."""
        record = dict(value)
        if (
            type(record.get("schema_version")) is not int
            or record["schema_version"] != 1
        ):
            raise ValueError("unsupported resource plan schema")
        identity = record.pop("identity")
        if canonical_hash(record) != identity:
            raise ValueError("resource plan checksum mismatch")
        requests = []
        for row in record["requests"]:
            candidates = tuple(
                ResourceCandidate(
                    **{
                        **candidate,
                        "estimates": tuple(
                            ResourceEstimate(**e) for e in candidate["estimates"]
                        ),
                    }
                )
                for candidate in row["candidates"]
            )
            requests.append(
                ResourceRequest(
                    **{
                        **row,
                        "identity": ResourceIdentity(**row["identity"]),
                        "candidates": candidates,
                    }
                )
            )
        plan = cls(
            ResourceBudget(**record["budget"]),
            tuple(requests),
            tuple(tuple(x) for x in record["selections"]),
            record["status"],
            record["diagnostic"],
        )
        if plan.identity != identity:
            raise ValueError("resource plan derived accounting mismatch")
        return plan


def plan_resources(requests, budget, *, maximum_combinations=65536):
    """Compose provider alternatives without inventing a scientific fallback.

    Enumerate the providers' finite choices in deterministic cost order. A
    global choice handles coupled host/device tradeoffs that greedy independent
    tile limits miss. Requests should describe shared bucket workspaces rather
    than duplicate them per item. A finite enumeration guard returns an honest
    unsupported diagnostic instead of falsely declaring a large search infeasible.
    """
    if not isinstance(budget, ResourceBudget):
        raise TypeError("a ResourceBudget is required")
    requests = tuple(sorted(requests, key=lambda r: r.name))
    if not requests or not all(isinstance(r, ResourceRequest) for r in requests):
        raise ValueError("at least one structured resource request is required")
    if len({r.name for r in requests}) != len(requests):
        raise ValueError("duplicate resource request owner")
    reasons = [
        f"{r.name}: {r.unsupported_reason}" for r in requests if r.unsupported_reason
    ]
    if reasons:
        return ResourcePlan(budget, requests, (), "unsupported", "; ".join(reasons))
    reasons = [
        f"{r.name}: {r.infeasible_reason}" for r in requests if r.infeasible_reason
    ]
    if reasons:
        return ResourcePlan(budget, requests, (), "infeasible", "; ".join(reasons))
    checked_bytes(maximum_combinations, "planning search limit")
    combinations = math.prod(len(r.candidates) for r in requests)
    if combinations > maximum_combinations:
        return ResourcePlan(
            budget,
            requests,
            (),
            "unsupported",
            f"{combinations} provider combinations exceed planning search limit {maximum_combinations}",
        )
    limits = budget.limits()
    feasible = []
    minima = {}
    closest = None
    for candidates in itertools.product(*(r.candidates for r in requests)):
        estimates = tuple(e for c in candidates for e in c.estimates)
        peaks, _ = _account(estimates)
        for space, value in peaks.items():
            minima[space] = min(minima.get(space, MAX_BYTES), value)
        overflow = sum(
            max(0, peaks.get(space, 0) - limit) for space, limit in limits.items()
        )
        selections = tuple(
            (r.name, c.name) for r, c in zip(requests, candidates, strict=True)
        )
        key = (overflow, sum(c.relative_cost for c in candidates), selections)
        if closest is None or key < closest[0]:
            closest = key, selections, peaks, estimates
        if not overflow:
            feasible.append((key, selections))
    if feasible:
        return ResourcePlan(budget, requests, min(feasible)[1], "feasible")
    _, selections, peaks, estimates = closest
    failures = {
        space: {
            "peak": peaks.get(space, 0),
            "limit": limit,
            "minimum_across_candidates": minima.get(space, 0),
        }
        for space, limit in limits.items()
        if peaks.get(space, 0) > limit
    }
    dominant = sorted(estimates, key=lambda e: (-e.bytes, e.name))[:3]
    return ResourcePlan(
        budget,
        requests,
        selections,
        "infeasible",
        f"no supported plan fits: {failures}; dominant allocations: "
        + ", ".join(f"{e.name}={e.bytes} {e.space}" for e in dominant),
    )


def lower_memory_plans(plan, failed_space, *, phase=None, maximum_combinations=65536):
    """Enumerate supported retries that reduce the failed space and still fit.

    The caller must destroy a failed provider before trying another plan and
    record the selected alternative and allocation error. This function never
    retries a solve, changes a backend, or decides that numerical failure is
    an allocation failure. Host/device tradeoffs remain subject to every cap.
    """
    plan.require_feasible()
    checked_bytes(maximum_combinations, "retry search limit")
    if math.prod(len(r.candidates) for r in plan.requests) > maximum_combinations:
        raise NotImplementedError("provider retries exceed the finite search limit")
    if failed_space not in plan.peak_bytes:
        raise ValueError("unknown failed allocation space")
    if phase is not None:
        checked_bytes(phase, "failed allocation phase")

    def failed_usage(estimates):
        live = (
            estimates
            if phase is None
            else tuple(e for e in estimates if e.first_phase <= phase <= e.last_phase)
        )
        return _account(live)[0].get(failed_space, 0)

    previous_usage = failed_usage(plan.estimates)
    result = []
    for candidates in itertools.product(*(r.candidates for r in plan.requests)):
        selected = tuple(
            (r.name, c.name) for r, c in zip(plan.requests, candidates, strict=True)
        )
        estimates = tuple(e for c in candidates for e in c.estimates)
        peaks = _account(estimates)[0]
        if failed_usage(estimates) >= previous_usage:
            continue
        if any(
            peaks.get(space, 0) > limit for space, limit in plan.budget.limits().items()
        ):
            continue
        alternative = ResourcePlan(plan.budget, plan.requests, selected, "feasible")
        result.append((sum(c.relative_cost for c in candidates), selected, alternative))
    return tuple(row[2] for row in sorted(result, key=lambda row: row[:2]))


class ResourceAllocationError(MemoryError):
    """A provider-identified allocation failure eligible for explicit retries.

    Providers must classify the failed memory space using an actual allocator
    status. Numerical errors, driver errors and message-string guesses are not
    allocation failures. A failed factory must release its partial resources
    before raising this exception.
    """

    def __init__(self, space, message):
        if space not in ("host", "pageable", "pinned", "device"):
            ResourceEstimate("failed allocation", 0, space, 0, 0)
        self.space = space
        super().__init__(message)


class ResourceSession:
    """Execute provider lifetimes and finite allocation retries under one plan.

    ``factories`` maps each request name to a callable taking the complete
    selected plan and returning an object with ``close()``. ``advance(phase)``
    releases expired owners before allocating newly live ones. A failed group
    is closed in reverse order before retrying an enumerated lower-memory
    alternative; already live owners never change their accepted selection.
    Scientific execution is explicit through ``provider(name)`` and is never
    retried here. Sessions belong to one synchronous orchestrator.

    This owner-level executor requires one common interval for every buffer
    in a provider candidate. Providers with internal phased arenas must manage
    those arenas themselves or expose separate request owners. The planner's
    general per-buffer lifetime model is otherwise unchanged.
    """

    def __init__(self, plan, factories):
        self.plan = plan.require_feasible()
        self.factories = dict(factories)
        if set(self.factories) != {r.name for r in plan.requests}:
            raise ValueError("resource factories must match every request owner")
        self.intervals = {}
        for request in plan.requests:
            intervals = {
                (e.first_phase, e.last_phase)
                for c in request.candidates
                for e in c.estimates
            }
            if len(intervals) != 1:
                raise ValueError(
                    "resource session requires a uniform provider lifetime"
                )
            self.intervals[request.name] = intervals.pop()
        self.phase = None
        self.fallbacks = []
        self._live = {}
        self._closed = False

    @staticmethod
    def _release(owners):
        # Continue releasing even when a provider's close reports a failure.
        # Propagate the first exception after every other owner was attempted.
        first_error = None
        for owner in reversed(tuple(owners.values())):
            try:
                owner.close()
            except Exception as error:  # noqa: BLE001 -- release other owners, then re-raise
                if first_error is None:
                    first_error = error
        owners.clear()
        if first_error is not None:
            raise first_error

    def advance(self, phase):
        """Enter a monotonic phase, rolling back failed preparation as a group."""
        checked_bytes(phase, "execution phase")
        if self._closed:
            raise RuntimeError("resource session is closed")
        if self.phase is not None and phase < self.phase:
            raise ValueError("resource phases cannot move backwards")
        expired = {
            name: self._live.pop(name)
            for name in tuple(self._live)
            if self.intervals[name][1] < phase
        }
        self._release(expired)
        self.phase = phase
        needed = sorted(
            name
            for name, (first, last) in self.intervals.items()
            if first <= phase <= last and name not in self._live
        )
        # Host/device tradeoffs can otherwise cycle: one candidate fails on
        # the device, its replacement fails on the host, then the first is
        # selected again. Each complete selection gets one attempt per phase.
        attempted = set()
        while needed:
            attempted.add(self.plan.selections)
            created = {}
            try:
                for name in needed:
                    owner = self.factories[name](self.plan)
                    if not callable(getattr(owner, "close", None)):
                        raise TypeError(
                            "prepared resource owner must implement close()"
                        )
                    created[name] = owner
            except ResourceAllocationError as error:
                self._release(created)
                fixed = {name: dict(self.plan.selections)[name] for name in self._live}
                alternatives = [
                    p
                    for p in lower_memory_plans(self.plan, error.space, phase=phase)
                    if p.selections not in attempted
                    and all(
                        dict(p.selections)[name] == choice
                        for name, choice in fixed.items()
                    )
                ]
                alternative = alternatives[0] if alternatives else None
                self.fallbacks.append(
                    {
                        "phase": phase,
                        "failed_owner": name,
                        "space": error.space,
                        "reason": str(error),
                        "from_plan": self.plan.identity,
                        "to_plan": alternative.identity
                        if alternative is not None
                        else None,
                    }
                )
                if alternative is None:
                    raise
                self.plan = alternative
                continue
            except BaseException:
                self._release(created)
                raise
            self._live.update(created)
            break
        return self

    def provider(self, name):
        """Return an allocated owner only while its declared lifetime is live."""
        if name not in self._live:
            raise RuntimeError(
                f"resource provider {name!r} is not live in phase {self.phase}"
            )
        return self._live[name]

    def close(self):
        """Release all owners, including after a scientific execution failure."""
        self._closed = True
        self._release(self._live)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("resource session is closed")
        return self

    def __exit__(self, *unused):
        self.close()


class CpuResourceObservation:
    """Synchronous optional samples from the native CPU SCF implementation.

    This is a per-item iteration-boundary observation of integral, density,
    Fock, orbital and DIIS vector capacities. It excludes transient recurrence
    and eigensolver allocations, other fleet items, and runtime overhead. The
    value must not be presented as the complete prepared plan's measured peak.
    Older libraries and unsampled backends report unavailable data as ``None``.
    """

    def __init__(self, library, *, enabled=True, cpu_workers=0, ledger=None):
        self.library = library
        self.ledger = ledger
        self.device_observation = None
        if type(cpu_workers) is not int or cpu_workers not in (0, 1):
            raise ValueError(
                "CPU resource scope supports default or serialized workers"
            )
        self.cpu_workers = cpu_workers
        self.available = enabled and all(
            hasattr(library, name)
            for name in (
                "vibeqc_resource_tracking_begin_v1",
                "vibeqc_resource_tracking_end_v1",
            )
        )
        self.peak_bytes = None
        self.samples = None
        self.cuda_arena_peak_bytes = None
        self.cuda_arena_samples = None

    def __enter__(self):
        if self.available:
            import ctypes

            self.library.vibeqc_resource_tracking_begin_v1.argtypes = [ctypes.c_uint]
            self.library.vibeqc_resource_tracking_begin_v1.restype = ctypes.c_int
            self.library.vibeqc_resource_tracking_end_v1.argtypes = [
                ctypes.POINTER(ctypes.c_uint64)
            ] * 2
            self.library.vibeqc_resource_tracking_end_v1.restype = ctypes.c_int
            if self.library.vibeqc_resource_tracking_begin_v1(self.cpu_workers):
                raise RuntimeError(
                    "CPU resource observation is already active on this thread"
                )
            if self.ledger is not None and self.library.vibeqc_resource_ledger_bind_v1(
                self.ledger.handle
            ):
                peak, samples = ctypes.c_uint64(), ctypes.c_uint64()
                self.library.vibeqc_resource_tracking_end_v1(
                    ctypes.byref(peak), ctypes.byref(samples)
                )
                raise RuntimeError("native resource ledger scope could not be bound")
        elif self.ledger is not None:
            raise NotImplementedError(
                "native device ledger requires an observation scope"
            )
        return self

    def __exit__(self, *unused):
        if self.available:
            import ctypes

            peak, samples = ctypes.c_uint64(), ctypes.c_uint64()
            cuda_query = getattr(self.library, "vibeqc_resource_tracking_cuda_v1", None)
            if cuda_query is not None:
                cuda_query.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 2
                cuda_query.restype = ctypes.c_int
                if (
                    cuda_query(ctypes.byref(peak), ctypes.byref(samples)) == 0
                    and samples.value
                ):
                    self.cuda_arena_peak_bytes, self.cuda_arena_samples = (
                        peak.value,
                        samples.value,
                    )
            if self.library.vibeqc_resource_tracking_end_v1(
                ctypes.byref(peak), ctypes.byref(samples)
            ):
                raise RuntimeError("CPU resource observation ownership changed")
            if samples.value:
                self.peak_bytes, self.samples = peak.value, samples.value
            if self.ledger is not None:
                self.device_observation = self.ledger.to_dict()

    def to_dict(self):
        return {
            "status": "observed"
            if self.samples is not None
            or self.cuda_arena_samples is not None
            or self.device_observation is not None
            else "unavailable",
            "sampled_item_peak_host_bytes": self.peak_bytes,
            "samples": self.samples,
            "sampled_cuda_arena_peak_bytes": self.cuda_arena_peak_bytes,
            "cuda_arena_samples": self.cuda_arena_samples,
            "device_ledger": self.device_observation,
            "cuda_scope": "direct-HF owned arenas at allocation and retained fleet-cache boundaries; excludes transient warm-retry overlap, runtime and library internals",
            "scope": "per-item integral and SCF/DIIS buffers at CPU iteration boundaries",
            "complete_plan_peak": False,
            "cpu_worker_limit": self.cpu_workers or None,
            "excludes": [
                "recurrence/eigensolver temporaries",
                "other fleet items",
                "runtime overhead",
            ],
        }

    def verify(self, plan):
        """Reject underestimated owned capacities without inventing missing data."""
        for space, observed in (
            ("host", self.peak_bytes),
            ("device", self.cuda_arena_peak_bytes),
        ):
            if observed is not None and observed > plan.peak_bytes[space]:
                raise RuntimeError(
                    f"sampled {space} capacity exceeds the accepted resource plan"
                )
        if (
            self.device_observation is not None
            and self.device_observation["peak_bytes"] > plan.peak_bytes["device"]
        ):
            raise RuntimeError(
                "owned CUDA allocation peak exceeds the accepted resource plan"
            )
