"""Bounded s/p/d primitive dispatch using exact center and axis permutations.

Only compiled code is shared. Every ordered consumer weight is still evaluated
once, and derivative center/axis slots are restored before physical scattering.
The existing first-derivative graphs remain the sole mathematical lowering.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product

from .eri_weights import eri_weight_orbit
from .first_derivative_native import (
    emit_first_derivative_cpu,
    emit_first_derivative_cuda,
)

AXES = tuple(permutations(range(3)))
ERI_CENTERS = eri_weight_orbit((0, 1, 2, 3))
REQUESTS_PER_UNIT = 8
CUDA_REQUESTS_PER_UNIT = 16
MAX_UNIT_BYTES = 4 << 20
MAX_PROGRAM_BYTES = 64 << 20
DerivativeRequest = tuple[str, tuple[str, ...]]
COMPONENT_LABELS = ("", "x", "xx", "xy", "xz", "y", "yy", "yz", "z", "zz")
COMPONENT_OPERATORS = ("overlap", "kinetic", "nuclear_attraction", "four_center_eri")
DISPATCH_ROWS = 3 * len(COMPONENT_LABELS) ** 2 + len(COMPONENT_LABELS) ** 4
DISPATCH_WIDTH = 9


@dataclass(frozen=True)
class DerivativeBinding:
    request: DerivativeRequest
    # Native center/axis slot -> original center/axis slot.
    centers: tuple[int, ...]
    axes: tuple[int, ...]


@lru_cache(maxsize=16384)
def derivative_binding(operator: str, components: tuple[str, ...]) -> DerivativeBinding:
    """Canonicalize a scalar Cartesian integral, preserving all derivative slots."""
    if operator == "nuclear":
        if components:
            raise ValueError("nuclear derivative has no AO components")
        return DerivativeBinding((operator, ()), (0, 1), (0, 1, 2))
    rank = 4 if operator == "four_center_eri" else 2
    if operator not in ("four_center_eri", "overlap", "kinetic", "nuclear_attraction"):
        raise ValueError("unsupported first derivative operator")
    if len(components) != rank or any(
        not isinstance(c, str)
        or len(c) > 2
        or c != "".join(sorted(c))
        or any(a not in "xyz" for a in c)
        for c in components
    ):
        raise ValueError("first derivative schedule requires s/p/d Cartesian labels")
    centers = ERI_CENTERS if rank == 4 else ((0, 1), (1, 0))
    transformed = {
        c: tuple(
            "".join("xyz"[j] * c.count("xyz"[i]) for j, i in enumerate(axes))
            for axes in AXES
        )
        for c in components
    }
    labels, order, axes = min(
        (tuple(transformed[components[i]][a] for i in order), order, axes)
        for order in centers
        for a, axes in enumerate(AXES)
    )
    if operator == "nuclear_attraction":
        order += (2,)
    return DerivativeBinding((operator, labels), order, axes)


@lru_cache(maxsize=4)
def derivative_requests(
    domain: tuple[str, ...],
) -> tuple[DerivativeRequest, ...]:
    """Finite metadata-only plan; full s/p/d needs 313 ERI representatives."""
    if not domain or len(domain) > 10 or tuple(sorted(set(domain))) != domain:
        raise ValueError(
            "component domain must contain one to ten sorted unique labels"
        )
    requests = {("nuclear", ())}
    for operator, rank in (
        ("overlap", 2),
        ("kinetic", 2),
        ("nuclear_attraction", 2),
        ("four_center_eri", 4),
    ):
        for components in product(domain, repeat=rank):
            requests.add(derivative_binding(operator, components).request)
    return tuple(sorted(requests))


@lru_cache(maxsize=4)
def derivative_sources(
    domain: tuple[str, ...],
) -> tuple[tuple[tuple[DerivativeRequest, ...], str], ...]:
    """Bound each translation unit and the whole finite generated source set.

    The small in-process source cache avoids regenerating graphs on warm calls.
    Native artifact consumers must still revalidate source/header/binary and
    toolchain identities through the common compile_runtime cache each time.
    """
    requests = derivative_requests(domain)
    units, total = [], 0
    for begin in range(0, len(requests), REQUESTS_PER_UNIT):
        selected = requests[begin : begin + REQUESTS_PER_UNIT]
        source = emit_first_derivative_cpu(selected)
        size = len(source.encode("utf-8"))
        total += size
        if size > MAX_UNIT_BYTES or total > MAX_PROGRAM_BYTES:
            raise ValueError("first derivative generated source budget exceeded")
        units.append((selected, source))
    return tuple(units)


@lru_cache(maxsize=4)
def derivative_cuda_sources(
    domain: tuple[str, ...],
) -> tuple[tuple[tuple[DerivativeRequest, ...], str], ...]:
    """Bound the s/p/d CUDA derivative inventory across relocatable objects."""

    requests = derivative_requests(domain)
    units, total = [], 0
    for unit, begin in enumerate(range(0, len(requests), CUDA_REQUESTS_PER_UNIT)):
        selected = requests[begin : begin + CUDA_REQUESTS_PER_UNIT]
        source = emit_first_derivative_cuda(
            selected, symbol=f"first_derivative_shard_{unit}"
        )
        size = len(source.encode("utf-8"))
        total += size
        if size > MAX_UNIT_BYTES or total > MAX_PROGRAM_BYTES:
            raise ValueError("first derivative generated CUDA source budget exceeded")
        units.append((selected, source))
    return tuple(units)


@lru_cache(maxsize=4)
def derivative_dispatch_table(domain: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    """Immutable runtime dispatch: library, kind, four centers and three axes.

    Slots use base-ten Cartesian label codes, independently of public AO order.
    Unrequested slots remain invalid. This is metadata, never consumer weights.
    """
    requests = {request: i for i, request in enumerate(derivative_requests(domain))}
    table = [(-1,) * DISPATCH_WIDTH] * DISPATCH_ROWS
    for op, operator in enumerate(COMPONENT_OPERATORS):
        rank = 4 if op == 3 else 2
        for labels in product(domain, repeat=rank):
            slot = 0
            for label in labels:
                slot = len(COMPONENT_LABELS) * slot + COMPONENT_LABELS.index(label)
            slot += op * len(COMPONENT_LABELS) ** 2
            binding = derivative_binding(operator, labels)
            kernel = requests[binding.request]
            table[slot] = (
                kernel // REQUESTS_PER_UNIT,
                kernel % REQUESTS_PER_UNIT,
                *binding.centers,
                *([-1] * (4 - len(binding.centers))),
                *binding.axes,
            )
    return tuple(table)
