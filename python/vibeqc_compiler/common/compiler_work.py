"""Deterministic, opt-in work bounds for finite compiler investigations.

A work unit is charged before one symbolic interning attempt, including reuse.
Limits constrain compilation work, never scientific precision or runtime work.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class CompilerWorkLimit(RuntimeError):
    """The requested compiler investigation exhausted its explicit budget."""


@dataclass(slots=True)
class CompilerWorkCounter:
    limit: int
    used: int = 0


_active: ContextVar[tuple[CompilerWorkCounter, ...]] = ContextVar(
    "vibeqc_compiler_work", default=()
)


@contextmanager
def compiler_work_budget(limit: int | None) -> Iterator[CompilerWorkCounter | None]:
    """Bound nested symbolic work; None does not disable an enclosing limit."""
    if limit is None:
        yield None
        return
    if type(limit) is not int or limit <= 0:
        raise ValueError("compiler work limit must be a positive integer or None")
    counter = CompilerWorkCounter(limit)
    token = _active.set((*_active.get(), counter))
    try:
        yield counter
    finally:
        _active.reset(token)


def charge_symbolic_intern() -> None:
    """Charge all active scopes before modifying a symbolic graph."""
    counters = _active.get()
    for counter in counters:
        if counter.used >= counter.limit:
            raise CompilerWorkLimit(
                f"compiler symbolic work budget exhausted ({counter.limit} intern attempts); "
                "emission is unqualified within this budget"
            )
    for counter in counters:
        counter.used += 1
