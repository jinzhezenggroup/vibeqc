"""Small runtime pages must not materialize large axes or alias changed work."""

import tracemalloc
from dataclasses import replace

from vibeqc_compiler.common.runtime_domain import RuntimeTaskDomain


def test_small_first_page_has_axis_independent_memory() -> None:
    domain = RuntimeTaskDomain.rectangular((200_000, 3))
    tracemalloc.start()
    try:
        page = next(domain.pages(2))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert page.coordinates == ((0, 0), (0, 1))
    assert peak < 512 * 1024, f"two-item page retained {peak} bytes"


def test_page_identity_binds_actual_ordered_coordinates() -> None:
    page = next(RuntimeTaskDomain.rectangular((3, 3)).pages(2))
    changed = replace(page, coordinates=tuple(reversed(page.coordinates)))
    assert page.identity != changed.identity
