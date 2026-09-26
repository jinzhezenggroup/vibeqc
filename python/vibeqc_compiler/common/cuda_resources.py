"""PTXAS resource records for arbitrary generated kernel symbols."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KernelResources:
    """Static PTXAS resources; stack_bytes bounds both frame and cumulative stack."""

    function: str
    registers: int
    stack_bytes: int
    spill_store_bytes: int
    spill_load_bytes: int
    shared_bytes: int
    local_bytes: int | None = None


def parse_resources(diagnostics: str) -> tuple[KernelResources, ...]:
    """Preserve zero-shared-memory entries omitted by shell-specific parsing."""
    pattern = re.compile(
        r"Function properties for (?P<function>\S+)\n"
        r"\s*(?P<stack>\d+) bytes stack frame, (?P<stores>\d+) bytes spill stores, (?P<loads>\d+) bytes spill loads\n"
        r"ptxas info\s*: Used (?P<registers>\d+) registers(?P<rest>[^\n]*)"
    )
    result = []
    # Windows compiler output uses CRLF even when the caller retains raw text.
    for match in pattern.finditer(diagnostics.replace("\r\n", "\n")):
        shared = re.search(r"(\d+) bytes smem", match["rest"])
        local = re.search(r"(\d+) bytes lmem", match["rest"])
        cumulative = re.search(r"(\d+) bytes cumulative stack size", match["rest"])
        stack = int(match["stack"])
        if cumulative is not None:
            stack = max(stack, int(cumulative[1]))
        result.append(
            KernelResources(
                function=match["function"],
                registers=int(match["registers"]),
                stack_bytes=stack,
                spill_store_bytes=int(match["stores"]),
                spill_load_bytes=int(match["loads"]),
                shared_bytes=int(shared[1]) if shared else 0,
                local_bytes=int(local[1]) if local else None,
            )
        )
    return tuple(result)
