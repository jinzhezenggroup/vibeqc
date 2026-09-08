"""PTXAS resource records for arbitrary generated tensor kernel symbols."""

import re

from tools.vibeqc_codegen.batch_benchmark import KernelResources


def parse_resources(diagnostics: str) -> tuple[KernelResources, ...]:
    """Preserve zero-shared-memory entries omitted by shell-specific parsing."""
    pattern = re.compile(
        r"Function properties for (?P<function>\S+)\n"
        r"\s*(?P<stack>\d+) bytes stack frame, (?P<stores>\d+) bytes spill stores, (?P<loads>\d+) bytes spill loads\n"
        r"ptxas info\s*: Used (?P<registers>\d+) registers(?P<rest>[^\n]*)"
    )
    result = []
    for match in pattern.finditer(diagnostics):
        shared = re.search(r"(\d+) bytes smem", match["rest"])
        result.append(
            KernelResources(
                function=match["function"],
                registers=int(match["registers"]),
                stack_bytes=int(match["stack"]),
                spill_store_bytes=int(match["stores"]),
                spill_load_bytes=int(match["loads"]),
                shared_bytes=int(shared[1]) if shared else 0,
            )
        )
    return tuple(result)
