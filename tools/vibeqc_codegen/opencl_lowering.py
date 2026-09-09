"""Structured OpenCL C lowering, with no textual CUDA syntax conversion.

The scalar arithmetic emitter is shared with CUDA. This module owns OpenCL
argument/address-space/work-item syntax and FP64 extension policy. Scientific
roots and their equation identity remain supplied by the existing integral IR.
"""

import hashlib
import re
from dataclasses import dataclass

from .expr import Expr, Graph
from .runtime_backend import ExecutionShape, RuntimeCapabilities
from .scalar_c import ScalarCEmitter

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_RESERVED = {
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "bool",
    "half",
    "uchar",
    "ushort",
    "uint",
    "ulong",
    "size_t",
    "ptrdiff_t",
    "intptr_t",
    "uintptr_t",
    "kernel",
    "global",
    "local",
    "constant",
    "private",
    "read_only",
    "write_only",
    "read_write",
    "sampler_t",
    "event_t",
    "inputs",
    "outputs",
    "count",
    "item",
    "exp",
    "log",
    "log1p",
    "expm1",
    "sqrt",
    "pow",
    "fma",
    "get_global_id",
}


@dataclass(frozen=True)
class ScalarKernel:
    """One row of independent primitive inputs and explicitly ordered outputs.

    ``scientific_hash`` identifies the operator/component intent, independent
    of workgroup shape, compiler options and output source language. Every
    reachable graph variable must appear exactly once in ``inputs``; unused
    input columns are retained so callers can preserve a stable primitive ABI.
    """

    graph: Graph
    roots: tuple[Expr, ...]
    inputs: tuple[str, ...]
    scientific_hash: str
    name: str = "integral_values"

    def __post_init__(self):
        object.__setattr__(self, "roots", tuple(self.roots))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if (
            not self.roots
            or not self.inputs
            or len(set(self.inputs)) != len(self.inputs)
        ):
            raise ValueError(
                "nonempty distinct input columns and output roots required"
            )
        for name in (*self.inputs, self.name):
            if (
                not isinstance(name, str)
                or not _IDENTIFIER.fullmatch(name)
                or name in _RESERVED
                or name.startswith("_")
                or name.startswith("v")
                and name[1:].isdigit()
            ):
                raise ValueError("kernel names must be safe, nonreserved C identifiers")
        if any(root.graph is not self.graph for root in self.roots):
            raise ValueError("all roots must belong to the same scientific graph")
        if len(self.scientific_hash) != 64 or any(
            c not in "0123456789abcdef" for c in self.scientific_hash
        ):
            raise ValueError("verified scientific SHA-256 identity required")
        reachable = {
            str(self.graph.nodes[i].payload)
            for i in self.graph.topological_order(self.roots)
            if self.graph.nodes[i].operation == "variable"
        }
        if not reachable <= set(self.inputs):
            raise ValueError("a reachable scientific input has no primitive ABI column")


def emit_opencl(
    kernel: ScalarKernel, target: RuntimeCapabilities, shape: ExecutionShape
):
    """Emit one FP64 scalar primitive kernel for ordinary native host submission."""
    if target.backend != "opencl":
        raise ValueError("OpenCL lowering requires an OpenCL target")
    shape.validate_for(target)
    if not shape.requires_fp64:
        raise ValueError("this scalar integral lowering requires explicit FP64")
    # Semantic input names must not shadow OpenCL builtins or emitted temporaries.
    # Their column order is the ABI; local source spellings carry no scientific meaning.
    variables = {name: f"input_column_{i}" for i, name in enumerate(kernel.inputs)}
    emitter = ScalarCEmitter(kernel.graph, variables)
    emitter.emit(kernel.roots)
    lines = [
        "// Scientific identity: " + kernel.scientific_hash,
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable",
        f"__attribute__((reqd_work_group_size({shape.workgroup_threads}, 1, 1)))",
        f"__kernel void {kernel.name}(__global const double* inputs,",
        "    __global double* outputs, ulong count) {",
        "  const size_t item = get_global_id(0);",
        "  if (item >= count) return;",
    ]
    for column, name in enumerate(kernel.inputs):
        lines.append(
            f"  const double {variables[name]} = inputs[item * {len(kernel.inputs)} + {column}];"
        )
    lines += emitter.lines
    for column, root in enumerate(kernel.roots):
        lines.append(
            f"  outputs[item * {len(kernel.roots)} + {column}] = {emitter.reference(root)};"
        )
    lines += ["}", ""]
    return "\n".join(lines)


def source_hash(source):
    """Separate executable source identity from the shared scientific identity."""
    return hashlib.sha256(source.encode()).hexdigest()
