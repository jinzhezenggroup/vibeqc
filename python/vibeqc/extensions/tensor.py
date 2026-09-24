"""Curated public subset of the versioned TensorIR construction API.

Mathematical TensorIR construction and inspection stay import-only operations.
Native compilation is an explicit advanced-user action: the public ``compile``
entry point activates the compiler lazily and never participates in built-in
AOT execution.
"""

from __future__ import annotations

import copy
import math
import numbers
import os
import typing
from pathlib import Path

from vibeqc_compiler.tensor import (
    DenseLayout,
    Index,
    IndexSpace,
    Node,
    PackedLayout,
    Program,
    Symmetry,
    TensorSpec,
    add,
    broadcast,
    capabilities,
    constant,
    divide,
    dot_test,
    einsum,
    execute,
    exp,
    gather,
    input_tensor,
    jvp,
    linearize,
    log,
    multiply,
    optimize,
    power,
    reduce_sum,
    reshape,
    scaled_bilinear,
    slice_tensor,
    sqrt,
    transpose,
    transpose_program,
    vjp,
)
from vibeqc_compiler.tensor.program import PRIMITIVE_VERSION
from vibeqc_compiler.tensor.program import VERSION as SCHEMA_VERSION

API_VERSION = 1


class CompiledTensorProgram:
    """Public handle for one explicitly JIT-compiled TensorIR program.

    The private native owner remains encapsulated so compiler implementation
    details can evolve without becoming part of the extension ABI. Artifact
    provenance is returned as a detached mapping for reproducibility.
    """

    __slots__ = ("_logical_hash", "_native")

    def __init__(self, program: Program, native: typing.Any) -> None:
        self._native = native
        self._logical_hash = program.logical_hash

    @property
    def logical_hash(self) -> str:
        """Return the immutable mathematical identity of the compiled program."""
        return self._logical_hash

    @property
    def target(self) -> str:
        """Return the execution target bound to this native owner."""
        return "cpu"

    @property
    def mode(self) -> str:
        """Return the explicit compilation mode used for this artifact."""
        return "jit"

    @property
    def identity(self) -> str:
        """Return the native source/program identity owned by the compiler."""
        return str(self._native.identity)

    @property
    def artifact(self) -> dict[str, typing.Any]:
        """Return detached verified-artifact provenance for this compilation."""
        artifact = self._native.artifact
        return {
            "target": self.target,
            "mode": self.mode,
            "logical_hash": self.logical_hash,
            "identity": self.identity,
            "library": str(artifact.library),
            "metadata": copy.deepcopy(artifact.metadata),
        }

    @property
    def resources(self) -> dict[str, typing.Any]:
        """Return detached compile-time byte/work requirements."""
        return copy.deepcopy(self._native.resources)

    def execute(self, feeds: typing.Any) -> typing.Any:
        """Execute through the compiled program's checked native ABI."""
        return self._native.execute(feeds)

    def inspect(self) -> dict[str, typing.Any]:
        """Return replay/provenance data without exposing compiler internals."""
        return {
            "extension_api_version": API_VERSION,
            "kind": "compiled-tensor",
            "target": self.target,
            "mode": self.mode,
            "logical_hash": self.logical_hash,
            "identity": self.identity,
            "resources": self.resources,
            "artifact": self.artifact,
        }


def inspect(program: Program) -> dict:
    """Return a detached, replayable TensorIR description."""
    if not isinstance(program, Program):
        raise TypeError("tensor inspection requires Program")
    return {
        "extension_api_version": API_VERSION,
        "kind": "tensor",
        "logical_hash": program.logical_hash,
        "program": program.to_payload(),
    }


def compile_capabilities(
    program: Program,
    *,
    target: str = "cpu",
    mode: str = "jit",
    max_bytes: int = 8 * 1024 * 1024,
    max_work: int = 100_000_000,
    max_nodes: int = 4096,
) -> dict[str, typing.Any]:
    """Inspect compiler admission without probing or invoking a local toolchain.

    ``compilable`` means that the public compiler can lower the requested
    TensorIR program under the supplied resource bounds. It does not claim that
    a suitable host compiler is installed. Successful lowering also reports the
    exact source/program identity that :func:`compile` will bind for the same
    request, before toolchain-specific artifact identity is added. Arbitrary
    user programs are not promoted to VibeQC's independently validated or
    built-in production domains merely because source lowering succeeds.
    """
    if not isinstance(program, Program):
        raise TypeError("tensor compilation capability query requires Program")
    report: dict[str, typing.Any] = {
        "extension_api_version": API_VERSION,
        "kind": "tensor-compile-capabilities",
        "target": target,
        "mode": mode,
        "logical_hash": program.logical_hash,
        "identity": None,
        "represented": True,
        "compilable": False,
        "lowering_validated": False,
        "validated": False,
        "production_promoted": False,
        "toolchain_checked": False,
        "resources": None,
        "reason": None,
    }
    if mode != "jit":
        report["reason"] = (
            "public TensorIR compilation currently supports mode='jit' only"
        )
        return report
    if target != "cpu":
        report["reason"] = "public TensorIR JIT currently supports target='cpu' only"
        return report

    # This is an explicit compiler query, but it remains toolchain-free: emit_cpu
    # performs semantic/resource lowering only and never creates an adapter,
    # probes an executable, starts a subprocess, or creates a cache artifact.
    # Use the production entry symbol so the reported identity is exactly the
    # identity NativeTensorProgram binds for the same compile request.
    from vibeqc_compiler.common.provenance import canonical_hash
    from vibeqc_compiler.tensor.cpu import emit_cpu

    try:
        source, resources = emit_cpu(
            program,
            max_bytes=max_bytes,
            max_work=max_work,
            max_nodes=max_nodes,
            symbol="tensor_cpu",
        )
    except (TypeError, ValueError) as error:
        report["reason"] = str(error)
        return report
    report["compilable"] = True
    report["lowering_validated"] = True
    report["identity"] = canonical_hash(
        {"program": program.to_payload(), "source": source}
    )
    report["resources"] = copy.deepcopy(resources)
    return report


def compile(
    program: Program,
    *,
    target: str = "cpu",
    mode: str = "jit",
    compiler: str | os.PathLike[str] | None = None,
    cache: str | os.PathLike[str] | None = None,
    compile_timeout: float = 300.0,
    max_bytes: int = 8 * 1024 * 1024,
    max_work: int = 100_000_000,
    max_nodes: int = 4096,
) -> CompiledTensorProgram:
    """Explicitly JIT-compile a supported TensorIR program.

    This call is intentionally separate from :func:`execute` and from all
    built-in method execution. Compiler/toolchain modules are imported only
    after the caller requests ``mode="jit"`` for a supported target. The first
    public slice is CPU-only and fails closed for every other target/mode.
    """
    if not isinstance(program, Program):
        raise TypeError("tensor compilation requires Program")
    if mode != "jit":
        raise ValueError(
            "public TensorIR compilation currently supports mode='jit' only"
        )
    if target != "cpu":
        raise ValueError("public TensorIR JIT currently supports target='cpu' only")
    if isinstance(compile_timeout, bool) or not isinstance(
        compile_timeout, numbers.Real
    ):
        raise TypeError("compile_timeout must be a finite positive real number")
    timeout = float(compile_timeout)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("compile_timeout must be finite and positive")
    if compiler is not None and not isinstance(compiler, (str, os.PathLike)):
        raise TypeError("compiler must be a filesystem path or executable name")
    if cache is not None and not isinstance(cache, (str, os.PathLike)):
        raise TypeError("cache must be a filesystem path")

    # Import lowering only on an explicit request. The native owner invokes the
    # compiler factory after its semantic/resource checks, so unsupported IR
    # does not probe a toolchain or create cache artifacts.
    from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
    from vibeqc_compiler.tensor.cpu import NativeTensorProgram

    compiler_path = Path(os.environ.get("CXX", "c++") if compiler is None else compiler)
    if cache is None:
        configured = os.environ.get("VIBEQC_TENSOR_CACHE")
        cache_path = (
            Path(configured) / "extensions"
            if configured
            else Path.home() / ".cache" / "vibeqc" / "tensor-cpu" / "extensions"
        )
    else:
        cache_path = Path(cache)
    native = NativeTensorProgram(
        program,
        compiler=lambda: CppCompilerAdapter(compiler_path, compile_timeout=timeout),
        cache=cache_path,
        max_bytes=max_bytes,
        max_work=max_work,
        max_nodes=max_nodes,
    )
    return CompiledTensorProgram(program, native)


__all__ = [
    "API_VERSION",
    "PRIMITIVE_VERSION",
    "SCHEMA_VERSION",
    "CompiledTensorProgram",
    "DenseLayout",
    "Index",
    "IndexSpace",
    "Node",
    "PackedLayout",
    "Program",
    "Symmetry",
    "TensorSpec",
    "add",
    "broadcast",
    "capabilities",
    "compile",
    "compile_capabilities",
    "constant",
    "divide",
    "dot_test",
    "einsum",
    "execute",
    "exp",
    "gather",
    "input_tensor",
    "inspect",
    "jvp",
    "linearize",
    "log",
    "multiply",
    "optimize",
    "power",
    "reduce_sum",
    "reshape",
    "scaled_bilinear",
    "slice_tensor",
    "sqrt",
    "transpose",
    "transpose_program",
    "vjp",
]
