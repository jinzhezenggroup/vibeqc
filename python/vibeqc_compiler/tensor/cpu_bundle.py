"""Eager native CPU bundle owner for exact TensorIR program sets.

The ordinary :class:`NativeTensorProgram` path remains the one-program API.  This
module adds an explicit prewarm boundary for consumers that know a bounded set of
TensorIR programs up front: every program is emitted as its own translation unit,
then the shared native-runtime cache links that exact ordered set into one verified
shared library.  Scientific equations remain owned by TensorIR program builders.
"""

from __future__ import annotations

import ctypes as ct
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime_bundle
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.source_cache import cache_source

from .cpu import emit_cpu
from .program import Program


@dataclass(frozen=True, slots=True)
class _BundleEntry:
    program: Program
    inputs: tuple[typing.Any, ...]
    resources: dict[str, typing.Any]
    identity: str
    symbol: str
    call: typing.Any


def _bind_call(library: typing.Any, symbol: str) -> typing.Any:
    call = getattr(library, symbol)
    call.argtypes = [
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    return call


class NativeTensorProgramBundle:
    """Prewarm one exact ordered TensorIR set into one verified CPU runtime."""

    def __init__(
        self,
        programs: Sequence[Program],
        *,
        compiler: typing.Any,
        cache: typing.Any,
        max_bytes: typing.Any = 8 * 1024 * 1024,
        max_work: typing.Any = 100_000_000,
        max_nodes: typing.Any = 4096,
    ) -> None:
        if not isinstance(compiler, CppCompilerAdapter):
            raise TypeError("native TensorIR bundles require a CPU compiler adapter")
        programs = tuple(programs)
        if not programs:
            raise ValueError("native TensorIR bundle requires at least one program")
        if any(not isinstance(program, Program) for program in programs):
            raise TypeError("native TensorIR bundle entries must be TensorIR Programs")

        logical_ids = tuple(program.logical_hash for program in programs)
        if len(set(logical_ids)) != len(logical_ids):
            raise ValueError(
                "native TensorIR bundle requires unique program identities"
            )

        cache = Path(cache)
        source_cache = cache / "sources"
        source_cache.mkdir(parents=True, exist_ok=True)
        emitted: list[
            tuple[Program, tuple[typing.Any, ...], dict[str, typing.Any], str, str]
        ] = []
        source_paths: list[Path] = []
        for program in programs:
            symbol = f"tensor_cpu_bundle_{program.logical_hash}"
            source, resources = emit_cpu(
                program,
                max_bytes=max_bytes,
                max_work=max_work,
                max_nodes=max_nodes,
                symbol=symbol,
            )
            identity = canonical_hash(
                {"program": program.to_payload(), "source": source}
            )
            source_path = source_cache / f"{identity}.cpp"
            cache_source(source_path, source)
            source_paths.append(source_path)
            emitted.append(
                (
                    program,
                    tuple(node for node in program.live_nodes if node.op == "input"),
                    resources,
                    identity,
                    symbol,
                )
            )

        header = asset_path("src/tensor/cpu_runtime.hpp")
        self.artifact = compile_runtime_bundle(
            compiler,
            cache / "bundles",
            source_paths,
            headers=(header,),
            options=("-ffp-contract=off", f"-I{header.parent}"),
        )
        self.library = ct.CDLL(str(self.artifact.library))
        self.max_bytes = max_bytes
        self.program_identities = logical_ids
        self._entries: dict[str, _BundleEntry] = {}
        for program, inputs, resources, identity, symbol in emitted:
            self._entries[program.logical_hash] = _BundleEntry(
                program=program,
                inputs=inputs,
                resources=resources,
                identity=identity,
                symbol=symbol,
                call=_bind_call(self.library, symbol),
            )

    @property
    def program_count(self) -> int:
        return len(self._entries)

    @property
    def artifact_count(self) -> int:
        """Number of shared-library artifacts owned by this exact bundle."""

        return 1

    def execute(
        self, program: Program, feeds: Mapping[str, typing.Any]
    ) -> dict[str, object]:
        """Execute one prewarmed program through its deterministic bundle entry."""

        if not isinstance(program, Program):
            raise TypeError(
                "native TensorIR bundle execution requires a TensorIR Program"
            )
        entry = self._entries.get(program.logical_hash)
        if entry is None or entry.program.to_payload() != program.to_payload():
            raise ValueError("TensorIR program is not present in this native bundle")
        if not isinstance(feeds, Mapping):
            raise TypeError("CPU tensor feeds must be a mapping")

        values = []
        for node in entry.inputs:
            name = node.attrs["name"]
            if name not in feeds:
                raise ValueError(f"missing tensor input: {name}")
            value = np.asarray(feeds[name])
            if (
                value.shape != node.spec.shape
                or value.dtype != np.float64
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"invalid float64 tensor input: {name}")
            for symmetry in node.spec.symmetries:
                if not np.allclose(
                    value,
                    symmetry.sign * value.transpose(symmetry.permutation),
                    atol=1e-11,
                    rtol=1e-10,
                ):
                    raise ValueError(f"input {name} violates its declared symmetry")
            values.append(value.reshape(-1))

        packed = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
        output = np.empty(entry.resources["output_count"], dtype=np.float64)
        code = entry.call(
            packed.ctypes.data_as(ct.POINTER(ct.c_double)),
            packed.size,
            output.ctypes.data_as(ct.POINTER(ct.c_double)),
            output.size,
            self.max_bytes,
        )
        if code:
            raise ValueError(f"native CPU tensor evaluation failed ({code})")

        result: dict[str, object] = {}
        cursor = 0
        for name, node in entry.program.outputs.items():
            result[name] = immutable(
                output[cursor : cursor + node.spec.size].reshape(node.spec.shape)
            )
            cursor += node.spec.size
        return result
