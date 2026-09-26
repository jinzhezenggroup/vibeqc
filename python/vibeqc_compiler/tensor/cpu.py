"""Explicit bounded FP64 CPU lowering of ordinary TensorIR primitives.

This materializing correctness backend accepts the ordinary FP64 primitives
needed by generated stationary/response programs, including indexing/reordering,
ragged accumulation, division and the range-safe bilinear quotient. It never
dispatches on method or source names. Unsupported semantics fail at generation,
before any allocation or compilation.
"""

import ctypes as ct
import re
import typing
from collections.abc import Mapping
from fractions import Fraction
from math import isfinite, prod
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.source_cache import cache_source

from .batch_schedule import scatter_add_inverted_table
from .cuda_emit import _coordinate, _flat
from .program import Program
from .types import checked_size


def _literal(pair: typing.Any) -> typing.Any:
    value = float(Fraction(*pair))
    if not isfinite(value):
        raise ValueError("CPU coefficient is not finite FP64")
    return value.hex()


CPU_PRIMITIVES = frozenset(
    {
        "input",
        "constant",
        "add",
        "multiply",
        "divide",
        "scaled_bilinear",
        "einsum",
        "transpose",
        "reshape",
        "slice",
        "gather",
        "indexed_gather",
        "scatter_add",
        "segment_sum",
        "reduce",
        "broadcast",
    }
)


def _scaled_bilinear_helper() -> str:
    """Emit the FP64 fused quotient used by TensorIR division VJPs."""

    return r"""
static inline bool tensor_scaled_bilinear(
    double a, double b, double c, double d, double e, double f, double& out) {
  if (e == 0.0 || f == 0.0) return false;
  int ea, eb, ec, ed, ee, ef;
  const double ma = std::frexp(a, &ea), mb = std::frexp(b, &eb);
  const double mc = std::frexp(c, &ec), md = std::frexp(d, &ed);
  const double me = std::frexp(e, &ee), mf = std::frexp(f, &ef);
  double p = ma * mb, q = mc * md;
  double pe = std::fma(ma, mb, -p), qe = std::fma(mc, md, -q);
  const int ep = ea + eb, eq = ec + ed;
  const int exponent = p == 0.0 ? eq : (q == 0.0 ? ep : std::max(ep, eq));
  constexpr int limit = 110;
  const int dp = ep - exponent, dq = eq - exponent;
  if (dp < -limit) {
    p = 0.0;
    pe = 0.0;
  } else {
    p = std::scalbn(p, dp);
    pe = std::scalbn(pe, dp);
  }
  if (dq < -limit) {
    q = 0.0;
    qe = 0.0;
  } else {
    q = std::scalbn(q, dq);
    qe = std::scalbn(qe, dq);
  }
  const double difference = p - q;
  const double tail = difference - p;
  const double residual = (p - (difference - tail)) - (q + tail);
  const double numerator = difference + ((pe - qe) + residual);
  out = std::scalbn(numerator / (me * mf), exponent - ee - ef);
  return std::isfinite(out);
}
"""


def emit_cpu(
    program: typing.Any,
    *,
    max_bytes: typing.Any = 8 * 1024 * 1024,
    max_work: typing.Any = 100_000_000,
    max_nodes: typing.Any = 4096,
    symbol: str = "tensor_cpu",
) -> typing.Any:
    """Return source and exact bounded storage/work requirements without runtime imports.

    No packed-storage semantics or implicit dtype conversion are admitted.
    Index expressions are shared with CUDA; CUDA emission bytes are unchanged.
    """
    if not isinstance(program, Program):
        raise TypeError("CPU lowering requires a TensorIR Program")
    if (
        not isinstance(symbol, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol) is None
    ):
        raise ValueError("CPU entry symbol must be a C identifier")
    checked_size(max_bytes, "CPU byte budget")
    checked_size(max_work, "CPU work budget")
    checked_size(max_nodes, "CPU node budget")
    if max_nodes == 0 or max_nodes > 16384:
        raise ValueError("CPU node budget must lie in [1, 16384]")
    nodes = program.live_nodes
    if len(nodes) > max_nodes:
        raise ValueError("CPU program exceeds node budget")
    offsets, cursor, inputs, work = {}, 0, [], 0
    for node in nodes:
        if node.op not in CPU_PRIMITIVES:
            raise ValueError(f"unsupported CPU primitive: {node.op}")
        if node.spec.dtype != "float64":
            raise ValueError("CPU lowering requires real float64 semantics")
        offsets[node], cursor = cursor, cursor + node.spec.size
        if node.op == "input":
            inputs.append(node)
        reduction = 1
        if node.op == "reduce":
            reduction = prod(node.inputs[0].spec.shape[i] for i in node.attrs["axes"])
        if node.op == "einsum":
            domains = {}
            for child, labels in zip(node.inputs, node.attrs["labels"], strict=True):
                domains.update(zip(labels, child.spec.shape, strict=True))
            reduction = prod(
                v for k, v in domains.items() if k not in node.attrs["output"]
            )
        if node.op in {"scatter_add", "segment_sum"}:
            work += node.spec.size + node.inputs[0].spec.size
        else:
            work += node.spec.size * reduction * max(1, len(node.inputs))
    ni = sum(n.spec.size for n in inputs)
    no = sum(n.spec.size for n in program.outputs.values())
    arena = cursor + no
    required = 8 * (arena + 2 * ni + 3 * no)
    if required > min(max_bytes, (1 << 63) - 1) or work > max_work:
        raise ValueError("CPU program exceeds byte/work budget")
    lines = [
        '#include "cpu_runtime.hpp"',
        *(
            [_scaled_bilinear_helper()]
            if any(n.op == "scaled_bilinear" for n in nodes)
            else []
        ),
        f"// TensorIR {program.logical_hash}; provenance {canonical_hash(program.provenance)}",
        f'extern "C" int {symbol}(const double* input, size_t ni, double* output, size_t no, size_t budget) noexcept {{',
        f"return vibeqc_tensor_cpu::run(input, ni, output, no, budget, {ni}ULL, {no}ULL, {arena}ULL, {required}ULL,",
        "[](const double* input, double* p) {",
    ]
    input_cursor = 0
    for i, node in enumerate(nodes):
        a, shape = node.attrs, node.spec.shape

        def read(child: typing.Any, index: typing.Any = "z") -> typing.Any:
            return f"p[{offsets[child]}ULL + ({index})]"

        c = [_coordinate("z", shape, axis) for axis in range(len(shape))]
        body = []
        if node.op == "input":
            expression = f"input[{input_cursor}ULL + z]"
            input_cursor += node.spec.size
        elif node.op == "constant":
            values = ", ".join(_literal(v) for v in a["values"]) or "0.0"
            lines.append(f"static const double constant_{i}[] = {{{values}}};")
            expression = f"constant_{i}[z]"
        elif node.op == "add":
            body.append("double value = 0.0;")
            for child, factor in zip(node.inputs, a["coefficients"], strict=True):
                body.append(f"value += {_literal(factor)} * {read(child)};")
            expression = "value"
        elif node.op == "multiply":
            expression = f"{read(node.inputs[0])} * {read(node.inputs[1])}"
        elif node.op == "divide":
            body.append(f"const double denominator = {read(node.inputs[1])};")
            body.append("if (denominator == 0.0) return false;")
            expression = f"{read(node.inputs[0])} / denominator"
        elif node.op == "scaled_bilinear":
            body.append("double value = 0.0;")
            arguments = ", ".join(read(child) for child in node.inputs)
            body.append(
                f"if (!tensor_scaled_bilinear({arguments}, value)) return false;"
            )
            expression = "value"
        elif node.op in {"reduce", "einsum"}:
            if node.op == "reduce":
                child = node.inputs[0]
                rs = tuple(child.spec.shape[axis] for axis in a["axes"])
                coords, kept = [], iter(c)
                for axis in range(len(child.spec.shape)):
                    coords.append(
                        _coordinate("r", rs, a["axes"].index(axis))
                        if axis in a["axes"]
                        else next(kept)
                    )
                term = read(child, _flat(coords, child.spec.shape))
                factor = "1.0"
            else:
                domains = {}
                for child, labels in zip(node.inputs, a["labels"], strict=True):
                    domains.update(zip(labels, child.spec.shape, strict=True))
                reduced = tuple(k for k in sorted(domains) if k not in a["output"])
                rs = tuple(domains[k] for k in reduced)
                mapping = dict(zip(a["output"], c, strict=True))
                mapping.update(
                    (k, _coordinate("r", rs, axis)) for axis, k in enumerate(reduced)
                )
                term = " * ".join(
                    read(child, _flat([mapping[k] for k in labels], child.spec.shape))
                    for child, labels in zip(node.inputs, a["labels"], strict=True)
                )
                factor = _literal(a["coefficient"])
            body += [
                "double value = 0.0;",
                f"for (I r = 0; r < {prod(rs)}LL; ++r) value += {term};",
                "if (!std::isfinite(value)) return false;",
            ]
            expression = f"value * {factor}"
        else:
            child = node.inputs[0]
            source_shape = child.spec.shape
            if node.op == "reshape":
                index = "z"
            elif node.op == "transpose":
                source = [c[a["axes"].index(axis)] for axis in range(len(source_shape))]
                index = _flat(source, source_shape)
            elif node.op == "slice":
                index = _flat(
                    [
                        f"({coord} + {start}LL)"
                        for coord, (start, _) in zip(c, a["ranges"], strict=True)
                    ],
                    source_shape,
                )
            elif node.op == "broadcast":
                index = _flat([c[axis] for axis in a["axes"]], source_shape)
            elif node.op in {"gather", "indexed_gather"}:
                values = ", ".join(f"{value}LL" for value in a["positions"]) or "0LL"
                lines.append(f"static const I index_{i}[] = {{{values}}};")
                mapped = list(c)
                mapped[a["axis"]] = f"index_{i}[{c[a['axis']]}]"
                index = _flat(mapped, source_shape)
            elif node.op == "scatter_add":
                axis = a["axis"]
                if not source_shape[axis]:
                    expression = "0.0"
                    index = None
                else:
                    table = scatter_add_inverted_table(node)
                    values = ", ".join(f"{value}LL" for value in table)
                    lines.append(f"static const I index_{i}[] = {{{values}}};")
                    target = c[axis]
                    source = list(c)
                    source[axis] = "r"
                    body += [
                        "double value = 0.0;",
                        f"const I begin = index_{i}[{target}];",
                        f"const I end = index_{i}[{target} + 1];",
                        "for (I q = begin; q < end; ++q) {",
                        f"  const I r = index_{i}[{shape[axis] + 1}LL + q];",
                        f"  value += {read(child, _flat(source, source_shape))};",
                        "}",
                    ]
                    expression = "value"
                    index = None
            elif node.op == "segment_sum":
                axis = a["axis"]
                values = ", ".join(f"{value}LL" for value in a["offsets"])
                lines.append(f"static const I index_{i}[] = {{{values}}};")
                segment = c[axis]
                source = list(c)
                source[axis] = "r"
                body += [
                    "double value = 0.0;",
                    f"const I begin = index_{i}[{segment}];",
                    f"const I end = index_{i}[{segment} + 1];",
                    f"for (I r = begin; r < end; ++r) value += {read(child, _flat(source, source_shape))};",
                ]
                expression = "value"
                index = None
            else:
                raise ValueError(f"unsupported CPU primitive: {node.op}")
            if index is not None:
                expression = read(child, index)
        lines += [
            f"for (I z = 0; z < {node.spec.size}LL; ++z) {{",
            *body,
            f"{read(node)} = {expression};",
            f"if (!std::isfinite({read(node)})) return false;",
            "}",
        ]
    for node in program.outputs.values():
        if node.spec.size:
            lines.append(
                f"std::copy_n(p + {offsets[node]}, {node.spec.size}, p + {cursor});"
            )
        cursor += node.spec.size
    lines += ["return true; });", "}"]
    source = (
        "\n".join(lines).replace(
            '#include "cpu_runtime.hpp"',
            '#include "cpu_runtime.hpp"\nusing I = long long;',
        )
        + "\n"
    )
    return source, {
        "input_count": ni,
        "output_count": no,
        "required_bytes": required,
        "scalar_work": work,
    }


class NativeTensorProgram:
    """Compiled CPU-only program, with checked feeds and transactional outputs.

    ``compiler`` accepts an adapter or a zero-argument adapter factory. The
    factory runs only after bounded lowering succeeds, allowing public JIT
    callers to reject unsupported IR without discovering a local toolchain.
    Existing callers may continue supplying an already constructed adapter.
    """

    def __init__(
        self,
        program: typing.Any,
        *,
        compiler: typing.Any,
        cache: typing.Any,
        max_bytes: typing.Any = 8 * 1024 * 1024,
        max_work: typing.Any = 100_000_000,
        max_nodes: typing.Any = 4096,
        symbol: str = "tensor_cpu",
    ) -> None:
        source, self.resources = emit_cpu(
            program,
            max_bytes=max_bytes,
            max_work=max_work,
            max_nodes=max_nodes,
            symbol=symbol,
        )
        if callable(compiler):
            compiler = compiler()
        if not isinstance(compiler, CppCompilerAdapter):
            raise TypeError("native TensorIR requires a CPU compiler adapter")
        self.program, self.max_bytes = program, max_bytes
        self.inputs = tuple(n for n in program.live_nodes if n.op == "input")
        self.identity = canonical_hash(
            {"program": program.to_payload(), "source": source}
        )
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (self.identity + ".cpp")
        cache_source(path, source)
        header = asset_path("src/tensor/cpu_runtime.hpp")
        self.artifact = compile_runtime(
            compiler,
            cache,
            path,
            headers=(header,),
            options=("-ffp-contract=off", f"-I{header.parent}"),
        )
        self.library = ct.CDLL(str(self.artifact.library))
        self.call = getattr(self.library, symbol)
        self.call.argtypes = [
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.c_size_t,
        ]
        self.call.restype = ct.c_int

    def execute(self, feeds: typing.Any) -> typing.Any:
        """Reject wrong shapes/dtypes/NaNs before calling the checked native ABI."""
        if not isinstance(feeds, Mapping):
            raise TypeError("CPU tensor feeds must be a mapping")
        values = []
        for node in self.inputs:
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
        packed = np.concatenate(values) if values else np.empty(0)
        output = np.empty(self.resources["output_count"])
        code = self.call(
            packed.ctypes.data_as(ct.POINTER(ct.c_double)),
            packed.size,
            output.ctypes.data_as(ct.POINTER(ct.c_double)),
            output.size,
            self.max_bytes,
        )
        if code:
            raise ValueError(f"native CPU tensor evaluation failed ({code})")
        result, cursor = {}, 0
        for name, node in self.program.outputs.items():
            result[name] = immutable(
                output[cursor : cursor + node.spec.size].reshape(node.spec.shape)
            )
            cursor += node.spec.size
        return result
