"""Generate runtime-shape native RCCSD evaluators from the audited TensorIR.

The emitted CPU and CUDA programs are a productization bridge for #149 C.  They
consume the same #148 physical residual DAGs as the Python validation path; the
only specialization removed here is the concrete occupied/virtual extent.
"""

from __future__ import annotations

import argparse
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "python"))

# Build-time generation must not execute the NumPy-backed TensorIR facade or
# tools.vibeqc_cc.__init__.  Load only the immutable IR/type/program modules
# needed to build #148's algebra, then expose their small public surface to the
# equation modules.  The optimized/interpreter/packing paths are never entered
# by this AOT generator (shared + expanded forms only).
import types
from fractions import Fraction

import vibeqc_compiler

_tensor_path = ROOT / "python" / "vibeqc_compiler" / "tensor"
_tensor_package = types.ModuleType("vibeqc_compiler.tensor")
_tensor_package.__path__ = [str(_tensor_path)]
_tensor_package.__package__ = "vibeqc_compiler.tensor"
sys.modules["vibeqc_compiler.tensor"] = _tensor_package
vibeqc_compiler.tensor = _tensor_package

from vibeqc_compiler.tensor.ir import add, divide, einsum, input_tensor
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import Index, IndexSpace, Symmetry, TensorSpec


class _BuildOnlyPackedLayout:
    @classmethod
    def from_spec(cls, *_args: typing.Any, **_kwargs: typing.Any) -> typing.NoReturn:
        raise RuntimeError(
            "packed layouts are not part of RCCSD AOT equation generation"
        )


for _name, _value in {
    "Index": Index,
    "IndexSpace": IndexSpace,
    "Symmetry": Symmetry,
    "TensorSpec": TensorSpec,
    "Program": Program,
    "add": add,
    "divide": divide,
    "einsum": einsum,
    "input_tensor": input_tensor,
    "PackedLayout": _BuildOnlyPackedLayout,
    "optimize": lambda program: program,
}.items():
    setattr(_tensor_package, _name, _value)

# The CC equation modules only need canonical_hash from the validation facade;
# providing it here avoids importing NumPy-backed evidence comparison helpers.
import hashlib
import json

_validation_schema = types.ModuleType("tools.vibeqc_validation.schema")


def _canonical_hash(value: typing.Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


_validation_schema.canonical_hash = _canonical_hash
sys.modules["tools.vibeqc_validation.schema"] = _validation_schema

_cc_path = Path(__file__).resolve().parent / "vibeqc_cc"
_cc_package = types.ModuleType("tools.vibeqc_cc")
_cc_package.__path__ = [str(_cc_path)]
_cc_package.__package__ = "tools.vibeqc_cc"
sys.modules.setdefault("tools.vibeqc_cc", _cc_package)

from tools.vibeqc_cc.doubles import build_ccsd_program

REPRESENTATIVE = (2, 3)


def iteration_program(nocc: int, nvir: int) -> Program:
    """Generated physical R plus the undamped Jacobi trial.

    Damping is a runtime solver control applied outside this TensorIR so one
    AOT program serves every legal damping value without changing R itself.
    """
    physical = build_ccsd_program(nocc, nvir, form="shared", diagnostics=False)
    inputs = {n.attrs["name"]: n for n in physical.live_nodes if n.op == "input"}
    outputs = dict(physical.outputs)
    for index, residual in enumerate(("singles_residual", "doubles_residual"), 1):
        t = inputs[f"t{index}"]
        d = input_tensor(f"d{index}", t.spec)
        outputs[f"next_t{index}"] = add(
            t,
            divide(physical.outputs[residual], d),
            coefficients=(1, Fraction(1)),
        )
    return Program(
        outputs,
        provenance={
            "physical_equation": physical.logical_hash,
            "iteration": "undamped Jacobi; runtime damping is a control",
        },
    )


INPUT_NAMES = (
    "foo",
    "fov",
    "fvv",
    "ovov",
    "ovvo",
    "oovv",
    "ovvv",
    "ovoo",
    "oooo",
    "vvvv",
    "d1",
    "d2",
    "t1",
    "t2",
)


def _kind(index: Index) -> str:
    kind = index.space.kind
    if kind not in ("occupied", "virtual") or index.start != 0:
        raise ValueError(f"unsupported RCCSD index domain: {index}")
    return kind


def _dim(index: Index) -> str:
    return "o" if _kind(index) == "occupied" else "v"


def _size(spec: TensorSpec) -> str:
    if not spec.indices:
        return "1"
    return "checked_product({" + ",".join(_dim(i) for i in spec.indices) + "})"


def _device_size(spec: TensorSpec) -> str:
    if not spec.indices:
        return "1"
    return "*".join(_dim(i) for i in spec.indices)


def _fraction(value: tuple[int, int]) -> str:
    num, den = value
    if den == 1:
        return f"{num}.0"
    return f"({num}.0/{den}.0)"


def _input_access(name: str, *, cuda: bool = False) -> str:
    if name not in INPUT_NAMES:
        raise ValueError(f"unexpected RCCSD input {name!r}")
    return f"s.{name}" if cuda else f"inputs.{name}"


def _label_kinds(node: typing.Any) -> dict[typing.Any, str]:
    result = {}
    for operand, labels in zip(node.inputs, node.attrs["labels"]):
        for index, label in zip(operand.spec.indices, labels):
            kind = _kind(index)
            previous = result.setdefault(label, kind)
            if previous != kind:
                raise ValueError("einsum label crosses occupied/virtual spaces")
    return result


def _flat_index(labels: tuple[typing.Any, ...], spec: TensorSpec) -> str:
    if not labels:
        return "0"
    expression = f"l{labels[0]}"
    for label, index in zip(labels[1:], spec.indices[1:]):
        expression = f"({expression}*{_dim(index)}+l{label})"
    return expression


def _cpu_node(node: typing.Any, number: int, names: dict[int, str]) -> list[str]:
    out = names[number]
    size = _size(node.spec)
    lines = [f"  double* {out}=allocate({size});"]
    if node.op == "add":
        terms = []
        for source, coefficient in zip(node.inputs, node.attrs["coefficients"]):
            terms.append(f"{_fraction(coefficient)}*{names[source._emit_index]}[i]")
        lines += [
            f"  for(std::size_t i=0;i<{size};++i){{",
            f"    const double value={' + '.join(terms)};",
            '    if(!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD add");',
            f"    {out}[i]=value;",
            "  }",
        ]
    elif node.op == "divide":
        a, b = (names[x._emit_index] for x in node.inputs)
        lines += [
            f"  for(std::size_t i=0;i<{size};++i){{",
            f'    if({b}[i]==0.0) throw std::runtime_error("zero RCCSD denominator");',
            f"    const double value={a}[i]/{b}[i];",
            '    if(!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD divide");',
            f"    {out}[i]=value;",
            "  }",
        ]
    elif node.op == "einsum":
        labels = node.attrs["labels"]
        output = tuple(node.attrs["output"])
        kinds = _label_kinds(node)
        all_labels = sorted(kinds)
        reduced = [label for label in all_labels if label not in output]
        lines.append(f"  for(std::size_t flat=0;flat<{size};++flat){{")
        if output:
            lines.append("    std::size_t rem=flat;")
        for label in reversed(output):
            dim = "o" if kinds[label] == "occupied" else "v"
            lines += [f"    const std::size_t l{label}=rem%{dim};", f"    rem/={dim};"]
        lines += ["    double sum=0.0;"]
        indent = "    "
        for label in reduced:
            dim = "o" if kinds[label] == "occupied" else "v"
            lines.append(
                f"{indent}for(std::size_t l{label}=0;l{label}<{dim};++l{label}){{"
            )
            indent += "  "
        factors = []
        for source, source_labels in zip(node.inputs, labels):
            factors.append(
                f"{names[source._emit_index]}[{_flat_index(tuple(source_labels), source.spec)}]"
            )
        lines.append(f"{indent}sum += {' * '.join(factors)};")
        for _ in reduced:
            indent = indent[:-2]
            lines.append(f"{indent}}}")
        lines += [
            f"    const double value={_fraction(node.attrs['coefficient'])}*sum;",
            '    if(!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD einsum");',
            f"    {out}[flat]=value;",
            "  }",
        ]
    else:
        raise ValueError(f"unsupported native RCCSD CPU op {node.op}")
    return lines


def _prepare_program(program: Program) -> dict[int, str]:
    names = {}
    for number, node in enumerate(program.live_nodes):
        object.__setattr__(node, "_emit_index", number)
        names[number] = f"n{number}"
    return names


def _cpu_function(program: Program, function_name: str, output_type: str) -> str:
    names = _prepare_program(program)
    lines = [
        f"inline {output_type} {function_name}(std::size_t o,std::size_t v,const Inputs& inputs,double* arena,std::size_t arena_elements){{",
        "  std::size_t cursor=0;",
        "  auto allocate=[&](std::size_t count)->double*{",
        "    const auto next=checked_add(cursor,count);",
        '    if(next>arena_elements) throw std::length_error("RCCSD generated CPU arena is too small");',
        "    double* result=arena+cursor; cursor=next; return result;",
        "  };",
    ]
    for number, node in enumerate(program.live_nodes):
        if node.op == "input":
            lines.append(
                f"  const double* {names[number]}={_input_access(node.attrs['name'])};"
            )
        else:
            lines += _cpu_node(node, number, names)
    outputs = {key: names[value._emit_index] for key, value in program.outputs.items()}
    if output_type == "IterationOutputs":
        lines.append(
            "  return {"
            + ",".join(
                [
                    f"*{outputs['correlation_energy']}",
                    outputs["singles_residual"],
                    outputs["doubles_residual"],
                    outputs["next_t1"],
                    outputs["next_t2"],
                ]
            )
            + "};"
        )
    else:
        lines.append(
            "  return {"
            + ",".join(
                [
                    f"*{outputs['correlation_energy']}",
                    outputs["singles_residual"],
                    outputs["doubles_residual"],
                ]
            )
            + "};"
        )
    lines.append("}")
    return "\n".join(lines)


def _required_function(program: Program, name: str) -> str:
    _prepare_program(program)
    pieces = [_size(node.spec) for node in program.live_nodes if node.op != "input"]
    body = "0"
    for piece in pieces:
        body = f"checked_add({body},{piece})"
    return f"inline std::size_t {name}(std::size_t o,std::size_t v){{return {body};}}"


def cpu_header() -> str:
    iteration = iteration_program(*REPRESENTATIVE)
    replay = build_ccsd_program(*REPRESENTATIVE, form="expanded", diagnostics=False)
    return "\n".join(
        [
            "// Generated by tools/generate_rccsd_native.py from #148 TensorIR.",
            "#pragma once",
            "#include <cmath>",
            "#include <cstddef>",
            "#include <initializer_list>",
            "#include <limits>",
            "#include <stdexcept>",
            "namespace vibeqc::cc::generated {",
            'inline std::size_t checked_add(std::size_t a,std::size_t b){if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("RCCSD size overflow");return a+b;}',
            'inline std::size_t checked_product(std::initializer_list<std::size_t> values){std::size_t x=1;for(auto v:values){if(v&&x>std::numeric_limits<std::size_t>::max()/v)throw std::length_error("RCCSD size overflow");x*=v;}return x;}',
            "struct Inputs {",
            *[f"  const double* {name}{{}};" for name in INPUT_NAMES],
            "};",
            "struct IterationOutputs { double energy{}; const double* r1{}; const double* r2{}; const double* next_t1{}; const double* next_t2{}; };",
            "struct ReplayOutputs { double energy{}; const double* r1{}; const double* r2{}; };",
            f'inline constexpr const char* iteration_equation_hash="{iteration.provenance["physical_equation"]}";',
            f'inline constexpr const char* iteration_program_hash="{iteration.logical_hash}";',
            f'inline constexpr const char* replay_equation_hash="{replay.logical_hash}";',
            _required_function(iteration, "iteration_arena_elements"),
            _required_function(replay, "replay_arena_elements"),
            _cpu_function(iteration, "run_iteration_cpu", "IterationOutputs"),
            _cpu_function(replay, "run_replay_cpu", "ReplayOutputs"),
            "}",
            "",
        ]
    )


def _cuda_kernel(
    node: typing.Any, number: int, prefix: str, names: dict[int, str]
) -> str:
    size = _device_size(node.spec)
    arguments = [f"const double* a{i}" for i in range(len(node.inputs))]
    arguments += ["double* out", "std::size_t o", "std::size_t v", "int* error"]
    lines = [
        f"__global__ void {prefix}_node_{number}({','.join(arguments)}){{",
        f"  const std::size_t count={size};",
        "  for(std::size_t flat=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;flat<count;flat+=std::size_t(blockDim.x)*gridDim.x){",
    ]
    if node.op == "add":
        terms = [
            f"{_fraction(c)}*a{i}[flat]"
            for i, c in enumerate(node.attrs["coefficients"])
        ]
        lines += [
            f"    const double value={' + '.join(terms)};",
            f"    out[flat]=vibeqc_tensor::finite(value,error,{number});",
        ]
    elif node.op == "divide":
        lines += [
            f"    out[flat]=vibeqc_tensor::quotient(a0[flat],a1[flat],error,{number});"
        ]
    elif node.op == "einsum":
        labels = node.attrs["labels"]
        output = tuple(node.attrs["output"])
        kinds = _label_kinds(node)
        all_labels = sorted(kinds)
        reduced = [label for label in all_labels if label not in output]
        if output:
            lines.append("    std::size_t rem=flat;")
        for label in reversed(output):
            dim = "o" if kinds[label] == "occupied" else "v"
            lines += [f"    const std::size_t l{label}=rem%{dim};", f"    rem/={dim};"]
        lines.append("    double sum=0.0;")
        indent = "    "
        for label in reduced:
            dim = "o" if kinds[label] == "occupied" else "v"
            lines.append(
                f"{indent}for(std::size_t l{label}=0;l{label}<{dim};++l{label}){{"
            )
            indent += "  "
        factors = []
        for i, (source, source_labels) in enumerate(zip(node.inputs, labels)):
            factors.append(f"a{i}[{_flat_index(tuple(source_labels), source.spec)}]")
        product = factors[0]
        for factor in factors[1:]:
            product = f"__dmul_rn({product},{factor})"
        lines.append(f"{indent}sum=__dadd_rn(sum,{product});")
        for _ in reduced:
            indent = indent[:-2]
            lines.append(f"{indent}}}")
        coefficient = _fraction(node.attrs["coefficient"])
        lines += [
            f"    const double value=__dmul_rn({coefficient},sum);",
            f"    out[flat]=vibeqc_tensor::finite(value,error,{number});",
        ]
    else:
        raise ValueError(f"unsupported native RCCSD CUDA op {node.op}")
    lines += ["  }", "}"]
    return "\n".join(lines)


def _cuda_program(program: Program, prefix: str, output_type: str) -> str:
    names = _prepare_program(program)
    kernels = []
    for number, node in enumerate(program.live_nodes):
        if node.op != "input":
            kernels.append(_cuda_kernel(node, number, prefix, names))
    lines = kernels + [
        f"static {output_type} run_{prefix}(CudaState& s){{",
        "  auto* arena=s."
        + ("iteration_arena" if prefix == "iteration" else "replay_arena")
        + ";",
        "  const auto o=s.o,v=s.v;",
        "  std::size_t cursor=0;",
        "  auto allocate=[&](std::size_t count)->double*{double* p=arena+cursor;cursor=checked_add(cursor,count);return p;};",
        "  vibeqc_tensor::cuda_check(cudaMemsetAsync(s.error,0,sizeof(int),s.stream));",
    ]
    for number, node in enumerate(program.live_nodes):
        if node.op == "input":
            lines.append(
                f"  const double* {names[number]}={_input_access(node.attrs['name'], cuda=True)};"
            )
            continue
        lines.append(f"  double* {names[number]}=allocate({_size(node.spec)});")
        sources = [names[x._emit_index] for x in node.inputs]
        count = _size(node.spec)
        launch_args = ",".join([*sources, names[number], "s.o", "s.v", "s.error"])
        lines += [
            f"  {prefix}_node_{number}<<<vibeqc_tensor::blocks(static_cast<vibeqc_tensor::I>({count}),256),256,0,s.stream>>>({launch_args});",
        ]
    lines.append("  vibeqc_tensor::cuda_check(cudaGetLastError());")
    outputs = {key: names[value._emit_index] for key, value in program.outputs.items()}
    if output_type == "DeviceIterationOutputs":
        lines.append(
            "  return {"
            + ",".join(
                [
                    outputs["correlation_energy"],
                    outputs["singles_residual"],
                    outputs["doubles_residual"],
                    outputs["next_t1"],
                    outputs["next_t2"],
                ]
            )
            + "};"
        )
    else:
        lines.append(
            "  return {"
            + ",".join(
                [
                    outputs["correlation_energy"],
                    outputs["singles_residual"],
                    outputs["doubles_residual"],
                ]
            )
            + "};"
        )
    lines.append("}")
    return "\n".join(lines)


def cuda_source() -> str:
    iteration = iteration_program(*REPRESENTATIVE)
    replay = build_ccsd_program(*REPRESENTATIVE, form="expanded", diagnostics=False)
    return "\n".join(
        [
            "// Generated by tools/generate_rccsd_native.py from #148 TensorIR.",
            '#include "cc/cuda_solver_support.cuh"',
            '#include "generated_rccsd_cpu.hpp"',
            "namespace vibeqc::cc::generated {",
            _cuda_program(iteration, "iteration", "DeviceIterationOutputs"),
            _cuda_program(replay, "replay", "DeviceReplayOutputs"),
            "DeviceIterationOutputs run_iteration_cuda(CudaState& state){return run_iteration(state);}",
            "DeviceReplayOutputs run_replay_cuda(CudaState& state){return run_replay(state);}",
            "}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-header", type=Path)
    parser.add_argument("--cuda-source", type=Path)
    args = parser.parse_args()
    if args.cpu_header:
        args.cpu_header.parent.mkdir(parents=True, exist_ok=True)
        args.cpu_header.write_text(cpu_header(), encoding="utf-8")
    if args.cuda_source:
        args.cuda_source.parent.mkdir(parents=True, exist_ok=True)
        args.cuda_source.write_text(cuda_source(), encoding="utf-8")
    if not args.cpu_header and not args.cuda_source:
        parser.error("select --cpu-header and/or --cuda-source")


if __name__ == "__main__":
    main()
