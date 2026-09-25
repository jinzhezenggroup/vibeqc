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

from vibeqc_compiler.tensor.ad_program import (
    JVPProgram,
    VJPProgram,
    linearize,
    transpose_program,
)
from vibeqc_compiler.tensor.ir import (
    add,
    broadcast,
    constant,
    divide,
    einsum,
    gather,
    input_tensor,
    multiply,
    reduce_sum,
    runtime_indexed_select,
    slice_tensor,
    transpose,
)
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
    "broadcast": broadcast,
    "constant": constant,
    "divide": divide,
    "einsum": einsum,
    "execute": lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("build-only RCCSD generator does not execute TensorIR")
    ),
    "gather": gather,
    "input_tensor": input_tensor,
    "multiply": multiply,
    "reduce_sum": reduce_sum,
    "runtime_indexed_select": runtime_indexed_select,
    "slice_tensor": slice_tensor,
    "transpose": transpose,
    "PackedLayout": _BuildOnlyPackedLayout,
    "JVPProgram": JVPProgram,
    "VJPProgram": VJPProgram,
    "linearize": linearize,
    "transpose_program": transpose_program,
    "optimize": lambda program: program,
}.items():
    setattr(_tensor_package, _name, _value)

# CC equation modules import the real canonical evidence module. Its numerical
# comparison routines load NumPy only when executed; no module replacement is
# required for immutable AOT equation construction.

_cc_path = Path(__file__).resolve().parent / "vibeqc_cc"
_cc_package = types.ModuleType("tools.vibeqc_cc")
_cc_package.__path__ = [str(_cc_path)]
_cc_package.__package__ = "tools.vibeqc_cc"
sys.modules.setdefault("tools.vibeqc_cc", _cc_package)

from tools.vibeqc_cc.doubles import build_ccsd_program
from tools.vibeqc_cc.gradient_equations import (
    build_fock_weight_program,
    build_hamiltonian_programs,
)
from tools.vibeqc_cc.lambda_equations import (
    PARAMETERS,
    build_lambda_programs,
    build_parameter_vjp,
)
from tools.vibeqc_cc.triples_tiles import build_runtime_tile_triples_program

REPRESENTATIVE = (2, 3)
REPRESENTATIVE_ORBITALS = sum(REPRESENTATIVE)
TRIPLES_RESPONSE_INPUTS = (
    "ovvv",
    "ovoo",
    "ovov",
    "fov",
    "t1",
    "t2",
    "eps_o",
    "eps_v",
)


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
    if index.space.kind in ("occupied", "virtual"):
        return "o" if _kind(index) == "occupied" else "v"
    if index.space.kind == "batch":
        return "q"
    if index.space.kind == "orbital" and index.space.size == REPRESENTATIVE_ORBITALS:
        bounds = (index.start, index.stop)
        if bounds == (0, REPRESENTATIVE[0]):
            return "o"
        if bounds == (REPRESENTATIVE[0], REPRESENTATIVE_ORBITALS):
            return "v"
        if bounds == (0, REPRESENTATIVE_ORBITALS):
            return "n"
    raise ValueError(f"unsupported runtime-shape RCCSD index domain: {index}")


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


def _label_dims(node: typing.Any) -> dict[typing.Any, str]:
    result = {}
    for operand, labels in zip(node.inputs, node.attrs["labels"]):
        for index, label in zip(operand.spec.indices, labels):
            dim = _dim(index)
            previous = result.setdefault(label, dim)
            if previous != dim:
                raise ValueError("einsum label crosses incompatible runtime domains")
    return result


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


def _flat_coords(coords: list[str], spec: TensorSpec) -> str:
    if not coords:
        return "0"
    expression = coords[0]
    for coord, index in zip(coords[1:], spec.indices[1:]):
        expression = f"({expression}*{_dim(index)}+{coord})"
    return expression


def _runtime_bound(value: int) -> str:
    if value == 0:
        return "0"
    if value == REPRESENTATIVE[0]:
        return "o"
    if value == REPRESENTATIVE_ORBITALS:
        return "n"
    raise ValueError(f"unsupported runtime RCCSD slice boundary {value}")


def _scaled_bilinear_cpp() -> str:
    return r"""inline bool generated_scaled_bilinear(
    double a,double b,double c,double d,double e,double f,double& out){
  if(e==0.0||f==0.0) return false;
  int ea,eb,ec,ed,ee,ef;
  const double ma=std::frexp(a,&ea), mb=std::frexp(b,&eb);
  const double mc=std::frexp(c,&ec), md=std::frexp(d,&ed);
  const double me=std::frexp(e,&ee), mf=std::frexp(f,&ef);
  double p=ma*mb,q=mc*md,pe=std::fma(ma,mb,-p),qe=std::fma(mc,md,-q);
  const int ep=ea+eb,eq=ec+ed,exponent=p==0.0?eq:(q==0.0?ep:std::max(ep,eq));
  constexpr int limit=110;
  const int dp=ep-exponent,dq=eq-exponent;
  if(dp < -limit){p=0.0;pe=0.0;} else {p=std::scalbn(p,dp);pe=std::scalbn(pe,dp);}
  if(dq < -limit){q=0.0;qe=0.0;} else {q=std::scalbn(q,dq);qe=std::scalbn(qe,dq);}
  const double difference=p-q,tail=difference-p;
  const double residual=(p-(difference-tail))-(q+tail);
  const double numerator=difference+((pe-qe)+residual);
  out=std::scalbn(numerator/(me*mf),exponent-ee-ef);
  return std::isfinite(out);
}"""


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
    elif node.op == "multiply":
        a, b = (names[x._emit_index] for x in node.inputs)
        lines += [
            f"  for(std::size_t i=0;i<{size};++i){{",
            f"    const double value={a}[i]*{b}[i];",
            '    if(!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD multiply");',
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
    elif node.op == "scaled_bilinear":
        args = [names[x._emit_index] for x in node.inputs]
        lines += [
            f"  for(std::size_t i=0;i<{size};++i){{",
            "    double value=0.0;",
            (
                f"    if(!generated_scaled_bilinear({','.join(f'{arg}[i]' for arg in args)},value)) "
                'throw std::runtime_error("nonfinite RCCSD scaled_bilinear");'
            ),
            f"    {out}[i]=value;",
            "  }",
        ]
    elif node.op == "einsum":
        labels = node.attrs["labels"]
        output = tuple(node.attrs["output"])
        dims = _label_dims(node)
        all_labels = sorted(dims)
        reduced = [label for label in all_labels if label not in output]
        lines.append(f"  for(std::size_t flat=0;flat<{size};++flat){{")
        if output:
            lines.append("    std::size_t rem=flat;")
        for label in reversed(output):
            dim = dims[label]
            lines += [f"    const std::size_t l{label}=rem%{dim};", f"    rem/={dim};"]
        lines += ["    double sum=0.0;"]
        indent = "    "
        for label in reduced:
            dim = dims[label]
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
    elif node.op == "broadcast":
        source = node.inputs[0]
        axes = tuple(node.attrs["axes"])
        lines += [
            f"  for(std::size_t flat=0;flat<{size};++flat){{",
            "    std::size_t rem=flat;",
        ]
        coords = [""] * len(node.spec.indices)
        for axis in reversed(range(len(node.spec.indices))):
            dim = _dim(node.spec.indices[axis])
            lines += [f"    const std::size_t c{axis}=rem%{dim};", f"    rem/={dim};"]
            coords[axis] = f"c{axis}"
        source_index = _flat_coords([coords[axis] for axis in axes], source.spec)
        lines += [
            f"    const double value={names[source._emit_index]}[{source_index}];",
            f"    {out}[flat]=value;",
            "  }",
        ]
    elif node.op == "reduce":
        source = node.inputs[0]
        reduced = set(node.attrs["axes"])
        source_size = _size(source.spec)
        lines.append(f"  std::fill_n({out},{size},0.0);")
        lines += [
            f"  for(std::size_t flat=0;flat<{source_size};++flat){{",
            "    std::size_t rem=flat;",
        ]
        source_coords = [""] * len(source.spec.indices)
        for axis in reversed(range(len(source.spec.indices))):
            dim = _dim(source.spec.indices[axis])
            lines += [f"    const std::size_t c{axis}=rem%{dim};", f"    rem/={dim};"]
            source_coords[axis] = f"c{axis}"
        kept = [
            coord for axis, coord in enumerate(source_coords) if axis not in reduced
        ]
        target_index = _flat_coords(kept, node.spec)
        lines += [
            f"    {out}[{target_index}]+={names[source._emit_index]}[flat];",
            "  }",
        ]
    elif node.op == "runtime_indexed_select":
        source = node.inputs[0]
        maps = node.inputs[1:]
        axes = tuple(node.attrs["axes"])
        selected = dict(zip(axes, maps, strict=True))
        lines += [
            f"  for(std::size_t flat=0;flat<{size};++flat){{",
            "    std::size_t rem=flat;",
        ]
        out_coords = [""] * len(node.spec.indices)
        for axis in reversed(range(len(node.spec.indices))):
            dim = _dim(node.spec.indices[axis])
            lines += [f"    const std::size_t c{axis}=rem%{dim};", f"    rem/={dim};"]
            out_coords[axis] = f"c{axis}"
        source_coords = []
        remaining = iter(out_coords[1:])
        for axis, index in enumerate(source.spec.indices):
            if axis in selected:
                map_name = names[selected[axis]._emit_index]
                dim = _dim(index)
                lines.append(f"    const auto m{axis}={map_name}[c0];")
                lines.append(
                    f'    if(m{axis}<0||static_cast<std::size_t>(m{axis})>={dim}) throw std::runtime_error("runtime triples index out of bounds");'
                )
                source_coords.append(f"static_cast<std::size_t>(m{axis})")
            else:
                source_coords.append(next(remaining))
        source_index = _flat_coords(source_coords, source.spec)
        lines += [
            f"    {out}[flat]={names[source._emit_index]}[{source_index}];",
            "  }",
        ]
    elif node.op == "runtime_indexed_scatter_add":
        source = node.inputs[0]
        maps = node.inputs[1:]
        axes = tuple(node.attrs["axes"])
        selected = dict(zip(axes, maps, strict=True))
        source_size = _size(source.spec)
        lines.append(f"  std::fill_n({out},{size},0.0);")
        lines += [
            f"  for(std::size_t flat=0;flat<{source_size};++flat){{",
            "    std::size_t rem=flat;",
        ]
        source_coords = [""] * len(source.spec.indices)
        for axis in reversed(range(len(source.spec.indices))):
            dim = _dim(source.spec.indices[axis])
            lines += [f"    const std::size_t c{axis}=rem%{dim};", f"    rem/={dim};"]
            source_coords[axis] = f"c{axis}"
        target_coords = []
        remaining = iter(source_coords[1:])
        for axis, index in enumerate(node.spec.indices):
            if axis in selected:
                map_name = names[selected[axis]._emit_index]
                dim = _dim(index)
                lines.append(f"    const auto m{axis}={map_name}[c0];")
                lines.append(
                    f'    if(m{axis}<0||static_cast<std::size_t>(m{axis})>={dim}) throw std::runtime_error("runtime triples scatter index out of bounds");'
                )
                target_coords.append(f"static_cast<std::size_t>(m{axis})")
            else:
                target_coords.append(next(remaining))
        target_index = _flat_coords(target_coords, node.spec)
        lines += [
            f"    {out}[{target_index}]+={names[source._emit_index]}[flat];",
            "  }",
        ]
    elif node.op == "slice":
        source = node.inputs[0]
        ranges = tuple(node.attrs["ranges"])
        rank = len(node.spec.indices)
        lines += [
            f"  for(std::size_t flat=0;flat<{size};++flat){{",
            "    std::size_t rem=flat;",
        ]
        coords = [""] * rank
        for axis in reversed(range(rank)):
            dim = _dim(node.spec.indices[axis])
            lines += [f"    const std::size_t c{axis}=rem%{dim};", f"    rem/={dim};"]
            coords[axis] = f"(c{axis}+{_runtime_bound(ranges[axis][0])})"
        source_index = _flat_coords(coords, source.spec)
        lines += [
            f"    const double value={names[source._emit_index]}[{source_index}];",
            '    if(!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD slice");',
            f"    {out}[flat]=value;",
            "  }",
        ]
    elif node.op == "scatter_add":
        source = node.inputs[0]
        axis = node.attrs["axis"]
        positions = tuple(node.attrs["positions"])
        if not positions or positions != tuple(range(positions[0], positions[-1] + 1)):
            raise ValueError("runtime RCCSD scatter requires contiguous positions")
        offset = _runtime_bound(positions[0])
        source_size = _size(source.spec)
        lines.append(f"  std::fill_n({out},{size},0.0);")
        lines += [
            f"  for(std::size_t flat=0;flat<{source_size};++flat){{",
            "    std::size_t rem=flat;",
        ]
        coords = [""] * len(source.spec.indices)
        for source_axis in reversed(range(len(source.spec.indices))):
            dim = _dim(source.spec.indices[source_axis])
            lines += [
                f"    const std::size_t c{source_axis}=rem%{dim};",
                f"    rem/={dim};",
            ]
            coords[source_axis] = (
                f"(c{source_axis}+{offset})"
                if source_axis == axis
                else f"c{source_axis}"
            )
        target_index = _flat_coords(coords, node.spec)
        lines += [
            f"    {out}[{target_index}]+={names[source._emit_index]}[flat];",
            "  }",
        ]
    elif node.op == "transpose":
        source = node.inputs[0]
        rank = len(node.spec.indices)
        axes = tuple(node.attrs["axes"])
        if len(axes) != rank or sorted(axes) != list(range(rank)):
            raise ValueError("invalid native RCCSD transpose permutation")
        source_coords: list[str | None] = [None] * rank
        lines += [
            f"  for(std::size_t flat=0;flat<{size};++flat){{",
            "    std::size_t rem=flat;",
        ]
        for axis in reversed(range(rank)):
            dim = _dim(node.spec.indices[axis])
            lines += [
                f"    const std::size_t c{axis}=rem%{dim};",
                f"    rem/={dim};",
            ]
        for out_axis, source_axis in enumerate(axes):
            source_coords[source_axis] = f"c{out_axis}"
        if any(coord is None for coord in source_coords):
            raise ValueError("invalid native RCCSD transpose coordinate map")
        index = typing.cast("list[str]", source_coords)[0] if source_coords else "0"
        for coord, spec_index in zip(source_coords[1:], source.spec.indices[1:]):
            index = f"({index}*{_dim(spec_index)}+{coord})"
        lines += [
            f"    const double value={names[source._emit_index]}[{index}];",
            '    if(!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD transpose");',
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


def _cpu_function(
    program: Program,
    function_name: str,
    output_type: str,
    *,
    signature: str = "const Inputs& inputs",
    input_overrides: dict[str, str] | None = None,
    batch_dim: bool = False,
) -> str:
    names = _prepare_program(program)
    input_overrides = {} if input_overrides is None else dict(input_overrides)
    dimensions = "std::size_t o,std::size_t v"
    if batch_dim:
        dimensions += ",std::size_t q"
    lines = [
        f"inline {output_type} {function_name}({dimensions},{signature},double* arena,std::size_t arena_elements){{",
        "  const std::size_t n=checked_add(o,v);",
        "  std::size_t cursor=0;",
        "  auto allocate=[&](std::size_t count)->double*{",
        "    const auto next=checked_add(cursor,count);",
        '    if(next>arena_elements) throw std::length_error("RCCSD generated CPU arena is too small");',
        "    double* result=arena+cursor; cursor=next; return result;",
        "  };",
    ]
    for number, node in enumerate(program.live_nodes):
        if node.op == "input":
            input_name = node.attrs["name"]
            access = input_overrides.get(input_name)
            if access is None:
                access = _input_access(input_name)
            ctype = "std::int64_t" if node.spec.dtype == "int64" else "double"
            lines.append(f"  const {ctype}* {names[number]}={access};")
        else:
            lines += _cpu_node(node, number, names)
    outputs = {key: names[value._emit_index] for key, value in program.outputs.items()}
    if output_type == "IterationOutputs":
        returned = [
            f"*{outputs['correlation_energy']}",
            outputs["singles_residual"],
            outputs["doubles_residual"],
            outputs["next_t1"],
            outputs["next_t2"],
        ]
    elif output_type == "ReplayOutputs":
        returned = [
            f"*{outputs['correlation_energy']}",
            outputs["singles_residual"],
            outputs["doubles_residual"],
        ]
    elif output_type == "LambdaOutputs":
        returned = [outputs["bar_t1"], outputs["bar_t2"]]
    elif output_type == "ParameterOutput":
        if len(outputs) != 1:
            raise ValueError("RCCSD parameter VJP must expose exactly one output")
        returned = [next(iter(outputs.values()))]
    elif output_type == "HamiltonianOutputs":
        returned = [
            outputs["hcore"],
            outputs["eri"],
            outputs["overlap"],
            outputs["rotation_gradient"],
            outputs["stationarity"],
            outputs["orbital_rhs"],
        ]
    elif output_type == "OrbitalJvpOutput":
        returned = [outputs["d_fov"]]
    elif output_type == "TriplesResponseOutputs":
        returned = [outputs[f"bar_{name}"] for name in TRIPLES_RESPONSE_INPUTS]
    else:
        raise ValueError(f"unsupported RCCSD generated CPU output type {output_type}")
    lines.append("  return {" + ",".join(returned) + "};")
    lines.append("}")
    return "\n".join(lines)


def _required_function(program: Program, name: str, *, batch_dim: bool = False) -> str:
    _prepare_program(program)
    pieces = [_size(node.spec) for node in program.live_nodes if node.op != "input"]
    # Emit sequential checked additions rather than an expression whose parser
    # nesting grows with the AD graph. Clang's default bracket limit is finite.
    body = "std::size_t required=0;"
    for piece in pieces:
        body += f"required=checked_add(required,{piece});"
    dimensions = "std::size_t o,std::size_t v"
    if batch_dim:
        dimensions += ",std::size_t q"
    return (
        f"inline std::size_t {name}({dimensions}){{"
        f"[[maybe_unused]] const std::size_t n=checked_add(o,v);{body}return required;}}"
    )


def cpu_header() -> str:
    iteration = iteration_program(*REPRESENTATIVE)
    replay = build_ccsd_program(*REPRESENTATIVE, form="expanded", diagnostics=False)
    lambda_programs = build_lambda_programs(*REPRESENTATIVE, form="shared")
    lambda_independent = build_lambda_programs(*REPRESENTATIVE, form="expanded")
    lambda_rhs = lambda_programs.energy_vjp.program
    lambda_transpose = lambda_programs.residual_vjp.program
    independent_rhs = lambda_independent.energy_vjp.program
    independent_transpose = lambda_independent.residual_vjp.program
    parameter_vjps = {
        parameter: build_parameter_vjp(lambda_programs.primal, parameter).program
        for parameter in PARAMETERS
    }
    hamiltonian = build_hamiltonian_programs(
        *REPRESENTATIVE, explicit_density_input=True
    )
    hamiltonian_weights = hamiltonian.weights
    orbital_jvp = hamiltonian.orbital_jvp.program
    fock_weights = build_fock_weight_program(
        *REPRESENTATIVE, explicit_density_input=True
    )
    hamiltonian_input_names = tuple(
        sorted(
            n.attrs["name"] for n in hamiltonian_weights.live_nodes if n.op == "input"
        )
    )
    orbital_jvp_input_names = tuple(
        sorted(n.attrs["name"] for n in orbital_jvp.live_nodes if n.op == "input")
    )
    fock_weight_input_names = tuple(
        sorted(n.attrs["name"] for n in fock_weights.live_nodes if n.op == "input")
    )
    triples_primal = build_runtime_tile_triples_program(*REPRESENTATIVE, capacity=6)
    triples_response = transpose_program(
        triples_primal,
        ("triples_energy",),
        inputs=TRIPLES_RESPONSE_INPUTS,
        max_elements=100_000_000,
    ).program
    triples_input_nodes = {
        n.attrs["name"]: n for n in triples_response.live_nodes if n.op == "input"
    }
    triples_input_names = tuple(sorted(triples_input_nodes))
    response_seed_signature = (
        "const Inputs& inputs,const double* bar_correlation_energy,"
        "const double* bar_singles_residual,const double* bar_doubles_residual"
    )
    response_seed_overrides = {
        "bar_correlation_energy": "bar_correlation_energy",
        "bar_singles_residual": "bar_singles_residual",
        "bar_doubles_residual": "bar_doubles_residual",
    }
    return "\n".join(
        [
            "// Generated by tools/generate_rccsd_native.py from #148 TensorIR.",
            "#pragma once",
            "#include <algorithm>",
            "#include <cmath>",
            "#include <cstddef>",
            "#include <cstdint>",
            "#include <initializer_list>",
            "#include <limits>",
            "#include <stdexcept>",
            "namespace vibeqc::cc::generated {",
            _scaled_bilinear_cpp(),
            'inline std::size_t checked_add(std::size_t a,std::size_t b){if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("RCCSD size overflow");return a+b;}',
            'inline std::size_t checked_product(std::initializer_list<std::size_t> values){std::size_t x=1;for(auto v:values){if(v&&x>std::numeric_limits<std::size_t>::max()/v)throw std::length_error("RCCSD size overflow");x*=v;}return x;}',
            "struct Inputs {",
            *[f"  const double* {name}{{}};" for name in INPUT_NAMES],
            "};",
            "struct IterationOutputs { double energy{}; const double* r1{}; const double* r2{}; const double* next_t1{}; const double* next_t2{}; };",
            "struct ReplayOutputs { double energy{}; const double* r1{}; const double* r2{}; };",
            "struct LambdaOutputs { const double* t1{}; const double* t2{}; };",
            "struct ParameterOutput { const double* values{}; };",
            "struct HamiltonianWeightInputs {",
            *[f"  const double* {name}{{}};" for name in hamiltonian_input_names],
            "};",
            "struct OrbitalJvpInputs {",
            *[f"  const double* {name}{{}};" for name in orbital_jvp_input_names],
            "};",
            "struct FockWeightInputs {",
            *[f"  const double* {name}{{}};" for name in fock_weight_input_names],
            "};",
            "struct TriplesResponseInputs {",
            *[
                f"  const {'std::int64_t' if triples_input_nodes[name].spec.dtype == 'int64' else 'double'}* {name}{{}};"
                for name in triples_input_names
            ],
            "};",
            "struct HamiltonianOutputs { const double* hcore{}; const double* eri{}; const double* overlap{}; const double* rotation_gradient{}; const double* stationarity{}; const double* orbital_rhs{}; };",
            "struct OrbitalJvpOutput { const double* d_fov{}; };",
            "struct TriplesResponseOutputs {",
            *[f"  const double* {name}{{}};" for name in TRIPLES_RESPONSE_INPUTS],
            "};",
            f'inline constexpr const char* iteration_equation_hash="{iteration.provenance["physical_equation"]}";',
            f'inline constexpr const char* iteration_program_hash="{iteration.logical_hash}";',
            f'inline constexpr const char* replay_equation_hash="{replay.logical_hash}";',
            f'inline constexpr const char* lambda_rhs_program_hash="{lambda_rhs.logical_hash}";',
            f'inline constexpr const char* lambda_transpose_program_hash="{lambda_transpose.logical_hash}";',
            f'inline constexpr const char* lambda_independent_rhs_program_hash="{independent_rhs.logical_hash}";',
            f'inline constexpr const char* lambda_independent_transpose_program_hash="{independent_transpose.logical_hash}";',
            *[
                f'inline constexpr const char* parameter_{parameter}_program_hash="{program.logical_hash}";'
                for parameter, program in parameter_vjps.items()
            ],
            f'inline constexpr const char* hamiltonian_weights_program_hash="{hamiltonian_weights.logical_hash}";',
            f'inline constexpr const char* orbital_jvp_program_hash="{orbital_jvp.logical_hash}";',
            f'inline constexpr const char* fock_weights_program_hash="{fock_weights.logical_hash}";',
            f'inline constexpr const char* triples_response_program_hash="{triples_response.logical_hash}";',
            _required_function(iteration, "iteration_arena_elements"),
            _required_function(replay, "replay_arena_elements"),
            _required_function(lambda_rhs, "lambda_rhs_arena_elements"),
            _required_function(lambda_transpose, "lambda_transpose_arena_elements"),
            _required_function(
                independent_rhs, "lambda_independent_rhs_arena_elements"
            ),
            _required_function(
                independent_transpose, "lambda_independent_transpose_arena_elements"
            ),
            *[
                _required_function(program, f"parameter_{parameter}_arena_elements")
                for parameter, program in parameter_vjps.items()
            ],
            _required_function(
                hamiltonian_weights, "hamiltonian_weights_arena_elements"
            ),
            _required_function(orbital_jvp, "orbital_jvp_arena_elements"),
            _required_function(fock_weights, "fock_weights_arena_elements"),
            _required_function(
                triples_response, "triples_response_arena_elements", batch_dim=True
            ),
            _cpu_function(iteration, "run_iteration_cpu", "IterationOutputs"),
            _cpu_function(replay, "run_replay_cpu", "ReplayOutputs"),
            _cpu_function(
                lambda_rhs,
                "run_lambda_rhs_cpu",
                "LambdaOutputs",
                signature="const Inputs& inputs,const double* bar_correlation_energy",
                input_overrides={"bar_correlation_energy": "bar_correlation_energy"},
            ),
            _cpu_function(
                lambda_transpose,
                "run_lambda_transpose_cpu",
                "LambdaOutputs",
                signature=(
                    "const Inputs& inputs,const double* bar_singles_residual,"
                    "const double* bar_doubles_residual"
                ),
                input_overrides={
                    "bar_singles_residual": "bar_singles_residual",
                    "bar_doubles_residual": "bar_doubles_residual",
                },
            ),
            _cpu_function(
                independent_rhs,
                "run_lambda_independent_rhs_cpu",
                "LambdaOutputs",
                signature="const Inputs& inputs,const double* bar_correlation_energy",
                input_overrides={"bar_correlation_energy": "bar_correlation_energy"},
            ),
            _cpu_function(
                independent_transpose,
                "run_lambda_independent_transpose_cpu",
                "LambdaOutputs",
                signature=(
                    "const Inputs& inputs,const double* bar_singles_residual,"
                    "const double* bar_doubles_residual"
                ),
                input_overrides={
                    "bar_singles_residual": "bar_singles_residual",
                    "bar_doubles_residual": "bar_doubles_residual",
                },
            ),
            *[
                _cpu_function(
                    program,
                    f"run_parameter_{parameter}_cpu",
                    "ParameterOutput",
                    signature=response_seed_signature,
                    input_overrides=response_seed_overrides,
                )
                for parameter, program in parameter_vjps.items()
            ],
            _cpu_function(
                hamiltonian_weights,
                "run_hamiltonian_weights_cpu",
                "HamiltonianOutputs",
                signature="const HamiltonianWeightInputs& inputs",
                input_overrides={
                    name: f"inputs.{name}" for name in hamiltonian_input_names
                },
            ),
            _cpu_function(
                orbital_jvp,
                "run_orbital_jvp_cpu",
                "OrbitalJvpOutput",
                signature="const OrbitalJvpInputs& inputs",
                input_overrides={
                    name: f"inputs.{name}" for name in orbital_jvp_input_names
                },
            ),
            _cpu_function(
                fock_weights,
                "run_fock_weights_cpu",
                "HamiltonianOutputs",
                signature="const FockWeightInputs& inputs",
                input_overrides={
                    name: f"inputs.{name}" for name in fock_weight_input_names
                },
            ),
            _cpu_function(
                triples_response,
                "run_triples_response_cpu",
                "TriplesResponseOutputs",
                signature="const TriplesResponseInputs& inputs",
                input_overrides={
                    name: f"inputs.{name}" for name in triples_input_names
                },
                batch_dim=True,
            ),
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
    uses_complete_orbital = any(
        _dim(index) == "n"
        for spec in (node.spec, *(source.spec for source in node.inputs))
        for index in spec.indices
    )
    lines = [
        f"__global__ void {prefix}_node_{number}({','.join(arguments)}){{",
        *(["  const std::size_t n=o+v;"] if uses_complete_orbital else []),
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
        dims = _label_dims(node)
        all_labels = sorted(dims)
        reduced = [label for label in all_labels if label not in output]
        if output:
            lines.append("    std::size_t rem=flat;")
        for label in reversed(output):
            dim = dims[label]
            lines += [f"    const std::size_t l{label}=rem%{dim};", f"    rem/={dim};"]
        lines.append("    double sum=0.0;")
        indent = "    "
        for label in reduced:
            dim = dims[label]
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
    elif node.op == "slice":
        source = node.inputs[0]
        ranges = tuple(node.attrs["ranges"])
        rank = len(node.spec.indices)
        if len(ranges) != rank:
            raise ValueError("runtime RCCSD CUDA slice rank mismatch")
        if rank:
            lines.append("    std::size_t rem=flat;")
        coords = [""] * rank
        for axis in reversed(range(rank)):
            dim = _dim(node.spec.indices[axis])
            lines += [
                f"    const std::size_t c{axis}=rem%{dim};",
                f"    rem/={dim};",
            ]
            coords[axis] = f"(c{axis}+{_runtime_bound(ranges[axis][0])})"
        source_index = _flat_coords(coords, source.spec)
        lines += [
            f"    const double value=a0[{source_index}];",
            f"    out[flat]=vibeqc_tensor::finite(value,error,{number});",
        ]
    elif node.op == "scatter_add":
        source = node.inputs[0]
        axis = node.attrs["axis"]
        positions = tuple(node.attrs["positions"])
        if not positions or positions != tuple(range(positions[0], positions[-1] + 1)):
            raise ValueError("runtime RCCSD CUDA scatter requires contiguous positions")
        offset = _runtime_bound(positions[0])
        source_dim = _dim(source.spec.indices[axis])
        rank = len(node.spec.indices)
        if rank:
            lines.append("    std::size_t rem=flat;")
        coords = [""] * rank
        for target_axis in reversed(range(rank)):
            dim = _dim(node.spec.indices[target_axis])
            lines += [
                f"    const std::size_t c{target_axis}=rem%{dim};",
                f"    rem/={dim};",
            ]
            coords[target_axis] = f"c{target_axis}"
        source_coords = list(coords)
        source_coords[axis] = f"(c{axis}-{offset})"
        source_index = _flat_coords(source_coords, source.spec)
        lines += [
            f"    if(c{axis}<{offset} || c{axis}>={offset}+{source_dim}){{",
            "      out[flat]=0.0;",
            "    }else{",
            f"      const double value=a0[{source_index}];",
            f"      out[flat]=vibeqc_tensor::finite(value,error,{number});",
            "    }",
        ]
    elif node.op == "transpose":
        source = node.inputs[0]
        rank = len(node.spec.indices)
        axes = tuple(node.attrs["axes"])
        if len(axes) != rank or sorted(axes) != list(range(rank)):
            raise ValueError("invalid native RCCSD CUDA transpose permutation")
        source_coords: list[str | None] = [None] * rank
        if rank:
            lines.append("    std::size_t rem=flat;")
        for axis in reversed(range(rank)):
            dim = _dim(node.spec.indices[axis])
            lines += [
                f"    const std::size_t c{axis}=rem%{dim};",
                f"    rem/={dim};",
            ]
        for out_axis, source_axis in enumerate(axes):
            source_coords[source_axis] = f"c{out_axis}"
        if any(coord is None for coord in source_coords):
            raise ValueError("invalid native RCCSD CUDA transpose coordinate map")
        index = typing.cast("list[str]", source_coords)[0] if source_coords else "0"
        for coord, spec_index in zip(source_coords[1:], source.spec.indices[1:]):
            index = f"({index}*{_dim(spec_index)}+{coord})"
        lines += [
            f"    const double value=a0[{index}];",
            f"    out[flat]=vibeqc_tensor::finite(value,error,{number});",
        ]
    else:
        raise ValueError(f"unsupported native RCCSD CUDA op {node.op}")
    lines += ["  }", "}"]
    return "\n".join(lines)


def _cuda_program(
    program: Program,
    prefix: str,
    output_type: str,
    *,
    input_overrides: dict[str, str] | None = None,
) -> str:
    names = _prepare_program(program)
    input_overrides = {} if input_overrides is None else dict(input_overrides)
    kernels = []
    for number, node in enumerate(program.live_nodes):
        if node.op != "input":
            kernels.append(_cuda_kernel(node, number, prefix, names))
    uses_complete_orbital = any(
        _dim(index) == "n"
        for node in program.live_nodes
        if node.op != "input"
        for index in node.spec.indices
    )
    lines = kernels + [
        f"static {output_type} run_{prefix}(CudaState& s){{",
        "  auto* arena=s."
        + (
            "iteration_arena"
            if prefix == "iteration"
            else "replay_arena"
            if prefix == "replay"
            else "response_arena"
        )
        + ";",
        "  const auto o=s.o,v=s.v;",
        *(["  const std::size_t n=checked_add(o,v);"] if uses_complete_orbital else []),
        "  std::size_t cursor=0;",
        "  auto allocate=[&](std::size_t count)->double*{double* p=arena+cursor;cursor=checked_add(cursor,count);return p;};",
        "  vibeqc_tensor::cuda_check(cudaMemsetAsync(s.error,0,sizeof(int),s.stream));",
    ]
    for number, node in enumerate(program.live_nodes):
        if node.op == "input":
            input_name = node.attrs["name"]
            access = (
                input_overrides[input_name]
                if input_name in input_overrides
                else _input_access(input_name, cuda=True)
            )
            lines.append(f"  const double* {names[number]}={access};")
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
        returned = [
            outputs["correlation_energy"],
            outputs["singles_residual"],
            outputs["doubles_residual"],
            outputs["next_t1"],
            outputs["next_t2"],
        ]
    elif output_type == "DeviceReplayOutputs":
        returned = [
            outputs["correlation_energy"],
            outputs["singles_residual"],
            outputs["doubles_residual"],
        ]
    elif output_type == "DeviceLambdaOutputs":
        returned = [outputs["bar_t1"], outputs["bar_t2"]]
    elif output_type == "DeviceParameterOutput":
        if len(outputs) != 1:
            raise ValueError("RCCSD CUDA parameter VJP must expose exactly one output")
        returned = [next(iter(outputs.values()))]
    elif output_type == "DeviceHamiltonianOutputs":
        returned = [
            outputs["hcore"],
            outputs["eri"],
            outputs["overlap"],
            outputs["rotation_gradient"],
            outputs["stationarity"],
            outputs["orbital_rhs"],
        ]
    elif output_type == "DeviceOrbitalJvpOutput":
        returned = [outputs["d_fov"]]
    else:
        raise ValueError(f"unsupported RCCSD generated CUDA output type {output_type}")
    lines.append("  return {" + ",".join(returned) + "};")
    lines.append("}")
    return "\n".join(lines)


def cuda_source() -> str:
    iteration = iteration_program(*REPRESENTATIVE)
    replay = build_ccsd_program(*REPRESENTATIVE, form="expanded", diagnostics=False)
    lambda_programs = build_lambda_programs(*REPRESENTATIVE, form="shared")
    lambda_independent = build_lambda_programs(*REPRESENTATIVE, form="expanded")
    lambda_rhs = lambda_programs.energy_vjp.program
    lambda_transpose = lambda_programs.residual_vjp.program
    independent_rhs = lambda_independent.energy_vjp.program
    independent_transpose = lambda_independent.residual_vjp.program
    parameter_vjps = {
        parameter: build_parameter_vjp(lambda_programs.primal, parameter).program
        for parameter in PARAMETERS
    }
    hamiltonian = build_hamiltonian_programs(
        *REPRESENTATIVE, explicit_density_input=True
    )
    hamiltonian_weights = hamiltonian.weights
    orbital_jvp = hamiltonian.orbital_jvp.program
    fock_weights = build_fock_weight_program(
        *REPRESENTATIVE, explicit_density_input=True
    )
    hamiltonian_input_names = tuple(
        sorted(
            n.attrs["name"] for n in hamiltonian_weights.live_nodes if n.op == "input"
        )
    )
    orbital_jvp_input_names = tuple(
        sorted(n.attrs["name"] for n in orbital_jvp.live_nodes if n.op == "input")
    )
    fock_weight_input_names = tuple(
        sorted(n.attrs["name"] for n in fock_weights.live_nodes if n.op == "input")
    )
    energy_seed = {"bar_correlation_energy": "s.bar_correlation_energy"}
    residual_seed = {
        "bar_singles_residual": "s.bar_singles_residual",
        "bar_doubles_residual": "s.bar_doubles_residual",
    }
    response_seed_overrides = {
        "bar_correlation_energy": "s.bar_correlation_energy",
        "bar_singles_residual": "s.bar_singles_residual",
        "bar_doubles_residual": "s.bar_doubles_residual",
    }
    return "\n".join(
        [
            "// Generated by tools/generate_rccsd_native.py from #148 TensorIR.",
            '#include "cc/cuda_solver_support.cuh"',
            '#include "generated_rccsd_cpu.hpp"',
            "namespace vibeqc::cc::generated {",
            _cuda_program(iteration, "iteration", "DeviceIterationOutputs"),
            _cuda_program(replay, "replay", "DeviceReplayOutputs"),
            _cuda_program(
                lambda_rhs,
                "lambda_rhs",
                "DeviceLambdaOutputs",
                input_overrides=energy_seed,
            ),
            _cuda_program(
                lambda_transpose,
                "lambda_transpose",
                "DeviceLambdaOutputs",
                input_overrides=residual_seed,
            ),
            _cuda_program(
                independent_rhs,
                "lambda_independent_rhs",
                "DeviceLambdaOutputs",
                input_overrides=energy_seed,
            ),
            _cuda_program(
                independent_transpose,
                "lambda_independent_transpose",
                "DeviceLambdaOutputs",
                input_overrides=residual_seed,
            ),
            *[
                _cuda_program(
                    program,
                    f"parameter_{parameter}",
                    "DeviceParameterOutput",
                    input_overrides=response_seed_overrides,
                )
                for parameter, program in parameter_vjps.items()
            ],
            _cuda_program(
                hamiltonian_weights,
                "hamiltonian_weights",
                "DeviceHamiltonianOutputs",
                input_overrides={name: f"s.{name}" for name in hamiltonian_input_names},
            ),
            _cuda_program(
                fock_weights,
                "fock_weights",
                "DeviceHamiltonianOutputs",
                input_overrides={name: f"s.{name}" for name in fock_weight_input_names},
            ),
            _cuda_program(
                orbital_jvp,
                "orbital_jvp",
                "DeviceOrbitalJvpOutput",
                input_overrides={name: f"s.{name}" for name in orbital_jvp_input_names},
            ),
            "DeviceIterationOutputs run_iteration_cuda(CudaState& state){return run_iteration(state);}",
            "DeviceReplayOutputs run_replay_cuda(CudaState& state){return run_replay(state);}",
            "DeviceLambdaOutputs run_lambda_rhs_cuda(CudaState& state){return run_lambda_rhs(state);}",
            "DeviceLambdaOutputs run_lambda_transpose_cuda(CudaState& state){return run_lambda_transpose(state);}",
            "DeviceLambdaOutputs run_lambda_independent_rhs_cuda(CudaState& state){return run_lambda_independent_rhs(state);}",
            "DeviceLambdaOutputs run_lambda_independent_transpose_cuda(CudaState& state){return run_lambda_independent_transpose(state);}",
            "DeviceParameterOutput run_parameter_foo_cuda(CudaState& state){return run_parameter_foo(state);}",
            "DeviceParameterOutput run_parameter_fov_cuda(CudaState& state){return run_parameter_fov(state);}",
            "DeviceParameterOutput run_parameter_fvv_cuda(CudaState& state){return run_parameter_fvv(state);}",
            "DeviceParameterOutput run_parameter_ovov_cuda(CudaState& state){return run_parameter_ovov(state);}",
            "DeviceParameterOutput run_parameter_ovvo_cuda(CudaState& state){return run_parameter_ovvo(state);}",
            "DeviceParameterOutput run_parameter_oovv_cuda(CudaState& state){return run_parameter_oovv(state);}",
            "DeviceParameterOutput run_parameter_ovvv_cuda(CudaState& state){return run_parameter_ovvv(state);}",
            "DeviceParameterOutput run_parameter_ovoo_cuda(CudaState& state){return run_parameter_ovoo(state);}",
            "DeviceParameterOutput run_parameter_oooo_cuda(CudaState& state){return run_parameter_oooo(state);}",
            "DeviceParameterOutput run_parameter_vvvv_cuda(CudaState& state){return run_parameter_vvvv(state);}",
            "DeviceHamiltonianOutputs run_hamiltonian_weights_cuda(CudaState& state){return run_hamiltonian_weights(state);}",
            "DeviceHamiltonianOutputs run_fock_weights_cuda(CudaState& state){return run_fock_weights(state);}",
            "DeviceOrbitalJvpOutput run_orbital_jvp_cuda(CudaState& state){return run_orbital_jvp(state);}",
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
