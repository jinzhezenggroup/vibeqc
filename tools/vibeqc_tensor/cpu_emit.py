"""Small native CPU lowering of elementwise/full-reduction TensorIR programs.

This is a deterministic correctness backend for bounded native consumers.
Unsupported primitives fail during generation; the NumPy interpreter remains
an independent implementation. There is no per-operation Python runtime.
"""

import re
from fractions import Fraction
from math import prod


def emit_cpu(program, *, function_name):
    if not re.fullmatch(r"[A-Za-z_]\w*", function_name, flags=re.ASCII):
        raise ValueError("invalid native CPU function name")
    nodes = program.live_nodes
    if any(n.spec.dtype != "float64" for n in nodes):
        raise ValueError("native CPU lowering supports FP64 only")
    if any(n.spec.symmetries for n in nodes):
        raise ValueError("native CPU symmetry validation is not implemented")
    if any(n.spec.size == 0 for n in nodes):
        raise ValueError("native CPU lowering requires nonempty tensors")
    ids = {n: k for k, n in enumerate(nodes)}
    inputs = [n for n in nodes if n.op == "input"]
    lines = [
        f"// TensorIR equation {program.logical_hash}",
        f"inline void {function_name}(const double* const* inputs, double* const* outputs) {{",
    ]
    for k, node in enumerate(nodes):
        size, op = node.spec.size, node.op
        if op == "input":
            lines.append(f"const double* v{k} = inputs[{inputs.index(node)}];")
            lines.append(
                f'if (!v{k}) throw std::invalid_argument("null tensor input");'
            )
        else:
            lines.append(
                f"std::vector<double> storage{k}({size}); double* v{k}=storage{k}.data();"
            )
            operands = [f"v{ids[n]}" for n in node.inputs]
            if op == "add":
                terms = [
                    f"({float(Fraction(*c)).hex()}*{v}[z])"
                    for v, c in zip(operands, node.attrs["coefficients"])
                ]
                expr = "+".join(terms)
            elif op in ("multiply", "divide"):
                if op == "divide":
                    lines.append(
                        f'for (size_t z=0;z<{size};++z) if ({operands[1]}[z]==0) throw std::runtime_error("zero tensor denominator");'
                    )
                expr = f"{operands[0]}[z]{'*' if op == 'multiply' else '/'}{operands[1]}[z]"
            elif op == "broadcast":
                source_shape = node.inputs[0].spec.shape
                terms = [
                    f"((z/{prod(node.spec.shape[axis + 1 :])})%{node.spec.shape[axis]})*{prod(source_shape[j + 1 :])}"
                    for j, axis in enumerate(node.attrs["axes"])
                ]
                expr = f"{operands[0]}[{'+'.join(terms) or '0'}]"
            elif op == "reduce" and node.spec.shape == ():
                lines.append(
                    f'for (size_t q=0;q<{node.inputs[0].spec.size};++q) {{ v{k}[0]+={operands[0]}[q]; if (!std::isfinite(v{k}[0])) throw std::runtime_error("nonfinite tensor reduction"); }}'
                )
                expr = None
            else:
                raise ValueError(f"native CPU lowering does not support {op}")
            if expr is not None:
                lines.append(f"for (size_t z=0;z<{size};++z) v{k}[z]={expr};")
        lines.append(
            f'for (size_t z=0;z<{size};++z) if (!std::isfinite(v{k}[z])) throw std::runtime_error("nonfinite TensorIR {op}");'
        )
    for slot, node in enumerate(program.outputs.values()):
        lines.append(f"std::copy_n(v{ids[node]}, {node.spec.size}, outputs[{slot}]);")
    lines.append("}")
    return "\n".join(lines) + "\n"
