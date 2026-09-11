"""CUDA scalar lowering of shared S/T/V first-derivative DAG roots."""

from itertools import product

from .cuda import CudaEmitter
from .expr import Graph
from .ir_serialization import integral_to_payload
from .one_electron_cuda import (
    _emit_component_index,
    _emit_pair_geometry,
    _emit_support_cuda,
    _geometry_boundary,
    _pair_geometry_inventory,
)
from .one_electron_derivatives import (
    build_one_electron_derivative_ir,
    build_one_electron_derivative_kernel,
)
from .shell_spec import cartesian_components


def one_electron_derivative_inventory():
    """Record mathematical center/sign/layout contracts before runtime scheduling."""
    return {
        "schema": "vibeqc.one_electron_derivatives",
        "version": 1,
        "precision": "fp64",
        "programs": [
            integral_to_payload(
                build_one_electron_derivative_ir(family, angular, weighted=weighted)
            )
            for family in ("overlap", "kinetic", "nuclear_attraction")
            for angular in product(range(4), repeat=2)
            for weighted in (False, True)
        ],
    }


def _emit_axis_permutations():
    """Exploit Cartesian-axis covariance without widening the public AO family.

    Every primitive operator is a scalar under an x/y or x/z coordinate swap.
    Swapping both Gaussian powers and all A/B/C positions turns its requested
    derivative into the emitted x derivative. Radial/component normalization is
    invariant under these permutations. This bounds live CSE roots to two.
    """
    components = tuple(c for l in range(4) for c in cartesian_components(l))
    lines = [
        "__device__ __forceinline__ unsigned permute_component(unsigned index, unsigned axis) {",
        "  if (axis == 0) return index;",
        "  switch (index) {",
    ]
    for index, component in enumerate(components):
        choices = []
        for axis in (1, 2):
            powers = [component.count(a) for a in "xyz"]
            powers[0], powers[axis] = powers[axis], powers[0]
            transformed = "".join(a * n for a, n in zip("xyz", powers))
            choices.append(components.index(transformed))
        lines.append(
            f"    case {index}U: return axis == 1 ? {choices[0]}U : {choices[1]}U;"
        )
    lines += [
        "  }",
        "  return 20U;",
        "}",
        "__device__ __forceinline__ PairGeometry permute_pair(const PairGeometry& pair, unsigned axis) {",
        "  PairGeometry result = pair;",
    ]
    _, fields = _pair_geometry_inventory()
    axis_fields = {field[:-2] for field in fields if field.endswith(("_x", "_y", "_z"))}
    for prefix in axis_fields:
        if not all(f"{prefix}_{axis}" in fields for axis in "xyz"):
            raise ValueError(f"incomplete Cartesian geometry field: {prefix}")
    for axis, name in ((1, "y"), (2, "z")):
        lines.append(f"  if (axis == {axis}) {{")
        for center in sorted(axis_fields):
            lines += [
                f"    result.{center}_x = pair.{center}_{name};",
                f"    result.{center}_{name} = pair.{center}_x;",
            ]
        lines.append("  }")
    return "\n".join(lines + ["  return result;", "}"])


def _emit_gradient_helpers(attraction):
    """Emit two x-axis roots; reuse them for y/z through exact permutations.

    Attraction evaluates Boys values once before all three axis calls. Keeping
    each helper to S/T or A/B on one axis avoids retaining six roots and their
    cross-axis CSE ancestors simultaneously. This trades modest recomputation
    for a smaller optimization unit and bounded register pressure.
    """
    name = "attraction_gradient" if attraction else "overlap_kinetic_gradient"
    arguments = "const PairGeometry& pair, unsigned component"
    if attraction:
        arguments += ", double c_x, double c_y, double c_z, const double* boys"
    lines = []
    for angular in product(range(4), repeat=2):
        lines += [
            f"__device__ __noinline__ GradientAxis {name}_x_{angular[0]}{angular[1]}({arguments}) {{",
            "  switch (component) {",
        ]
        for index, components in enumerate(
            product(*(cartesian_components(l) for l in angular))
        ):
            graph = Graph()
            if attraction:
                kernel = build_one_electron_derivative_kernel(
                    build_one_electron_derivative_ir("nuclear_attraction", angular),
                    components,
                    graph=graph,
                )
                roots = (kernel.gradients[0][0], kernel.gradients[1][0])
            else:
                overlap = build_one_electron_derivative_kernel(
                    build_one_electron_derivative_ir("overlap", angular),
                    components,
                    graph=graph,
                )
                kernel = build_one_electron_derivative_kernel(
                    build_one_electron_derivative_ir("kinetic", angular),
                    components,
                    graph=graph,
                )
                roots = (overlap.gradients[0][0], kernel.gradients[0][0])
            target, roots = _geometry_boundary(kernel, roots)
            emitter = CudaEmitter(target, {})
            emitter.emit(roots)
            lines += [
                f"    case {index}U: {{",
                *emitter.lines,
                "      return {"
                + ", ".join(emitter.reference(r) for r in roots)
                + "};",
                "    }",
            ]
        lines += ["  }", '  return {nan(""),nan("")};', "}"]
    external = (
        ", double c_x, double c_y, double c_z, const double* boys" if attraction else ""
    )
    lines += [
        f"__device__ __forceinline__ GradientAxis {name}_x(",
        "    const PairGeometry& pair, unsigned first, unsigned second"
        + external
        + ") {",
        '  if (first >= 20 || second >= 20) return {nan(""),nan("")};',
        "  const unsigned a = first < 1 ? 0 : first < 4 ? 1 : first < 10 ? 2 : 3;",
        "  const unsigned b = second < 1 ? 0 : second < 4 ? 1 : second < 10 ? 2 : 3;",
        "  const unsigned offsets[] = {0,1,4,10}, counts[] = {1,3,6,10};",
        "  const unsigned component = (first-offsets[a])*counts[b]+second-offsets[b];",
        "  switch (a*4U+b) {",
    ]
    for a, b in product(range(4), repeat=2):
        args = "pair, component" + (", c_x, c_y, c_z, boys" if attraction else "")
        lines.append(f"    case {a * 4 + b}U: return {name}_x_{a}{b}({args});")
    lines += [
        "  }",
        '  return {nan(""),nan("")};',
        "}",
        f"__device__ __forceinline__ GradientPair {name}(",
        "    const PairGeometry& pair, unsigned first, unsigned second"
        + (", double c_x, double c_y, double c_z" if attraction else "")
        + ") {",
        "  GradientPair result{};",
    ]
    if attraction:
        # The Boys argument is rotationally invariant and independent of the
        # shell powers. Use its validated graph root once for all axis calls.
        kernel = build_one_electron_derivative_kernel(
            build_one_electron_derivative_ir("nuclear_attraction", (0, 0)), ("", "")
        )
        target, roots = _geometry_boundary(kernel, (kernel.boys_argument,))
        emitter = CudaEmitter(target, {})
        emitter.emit(roots)
        lines += [
            *emitter.lines,
            "  double boys[8];",
            f"  boys_values<7>({emitter.reference(roots[0])},boys);",
        ]
    lines += [
        "  for (unsigned axis=0; axis<3; ++axis) {",
        "    const auto rotated = permute_pair(pair,axis);",
        "    const auto a = permute_component(first,axis), b = permute_component(second,axis);",
    ]
    args = "rotated, a, b"
    if attraction:
        args += ", axis==0 ? c_x : axis==1 ? c_y : c_z, axis==1 ? c_x : c_y, axis==2 ? c_x : c_z, boys"
    lines += [
        f"    const auto value = {name}_x({args});",
        "    result.first[axis]=value.first; result.second[axis]=value.second;",
        "  }",
        "  return result;",
        "}",
    ]
    return "\n".join(lines)


def emit_one_electron_derivatives_cuda():
    """Emit primitive gradient roots; normalized AO weights belong to the caller."""
    prefix = (
        _emit_support_cuda()
        .replace(
            "VIBEQC_GENERATED_ONE_ELECTRON_VALUES_CUH",
            "VIBEQC_GENERATED_ONE_ELECTRON_DERIVATIVES_CUH",
        )
        .replace("generated_one_electron {", "generated_one_electron_derivatives {")
        .replace("Order <= 6", "Order <= 7")
        .replace("through order six", "through order seven")
    )
    return "\n".join(
        [
            prefix,
            "struct GradientPair { double first[3], second[3]; };",
            "struct GradientAxis { double first, second; };",
            '__device__ __forceinline__ GradientPair invalid_gradient() { const double n = nan(""); return {{n,n,n},{n,n,n}}; }',
            _emit_pair_geometry(),
            _emit_component_index(),
            _emit_axis_permutations(),
            _emit_gradient_helpers(False),
            _emit_gradient_helpers(True),
            "}  // namespace vibeqc::scf::generated_one_electron_derivatives",
            "#endif",
            "",
        ]
    )
