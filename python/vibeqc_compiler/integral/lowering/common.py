"""Structured CUDA component decoding, array declarations and geometry setup.

These helpers share shell component ordering across Fock and force consumers;
they do not choose a production schedule or execute a device program."""

from __future__ import annotations

from collections.abc import Sequence

from ..shell_spec import (
    DPPP_SPEC,
    ShellClassSpec,
)


def _format_cuda_array(values: Sequence[int], columns: int = 12) -> str:
    """Format a deterministic wrapped CUDA initializer."""

    rows = []
    for start in range(0, len(values), columns):
        rows.append(
            "    " + ", ".join(str(value) for value in values[start : start + columns])
        )
    return ",\n".join(rows)


def _shell_letter(angular_momentum: int) -> str:
    """Return conventional shell notation for the supported AOT range."""

    labels = "spdfgh"
    if not 0 <= angular_momentum < len(labels):
        raise ValueError("fused CUDA emitter supports shell labels through h")
    return labels[angular_momentum]


def _component_names(spec: ShellClassSpec) -> tuple[str, str, str, str]:
    """Create readable, unique CUDA names for decoded center components."""

    ordinals = ("first", "second", "third", "fourth")
    return tuple(
        f"{ordinal}_{_shell_letter(order)}"
        for ordinal, order in zip(ordinals, spec.angular, strict=True)
    )


def _emitted_component_names(
    spec: ShellClassSpec,
) -> tuple[str, str, str, str]:
    """Return the actual scalar names present in specialized CUDA source."""

    if spec == DPPP_SPEC:
        return ("d_component", "first_p", "third_p", "fourth_p")
    return _component_names(spec)


def _specialize_dppp_identifiers(source: str, spec: ShellClassSpec) -> str:
    """Rename the shared CUDA skeleton for a non-dppp shell class."""

    if spec == DPPP_SPEC:
        return source
    notation = (
        f"({_shell_letter(spec.angular[0])} "
        f"{_shell_letter(spec.angular[1])}|"
        f"{_shell_letter(spec.angular[2])} "
        f"{_shell_letter(spec.angular[3])})"
    )
    class_name = spec.name[0].upper() + spec.name[1:]
    source = source.replace("(d p|p p)", notation)
    source = source.replace("Dppp", class_name)
    source = source.replace("DPPP", spec.name.upper())
    return source.replace("dppp", spec.name)


def _generic_component_decode(
    spec: ShellClassSpec, *, include_s: bool = True
) -> tuple[str, ...]:
    """Emit compile-time division/modulo lane decoding from spec strides."""

    names = _component_names(spec)
    counts = tuple(map(len, spec.center_components))
    lines = []
    for angular, name, count, stride in zip(
        spec.angular, names, counts, spec.component_strides, strict=True
    ):
        if angular == 0 and not include_s:
            continue
        if count == 1:
            expression = "0U"
        else:
            expression = "component"
            if stride != 1:
                expression = f"({expression} / {stride}U)"
            expression = f"{expression} % {count}U"
        lines.append(f"  const unsigned {name} = {expression};")
    return tuple(lines)


def _component_axis_expression(
    spec: ShellClassSpec,
    center: int,
    quantum: int,
    component_name: str,
) -> str:
    """Lower one component quantum without a runtime shell-class branch."""

    angular_momentum = spec.angular[center]
    if angular_momentum == 1:
        return component_name
    if angular_momentum == 2:
        return f"generated_dppp_d_axes[{component_name}][{quantum}]"
    if angular_momentum == 3:
        return f"generated_dppp_f_axes[{component_name}][{quantum}]"
    raise ValueError(
        "current fused CUDA candidate supports s, p, d, and f centers only"
    )


def _cuda_array_declaration(declaration: str, values: Sequence[str]) -> list[str]:
    """Format a small local CUDA initializer with stable indentation."""

    if not values:
        raise ValueError("CUDA local arrays cannot be empty")
    lines = [f"  {declaration} = {{"]
    lines.extend(
        f"      {value}{',' if index + 1 < len(values) else ''}"
        for index, value in enumerate(values)
    )
    lines[-1] += "};"
    return lines


def _generic_component_gradient_setup(spec: ShellClassSpec) -> str:
    """Generate lane decoding and both Gaussian-pair recurrence inputs."""

    if spec == DPPP_SPEC:
        # Preserve the production golden source byte-for-byte while other
        # shell classes use the fully generated center naming below.
        return """  const unsigned d_component = component / 27U;
  const unsigned p_components = component % 27U;
  const unsigned first_p = p_components / 9U;
  const unsigned third_p = (p_components / 3U) % 3U;
  const unsigned fourth_p = p_components % 3U;
  const unsigned first_axes[3] = {
      generated_dppp_d_axes[d_component][0],
      generated_dppp_d_axes[d_component][1],
      first_p};
  const double first_shifts[3] = {
      geometry.pair_shifts[0][first_axes[0]],
      geometry.pair_shifts[0][first_axes[1]],
      geometry.pair_shifts[1][first_axes[2]]};
  const double first_shift_gradients[3] = {
      geometry.product_scales[0] - 1.0,
      geometry.product_scales[0] - 1.0,
      geometry.product_scales[0]};
  const unsigned second_axes[2] = {third_p, fourth_p};
  const double second_shifts[2] = {
      geometry.pair_shifts[2][third_p],
      geometry.pair_shifts[3][fourth_p]};
  const double second_shift_gradients[2] = {
      geometry.product_scales[2] - 1.0,
      geometry.product_scales[2]};"""

    names = _component_names(spec)
    lines = list(_generic_component_decode(spec, include_s=False))
    pair_definitions = (
        ((0, 1), "first", "geometry.product_scales[0]"),
        ((2, 3), "second", "geometry.product_scales[2]"),
    )
    for centers, pair_name, scale in pair_definitions:
        axes = []
        shifts = []
        gradients = []
        for pair_center, center in enumerate(centers):
            for quantum in range(spec.angular[center]):
                axis = _component_axis_expression(spec, center, quantum, names[center])
                axis_position = len(axes)
                axes.append(axis)
                shifts.append(
                    f"geometry.pair_shifts[{center}][{pair_name}_axes[{axis_position}]]"
                )
                gradients.append(f"{scale} - 1.0" if pair_center == 0 else scale)
        order = sum(spec.angular[center] for center in centers)
        storage_order = max(order, 1)
        lines.extend(
            _cuda_array_declaration(
                f"const unsigned {pair_name}_axes[{storage_order}]",
                axes or ["0U"],
            )
        )
        lines.extend(
            _cuda_array_declaration(
                f"const double {pair_name}_shifts[{storage_order}]",
                shifts or ["0.0"],
            )
        )
        lines.extend(
            _cuda_array_declaration(
                f"const double {pair_name}_shift_gradients[{storage_order}]",
                gradients or ["0.0"],
            )
        )
    return "\n".join(lines)


def _generic_component_value_setup(spec: ShellClassSpec) -> str:
    """Generate lane decoding and coefficient-only pair recurrence inputs."""

    names = _component_names(spec)
    lines = list(_generic_component_decode(spec, include_s=False))
    for centers, pair_name in (((0, 1), "first"), ((2, 3), "second")):
        axes = []
        shifts = []
        for center in centers:
            for quantum in range(spec.angular[center]):
                axis = _component_axis_expression(spec, center, quantum, names[center])
                axis_position = len(axes)
                axes.append(axis)
                shifts.append(
                    f"geometry.pair_shifts[{center}][{pair_name}_axes[{axis_position}]]"
                )
        order = sum(spec.angular[center] for center in centers)
        storage_order = max(order, 1)
        lines.extend(
            _cuda_array_declaration(
                f"const unsigned {pair_name}_axes[{storage_order}]",
                axes or ["0U"],
            )
        )
        lines.extend(
            _cuda_array_declaration(
                f"const double {pair_name}_shifts[{storage_order}]",
                shifts or ["0.0"],
            )
        )
    return "\n".join(lines)


def _generic_task_component_setup(spec: ShellClassSpec) -> str:
    """Generate AO/density routing for one automatically decoded lane."""

    if spec == DPPP_SPEC:
        return """  const unsigned d_component = component / 27U;
  const unsigned p_components = component % 27U;
  const unsigned first_p = p_components / 9U;
  const unsigned third_p = (p_components / 3U) % 3U;
  const unsigned fourth_p = p_components % 3U;
  const bool unique_ket_component =
      shared.task.shell[2] != shared.task.shell[3] || third_p >= fourth_p;
  const std::size_t i = shared.task.ao_begin[0] + d_component;
  const std::size_t j = shared.task.ao_begin[1] + first_p;
  const std::size_t k = shared.task.ao_begin[2] + third_p;
  const std::size_t l = shared.task.ao_begin[3] + fourth_p;"""

    names = _component_names(spec)
    lines = list(_generic_component_decode(spec))
    symmetry_conditions = []
    for first, second in ((0, 1), (2, 3)):
        if spec.angular[first] == spec.angular[second]:
            symmetry_conditions.append(
                f"shared.task.shell[{first}] != shared.task.shell[{second}] || "
                f"{names[first]} >= {names[second]}"
            )
    if spec.angular[:2] == spec.angular[2:]:
        # The active-tile builder triangularizes AO-pair quartets when the
        # bra and ket refer to the same shell pair.  Generated shell-wide
        # workers must reproduce that domain; density permutation de-dup only
        # removes equal AO-index permutations and cannot prevent evaluating
        # both (ij|kl) and (kl|ij) component lanes.
        second_component_count = len(spec.center_components[1])
        fourth_component_count = len(spec.center_components[3])
        symmetry_conditions.append(
            "shared.task.shell_pair[0] != shared.task.shell_pair[1] || "
            f"({names[0]} * {second_component_count}U + {names[1]}) >= "
            f"({names[2]} * {fourth_component_count}U + {names[3]})"
        )
    if symmetry_conditions:
        expression = ") && (".join(symmetry_conditions)
        lines.extend(
            [
                "  const bool unique_ket_component =",
                f"      ({expression});",
            ]
        )
    else:
        lines.append("  constexpr bool unique_ket_component = true;")
    for center, (ao_name, component_name) in enumerate(
        zip(("i", "j", "k", "l"), names, strict=True)
    ):
        lines.append(
            f"  const std::size_t {ao_name} = "
            f"shared.task.ao_begin[{center}] + {component_name};"
        )
    return "\n".join(lines)


def emit_uncached_primitive_geometry_cuda(spec: ShellClassSpec) -> str:
    """Emit the primitive-quartet setup retained by standalone baselines."""

    maximum_order = spec.maximum_force_coulomb_order
    source = f"""
__device__ __forceinline__ void generated_dppp_make_primitive_geometry_uncached(
    double alpha, const GeneratedDpppVec3& first,
    double beta, const GeneratedDpppVec3& second,
    double gamma, const GeneratedDpppVec3& third,
    double delta, const GeneratedDpppVec3& fourth,
    double primitive_coefficient,
    GeneratedDpppPrimitiveGeometry& geometry) {{
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  geometry.rho = p * q / (p + q);
  geometry.inverse_two_p = 0.5 / p;
  geometry.inverse_two_q = 0.5 / q;
  geometry.product_scales[0] = alpha / p;
  geometry.product_scales[1] = beta / p;
  geometry.product_scales[2] = gamma / q;
  double pair_decay_exponent = 0.0;
  double argument_squared_distance = 0.0;
#pragma unroll
  for (unsigned axis = 0; axis < 3U; ++axis) {{
    const double first_coordinate = generated_dppp_axis(first, axis);
    const double second_coordinate = generated_dppp_axis(second, axis);
    const double third_coordinate = generated_dppp_axis(third, axis);
    const double fourth_coordinate = generated_dppp_axis(fourth, axis);
    const double product_p =
        (alpha * first_coordinate + beta * second_coordinate) / p;
    const double product_q =
        (gamma * third_coordinate + delta * fourth_coordinate) / q;
    geometry.pair_shifts[0][axis] = product_p - first_coordinate;
    geometry.pair_shifts[1][axis] = product_p - second_coordinate;
    geometry.pair_shifts[2][axis] = product_q - third_coordinate;
    geometry.pair_shifts[3][axis] = product_q - fourth_coordinate;
    geometry.difference[axis] = product_p - product_q;
    geometry.decay_gradients[0][axis] =
        -2.0 * mu * (first_coordinate - second_coordinate);
    geometry.decay_gradients[1][axis] =
        2.0 * mu * (first_coordinate - second_coordinate);
    geometry.decay_gradients[2][axis] =
        -2.0 * nu * (third_coordinate - fourth_coordinate);
    pair_decay_exponent +=
        -mu * (first_coordinate - second_coordinate) *
            (first_coordinate - second_coordinate) -
        nu * (third_coordinate - fourth_coordinate) *
            (third_coordinate - fourth_coordinate);
    argument_squared_distance +=
        geometry.difference[axis] * geometry.difference[axis];
    geometry.coordinate_powers[axis][0] = 1.0;
#pragma unroll
    for (unsigned power = 1; power <= {maximum_order}U; ++power) {{
      geometry.coordinate_powers[axis][power] =
          geometry.coordinate_powers[axis][power - 1U] *
          geometry.difference[axis];
    }}
  }}
  boys_values<{maximum_order}>(
      geometry.rho * argument_squared_distance, geometry.boys);
  geometry.negative_two_rho_powers[0] = 1.0;
#pragma unroll
  for (unsigned power = 1; power <= {maximum_order}U; ++power) {{
    geometry.negative_two_rho_powers[power] =
        geometry.negative_two_rho_powers[power - 1U] *
        (-2.0 * geometry.rho);
  }}
  geometry.prefactor =
      34.986836655249725 / (p * q * sqrt(p + q)) *
      exp(pair_decay_exponent);
  geometry.primitive_coefficient = primitive_coefficient;
}}
"""
    return _specialize_dppp_identifiers(source, spec)
