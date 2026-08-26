"""Compact Rys/TRR/HRR programs for shell-specific force kernels.

The symbolic shell compiler expands final Cartesian derivatives into a scalar
expression DAG.  That representation is useful as a correctness oracle, but
it obscures the short-lived one-dimensional recurrence used by mature ERI
kernels.  This module keeps the recurrence states explicit so CUDA lowering
can generate each state once and contract it immediately with density weights.

The initial implementation deliberately specializes the mathematical oracle
to ``(p p|p s)`` and three Rys roots.  It preserves VibeQC's shell-task ABI and
component order; only the primitive recurrence is different.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .fused_schedule import FusedShellResult
from .rys3_data import (
    RYS3_DEGREE,
    RYS3_INTERVALS,
    RYS3_LARGEX_R_DATA,
    RYS3_LARGEX_W_DATA,
    RYS3_RW_DATA,
    RYS3_SMALLX_R0,
    RYS3_SMALLX_R1,
    RYS3_SMALLX_W0,
    RYS3_SMALLX_W1,
)
from .shell_spec import AXES, FUSED_SHELL_SPEC_BY_NAME, ShellClassSpec


class RysRecurrenceKind(str, Enum):
    """One operation in a one-dimensional Rys integral program."""

    SEED = "seed"
    TRR_BRA = "trr_bra"
    TRR_KET = "trr_ket"
    HRR_BRA = "hrr_bra"
    HRR_KET = "hrr_ket"


@dataclass(frozen=True, order=True, slots=True)
class RysState:
    """Angular momenta ``(a, b|c, d)`` for one Cartesian coordinate."""

    a: int
    b: int
    c: int
    d: int

    def __post_init__(self) -> None:
        if min(self.a, self.b, self.c, self.d) < 0:
            raise ValueError("Rys angular momenta must be non-negative")

    def replace(self, **changes: int) -> RysState:
        """Return a state with selected angular momenta replaced."""

        values = {"a": self.a, "b": self.b, "c": self.c, "d": self.d}
        values.update(changes)
        return RysState(**values)


@dataclass(frozen=True, slots=True)
class RysRecurrenceInstruction:
    """A unique recurrence state and its already-generated dependencies."""

    kind: RysRecurrenceKind
    state: RysState
    dependencies: tuple[RysState, ...]


@dataclass(frozen=True, slots=True)
class RysAxisProgram:
    """Topologically ordered state program shared by x, y, and z axes."""

    requested_states: tuple[RysState, ...]
    instructions: tuple[RysRecurrenceInstruction, ...]


@dataclass(frozen=True, slots=True)
class RysForceProgram:
    """Backend-independent Rys state program for one shell-class force."""

    spec: ShellClassSpec
    nroots: int
    independent_force_centers: tuple[int, ...]
    component_order: tuple[tuple[str, str, str, str], ...]
    axis_program: RysAxisProgram


def _instruction_for_state(state: RysState) -> RysRecurrenceInstruction:
    """Return the recurrence operation selected by GPU4PySCF's HRR order."""

    if state.b:
        dependencies = (
            state.replace(a=state.a + 1, b=state.b - 1),
            state.replace(b=state.b - 1),
        )
        kind = RysRecurrenceKind.HRR_BRA
    elif state.d:
        dependencies = (
            state.replace(c=state.c + 1, d=state.d - 1),
            state.replace(d=state.d - 1),
        )
        kind = RysRecurrenceKind.HRR_KET
    elif state.c:
        dependency_list = [state.replace(c=state.c - 1)]
        if state.c > 1:
            dependency_list.append(state.replace(c=state.c - 2))
        if state.a:
            dependency_list.append(
                state.replace(a=state.a - 1, c=state.c - 1)
            )
        dependencies = tuple(dependency_list)
        kind = RysRecurrenceKind.TRR_KET
    elif state.a:
        dependency_list = [state.replace(a=state.a - 1)]
        if state.a > 1:
            dependency_list.append(state.replace(a=state.a - 2))
        dependencies = tuple(dependency_list)
        kind = RysRecurrenceKind.TRR_BRA
    else:
        dependencies = ()
        kind = RysRecurrenceKind.SEED
    return RysRecurrenceInstruction(kind, state, dependencies)


def build_rys_axis_program(
    requested_states: Sequence[RysState],
) -> RysAxisProgram:
    """Build a deterministic, duplicate-free TRR/HRR dependency program."""

    normalized = tuple(dict.fromkeys(requested_states))
    emitted: set[RysState] = set()
    instructions: list[RysRecurrenceInstruction] = []

    def emit(state: RysState) -> None:
        if state in emitted:
            return
        instruction = _instruction_for_state(state)
        for dependency in instruction.dependencies:
            emit(dependency)
        emitted.add(state)
        instructions.append(instruction)

    for state in normalized:
        emit(state)
    return RysAxisProgram(normalized, tuple(instructions))


def _angular_counts(label: str) -> tuple[int, int, int]:
    return tuple(label.count(axis) for axis in AXES)


def build_rys_force_program(spec: ShellClassSpec) -> RysForceProgram:
    """Build compact value/first-derivative states for any catalog shell class.

    Only centers A/B/C are requested explicitly.  Center D is recovered from
    translation invariance, matching the existing VibeQC force ABI and avoiding
    the extra ket-HRR states used by GPU4PySCF's twelve-output implementation.
    """

    requested: list[RysState] = []
    for component in spec.components:
        quantums = tuple(_angular_counts(label) for label in component)
        for coordinate in range(3):
            base = RysState(*(item[coordinate] for item in quantums))
            requested.append(base)
            for center in range(3):
                values = [base.a, base.b, base.c, base.d]
                values[center] += 1
                requested.append(RysState(*values))
                if values[center] > 1:
                    values[center] -= 2
                    requested.append(RysState(*values))
    maximum_polynomial_order = sum(spec.angular) + 1
    return RysForceProgram(
        spec=spec,
        nroots=maximum_polynomial_order // 2 + 1,
        independent_force_centers=(0, 1, 2),
        component_order=spec.components,
        axis_program=build_rys_axis_program(requested),
    )


# Compatibility name retained for the original specialized prototype API.
PppsRysForceProgram = RysForceProgram


def build_ppps_rys_force_program() -> RysForceProgram:
    """Build the three-root program used by the experimental CUDA emitter."""

    return build_rys_force_program(FUSED_SHELL_SPEC_BY_NAME["ppps"])


def boys_values(argument: float, count: int) -> tuple[float, ...]:
    """Return a stable reference Boys sequence for code-generation tests."""

    if argument < 0.0:
        raise ValueError("the Boys argument must be non-negative")
    if count < 1:
        raise ValueError("at least one Boys value is required")
    # Upward recurrence loses the highest moments long before underflow when
    # x is small.  The alternating series is cheap in this host-only oracle
    # and remains accurate throughout the complete small-argument interval.
    if argument < 1.0:
        return tuple(
            sum(
                (-argument) ** term
                / (math.factorial(term) * (2 * order + 2 * term + 1))
                for term in range(32)
            )
            for order in range(count)
        )
    values = [
        0.5
        * math.sqrt(math.pi / argument)
        * math.erf(math.sqrt(argument))
    ]
    exponential = math.exp(-argument)
    for order in range(count - 1):
        values.append(
            ((2 * order + 1) * values[-1] - exponential)
            / (2.0 * argument)
        )
    return tuple(values)


def rys3_roots_weights(argument: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Construct a three-point Rys rule from its first six Boys moments.

    This moment-based solver is intentionally a host-side correctness oracle.
    The CUDA performance path uses a dedicated three-root approximation table;
    it must not lower this dense eigensolve into a device kernel.
    """

    moments = np.asarray(boys_values(argument, 6), dtype=np.float64)
    moment_matrix = np.asarray(
        [[moments[row + column] for column in range(3)] for row in range(3)]
    )
    shifted_matrix = np.asarray(
        [
            [moments[row + column + 1] for column in range(3)]
            for row in range(3)
        ]
    )
    cholesky = np.linalg.cholesky(moment_matrix)
    jacobi = np.linalg.solve(cholesky, shifted_matrix)
    jacobi = np.linalg.solve(cholesky, jacobi.T).T
    roots = np.linalg.eigvalsh(jacobi)
    vandermonde = np.vstack([roots**order for order in range(3)])
    weights = np.linalg.solve(vandermonde, moments[:3])
    return tuple(map(float, roots)), tuple(map(float, weights))


def rys3_table_roots_weights(
    argument: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Evaluate the GPU4PySCF-compatible three-root interpolation table."""

    if argument < 0.0:
        raise ValueError("the Rys argument must be non-negative")
    if argument < 3.0e-7:
        roots = tuple(
            RYS3_SMALLX_R0[index] + RYS3_SMALLX_R1[index] * argument
            for index in range(3)
        )
        weights = tuple(
            RYS3_SMALLX_W0[index] + RYS3_SMALLX_W1[index] * argument
            for index in range(3)
        )
        return roots, weights
    if argument > 50.0:
        scale = math.sqrt(0.7853981633974483096 / argument)
        roots = tuple(value / argument for value in RYS3_LARGEX_R_DATA)
        weights = tuple(value * scale for value in RYS3_LARGEX_W_DATA)
        return roots, weights

    interval = int(argument * 0.4)
    transformed = (argument - interval * 2.5) * 0.8 - 1.0
    twice_transformed = 2.0 * transformed

    def interpolate(series: int) -> float:
        offset = series * (RYS3_DEGREE + 1) * RYS3_INTERVALS
        c0 = RYS3_RW_DATA[
            offset + interval + RYS3_DEGREE * RYS3_INTERVALS
        ]
        c1 = RYS3_RW_DATA[
            offset + interval + (RYS3_DEGREE - 1) * RYS3_INTERVALS
        ]
        for degree in range(RYS3_DEGREE - 2, 0, -2):
            c2 = RYS3_RW_DATA[
                offset + interval + degree * RYS3_INTERVALS
            ] - c1
            c3 = c0 + c1 * twice_transformed
            c1 = c2 + c3 * twice_transformed
            c0 = RYS3_RW_DATA[
                offset + interval + (degree - 1) * RYS3_INTERVALS
            ] - c3
        return c0 + c1 * transformed

    values = tuple(interpolate(series) for series in range(6))
    return values[::2], values[1::2]


def _format_cuda_values(values: Sequence[float], columns: int = 4) -> str:
    """Format deterministic double literals for generated CUDA tables."""

    lines = []
    for begin in range(0, len(values), columns):
        lines.append(
            "    "
            + ", ".join(f"{value:.17e}" for value in values[begin : begin + columns])
            + ","
        )
    return "\n".join(lines)


def emit_rys3_roots_cuda() -> str:
    """Emit the Apache-attributed, GPU4PySCF-compatible Rys3 evaluator."""

    small_r0 = _format_cuda_values(RYS3_SMALLX_R0, columns=3)
    small_r1 = _format_cuda_values(RYS3_SMALLX_R1, columns=3)
    small_w0 = _format_cuda_values(RYS3_SMALLX_W0, columns=3)
    small_w1 = _format_cuda_values(RYS3_SMALLX_W1, columns=3)
    large_r = _format_cuda_values(RYS3_LARGEX_R_DATA, columns=3)
    large_w = _format_cuda_values(RYS3_LARGEX_W_DATA, columns=3)
    table = _format_cuda_values(RYS3_RW_DATA)
    return f"""/*
 * Three-root interpolation adapted from GPU4PySCF.
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */
__device__ double generated_ppps_rys3_small_r0[3] = {{
{small_r0}
}};
__device__ double generated_ppps_rys3_small_r1[3] = {{
{small_r1}
}};
__device__ double generated_ppps_rys3_small_w0[3] = {{
{small_w0}
}};
__device__ double generated_ppps_rys3_small_w1[3] = {{
{small_w1}
}};
__device__ double generated_ppps_rys3_large_r[3] = {{
{large_r}
}};
__device__ double generated_ppps_rys3_large_w[3] = {{
{large_w}
}};
__device__ double generated_ppps_rys3_rw[{len(RYS3_RW_DATA)}] = {{
{table}
}};

__device__ __noinline__ void generated_ppps_rys3_roots(
    double argument, double* roots_weights, unsigned stride) {{
  if (argument < 3.0e-7) {{
#pragma unroll
    for (unsigned root = 0; root < 3U; ++root) {{
      roots_weights[(2U * root) * stride] =
          generated_ppps_rys3_small_r0[root] +
          generated_ppps_rys3_small_r1[root] * argument;
      roots_weights[(2U * root + 1U) * stride] =
          generated_ppps_rys3_small_w0[root] +
          generated_ppps_rys3_small_w1[root] * argument;
    }}
    return;
  }}
  if (argument > 50.0) {{
    const double scale = sqrt(0.7853981633974483096 / argument);
#pragma unroll
    for (unsigned root = 0; root < 3U; ++root) {{
      roots_weights[(2U * root) * stride] =
          generated_ppps_rys3_large_r[root] / argument;
      roots_weights[(2U * root + 1U) * stride] =
          generated_ppps_rys3_large_w[root] * scale;
    }}
    return;
  }}

  const int interval = static_cast<int>(argument * 0.4);
  const double transformed =
      (argument - static_cast<double>(interval) * 2.5) * 0.8 - 1.0;
  const double twice_transformed = 2.0 * transformed;
#pragma unroll
  for (unsigned series = 0; series < 6U; ++series) {{
    const double* coefficients = generated_ppps_rys3_rw +
        series * {RYS3_DEGREE + 1}U * {RYS3_INTERVALS}U;
    double c0 = coefficients[
        interval + {RYS3_DEGREE}U * {RYS3_INTERVALS}U];
    double c1 = coefficients[
        interval + {RYS3_DEGREE - 1}U * {RYS3_INTERVALS}U];
    double c2 = 0.0;
    double c3 = 0.0;
#pragma unroll
    for (int degree = {RYS3_DEGREE - 2}; degree > 0; degree -= 2) {{
      c2 = coefficients[interval + degree * {RYS3_INTERVALS}] - c1;
      c3 = c0 + c1 * twice_transformed;
      c1 = c2 + c3 * twice_transformed;
      c0 = coefficients[
          interval + (degree - 1) * {RYS3_INTERVALS}] - c3;
    }}
    roots_weights[series * stride] = c0 + c1 * transformed;
  }}
}}
"""


def _state_expression(
    axis: str,
    instruction: RysRecurrenceInstruction,
    slots: Mapping[tuple[str, RysState], int],
) -> str:
    """Lower one recurrence instruction to the compact scalar CUDA form."""

    state = instruction.state
    dependency = tuple(
        f"rys_state_{slots[(axis, item)]}" for item in instruction.dependencies
    )
    if instruction.kind == RysRecurrenceKind.SEED:
        return "weighted_root" if axis == "z" else "1.0"
    if instruction.kind == RysRecurrenceKind.TRR_BRA:
        expression = f"c0{axis} * {dependency[0]}"
        if state.a > 1:
            expression += f" + {state.a - 1}.0 * b10 * {dependency[1]}"
        return expression
    if instruction.kind == RysRecurrenceKind.TRR_KET:
        expression = f"cp{axis} * {dependency[0]}"
        dependency_index = 1
        if state.c > 1:
            expression += (
                f" + {state.c - 1}.0 * b01 * {dependency[dependency_index]}"
            )
            dependency_index += 1
        if state.a:
            expression += f" + {state.a}.0 * b00 * {dependency[dependency_index]}"
        return expression
    if instruction.kind == RysRecurrenceKind.HRR_BRA:
        return f"{dependency[0]} - ab{axis} * {dependency[1]}"
    return f"{dependency[0]} - cd{axis} * {dependency[1]}"


def emit_ppps_rys3_root_body_cuda() -> str:
    """Emit one root's state-on-first-use recurrence and force contraction.

    The caller provides compact primitive scalars, ``weighted_root`` (Rys
    weight times the signed primitive prefactor), shared component weights,
    and nine register force accumulators.  States are introduced immediately
    before their first component use so PTXAS can reuse slots after last use.
    """

    program = build_ppps_rys_force_program()

    def emit_group(
        component_begin: int,
        components: Sequence[tuple[str, str, str, str]],
    ) -> list[str]:
        # Nine adjacent components retain the useful i/j/k recurrence reuse.
        # Explicit slot reuse bounds the group to roughly 31 state doubles.
        emitted: set[tuple[str, RysState]] = set()
        events: list[dict[str, object]] = []

        def ensure(axis: str, state: RysState) -> tuple[str, RysState]:
            key = (axis, state)
            if key in emitted:
                return key
            instruction = _instruction_for_state(state)
            for dependency in instruction.dependencies:
                ensure(axis, dependency)
            emitted.add(key)
            events.append(
                {
                    "kind": "define",
                    "key": key,
                    "axis": axis,
                    "instruction": instruction,
                }
            )
            return key

        for local_index, component in enumerate(components):
            quantums = tuple(_angular_counts(label) for label in component)
            base_states = tuple(
                RysState(*(item[coordinate] for item in quantums))
                for coordinate in range(3)
            )
            base = tuple(
                ensure(axis, state)
                for axis, state in zip(AXES, base_states, strict=True)
            )
            derivatives: dict[
                tuple[int, int],
                tuple[
                    str,
                    tuple[str, RysState],
                    int,
                    tuple[str, RysState] | None,
                ],
            ] = {}
            for center in program.independent_force_centers:
                for coordinate, axis in enumerate(AXES):
                    state = base_states[coordinate]
                    values = [state.a, state.b, state.c, state.d]
                    values[center] += 1
                    raised = ensure(axis, RysState(*values))
                    exponent = ("alpha2", "beta2", "gamma2")[center]
                    angular = (state.a, state.b, state.c, state.d)[center]
                    lowered = None
                    if angular:
                        values[center] -= 2
                        lowered = ensure(axis, RysState(*values))
                    derivatives[(center, coordinate)] = (
                        exponent,
                        raised,
                        angular,
                        lowered,
                    )
            events.append(
                {
                    "kind": "contract",
                    "component": component_begin + local_index,
                    "base": base,
                    "derivatives": derivatives,
                }
            )

        last_use: dict[tuple[str, RysState], int] = {}
        for event_index, event in enumerate(events):
            if event["kind"] == "define":
                instruction = event["instruction"]
                axis = event["axis"]
                assert isinstance(instruction, RysRecurrenceInstruction)
                assert isinstance(axis, str)
                used = tuple((axis, item) for item in instruction.dependencies)
            else:
                base = event["base"]
                derivatives = event["derivatives"]
                assert isinstance(base, tuple)
                assert isinstance(derivatives, dict)
                used_list = list(base)
                for _, raised, _, lowered in derivatives.values():
                    used_list.append(raised)
                    if lowered is not None:
                        used_list.append(lowered)
                used = tuple(used_list)
            for key in used:
                last_use[key] = event_index

        slots: dict[tuple[str, RysState], int] = {}
        available: list[int] = []
        active: dict[tuple[str, RysState], int] = {}
        next_slot = 0
        for event_index, event in enumerate(events):
            if event["kind"] == "define":
                key = event["key"]
                assert isinstance(key, tuple)
                slot = available.pop() if available else next_slot
                if slot == next_slot:
                    next_slot += 1
                slots[key] = slot
                active[key] = slot
            expired = [
                key
                for key in active
                if last_use.get(key, event_index) == event_index
            ]
            for key in expired:
                available.append(active.pop(key))

        lines = ["    {"]
        lines.extend(
            f"      double rys_state_{slot};" for slot in range(next_slot)
        )
        for event in events:
            if event["kind"] == "define":
                key = event["key"]
                axis = event["axis"]
                instruction = event["instruction"]
                assert isinstance(key, tuple)
                assert isinstance(axis, str)
                assert isinstance(instruction, RysRecurrenceInstruction)
                lines.append(
                    f"      rys_state_{slots[key]} = "
                    f"{_state_expression(axis, instruction, slots)};"
                )
                continue

            component_index = event["component"]
            base = event["base"]
            derivatives = event["derivatives"]
            assert isinstance(component_index, int)
            assert isinstance(base, tuple)
            assert isinstance(derivatives, dict)
            base_names = tuple(f"rys_state_{slots[key]}" for key in base)
            lines.extend(
                [
                    "      {",
                    f"        const double density_weight = component_weights[{component_index}U][lane];",
                    f"        const double product_xy = {base_names[0]} * {base_names[1]} * density_weight;",
                    f"        const double product_xz = {base_names[0]} * {base_names[2]} * density_weight;",
                    f"        const double product_yz = {base_names[1]} * {base_names[2]} * density_weight;",
                ]
            )
            products = ("product_yz", "product_xz", "product_xy")
            for center in program.independent_force_centers:
                for coordinate in range(3):
                    exponent, raised, angular, lowered = derivatives[
                        (center, coordinate)
                    ]
                    expression = f"{exponent} * rys_state_{slots[raised]}"
                    if lowered is not None:
                        expression += (
                            f" - {angular}.0 * rys_state_{slots[lowered]}"
                        )
                    force = center * 3 + coordinate
                    lines.append(
                        f"        force_{force} += ({expression}) * "
                        f"{products[coordinate]};"
                    )
            lines.append("      }")
        lines.append("    }")
        return lines

    lines: list[str] = []
    component_group = 9
    for begin in range(0, len(program.component_order), component_group):
        lines.extend(
            emit_group(
                begin,
                program.component_order[begin : begin + component_group],
            )
        )
    return "\n".join(lines)


def _evaluate_axis_state(
    requested: RysState,
    axis: str,
    root: float,
    variables: Mapping[str, float],
) -> float:
    """Evaluate one one-dimensional state with compact TRR followed by HRR."""

    p = 0.5 / variables["inverse_two_p"]
    q = 0.5 / variables["inverse_two_q"]
    pair_difference = variables[f"difference_{axis}"]
    ab = variables[f"pa_{axis}"] - variables[f"pb_{axis}"]
    cd = variables[f"qc_{axis}"] - variables[f"qd_{axis}"]
    root_over_sum = root / (p + q)
    root_bra = root_over_sum * q
    root_ket = root_over_sum * p
    c0 = variables[f"pa_{axis}"] - pair_difference * root_bra
    cp = variables[f"qc_{axis}"] + pair_difference * root_ket
    b10 = 0.5 / p * (1.0 - root_bra)
    b00 = 0.5 * root_over_sum
    b01 = 0.5 / q * (1.0 - root_ket)
    cache: dict[RysState, float] = {}

    def evaluate(state: RysState) -> float:
        cached = cache.get(state)
        if cached is not None:
            return cached
        instruction = _instruction_for_state(state)
        if instruction.kind == RysRecurrenceKind.SEED:
            value = 1.0
        elif instruction.kind == RysRecurrenceKind.TRR_BRA:
            value = c0 * evaluate(instruction.dependencies[0])
            if state.a > 1:
                value += (state.a - 1) * b10 * evaluate(
                    instruction.dependencies[1]
                )
        elif instruction.kind == RysRecurrenceKind.TRR_KET:
            value = cp * evaluate(instruction.dependencies[0])
            dependency = 1
            if state.c > 1:
                value += (state.c - 1) * b01 * evaluate(
                    instruction.dependencies[dependency]
                )
                dependency += 1
            if state.a:
                value += state.a * b00 * evaluate(
                    instruction.dependencies[dependency]
                )
        elif instruction.kind == RysRecurrenceKind.HRR_BRA:
            value = evaluate(instruction.dependencies[0]) - ab * evaluate(
                instruction.dependencies[1]
            )
        else:
            value = evaluate(instruction.dependencies[0]) - cd * evaluate(
                instruction.dependencies[1]
            )
        cache[state] = value
        return value

    return evaluate(requested)


def evaluate_ppps_rys_component(
    component: Sequence[str],
    variables: Mapping[str, float],
) -> FusedShellResult:
    """Evaluate one primitive ``ppps`` component through three-root Rys HRR."""

    program = build_ppps_rys_force_program()
    normalized = program.spec.validate_component(component)
    quantums = tuple(_angular_counts(label) for label in normalized)
    argument = variables["rho"] * sum(
        variables[f"difference_{axis}"] ** 2 for axis in AXES
    )
    roots, weights = rys3_roots_weights(argument)
    p = 0.5 / variables["inverse_two_p"]
    q = 0.5 / variables["inverse_two_q"]
    exponents = (
        p * variables["first_product_scale"],
        p * variables["second_product_scale"],
        q * variables["third_product_scale"],
        q * (1.0 - variables["third_product_scale"]),
    )
    value = 0.0
    gradients = [[0.0, 0.0, 0.0] for _ in range(3)]
    for root, weight in zip(roots, weights, strict=True):
        base_states = tuple(
            RysState(*(item[coordinate] for item in quantums))
            for coordinate in range(3)
        )
        base_values = tuple(
            _evaluate_axis_state(state, axis, root, variables)
            for state, axis in zip(base_states, AXES, strict=True)
        )
        value += weight * math.prod(base_values)
        for center in program.independent_force_centers:
            for coordinate, axis in enumerate(AXES):
                base = base_states[coordinate]
                values = [base.a, base.b, base.c, base.d]
                values[center] += 1
                derivative = 2.0 * exponents[center] * _evaluate_axis_state(
                    RysState(*values), axis, root, variables
                )
                angular_order = (base.a, base.b, base.c, base.d)[center]
                if angular_order:
                    values[center] -= 2
                    derivative -= angular_order * _evaluate_axis_state(
                        RysState(*values), axis, root, variables
                    )
                other_axes = tuple(
                    index for index in range(3) if index != coordinate
                )
                gradients[center][coordinate] += (
                    weight
                    * derivative
                    * base_values[other_axes[0]]
                    * base_values[other_axes[1]]
                )
    prefactor = variables["prefactor"]
    independent = tuple(
        tuple(prefactor * gradients[center][axis] for axis in range(3))
        for center in range(3)
    )
    fourth = tuple(
        -sum(independent[center][axis] for center in range(3))
        for axis in range(3)
    )
    return FusedShellResult(
        value=prefactor * value,
        gradients=independent + (fourth,),
    )
