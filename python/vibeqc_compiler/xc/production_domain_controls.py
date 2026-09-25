"""Shared control-case qualification for bulk Libxc production admission.

Control rows are compiler/runtime invariants rather than functional numerical
coordinates.  They are still bound to each functional's exact capability
identity by the surrounding production-domain receipt.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.cuda import CudaEmitter
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter

from .bulk_runtime import build_bulk_runtime_program
from .libxc_bulk_capabilities import functional_capability
from .spec import UnsupportedXC

if TYPE_CHECKING:
    from .bulk_runtime import BulkRuntimeProgram

CONTROL_SEMANTICS = "libxc-production-domain-controls/v1"
_LAZY_CASE = "control/lazy-inactive-branch"
_NONFINITE_CASE = "control/invalid-nonfinite"


def _pass_row(
    name: str, spin: str, case_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    capability = functional_capability(name)
    outputs = list(capability.production_domain_profile.outputs)
    row = {
        "spin": spin,
        "case_id": case_id,
        "status": "pass",
        "outputs": outputs,
        "reason": None,
    }
    detail = {
        "spin": spin,
        "case_id": case_id,
        "status": "pass",
        "reason": None,
        "control_semantics": CONTROL_SEMANTICS,
    }
    return row, detail


def _fail_row(
    name: str,
    spin: str,
    case_id: str,
    reason: str,
    *,
    detail: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capability = functional_capability(name)
    outputs = list(capability.production_domain_profile.outputs)
    row = {
        "spin": spin,
        "case_id": case_id,
        "status": "fail",
        "outputs": outputs,
        "reason": reason,
    }
    payload = {
        "spin": spin,
        "case_id": case_id,
        "status": "fail",
        "reason": reason,
        "control_semantics": CONTROL_SEMANTICS,
    }
    if detail:
        payload.update(detail)
    return row, payload


def _lazy_inactive_branch(
    name: str, spin: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove value/vxc/fxc-style derivatives never touch the inactive branch."""
    graph = Graph()
    x = graph.variable("x")
    selected = graph.select_le(x, 0, x * x, 1 / x)
    first = graph.differentiate(selected, x)
    second = graph.differentiate(first, x)
    roots = (selected, first, second)

    try:
        scalar = tuple(graph.evaluate(root, {"x": 0.0}) for root in roots)
        array = evaluate_array_graph(
            graph,
            roots,
            {"x": np.asarray([-1.0, 0.0, 2.0], dtype=np.float64)},
        )
        scalar_emitter = ScalarCEmitter(graph, {"x": "x"})
        scalar_emitter.emit(roots)
        cuda_emitter = CudaEmitter(graph, {"x": "x"})
        cuda_emitter.emit(roots)
    except (ArithmeticError, FloatingPointError, RuntimeError, ValueError) as exc:
        return _fail_row(
            name,
            spin,
            _LAZY_CASE,
            f"lazy branch contract raised {type(exc).__name__}: {exc}",
        )

    expected_scalar = (0.0, 0.0, 2.0)
    expected_value = np.asarray([1.0, 0.0, 0.5])
    expected_first = np.asarray([-2.0, 0.0, -0.25])
    expected_second = np.asarray([2.0, 2.0, 0.25])
    source_c = "\n".join(scalar_emitter.lines)
    source_cuda = "\n".join(cuda_emitter.lines)

    checks = {
        "scalar_exact": scalar == expected_scalar,
        "array_value": bool(np.array_equal(array[0], expected_value)),
        "array_first": bool(np.array_equal(array[1], expected_first)),
        "array_second": bool(np.array_equal(array[2], expected_second)),
        "scalar_emits_branch": "if (x <= 0.0)" in source_c,
        "cuda_emits_branch": "if (x <= 0.0)" in source_cuda,
        "singular_branch_retained": "1.0 / x" in source_c and "1.0 / x" in source_cuda,
    }
    if not all(checks.values()):
        return _fail_row(
            name,
            spin,
            _LAZY_CASE,
            "lazy branch value/derivative/emitter contract mismatch",
            detail={"checks": checks, "scalar": list(scalar)},
        )

    row, detail = _pass_row(name, spin, _LAZY_CASE)
    detail.update({"checks": checks, "scalar": list(scalar)})
    return row, detail


def _invalid_nonfinite(name: str, spin: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require every nonfinite feature to fail before mathematical evaluation."""
    program = build_bulk_runtime_program(name, spin=spin, order=2)
    feature_count = len(program.spec.features)
    baseline = np.ones((feature_count, 1), dtype=np.float64)
    rejected = 0
    failures: list[str] = []

    for index, feature in enumerate(program.spec.features):
        for label, value in (
            ("nan", math.nan),
            ("positive-infinity", math.inf),
            ("negative-infinity", -math.inf),
        ):
            candidate = baseline.copy()
            candidate[index, 0] = value
            try:
                program.evaluate(candidate)
            except UnsupportedXC as exc:
                if "nonfinite" not in str(exc):
                    failures.append(
                        f"{feature}:{label}: unexpected rejection reason {exc}"
                    )
                else:
                    rejected += 1
            except (
                ArithmeticError,
                FloatingPointError,
                RuntimeError,
                ValueError,
            ) as exc:
                failures.append(
                    f"{feature}:{label}: escaped validator as "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                failures.append(f"{feature}:{label}: nonfinite input was accepted")

    expected = feature_count * 3
    if failures or rejected != expected:
        return _fail_row(
            name,
            spin,
            _NONFINITE_CASE,
            "nonfinite feature rejection contract failed",
            detail={
                "features": list(program.spec.features),
                "rejected": rejected,
                "expected": expected,
                "failures": failures,
            },
        )

    row, detail = _pass_row(name, spin, _NONFINITE_CASE)
    detail.update(
        {
            "features": list(program.spec.features),
            "rejected": rejected,
            "expected": expected,
        }
    )
    return row, detail


def run_control_case(
    name: str,
    *,
    spin: str,
    case_id: str,
    program: BulkRuntimeProgram | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one exact v2 control row without fabricating numerical XC data."""
    capability = functional_capability(name)
    profile = capability.production_domain_profile
    if spin not in profile.spin_layouts:
        raise ValueError(f"unsupported production-domain spin layout {spin!r}")
    if case_id not in profile.case_ids_for_spin(spin):
        raise ValueError(f"control case is outside the exact profile: {case_id!r}")
    if case_id == _LAZY_CASE:
        return _lazy_inactive_branch(capability.name, spin)
    if case_id == _NONFINITE_CASE:
        return _invalid_nonfinite(capability.name, spin, program)
    raise ValueError(f"unsupported production-domain control case {case_id!r}")


__all__ = ["CONTROL_SEMANTICS", "run_control_case"]
