"""Qualify one imported Libxc functional against the production-domain matrix.

This is an optional evidence producer.  The independent oracle is PySCF 2.14.0
backed by Libxc 7.0.0; the candidate is the pinned VibeQC bulk Graph/runtime
contract.  The tool writes a complete fail-closed receipt even when boundary or
control rows are not yet admissible.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from vibeqc_compiler.xc.bulk_runtime import (
    PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    build_bulk_runtime_program,
)
from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability
from vibeqc_compiler.xc.production_domain_cases import (
    ProductionDomainCase,
    control_case_ids,
    numerical_cases,
)
from vibeqc_compiler.xc.production_domain_controls import run_control_case
from vibeqc_compiler.xc.production_domain_evidence import (
    build_execution_binding,
    build_result,
    stage_evidence,
)

if TYPE_CHECKING:
    from vibeqc_compiler.xc.bulk_runtime import BulkRuntimeProgram
    from vibeqc_compiler.xc.libxc_production_domain import ProductionDomainProfile

CAMPAIGN_SCHEMA = "vibeqc.libxc-production-domain-campaign/v2"


def _reference(
    name: str,
    case: ProductionDomainCase,
    *,
    family: str,
    libxc: Any,
) -> np.ndarray:
    """Return energy per volume plus first physical feature derivatives."""
    polarized = case.spin == "polarized"
    raw = np.asarray(case.pyscf_rho(), dtype=np.float64)
    rho = raw[..., None] if polarized else raw[0, :, None]

    if family == "lda":
        libxc_rho = rho[:, 0] if polarized else rho[0]
        size = 2 if polarized else 1
    elif family == "gga":
        libxc_rho = rho[:, :4] if polarized else rho[:4]
        size = 5 if polarized else 2
    elif family == "mgga":
        libxc_rho = rho
        size = 7 if polarized else 3
    else:
        raise ValueError(f"unsupported semilocal family {family!r}")

    exc, vxc, _, _ = libxc.eval_xc(
        name,
        libxc_rho,
        spin=1 if polarized else 0,
        deriv=1,
    )
    total_rho = rho[:, 0].sum(axis=0) if polarized else rho[0]
    gradient = np.zeros((size, 1), dtype=np.float64)

    if polarized:
        gradient[:2] = vxc[0].T
        if family != "lda":
            gradient[2:5] = vxc[1].T
        if family == "mgga":
            gradient[5:7] = vxc[3].T
    else:
        gradient[0] = vxc[0]
        if family != "lda":
            gradient[1] = vxc[1]
        if family == "mgga":
            gradient[2] = vxc[3]

    return np.concatenate(((total_rho * exc)[None], gradient))[:, 0]


def _relative_error(
    observed: np.ndarray, expected: np.ndarray, *, atol: float
) -> float | None:
    """Return zero for exact zeros, or null when no finite ratio exists."""
    scale = np.maximum(np.abs(expected), atol)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        difference = np.abs(observed - expected)
        ratios = np.divide(
            difference, scale, out=np.zeros_like(difference), where=scale != 0.0
        )
    if np.any((scale == 0.0) & (difference != 0.0)):
        return None
    maximum = float(np.max(ratios))
    return maximum if math.isfinite(maximum) else None


def _validate_tolerances(rtol: float, atol: float) -> None:
    """Prevent nonfinite comparison gates from admitting arbitrary candidates."""
    if any(
        isinstance(value, bool) or not math.isfinite(value) or value < 0.0
        for value in (rtol, atol)
    ):
        raise ValueError("qualification tolerances must be finite and nonnegative")


def _run_numeric_case(
    name: str,
    case: ProductionDomainCase,
    *,
    family: str,
    profile: ProductionDomainProfile,
    program: BulkRuntimeProgram,
    libxc: Any,
    rtol: float,
    atol: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_tolerances(rtol, atol)
    feature_names, feature_values = case.runtime_features(profile)
    detail: dict[str, Any] = {
        "spin": case.spin,
        "case_id": case.case_id,
        "feature_names": list(feature_names),
        "features": list(feature_values),
    }
    row = {
        "spin": case.spin,
        "case_id": case.case_id,
        "outputs": list(profile.outputs),
    }

    try:
        expected = _reference(name, case, family=family, libxc=libxc)
    except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
        reason = f"independent oracle failed: {type(exc).__name__}: {exc}"
        return {**row, "status": "fail", "reason": reason}, {
            **detail,
            "status": "fail",
            "reason": reason,
        }
    if not np.all(np.isfinite(expected)):
        reason = "independent Libxc oracle produced nonfinite E/vxc"
        return {**row, "status": "fail", "reason": reason}, {
            **detail,
            "status": "fail",
            "reason": reason,
        }

    try:
        if program.spec.features != feature_names:
            raise ValueError(
                "candidate feature layout mismatch: "
                f"{program.spec.features!r} != {feature_names!r}"
            )
        observed = program.evaluate(
            np.asarray(feature_values, dtype=np.float64).reshape(-1, 1)
        )[:, 0]
    except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
        reason = f"production candidate failed: {type(exc).__name__}: {exc}"
        return {**row, "status": "fail", "reason": reason}, {
            **detail,
            "status": "fail",
            "reason": reason,
            "reference": expected.tolist(),
        }

    shape_ok = observed.shape == expected.shape
    finite = bool(np.all(np.isfinite(observed)))
    passed = (
        shape_ok
        and finite
        and bool(np.allclose(observed, expected, rtol=rtol, atol=atol))
    )
    with np.errstate(over="ignore", invalid="ignore"):
        max_abs = float(np.max(np.abs(observed - expected))) if shape_ok else None
    if max_abs is not None and not math.isfinite(max_abs):
        max_abs = None
    max_rel = _relative_error(observed, expected, atol=atol) if shape_ok else None
    reason = None
    if not finite:
        reason = "production candidate produced nonfinite E/vxc"
    elif not shape_ok:
        reason = (
            f"candidate/reference shape mismatch: {observed.shape!r} "
            f"!= {expected.shape!r}"
        )
    elif not passed:
        reason = (
            f"candidate/reference mismatch: max_abs={max_abs!r}, max_rel={max_rel!r}"
        )

    status = "pass" if passed else "fail"
    return {**row, "status": status, "reason": reason}, {
        **detail,
        "status": status,
        "reason": reason,
        "reference": expected.tolist(),
        "observed": [
            float(value) if np.isfinite(value) else None for value in observed
        ],
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="imported Libxc registration name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        required=True,
        help="retained evidence reference embedded into the exact receipt",
    )
    parser.add_argument("--rtol", type=float, default=2.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="return nonzero when the exact matrix is not fully qualified",
    )
    return parser.parse_args()


def qualify_functional(
    name: str,
    *,
    evidence: str,
    pyscf_version: str,
    libxc: Any,
    rtol: float = 2.0e-6,
    atol: float = 1.0e-8,
) -> dict[str, Any]:
    """Run one exact production-domain campaign without writing files."""
    if pyscf_version != "2.14.0" or libxc.__version__ != "7.0.0":
        raise RuntimeError("qualification requires exactly PySCF 2.14.0 / Libxc 7.0.0")
    _validate_tolerances(rtol, atol)
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("qualification requires a nonempty evidence reference")

    capability = functional_capability(name)
    profile = capability.production_domain_profile
    if not profile.eligible:
        raise RuntimeError(
            f"{capability.name} is structurally blocked: {profile.blocker}"
        )
    if libxc.xc_type(capability.name).lower() != profile.family:
        raise RuntimeError("independent Libxc family disagrees with imported catalog")

    programs = {
        spin: build_bulk_runtime_program(
            capability.name,
            spin=spin,
            order=1,
            domain=PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
        )
        for spin in profile.spin_layouts
    }
    execution = build_execution_binding(capability.name, programs)

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for spin in profile.spin_layouts:
        by_id = {case.case_id: case for case in numerical_cases(profile, spin=spin)}
        controls = set(control_case_ids(profile, spin=spin))
        for case_id in profile.case_ids_for_spin(spin):
            if case_id in controls:
                row, detail = run_control_case(
                    capability.name,
                    spin=spin,
                    case_id=case_id,
                    program=programs[spin],
                )
            else:
                row, detail = _run_numeric_case(
                    capability.name,
                    by_id[case_id],
                    family=profile.family,
                    profile=profile,
                    program=programs[spin],
                    libxc=libxc,
                    rtol=rtol,
                    atol=atol,
                )
            rows.append(row)
            details.append(detail)

    receipt = build_result(
        capability.name,
        rows,
        evidence=evidence,
        execution=execution,
    )
    envelope = stage_evidence(capability.name, receipt)
    return {
        "schema": CAMPAIGN_SCHEMA,
        "functional": capability.name,
        "capability_identity": capability.identity,
        "profile": profile.to_payload(),
        "execution": execution,
        "oracle": {
            "pyscf": pyscf_version,
            "libxc": libxc.__version__,
            "api": "pyscf.dft.libxc.eval_xc deriv=1",
        },
        "tolerance": {"rtol": rtol, "atol": atol},
        "details": details,
        "receipt": receipt,
        "stage_evidence": envelope,
    }


def main() -> int:
    args = _parse_args()
    _validate_tolerances(args.rtol, args.atol)
    import pyscf
    from pyscf.dft import libxc

    payload = qualify_functional(
        args.name,
        evidence=args.evidence,
        pyscf_version=pyscf.__version__,
        libxc=libxc,
        rtol=args.rtol,
        atol=args.atol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    envelope = payload["stage_evidence"]
    rows = payload["receipt"]["cases"]
    print(
        f"{payload['functional']}: {envelope['status']} -> {args.output} "
        f"({len(rows)} exact matrix rows)"
    )
    return 1 if args.require_pass and envelope["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
