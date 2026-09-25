"""Qualify one imported Libxc functional against the production-domain matrix.

This is an optional evidence producer.  The independent oracle is PySCF 2.14.0
backed by Libxc 7.0.0; the candidate is the pinned VibeQC bulk Graph/runtime
contract.  The tool writes a complete fail-closed receipt even when boundary or
control rows are not yet admissible.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from vibeqc_compiler.xc.bulk_runtime import build_bulk_runtime_program
from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability
from vibeqc_compiler.xc.production_domain_cases import (
    ProductionDomainCase,
    control_case_ids,
    numerical_cases,
)
from vibeqc_compiler.xc.production_domain_evidence import (
    build_result,
    stage_evidence,
)

if TYPE_CHECKING:
    from vibeqc_compiler.xc.libxc_production_domain import ProductionDomainProfile

CAMPAIGN_SCHEMA = "vibeqc.libxc-production-domain-campaign/v1"


def _reference(
    name: str,
    case: ProductionDomainCase,
    *,
    family: str,
    libxc: Any,
) -> np.ndarray:
    """Return energy, physical-feature gradient, and packed Hessian."""
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

    exc, vxc, fxc, _ = libxc.eval_xc(
        name,
        libxc_rho,
        spin=1 if polarized else 0,
        deriv=2,
    )
    total_rho = rho[:, 0].sum(axis=0) if polarized else rho[0]
    gradient = np.zeros((size, 1), dtype=np.float64)
    hessian = np.zeros((size, size, 1), dtype=np.float64)

    if polarized:
        gradient[:2] = vxc[0].T
        for index, (left, right) in enumerate(((0, 0), (0, 1), (1, 1))):
            hessian[left, right] = hessian[right, left] = fxc[0][:, index]
        if family != "lda":
            gradient[2:5] = vxc[1].T
            for index, (left, right) in enumerate(
                (left, right) for left in range(2) for right in range(2, 5)
            ):
                hessian[left, right] = hessian[right, left] = fxc[1][:, index]
            for index, (left, right) in enumerate(
                combinations_with_replacement(range(2, 5), 2)
            ):
                hessian[left, right] = hessian[right, left] = fxc[2][:, index]
        if family == "mgga":
            gradient[5:7] = vxc[3].T
            for index, (left, right) in enumerate(((5, 5), (5, 6), (6, 6))):
                hessian[left, right] = hessian[right, left] = fxc[4][:, index]
            for index, (left, right) in enumerate(((0, 5), (0, 6), (1, 5), (1, 6))):
                hessian[left, right] = hessian[right, left] = fxc[6][:, index]
            for index, (left, right) in enumerate(
                ((2, 5), (2, 6), (3, 5), (3, 6), (4, 5), (4, 6))
            ):
                hessian[left, right] = hessian[right, left] = fxc[9][:, index]
    else:
        gradient[0], hessian[0, 0] = vxc[0], fxc[0]
        if family != "lda":
            gradient[1], hessian[1, 1] = vxc[1], fxc[2]
            hessian[0, 1] = hessian[1, 0] = fxc[1]
        if family == "mgga":
            gradient[2], hessian[2, 2] = vxc[3], fxc[4]
            hessian[0, 2] = hessian[2, 0] = fxc[6]
            hessian[1, 2] = hessian[2, 1] = fxc[9]

    packed = np.stack(
        [
            hessian[left, right]
            for left, right in combinations_with_replacement(range(size), 2)
        ]
    )
    return np.concatenate(((total_rho * exc)[None], gradient, packed))[:, 0]


def _relative_error(
    observed: np.ndarray, expected: np.ndarray, *, atol: float
) -> float:
    scale = np.maximum(np.abs(expected), atol)
    return float(np.max(np.abs(observed - expected) / scale))


def _run_numeric_case(
    name: str,
    case: ProductionDomainCase,
    *,
    family: str,
    profile: ProductionDomainProfile,
    libxc: Any,
    rtol: float,
    atol: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        reason = "independent Libxc oracle produced nonfinite E/vxc/fxc"
        return {**row, "status": "fail", "reason": reason}, {
            **detail,
            "status": "fail",
            "reason": reason,
        }

    try:
        program = build_bulk_runtime_program(name, spin=case.spin, order=2)
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
    max_abs = float(np.max(np.abs(observed - expected))) if shape_ok else None
    max_rel = _relative_error(observed, expected, atol=atol) if shape_ok else None
    reason = None
    if not finite:
        reason = "production candidate produced nonfinite E/vxc/fxc"
    elif not shape_ok:
        reason = (
            f"candidate/reference shape mismatch: {observed.shape!r} "
            f"!= {expected.shape!r}"
        )
    elif not passed:
        reason = (
            "candidate/reference mismatch: "
            f"max_abs={max_abs:.17g}, max_rel={max_rel:.17g}"
        )

    status = "pass" if passed else "fail"
    return {**row, "status": status, "reason": reason}, {
        **detail,
        "status": status,
        "reason": reason,
        "reference": expected.tolist(),
        "observed": observed.tolist(),
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


def main() -> int:
    args = _parse_args()
    import pyscf
    from pyscf.dft import libxc

    if pyscf.__version__ != "2.14.0" or libxc.__version__ != "7.0.0":
        raise RuntimeError("qualification requires exactly PySCF 2.14.0 / Libxc 7.0.0")
    if args.rtol < 0.0 or args.atol < 0.0:
        raise ValueError("qualification tolerances must be nonnegative")

    capability = functional_capability(args.name)
    profile = capability.production_domain_profile
    if not profile.eligible:
        raise RuntimeError(
            f"{capability.name} is structurally blocked: {profile.blocker}"
        )
    if libxc.xc_type(capability.name).lower() != profile.family:
        raise RuntimeError("independent Libxc family disagrees with imported catalog")

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for spin in profile.spin_layouts:
        by_id = {case.case_id: case for case in numerical_cases(profile, spin=spin)}
        controls = set(control_case_ids(profile, spin=spin))
        for case_id in profile.case_ids_for_spin(spin):
            if case_id in controls:
                reason = (
                    "generic control policy is not a numeric oracle case; "
                    "qualification remains fail-closed until #1120 B3"
                )
                rows.append(
                    {
                        "spin": spin,
                        "case_id": case_id,
                        "status": "not-run",
                        "outputs": list(profile.outputs),
                        "reason": reason,
                    }
                )
                details.append(
                    {
                        "spin": spin,
                        "case_id": case_id,
                        "status": "not-run",
                        "reason": reason,
                    }
                )
                continue
            row, detail = _run_numeric_case(
                capability.name,
                by_id[case_id],
                family=profile.family,
                profile=profile,
                libxc=libxc,
                rtol=args.rtol,
                atol=args.atol,
            )
            rows.append(row)
            details.append(detail)

    receipt = build_result(capability.name, rows, evidence=args.evidence)
    envelope = stage_evidence(capability.name, receipt)
    payload = {
        "schema": CAMPAIGN_SCHEMA,
        "functional": capability.name,
        "capability_identity": capability.identity,
        "profile": profile.to_payload(),
        "oracle": {
            "pyscf": pyscf.__version__,
            "libxc": libxc.__version__,
            "api": "pyscf.dft.libxc.eval_xc deriv=2",
        },
        "tolerance": {"rtol": args.rtol, "atol": args.atol},
        "details": details,
        "receipt": receipt,
        "stage_evidence": envelope,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{capability.name}: {envelope['status']} -> {args.output} "
        f"({len(rows)} exact matrix rows)"
    )
    return 1 if args.require_pass and envelope["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
