"""Independent pinned-PySCF RCCSD(T) analytic-gradient validation for issue #155.

This module is test/qualification infrastructure only. It deliberately does not
call VibeQC's incomplete RCCSD(T) force endpoint. PySCF 2.14.0 provides an
independent canonical-RHF analytic gradient oracle, and central finite
differences re-solve RHF, RCCSD, and (T) at every displaced geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import typing
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np

from tools.cc_gradient_fixtures import inputs
from tools.generate_validation_references import pyscf_molecule
from tools.vibeqc_validation.schema import canonical_hash

PYSCF_VERSION = "2.14.0"
CASES = ("h2o", "nh3")
FD_STEPS = (1.0e-3, 3.0e-4, 1.0e-4)


def _source_sha256(module: typing.Any) -> str:
    return hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()


def _check_inputs(value: dict[str, typing.Any]) -> None:
    if (
        value["method"] != "rhf"
        or value["multiplicity"] != 1
        or value["auxiliary_centers"]
    ):
        raise ValueError("CCSD(T) validation requires an all-electron RHF molecule")
    settings = value["cc_settings"]
    if settings["method"] != "CCSD" or settings["frozen_core"] != 0:
        raise ValueError("CCSD(T) validation currently requires unfrozen RCCSD")


def _solve(
    value: dict[str, typing.Any],
    *,
    gradients: bool,
) -> dict[str, typing.Any]:
    import pyscf
    from pyscf import cc, scf
    from pyscf.cc import ccsd_t, ccsd_t_lambda
    from pyscf.grad import ccsd as grad_ccsd
    from pyscf.grad import ccsd_t as grad_ccsd_t
    from threadpoolctl import threadpool_limits

    if pyscf.__version__ != PYSCF_VERSION:
        raise RuntimeError(
            f"CCSD(T) gradient validation requires PySCF {PYSCF_VERSION}, "
            f"not {pyscf.__version__}"
        )
    _check_inputs(value)

    with threadpool_limits(limits=1):
        mol, _, _ = pyscf_molecule(value)
        scf_policy = value["scf_settings"]
        mf = scf.RHF(mol)
        mf.conv_tol = scf_policy["energy_tolerance"]
        mf.conv_tol_grad = scf_policy["gradient_tolerance"]
        mf.direct_scf_tol = scf_policy["direct_scf_tol"]
        mf.max_cycle = scf_policy["max_iterations"]
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("independent PySCF RHF failed")

        cc_policy = value["cc_settings"]
        coupled = cc.CCSD(mf, frozen=0)
        coupled.conv_tol = cc_policy["conv_tol"]
        coupled.conv_tol_normt = cc_policy["conv_tol_normt"]
        coupled.max_cycle = cc_policy["max_cycle"]
        coupled.kernel()
        if not coupled.converged:
            raise RuntimeError("independent PySCF RCCSD failed")

        eris = coupled.ao2mo(mf.mo_coeff)
        fock_mo = np.asarray(eris.fock, dtype=np.float64)
        canonical_offdiag = fock_mo - np.diag(np.diag(fock_mo))
        canonical_offdiag_max = float(np.max(np.abs(canonical_offdiag)))
        if canonical_offdiag_max > 1.0e-9:
            raise RuntimeError(
                "PySCF CCSD(T) gradient oracle requires canonical RHF orbitals"
            )
        triples_energy = float(
            ccsd_t.kernel(
                coupled,
                eris,
                coupled.t1,
                np.ascontiguousarray(coupled.t2),
            )
        )
        total_energy = float(coupled.e_tot + triples_energy)
        result: dict[str, typing.Any] = {
            "hf_energy": float(mf.e_tot),
            "ccsd_correlation_energy": float(coupled.e_corr),
            "triples_energy": triples_energy,
            "total_energy": total_energy,
            "scf_converged": bool(mf.converged),
            "ccsd_converged": bool(coupled.converged),
            "canonical_fock_offdiag_max": canonical_offdiag_max,
        }
        if not gradients:
            return result

        l1_ccsd, l2_ccsd = coupled.solve_lambda(eris=eris)
        if not coupled.converged_lambda:
            raise RuntimeError("independent PySCF RCCSD Lambda failed")

        corrected_converged, l1_t, l2_t = ccsd_t_lambda.kernel(
            coupled,
            eris,
            coupled.t1,
            coupled.t2,
            max_cycle=cc_policy["max_cycle"],
            tol=cc_policy["conv_tol_normt"],
        )
        if not corrected_converged:
            raise RuntimeError("independent PySCF RCCSD(T) Lambda failed")

        ccsd_gradient = np.asarray(
            grad_ccsd.Gradients(coupled).kernel(
                coupled.t1,
                coupled.t2,
                l1_ccsd,
                l2_ccsd,
                eris=eris,
            ),
            dtype=np.float64,
        )
        ccsdt_gradient = np.asarray(
            grad_ccsd_t.Gradients(coupled).kernel(
                coupled.t1,
                coupled.t2,
                l1_t,
                l2_t,
                eris=eris,
            ),
            dtype=np.float64,
        )
        without_delta_lambda = np.asarray(
            grad_ccsd_t.Gradients(coupled).kernel(
                coupled.t1,
                coupled.t2,
                l1_ccsd,
                l2_ccsd,
                eris=eris,
            ),
            dtype=np.float64,
        )
        for name, array in (
            ("ccsd_gradient", ccsd_gradient),
            ("ccsdt_gradient", ccsdt_gradient),
            ("without_delta_lambda_gradient", without_delta_lambda),
        ):
            if array.shape != (len(value["atomic_numbers"]), 3):
                raise RuntimeError(f"unexpected {name} shape {array.shape}")
            if not np.isfinite(array).all():
                raise RuntimeError(f"nonfinite independent {name}")

        dm = np.asarray(mf.make_rdm1())
        overlap = np.asarray(mf.get_ovlp())
        fock = np.asarray(mf.get_fock(dm=dm))
        scf_commutator = fock @ dm @ overlap - overlap @ dm @ fock
        updated = coupled.update_amps(coupled.t1, coupled.t2, eris)
        amplitude_update_max = max(
            float(np.max(np.abs(updated[0] - coupled.t1))),
            float(np.max(np.abs(updated[1] - coupled.t2))),
        )

        result.update(
            {
                "ccsd_gradient": ccsd_gradient.tolist(),
                "gradient": ccsdt_gradient.tolist(),
                "triples_gradient": (ccsdt_gradient - ccsd_gradient).tolist(),
                "without_delta_lambda_gradient": without_delta_lambda.tolist(),
                "delta_lambda_gradient": (
                    ccsdt_gradient - without_delta_lambda
                ).tolist(),
                "scf_commutator_max": float(np.max(np.abs(scf_commutator))),
                "ccsd_amplitude_update_max": amplitude_update_max,
                "ccsd_lambda_converged": bool(coupled.converged_lambda),
                "ccsdt_lambda_converged": bool(corrected_converged),
                "source_sha256": {
                    "pyscf.cc.ccsd_t": _source_sha256(ccsd_t),
                    "pyscf.cc.ccsd_t_lambda": _source_sha256(ccsd_t_lambda),
                    "pyscf.grad.ccsd": _source_sha256(grad_ccsd),
                    "pyscf.grad.ccsd_t": _source_sha256(grad_ccsd_t),
                },
            }
        )
        return result


@lru_cache(maxsize=len(CASES))
def analytic_oracle(name: str) -> dict[str, typing.Any]:
    if name not in CASES:
        raise ValueError(f"unsupported CCSD(T) gradient validation case: {name}")
    record = {
        "schema": "vibeqc.ccsd_t.gradient_validation",
        "schema_version": 1,
        "pyscf": PYSCF_VERSION,
        "case": name,
        "inputs": inputs(name),
        "analytic": _solve(inputs(name), gradients=True),
        "units": {
            "energy": "Eh",
            "gradient": "Eh/bohr",
            "coordinates": "bohr",
        },
        "restrictions": {
            "reference": "canonical RHF",
            "frozen_core": 0,
            "density_fitting": False,
            "open_shell": False,
        },
    }
    record["content_hash"] = canonical_hash(record)
    return record


def _direction(name: str) -> np.ndarray:
    shape = (len(inputs(name)["atomic_numbers"]), 3)
    direction = np.random.default_rng(15500 + CASES.index(name)).normal(size=shape)
    direction -= direction.mean(axis=0, keepdims=True)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise RuntimeError("invalid finite-difference direction")
    return direction / norm


def finite_difference(
    name: str,
    *,
    steps: tuple[float, ...] = FD_STEPS,
) -> dict[str, typing.Any]:
    if len(steps) < 3 or any(step <= 0 for step in steps):
        raise ValueError("finite-difference validation requires three positive steps")
    if len(set(steps)) != len(steps):
        raise ValueError("finite-difference steps must be distinct")

    oracle = analytic_oracle(name)
    gradient = np.asarray(oracle["analytic"]["gradient"], dtype=np.float64)
    direction = _direction(name)
    analytic = float(np.vdot(gradient, direction))
    base = inputs(name)
    points = []
    errors = []
    displaced_hashes = set()
    for step in steps:
        energies = []
        for sign in (-1.0, 1.0):
            displaced = deepcopy(base)
            displaced["coordinates"] = (
                np.asarray(base["coordinates"], dtype=np.float64)
                + sign * step * direction
            ).tolist()
            displaced_hashes.add(canonical_hash(displaced))
            energies.append(_solve(displaced, gradients=False)["total_energy"])
        derivative = float((energies[1] - energies[0]) / (2.0 * step))
        error = abs(derivative - analytic)
        errors.append(error)
        points.append(
            {
                "step_bohr": float(step),
                "minus_energy": energies[0],
                "plus_energy": energies[1],
                "directional_derivative": derivative,
                "absolute_error": error,
            }
        )
    return {
        "direction": direction.tolist(),
        "analytic_directional_derivative": analytic,
        "points": points,
        "errors": errors,
        "fresh_displaced_geometries": len(displaced_hashes),
    }


def validate(
    name: str, *, include_finite_difference: bool = True
) -> dict[str, typing.Any]:
    record = dict(analytic_oracle(name))
    analytic = record["analytic"]
    gradient = np.asarray(analytic["gradient"])
    triples_gradient = np.asarray(analytic["triples_gradient"])
    delta_lambda_gradient = np.asarray(analytic["delta_lambda_gradient"])

    if abs(analytic["triples_energy"]) <= 1.0e-7:
        raise RuntimeError("validation case has a trivial (T) energy")
    if float(np.max(np.abs(triples_gradient))) <= 1.0e-8:
        raise RuntimeError("validation case has a trivial triples gradient")
    if float(np.max(np.abs(delta_lambda_gradient))) <= 1.0e-10:
        raise RuntimeError("omitting corrected Lambda is not detectable")
    if float(np.max(np.abs(gradient.sum(axis=0)))) > 2.0e-8:
        raise RuntimeError("independent CCSD(T) gradient violates translation")

    if include_finite_difference:
        fd = finite_difference(name)
        record["finite_difference"] = fd
        if fd["fresh_displaced_geometries"] != 2 * len(FD_STEPS):
            raise RuntimeError("finite-difference run reused a displaced geometry")
        if min(fd["errors"][1:]) > 1.0e-6 or fd["errors"][-1] > 2.0e-6:
            raise RuntimeError(
                f"CCSD(T) finite-difference convergence failed: {fd['errors']}"
            )
    record["validation_hash"] = canonical_hash(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument(
        "--skip-finite-difference",
        action="store_true",
        help="generate only the independent analytic oracle",
    )
    args = parser.parse_args()
    selected = tuple(args.case or CASES)
    payload = {
        "schema": "vibeqc.ccsd_t.gradient_validation_bundle",
        "schema_version": 1,
        "pyscf": PYSCF_VERSION,
        "cases": [
            validate(name, include_finite_difference=not args.skip_finite_difference)
            for name in selected
        ],
    }
    payload["content_hash"] = canonical_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
