"""Run the native small-system CPU CCSD analytic-gradient validation endpoint.

Coordinates are Bohr; gradients are Eh/bohr. This driver does not enable public
Calculator forces or call PySCF. --input accepts the explicit molecular-input
schema used by the gradient fixtures; --case selects a retained small example.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tools.cc_gradient_fixtures import CASES, inputs, source_arguments
from tools.vibeqc_cc.complete_gradient import (
    CCSDGradientOptions,
    complete_gradient_validation,
)
from tools.vibeqc_posthf.sources import NativeSource


def record(result, options):
    return {
        "schema": "vibeqc.ccsd.complete_gradient_validation",
        "schema_version": 1,
        "method": "conventional closed-shell all-electron CCSD",
        "backend": "cpu",
        "total_energy": result.total_energy,
        "correlation_energy": result.correlation_energy,
        "gradient": result.gradient.tolist(),
        "forces": result.forces.tolist(),
        "units": {"energy": "Eh", "gradient": "Eh/bohr", "coordinates": "bohr"},
        "physical_components": {
            k: v.tolist() for k, v in result.physical_components.items()
        },
        "integral_components": {
            k: v.tolist() for k, v in result.integral_components.items()
        },
        "residuals": {
            k: getattr(result, k)
            for k in (
                "scf_residual",
                "cc_residual",
                "lambda_residual",
                "z_residual",
                "orbital_stationarity",
            )
        },
        "minimum_orbital_curvature": result.minimum_orbital_curvature,
        "identities": {
            k: getattr(result, k)
            for k in (
                "reference_identity",
                "cc_state_identity",
                "response_identity",
                "operator_identity",
                "source_identity",
            )
        },
        "options": asdict(options),
        "diagnostics": dict(result.diagnostics),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", choices=CASES)
    selection.add_argument("--input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="write a new JSON file; existing files are never overwritten",
    )
    parser.add_argument("--max-bytes", type=int, default=256 << 20)
    parser.add_argument("--provider-budget-bytes", type=int, default=64 << 20)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new path")
    value = (
        inputs(args.case)
        if args.case is not None
        else json.loads(args.input.read_text())
    )
    options = CCSDGradientOptions(
        max_bytes=args.max_bytes, provider_budget_bytes=args.provider_budget_bytes
    )
    with NativeSource(**source_arguments(value)) as source:
        result = complete_gradient_validation(source, options=options)
    text = (
        json.dumps(record(result, options), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    if args.output is None:
        print(text, end="")
    else:
        with args.output.open("x") as stream:
            stream.write(text)


if __name__ == "__main__":
    main()
