"""Freeze independent LDA/GGA Libxc feature-Hessian cases, without VibeQC imports.

Use PySCF 2.14.0 / Libxc 7.0.0. Repeat --functional NAME=LIBXC_CODE and pass
--output to a small tests/data/xc JSON fixture. Cartesian gradient vectors define
physical sigma triples. Densities include both sides of the PZ rs=1 branch.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pyscf
from generate_xc_references import reference
from pyscf.dft import libxc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functional", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if pyscf.__version__ != "2.14.0" or libxc.__version__ != "7.0.0":
        raise RuntimeError("requires PySCF 2.14.0 and Libxc 7.0.0")
    definitions = {}
    for item in args.functional:
        name, separator, code = item.partition("=")
        if not separator or not name or not code or name in definitions:
            parser.error("each functional must be a unique NAME=LIBXC_CODE")
        if libxc.xc_type(code) not in ("LDA", "GGA"):
            parser.error("only LDA/GGA feature-Hessian cases are supported")
        definitions[name] = code
    rng = np.random.default_rng(609610)
    transition = 3 / (4 * np.pi)
    n = np.array([0.02, 0.08, transition * 0.999, transition * 1.001, 0.5, 1, 2, 4])
    z = rng.uniform(-0.8, 0.8, len(n))
    ra, rb = n * (1 + z) / 2, n * (1 - z) / 2
    ga = rng.normal(size=(3, len(n))) * ra ** (4 / 3) * 0.2
    gb = rng.normal(size=(3, len(n))) * rb ** (4 / 3) * 0.2
    aa, ab, bb = (np.sum(v, axis=0) for v in (ga * ga, ga * gb, gb * gb))
    zeros = np.zeros((2, len(n)))
    rho_p = np.stack((np.vstack((ra, ga, zeros)), np.vstack((rb, gb, zeros))))
    rho_u = np.vstack((n, ga + gb, zeros))
    features_p = np.vstack((ra, rb, aa, ab, bb, zeros))
    features_u = np.vstack((n, aa + 2 * ab + bb, zeros[0]))
    cases = []
    for name, code in definitions.items():
        for spin, rho, features in ((0, rho_u, features_u), (1, rho_p, features_p)):
            expected = reference(code, rho, spin)
            if not np.isfinite(expected).all():
                raise ArithmeticError("nonfinite independent oracle")
            cases.append(
                {
                    "name": name,
                    "code": code,
                    "spin": "polarized" if spin else "unpolarized",
                    "features": features.tolist(),
                    "expected": expected.tolist(),
                }
            )
    payload = {
        "schema": "vibeqc.independent-semilocal-hessian.v1",
        "pyscf": pyscf.__version__,
        "libxc": libxc.__version__,
        "seed": 609610,
        "generator": Path(__file__).name,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "converter_function_sha256": hashlib.sha256(
            inspect.getsource(reference).encode()
        ).hexdigest(),
        "conventions": "energy density; feature gradient; packed upper feature Hessian; physical Cartesian sigma; tau=0 for LDA/GGA",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
