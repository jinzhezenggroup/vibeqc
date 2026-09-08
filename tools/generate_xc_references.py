"""Regenerate independent, pinned Libxc XC energy/feature derivative fixtures.

Run with PySCF 2.14.0 / Libxc 7.0.0. Inputs come from actual Cartesian gradient
vectors, never arbitrary sigma triples. This script imports no VibeQC XC code.
"""

import argparse
import hashlib
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pyscf
from pyscf.dft import libxc

CODES = {
    "LDA_X": "LDA_X",
    "LDA_C_PW": "LDA_C_PW",
    "LDA_C_PW_MOD": "LDA_C_PW_MOD",
    "GGA_X_PBE": "GGA_X_PBE",
    "GGA_C_PBE": "GGA_C_PBE",
    "LDA_XC_PW": "LDA_X,LDA_C_PW",
    "PBE": "GGA_X_PBE,GGA_C_PBE",
}


def reference(code, rho, spin):
    """Convert documented Libxc vxc/fxc order to full feature Hessians."""
    lda = libxc.xc_type(code) == "LDA"
    n = rho[:, 0].sum(axis=0) if spin else rho[0]
    exc, vxc, fxc, _ = libxc.eval_xc(
        code, rho[:, 0] if spin and lda else rho[0] if lda else rho, spin=spin, deriv=2
    )
    size = 7 if spin else 3
    v = np.zeros((size, len(n)))
    h = np.zeros((size, size, len(n)))
    if spin:
        v[:2] = vxc[0].T
        for k, (i, j) in enumerate(((0, 0), (0, 1), (1, 1))):
            h[i, j] = h[j, i] = fxc[0][:, k]
        if not lda:
            v[2:5] = vxc[1].T
            for k, (i, j) in enumerate((i, j) for i in range(2) for j in range(2, 5)):
                h[i, j] = h[j, i] = fxc[1][:, k]
            for k, (i, j) in enumerate(combinations_with_replacement(range(2, 5), 2)):
                h[i, j] = h[j, i] = fxc[2][:, k]
    else:
        v[0], h[0, 0] = vxc[0], fxc[0]
        if not lda:
            v[1], h[1, 1] = vxc[1], fxc[2]
            h[0, 1] = h[1, 0] = fxc[1]
    return np.concatenate(
        (
            (n * exc)[None],
            v,
            np.stack(
                [h[i, j] for i, j in combinations_with_replacement(range(size), 2)]
            ),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if pyscf.__version__ != "2.14.0" or libxc.__version__ != "7.0.0":
        raise RuntimeError("reference requires PySCF 2.14.0 and Libxc 7.0.0")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(161)
    arrays = {}
    for domain in ("typical", "boundary"):
        if domain == "typical":
            densities = np.exp(rng.uniform(np.log(0.01), np.log(10), (2, 36)))
            gradient = rng.normal(size=(2, 3, 36)) * densities[:, None, :] ** (4 / 3)
            gradient[:, :, :3] = 0
            gradient[:, :, 3:6] *= 1e-8
        else:
            n = np.array([1e-8, 1e-4, 1, 1e4, 1e8, 1e-3, 1, 1e3, 0.4, 0.4])
            fraction = np.array([0.3, 0.4, 0.5, 0.3, 0.6, 1e-9, 1e-9, 1e-9, 0.2, 0.7])
            densities = np.stack((n * fraction, n * (1 - fraction)))
            gradient = rng.normal(size=(2, 3, len(n))) * densities[:, None, :] ** (
                4 / 3
            )
            gradient[:, :, -2:] *= 1e4
        for spin in (0, 1):
            tag = f"{domain}_{spin}"
            if spin:
                rho = np.concatenate((densities[:, None, :], gradient), axis=1)
                aa = np.sum(gradient[0] ** 2, axis=0)
                ab = np.sum(gradient[0] * gradient[1], axis=0)
                bb = np.sum(gradient[1] ** 2, axis=0)
                features = np.concatenate(
                    (densities, np.stack((aa, ab, bb)), densities ** (5 / 3))
                )
            else:
                rho = np.concatenate(
                    (densities.sum(axis=0)[None], gradient.sum(axis=0))
                )
                features = np.stack(
                    (rho[0], np.sum(rho[1:] ** 2, axis=0), rho[0] ** (5 / 3))
                )
            arrays[f"{tag}_features"] = features
            arrays[f"{tag}_rho_gradient"] = rho
            for name, code in CODES.items():
                arrays[f"{tag}_{name}"] = reference(code, rho, spin)
    path = args.output / "libxc.npz"
    np.savez_compressed(path, **arrays)
    metadata = {
        "schema": 1,
        "pyscf": pyscf.__version__,
        "libxc": libxc.__version__,
        "numpy": np.__version__,
        "codes": CODES,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arrays_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "conventions": "energy per volume; spin sigma_ab without factor two; tau=one-half gradient square; packed upper Hessian",
        "typical_tolerance": {"atol": 1e-11, "rtol": 1e-10},
        "boundary_tolerance": {"atol": 1e-8, "rtol": 2e-6},
        "boundary_rationale": "Nearly polarized spin interpolation loses relative accuracy in Libxc's 1-z subtraction; large density and gradient ratios amplify high derivatives. Scale-aware gate applies only to named boundary fixtures; no clipping is applied by VibeQC.",
    }
    (args.output / "libxc.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
