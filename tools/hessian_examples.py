"""Validate the independent semi-numerical RHF Hessian oracle for #180.

Retain analytic-PySCF, energy-difference and three gradient-difference
comparisons, basis/geometry provenance, component gates and raw symmetry.
This CPU reference does not qualify the production analytic-provider chain.
"""

from __future__ import annotations

# Source-tree CLI bootstrap for transitive compiler clients.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_root = _CompilerPath(__file__).resolve().parents[1]
_compiler_sys.path.insert(0, str(_root / "python"))
_compiler_sys.path.insert(0, str(_root))

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pyscf

from tools.vibeqc_hessian.reference import (
    System,
    _converged_rhf,
    build_mol,
    fd_hessian,
    hessian_components,
    hessian_total,
)

# H uses STO-3G. Oxygen uses a custom Cartesian s/p/d stress basis.
_S = [
    [
        0,
        (3.425250914, 0.1543289673),
        (0.6239137298, 0.5353281423),
        (0.168855404, 0.4446345422),
    ]
]
_O = [
    [
        0,
        (18.5950316763, 0.0499684015),
        (5.6203025291, 0.1505027689),
        (1.3720603452, 0.4117903918),
        (0.3434943771, 0.4106611356),
    ],
    [
        1,
        (7.1153375795, 0.0102428309),
        (2.0218099978, 0.0314286582),
        (0.5271803969, 0.0814495192),
        (0.1703101032, 0.1646776769),
        (0.0628267133, 0.2917579986),
    ],
    [2, (0.0628267133, 0.1506938663), (0.0351485943, 0.2198166377)],
]

CASES = {
    "h2": {
        "basis_label": "STO-3G",
        "atoms": [(1, [0.0, 0.0, 0.0]), (1, [0.1, 0.2, 1.4])],
        "basis": {"H0": _S, "H1": _S},
    },
    "water": {
        "basis_label": "custom O(s,p,d) + H/STO-3G, 12 Cartesian AOs",
        "atoms": [
            (8, [0.0, 0.0, 0.0]),
            (1, [0.0, 0.958, 0.587]),
            (1, [0.0, -0.958, 0.587]),
        ],
        "basis": {"O0": _O, "H1": _S, "H2": _S},
    },
}


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(v) for v in value]
    return value


def _evidence_case(args, name):
    spec = CASES[name]
    mol = build_mol(spec["atoms"], spec["basis"], charge=0, spin=0)
    start = time.perf_counter()
    s = System(mol, h2=args.h2)
    s.derive()
    derive_s = time.perf_counter() - start

    H = hessian_total(s)
    H_fd = fd_hessian(mol, h=args.fd_h)
    H_analytic = _converged_rhf(mol).Hessian().kernel()
    from tools.vibeqc_hessian.numerical import numerical_hessian

    def gradient(coords, settings):
        moved = mol.copy().set_geom_(coords, unit="Bohr")
        return _converged_rhf(moved).nuc_grad_method().kernel()

    gradient_curve = numerical_hessian(
        gradient,
        mol.atom_coords(),
        settings={"method": "RHF", "conv_tol": 1e-13, "conv_tol_grad": 1e-10},
        steps=(2e-3, 7e-4, 2e-4),
    )
    for sample in gradient_curve["samples"]:
        # numerical_hessian stores displacement first, gradient second.
        numerical = np.asarray(sample["hessian"]).transpose(2, 0, 3, 1)
        sample["reference_error"] = float(np.max(np.abs(H - numerical)))
        sample["passed"] = sample["reference_error"] < 2e-4
    nd = 3 * mol.natm
    H6 = H.transpose(0, 2, 1, 3).reshape(nd, nd)

    comps = hessian_components(s)
    H_frozen = hessian_total(s, with_relax=False)

    record = {
        "case": name,
        "basis_label": spec["basis_label"],
        "basis": spec["basis"],
        "atoms_bohr": spec["atoms"],
        "hessian_layout": "atom,xyz,atom,xyz flattened to 3N by 3N",
        "units": "Eh/Bohr^2",
        "reference_vs_analytic": float(np.max(np.abs(H - H_analytic))),
        "gradient_difference_curve": gradient_curve,
        "nao": int(mol.nao),
        "nocc": int(s.nocc),
        "nvirt": int(s.nmo - s.nocc),
        "h2_step": float(args.h2),
        "fd_step": float(args.fd_h),
        "derive_seconds": float(derive_s),
        "reference_vs_energy_fd": float(np.abs(H - H_fd).max()),
        "symmetry": float(np.abs(H - H.transpose(1, 0, 3, 2)).max()),
        "translation_row": float(np.abs(H6.sum(axis=1)).max()),
        "translation_col": float(np.abs(H6.sum(axis=0)).max()),
        "component_magnitude": {
            k: float(np.abs(v).max()) for k, v in comps.items() if k != "total"
        },
        "component_sum_vs_total": float(
            np.abs(
                (
                    comps["nuclear"]
                    + comps["core"]
                    + comps["pulay"]
                    + comps["two_electron"]
                    + comps["relaxation"]
                )
                - comps["total"]
            ).max()
        ),
        "frozen_density_shift": float(np.abs(H - H_frozen).max()),
        "reference_hessian": _serializable(H6),
    }
    record["gates"] = {
        "analytic_reference": record["reference_vs_analytic"] < 5e-6,
        "energy_difference": record["reference_vs_energy_fd"] < 5e-4,
        "gradient_difference": all(s["passed"] for s in gradient_curve["samples"]),
        "raw_symmetry": record["symmetry"] < 1e-8,
        "translation": max(record["translation_row"], record["translation_col"]) < 1e-4,
        "component_sum": record["component_sum_vs_total"] < 1e-10,
        "frozen_density_negative": record["frozen_density_shift"]
        > 3 * record["reference_vs_energy_fd"],
    }
    record["passed"] = all(record["gates"].values())
    return record


def run(args):
    evidence = {
        "title": "issue #180: independent semi-numerical RHF Hessian oracle",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pyscf": pyscf.__version__,
        "source_sha256": {
            str(p.relative_to(_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (_root / "tools/vibeqc_hessian/reference.py", Path(__file__))
        },
        "cases": {},
    }
    for name in args.cases:
        t0 = time.perf_counter()
        evidence["cases"][name] = _evidence_case(args, name)
        evidence["cases"][name]["total_seconds"] = float(time.perf_counter() - t0)
        print(
            f"[{name}] reference-vs-energy-FD = "
            f"{evidence['cases'][name]['reference_vs_energy_fd']:.3e}",
            flush=True,
        )
    evidence["passed"] = bool(evidence["cases"]) and all(
        case["passed"] for case in evidence["cases"].values()
    )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", default="h2,water", help="comma list from: " + ",".join(CASES)
    )
    parser.add_argument(
        "--h2", type=float, default=3e-4, help="2nd-derivative integral FD step (Bohr)"
    )
    parser.add_argument(
        "--fd-h", type=float, default=1e-4, help="total-energy FD oracle step (Bohr)"
    )
    parser.add_argument(
        "--output", default=Path("hessian_reference_evidence.json"), type=Path
    )
    args = parser.parse_args()
    args.cases = [c.strip() for c in args.case.split(",") if c.strip()]
    for c in args.cases:
        if c not in CASES:
            raise SystemExit(f"unknown case {c!r}; choose from {list(CASES)}")
    evidence = run(args)
    args.output.write_text(json.dumps(_serializable(evidence), indent=2) + "\n")
    print(f"wrote {args.output}")
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
