"""Validate the native arbitrary-weight ERI boundary against libcint under Slurm.

Fixtures remain small shell blocks. No molecular four-index response or HF
density is supplied to CUDA. Both native routes consume the same streamed
records, including public spherical weights and mixed output-tile indices.
"""

from __future__ import annotations

# Source-tree CLI bootstrap; importing the compiler needs no native runtime.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import os
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vibeqc_compiler.integral.blocks import (
    BlockRequest,
    ShellTile,
    TensorLayout,
    WeightTile,
)
from vibeqc_compiler.integral.eri_weights import (
    fold_dense_eri_weight,
    fold_normalized_pair_weight,
)
from vibeqc_compiler.integral.ir import four_center_eri_operator
from vibeqc_compiler.integral.range_separation import CoulombKernel, CoulombKernelFamily
from vibeqc_compiler.integral.shell_signature import BasisConvention, CenterBinding
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir
from vibeqc_compiler.integral.weighted_eri_inputs import (
    prepare_weighted_eri_stream,
    weighted_eri_response,
)

from tools.generate_validation_references import pyscf_molecule, quartet_data
from tools.vibeqc_validation.f_shell_numerics import (
    _normalized_primitives,
    numerical_error,
)
from tools.vibeqc_validation.schema import canonical_hash, file_hash

CENTERS = np.array(
    [
        [0.13, -0.31, 0.24],
        [-0.43, 0.27, 0.51],
        [0.68, -0.14, -0.22],
        [-0.21, 0.48, -0.63],
    ]
)


def make_fixture(
    name, variant, *, displacement=None, unit_component=None, coulomb_kernel=None
):
    """Use independent libcint center derivatives and normalized public AOs."""
    radial = CoulombKernel() if coulomb_kernel is None else coulomb_kernel
    if not isinstance(radial, CoulombKernel):
        raise TypeError("fixture requires explicit CoulombKernel semantics")
    angular = tuple("spdf".index(letter) for letter in name)
    centers = CENTERS.copy()
    atoms = (0, 1, 2, 3)
    if variant == "coincident":
        centers[1] = centers[0]
        atoms = (0, 0, 2, 3)
    if variant == "changed":
        centers += np.arange(12).reshape(4, 3) * 0.017
    if displacement is not None:
        centers += displacement
    shells = []
    for slot, l in enumerate(angular):
        primitives = [[0.57 + 0.13 * slot, 0.83]]
        if slot % 2 or variant == "long":
            primitives.append([1.31 + 0.19 * slot, -0.17])
        if variant == "long":
            primitives.append([2.17 + 0.11 * slot, 0.08])
        shells.append(
            {"atom_index": slot, "angular_momentum": l, "primitives": primitives}
        )
    inputs = {
        "atomic_numbers": [1] * 4,
        "coordinates": centers.tolist(),
        "shells": shells,
        "basis_representation": "cartesian",
        "charge": 0,
        "multiplicity": 1,
    }
    cart_mol, scales, _ = pyscf_molecule(inputs)
    projections = None
    integral = build_weighted_eri_ir(angular, operator=four_center_eri_operator(radial))
    if variant == "spherical":
        # cart2sph maps libcint Cartesian functions; divide by their unit-AO
        # scales to obtain the normalized Cartesian-to-public pullback.
        matrix = cart_mol.cart2sph_coeff() / scales[:, None]
        inputs = {**inputs, "basis_representation": "spherical"}
        mol, scales, _ = pyscf_molecule(inputs)
        cart_offsets, public_offsets = cart_mol.ao_loc_nr(), mol.ao_loc_nr()
        projections = tuple(
            matrix[
                cart_offsets[i] : cart_offsets[i + 1],
                public_offsets[i] : public_offsets[i + 1],
            ]
            for i in range(4)
        )
        signature = replace(
            integral.signature,
            legacy_class=None,
            shells=tuple(
                replace(s, convention=BasisConvention.REAL_SPHERICAL)
                for s in integral.signature.shells
            ),
        )
        consumer = integral.contractions[0]
        layout = TensorLayout(signature.tensor_indices, signature.component_shape)
        integral = replace(
            integral,
            spec=signature,
            contractions=(
                replace(consumer, weights=replace(consumer.weights, layout=layout)),
            ),
        )
    else:
        mol = cart_mol
    with mol.with_range_coulomb(
        -radial.omega
        if radial.family == CoulombKernelFamily.SHORT_RANGE
        else radial.omega
    ):
        reference = quartet_data(mol, scales)
    eri, gradient = np.asarray(reference["eri"]), np.asarray(reference["gradient"])
    if radial.family == CoulombKernelFamily.LONG_RANGE and radial.omega == 0:
        # Libcint uses signed zero for ordinary Coulomb; the exact LR limit
        # instead vanishes. Do not mislabel a full-Coulomb reference as LR.
        eri, gradient = np.zeros_like(eri), np.zeros_like(gradient)
    weights = np.random.default_rng(144).normal(size=eri.shape)
    reference_weights = weights.copy()
    offsets = mol.ao_loc_nr()
    if variant in ("orbit", "normalized_pair", "exchange"):
        # Explicit independent orbit / pair-matrix contractions serve as the
        # scalar/gradient oracle, while production folding prepares CUDA input.
        dense = lambda q: float(
            np.sin(0.7 * q[0] + 1.1 * q[1] - 0.3 * q[2] + 0.2 * q[3])
        )
        if variant == "exchange":
            # A fixed, nontrivial restricted density and an explicit exchange
            # coefficient. The range family belongs to the integral; it does
            # not change the existing ordered/orbit cotangent convention.
            orbitals = np.random.default_rng(166).normal(size=(mol.nao_nr(), 3))
            density = orbitals @ orbitals.T / 7
            dense = lambda q: float(
                -0.25 * 0.37 * density[q[0], q[2]] * density[q[1], q[3]]
            )
        pair = lambda i, j: float(np.sin(0.37 * i - 0.21 * j))
        for component in np.ndindex(eri.shape):
            i, j, k, l = (int(offsets[s] + c) for s, c in enumerate(component))
            if variant in ("orbit", "exchange"):
                weights[component] = fold_dense_eri_weight(dense, (i, j, k, l))
                orbit = {
                    (i, j, k, l),
                    (j, i, k, l),
                    (i, j, l, k),
                    (j, i, l, k),
                    (k, l, i, j),
                    (l, k, i, j),
                    (k, l, j, i),
                    (l, k, j, i),
                }
                reference_weights[component] = sum(dense(q) for q in orbit)
            else:
                weights[component] = fold_normalized_pair_weight(pair, (i, j, k, l))
                a, b = (
                    max(i, j) * (max(i, j) + 1) // 2 + min(i, j),
                    max(k, l) * (max(k, l) + 1) // 2 + min(k, l),
                )
                reference_weights[component] = np.sqrt(
                    (1 if i == j else 2) * (1 if k == l else 2)
                ) * (pair(a, b) + (pair(b, a) if a != b else 0))
    if unit_component is not None:
        weights[:] = 0
        weights.flat[unit_component] = 1
        reference_weights = weights.copy()
    if variant == "zero":
        weights[:] = 0
        reference_weights[:] = 0
    request = BlockRequest(
        f"{name}/{variant}/{unit_component}",
        integral,
        ShellTile((0,) * 4, eri.shape),
        center_bindings=tuple(CenterBinding(i, atom) for i, atom in enumerate(atoms)),
    )
    result = np.r_[
        np.sum(reference_weights * eri),
        np.sum(gradient * reference_weights, axis=(2, 3, 4, 5)).ravel(),
    ]
    return {
        "request": request,
        "inputs": inputs,
        "primitives": tuple(_normalized_primitives(s) for s in shells),
        "centers": centers,
        "projections": projections,
        "weights": weights,
        "reference": result,
        "input_hash": canonical_hash(
            {
                "inputs": inputs,
                "weights": weights.tolist(),
                "atoms": atoms,
                **(
                    {"coulomb_kernel": radial.to_payload()}
                    if radial.family != CoulombKernelFamily.FULL_RANGE
                    else {}
                ),
            }
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run this numerical validator inside an allocated Slurm GPU job")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("psss", v)
        for v in (
            "cartesian",
            "spherical",
            "coincident",
            "changed",
            "long",
            "orbit",
            "normalized_pair",
            "zero",
        )
    ]
    cases += [
        ("psps", "cartesian"),
        ("ppss", "normalized_pair"),
        ("dsss", "spherical"),
        ("fsss", "spherical"),
    ]
    fixtures = [make_fixture(*case) for case in cases]
    # Unit weights exercise raw component diagnostics through the same API.
    fixtures.extend(
        make_fixture("psss", "cartesian", unit_component=i) for i in range(3)
    )
    differences = []
    for name, variant in (("psss", "coincident"), ("fsss", "spherical")):
        base = cases.index((name, variant))
        atoms = tuple(b.atom_index for b in fixtures[base]["request"].center_bindings)
        for step in (2e-3, 1e-3, 5e-4):
            for atom in sorted(set(atoms)):
                for axis in range(3):
                    displacement = np.zeros((4, 3))
                    displacement[np.array(atoms) == atom, axis] = step
                    plus = len(fixtures)
                    fixtures.extend(
                        (
                            make_fixture(name, variant, displacement=displacement),
                            make_fixture(name, variant, displacement=-displacement),
                        )
                    )
                    differences.append(
                        {
                            "base": base,
                            "step": step,
                            "atom": atom,
                            "axis": axis,
                            "plus": plus,
                            "minus": plus + 1,
                        }
                    )
    records = args.output / "records.bin"
    streams = []
    for tile, fixture in enumerate(fixtures):
        request = fixture["request"]
        stream = prepare_weighted_eri_stream(
            request,
            fixture["primitives"],
            fixture["centers"],
            lambda descriptor, _, f=fixture: WeightTile(
                descriptor.layout, f["weights"].ravel()
            ),
            projections=fixture["projections"],
            output_tile=tile,
        )
        streams.append(stream)
    count = sum(s.record_count for s in streams)
    with records.open("wb") as output:
        output.write(b"VQWE1441" + struct.pack("<QQ", count, len(fixtures)))
        for stream in streams:
            for record in stream.records():
                output.write(record)
    expected = np.array([f["reference"] for f in fixtures])
    report = {
        "schema_version": 1,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "records": count,
        "tiles": len(fixtures),
        "probe_sha256": file_hash(args.probe),
        "records_sha256": file_hash(records),
        "library_sha256": file_hash(Path(os.environ["VIBEQC_LIBRARY"])),
        "fixtures": [
            {
                "name": f["request"].request_id,
                "input_hash": f["input_hash"],
                "host_numeric_peak_bytes": s.host_peak_bytes,
            }
            for f, s in zip(fixtures, streams)
        ],
        "runs": [],
    }
    previous = None
    for capacity in (7, 113):
        budget = len(fixtures) * 208 + capacity * 208
        for generated in (False, True):
            path = args.output / f"output-{capacity}-{int(generated)}.bin"
            completed = subprocess.run(
                [
                    str(args.probe.resolve()),
                    str(records.resolve()),
                    str(path.resolve()),
                    str(budget),
                    str(int(generated)),
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            actual = np.fromfile(path, dtype="<f8").reshape(-1, 13)
            error = numerical_error(actual, expected, atol=3e-10, rtol=3e-9)
            cross = numerical_error(
                actual,
                previous if previous is not None else expected,
                atol=3e-10,
                rtol=3e-9,
            )
            fd = []
            for difference in differences:
                base, axis, atom = (
                    difference["base"],
                    difference["axis"],
                    difference["atom"],
                )
                finite = (
                    actual[difference["plus"], 0] - actual[difference["minus"], 0]
                ) / (2 * difference["step"])
                bindings = fixtures[base]["request"].center_bindings
                derivative = sum(
                    actual[base, 1 + 3 * slot + axis]
                    for slot, binding in enumerate(bindings)
                    if binding.atom_index == atom
                )
                fd.append({**difference, "absolute_error": abs(finite - derivative)})
            # Response metadata/center ordering is also checked against the
            # independently contracted native oracle for every public tile.
            for fixture, values, reference in zip(fixtures, actual, expected):
                np.testing.assert_allclose(
                    weighted_eri_response(fixture["request"], values).values,
                    weighted_eri_response(fixture["request"], reference).values,
                    atol=3e-10,
                    rtol=3e-9,
                )
            fd_passed = max(d["absolute_error"] for d in fd if d["step"] == 5e-4) < 2e-7
            run = {
                "generated": generated,
                "budget_bytes": budget,
                "native": json.loads(completed.stdout),
                "libcint_error": error,
                "route_budget_error": cross,
                "finite_differences": fd,
                "passed": bool(error["passed"] and cross["passed"] and fd_passed),
            }
            report["runs"].append(run)
            previous = actual
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            if not run["passed"]:
                raise RuntimeError(f"weighted ERI numerical gate failed: {run}")
    report["passed"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "records": count,
                "tiles": len(fixtures),
                "runs": len(report["runs"]),
            }
        )
    )


if __name__ == "__main__":
    main()
