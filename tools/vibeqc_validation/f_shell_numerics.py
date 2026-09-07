"""Independent libcint fixtures and contraction oracles for generated f shells.

Only normalization/task plumbing follows VibeQC conventions. Integral values
and all four center derivatives come from libcint, and explicit eightfold
orbit loops implement RHF/UHF contractions without generator contraction code.
"""

from __future__ import annotations

import json
import os
import platform
import struct
import subprocess
import time
from dataclasses import dataclass
from itertools import product
from math import exp, pi, prod, sqrt
from pathlib import Path

import numpy as np

from tools.vibeqc_codegen.shell_spec import (
    FUSED_SHELL_SPEC_BY_NAME,
    cartesian_components,
)

from .schema import block_error, canonical_hash

FIXTURE_VARIANTS = (
    "cartesian",
    "reversed_pairs",
    "coincident",
    "spherical",
    "atom_permutation",
    "reversed_bra",
    "reversed_ket",
    *(f"shell_permutation_{index}" for index in range(1, 8)),
)


def numerical_error(actual, reference, *, atol, rtol) -> dict:
    """Retain absolute/scaled gates and raw relative errors on nonzero entries.

    Exactly zero reference entries have no defined relative error; their
    absolute errors remain part of the all-entry tolerance gate.
    """
    result = block_error(actual, reference, atol=atol, rtol=rtol)
    actual, reference = np.asarray(actual), np.asarray(reference)
    nonzero = reference != 0
    relative = np.divide(
        np.abs(actual - reference),
        np.abs(reference),
        out=np.zeros_like(actual, dtype=float),
        where=nonzero,
    )
    result["max_relative_error_nonzero_reference"] = float(relative.max())
    result["zero_reference_entries"] = int(np.count_nonzero(~nonzero))
    return result


@dataclass(frozen=True)
class ShellFixture:
    """One bounded task, external densities, and independent expected outputs."""

    name: str
    inputs: dict
    positions: np.ndarray
    atom_indices: tuple[int, ...]
    ao_offsets: tuple[int, ...]
    pairs: np.ndarray
    pair_split: int
    reversed_mask: int
    ao_coefficients: np.ndarray
    density: np.ndarray
    spin_density: np.ndarray
    projection: np.ndarray
    reference: dict[str, np.ndarray]

    @property
    def inputs_hash(self) -> str:
        """Include densities, layout/projection, geometry, and every normalization."""
        return canonical_hash(
            {
                "inputs": self.inputs,
                "atom_indices": self.atom_indices,
                "pairs": self.pairs.tolist(),
                "reversed_mask": self.reversed_mask,
                "ao_coefficients": self.ao_coefficients.tolist(),
                "density": self.density.tolist(),
                "spin_density": self.spin_density.tolist(),
                "projection": self.projection.tolist(),
            }
        )


def eri_orbit(indices) -> tuple[tuple[int, ...], ...]:
    """Enumerate unique chemists' ERI permutations using explicit set equality."""
    i, j, k, l = indices
    return tuple(
        sorted(
            {
                (i, j, k, l),
                (j, i, k, l),
                (i, j, l, k),
                (j, i, l, k),
                (k, l, i, j),
                (l, k, i, j),
                (k, l, j, i),
                (l, k, j, i),
            }
        )
    )


def contract_reference(eri, derivatives, density, spin_density, offsets, atom_indices):
    """Contract one shell-orbit contribution with independent RHF/UHF formulas."""
    n = density.shape[0]
    result = {
        "rhf_fock": np.zeros((n, n)),
        "uhf_fock": np.zeros((2, n, n)),
        "rhf_force": np.zeros((4, 3)),
        "uhf_force": np.zeros((4, 3)),
    }
    alpha, beta = spin_density
    total = alpha + beta
    for component in np.ndindex(eri.shape):
        indices = tuple(offset + i for offset, i in zip(offsets, component))
        value = eri[component]
        rhf_weight, uhf_weight = 0.0, 0.0
        for a, b, c, d in eri_orbit(indices):
            result["rhf_fock"][a, b] += density[c, d] * value
            result["rhf_fock"][a, c] -= 0.5 * density[b, d] * value
            rhf_weight += (
                0.5 * density[a, b] * density[c, d]
                - 0.25 * density[a, c] * density[b, d]
            )
            for spin, one_spin in enumerate((alpha, beta)):
                result["uhf_fock"][spin, a, b] += total[c, d] * value
                result["uhf_fock"][spin, a, c] -= one_spin[b, d] * value
            uhf_weight += 0.5 * total[a, b] * total[c, d] - 0.5 * (
                alpha[a, c] * alpha[b, d] + beta[a, c] * beta[b, d]
            )
        for center, atom in enumerate(atom_indices):
            gradient = derivatives[(center, slice(None), *component)]
            result["rhf_force"][atom] -= rhf_weight * gradient
            result["uhf_force"][atom] -= uhf_weight * gradient
    return result


def _normalized_primitives(shell):
    """Factor unit-Cartesian normalization into radial and angular parts."""
    l, primitives = shell["angular_momentum"], shell["primitives"]
    norm2 = sum(
        ca * cb * (2 * sqrt(a * b) / (a + b)) ** (l + 1.5)
        for a, ca in primitives
        for b, cb in primitives
    )
    return [
        (a, c * (2 * a / pi) ** 0.75 * (4 * a) ** (0.5 * l) / sqrt(norm2))
        for a, c in primitives
    ]


def _pair_rows(first, second, A, B, *, reverse=False):
    rows = []
    for (a, ca), (b, cb) in product(
        _normalized_primitives(first), _normalized_primitives(second)
    ):
        p, mu = a + b, a * b / (a + b)
        P = (a * A + b * B) / p
        rows.append(
            [
                p,
                mu,
                *P,
                ca * cb * exp(-mu * sum((A - B) ** 2)),
                b / p if reverse else a / p,
                a / p if reverse else b / p,
            ]
        )
    return rows


def _reference_integrals(inputs):
    # Reuse the audited CG01 libcint normalization and derivative-center
    # adapter; this import is optional outside the manual numerical tier.
    from tools.generate_validation_references import pyscf_molecule, quartet_data

    mol, scale, angular = pyscf_molecule(inputs)
    data = quartet_data(mol, scale)
    if angular != [shell["angular_momentum"] for shell in inputs["shells"]]:
        raise ValueError("loaded reference shells differ from the requested f class")
    return mol, scale, np.array(data["eri"]), np.array(data["gradient"])


def make_fixture(
    name: str, variant: str = "cartesian", *, displacement=None
) -> ShellFixture:
    """Build asymmetric, reversed-cache, coincident-atom, or spherical fixtures."""
    if variant not in FIXTURE_VARIANTS:
        raise ValueError("unsupported f-shell numerical fixture")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    coordinates = np.array(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )
    if variant == "coincident":
        coordinates[1] = coordinates[0]
    if displacement is not None:
        coordinates += np.asarray(displacement)
    shells = []
    for slot, angular in enumerate(spec.angular):
        primitives = [[0.57 + 0.13 * slot, 0.83]]
        if slot % 2:
            primitives.append([1.31 + 0.19 * slot, -0.17])
        shells.append(
            {"atom_index": slot, "angular_momentum": angular, "primitives": primitives}
        )
    inputs = {
        "name": f"{name}/{variant}",
        "atomic_numbers": [1] * 4,
        "coordinates": coordinates.tolist(),
        "shells": shells,
        "basis_representation": "cartesian",
        "charge": 0,
        "multiplicity": 1,
        "units": {"coordinates": "bohr", "energy": "hartree"},
        "variant": variant,
        "seed": 135,
        "reference_version": 1,
    }
    mol, scales, eri, derivatives = _reference_integrals(inputs)
    n = mol.nao_nr()
    projection = (
        mol.cart2sph_coeff() / scales[:, None] if variant == "spherical" else np.eye(n)
    )
    offsets = tuple(int(i) for i in mol.ao_loc_nr()[:-1])
    task_offsets = offsets
    ao_order = np.arange(n)
    if variant.startswith("shell_permutation_"):
        # Reorder the external shell layout through all eight ERI symmetries,
        # then feed the canonical angular task with the corresponding offsets.
        # The oracle stays in the original layout; projection independently
        # transports its density/Fock and tests canonicalization at the ABI.
        order = eri_orbit((0, 1, 2, 3))[int(variant.rsplit("_", 1)[1])]
        counts = np.diff(mol.ao_loc_nr())
        task_offsets_list = [0] * 4
        cursor = 0
        for slot in order:
            task_offsets_list[slot] = cursor
            ao_order[offsets[slot] : offsets[slot] + counts[slot]] = np.arange(
                cursor, cursor + counts[slot]
            )
            cursor += counts[slot]
        task_offsets = tuple(task_offsets_list)
        projection = np.zeros((n, n))
        projection[ao_order, np.arange(n)] = 1
    rng = np.random.default_rng(135)
    density_order = projection.shape[1]
    factors = rng.normal(scale=0.2, size=(2, density_order, density_order))
    spin_density = np.array([matrix @ matrix.T / density_order for matrix in factors])
    density = spin_density.sum(axis=0)
    cart_spin = np.array([projection @ block @ projection.T for block in spin_density])
    cart_density = projection @ density @ projection.T
    atom_indices = (
        (0, 0, 2, 3)
        if variant == "coincident"
        else (2, 3, 0, 1)
        if variant == "atom_permutation"
        else (0, 1, 2, 3)
    )
    positions = coordinates.copy()
    for center, atom in enumerate(atom_indices):
        positions[atom] = coordinates[center]
    if variant == "spherical":
        spherical_inputs = {**inputs, "basis_representation": "spherical"}
        spherical_mol, _, eri, derivatives = _reference_integrals(spherical_inputs)
        reference_offsets = tuple(int(i) for i in spherical_mol.ao_loc_nr()[:-1])
    else:
        reference_offsets = offsets
    reference = contract_reference(
        eri, derivatives, density, spin_density, reference_offsets, atom_indices
    )
    reverse_bra = variant in ("reversed_pairs", "reversed_bra")
    reverse_ket = variant in ("reversed_pairs", "reversed_ket")
    first_pairs = _pair_rows(
        shells[0], shells[1], coordinates[0], coordinates[1], reverse=reverse_bra
    )
    second_pairs = _pair_rows(
        shells[2], shells[3], coordinates[2], coordinates[3], reverse=reverse_ket
    )
    ao_coefficients = np.array(
        [
            1
            / sqrt(prod(prod(range(1, 2 * component.count(axis), 2)) for axis in "xyz"))
            for angular in spec.angular
            for component in cartesian_components(angular)
        ]
    )
    reordered_coefficients = np.empty_like(ao_coefficients)
    reordered_coefficients[ao_order] = ao_coefficients
    return ShellFixture(
        f"{name}_{variant}",
        inputs,
        positions,
        atom_indices,
        task_offsets,
        np.array(first_pairs + second_pairs),
        len(first_pairs),
        int(reverse_bra) | (int(reverse_ket) << 1),
        reordered_coefficients,
        cart_density,
        cart_spin,
        projection,
        reference,
    )


def write_fixture(fixture: ShellFixture, path: Path) -> None:
    """Write a bounded little-endian binary task for the CUDA-only host driver.

    This format is local executable I/O; the report preserves its mathematical
    input hash and reconstructible fixture conventions instead of relying on
    native struct padding or machine-dependent NumPy serialization.
    """
    n = fixture.density.shape[0]
    with path.open("wb") as stream:
        stream.write(b"VQF13501")
        stream.write(
            struct.pack(
                "<4I", n, len(fixture.pairs), fixture.pair_split, fixture.reversed_mask
            )
        )
        stream.write(struct.pack("<4Q", *fixture.ao_offsets))
        stream.write(struct.pack("<4I", *fixture.atom_indices))
        stream.write(
            struct.pack(
                "<4I", *(len(s["primitives"]) for s in fixture.inputs["shells"])
            )
        )
        for data in (fixture.positions, fixture.ao_coefficients, fixture.pairs):
            stream.write(np.asarray(data, dtype="<f8").tobytes(order="C"))
        stream.write(np.asarray(fixture.density, dtype="<f8").tobytes(order="F"))
        for block in fixture.spin_density:
            stream.write(np.asarray(block, dtype="<f8").tobytes(order="F"))


def decoded_outputs(fixture: ShellFixture, row: dict) -> dict[str, np.ndarray]:
    """Project generated Cartesian Fock blocks into the fixture's AO convention."""
    n, C = fixture.density.shape[0], fixture.projection
    result = {}
    for name, values in row["outputs"].items():
        if "force" in name:
            result[name] = np.asarray(values).reshape(4, 3)
        elif "uhf" in name:
            result[name] = np.array(
                [
                    C.T @ block.reshape(n, n, order="F") @ C
                    for block in np.asarray(values).reshape(2, n * n)
                ]
            )
        else:
            result[name] = C.T @ np.asarray(values).reshape(n, n, order="F") @ C
    return result


def class_fixtures(name: str, *, finite_difference: bool = False):
    """Reconstruct the exact ordered fixture set and finite-difference indices."""
    fixtures = [make_fixture(name, variant) for variant in FIXTURE_VARIANTS]
    fd_indices = []
    if finite_difference:
        # Freeze density, basis coefficients, recurrence, and screening;
        # both signs and all three step sizes are executed, never selected
        # afterwards for an accidentally favorable error.
        for step in (1e-2, 3e-3, 1e-3):
            indices = {}
            for center, xyz, sign in product(range(4), range(3), (-1, 1)):
                displacement = np.zeros((4, 3))
                displacement[center, xyz] = sign * step
                indices[(center, xyz, sign)] = len(fixtures)
                fixtures.append(make_fixture(name, displacement=displacement))
            fd_indices.append((step, indices))
    return fixtures, fd_indices


def numerical_matrix(
    report: dict,
    *,
    nvcc: Path,
    cache: Path,
    slurm_time: str = "00:10:00",
    timeout: int = 900,
    finite_difference_classes=("fsss", "fsps", "fpps"),
    progress=None,
) -> dict:
    """Gate 4: all ordinary/persistent RHF/UHF wrappers against libcint.

    One finite Slurm allocation per class runs every fixture, including
    displaced fixed-density Fock energies for the selected finite-difference
    subset. Existing GPU visibility is passed through without modification.
    """
    import pyscf

    from tools.vibeqc_codegen.cuda_adapter import (
        CudaBenchmarkExecutor,
        CudaCompilerAdapter,
    )
    from tools.vibeqc_codegen.cuda_target import cuda_target_info

    from .f_shell import ROOT, source_audit
    from .f_shell_cuda import emit_numerical_driver
    from .schema import file_hash, new_evidence, outcome, validate_evidence

    report = json.loads(json.dumps(report))
    compiler = CudaCompilerAdapter(
        nvcc.resolve(), cuda_target_info(report["architecture"])
    )
    executor = CudaBenchmarkExecutor(
        timeout, partition="main", gres="gpu:5090:1", slurm_time=slurm_time
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    for row in report["rows"]:
        name, compilation = row["shell_class"], row["compilation"]
        if compilation["status"] != "pass":
            row["numerical"] = outcome("not-run", "class object did not compile")
            continue
        directory = cache / compilation["cache_key"]
        obj, driver, executable = (
            directory / "kernel.o",
            directory / "driver.cu",
            directory / "numerical",
        )
        if not obj.is_file() or file_hash(obj) != compilation["object_hash"]:
            raise ValueError("numerical gate cannot use an unverified cached object")
        if source_audit(name, report["architecture"])[0] != row["source"]:
            raise ValueError("source changed between compile and numerical tiers")
        driver_source = emit_numerical_driver(name, report["architecture"])
        driver.write_text(driver_source)
        link_started = time.monotonic()
        link = compiler.link(driver, [obj], executable)
        link_seconds = time.monotonic() - link_started
        if link.returncode != 0:
            row["numerical"] = outcome(
                "fail",
                "numerical host-driver link failed",
                log=link.stdout + link.stderr,
            )
            continue
        fixtures, fd_indices = class_fixtures(
            name, finite_difference=name in finite_difference_classes
        )
        fixture_paths = []
        for ordinal, fixture in enumerate(fixtures):
            path = directory / f"fixture-{ordinal}.bin"
            write_fixture(fixture, path)
            fixture_paths.append(path)
        command = executor.command(executable.resolve()) + [
            str(p.resolve()) for p in fixture_paths
        ]
        run = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ),
        )
        (directory / "numerical.jsonl").write_text(run.stdout)
        (directory / "numerical.stderr").write_text(run.stderr)
        if run.returncode != 0:
            row["numerical"] = outcome(
                "fail",
                "Slurm CUDA fixture execution failed",
                returncode=run.returncode,
                log=run.stderr,
            )
            continue
        records = [json.loads(line) for line in run.stdout.splitlines()]
        if (
            len(records) != len(fixtures) + 1
            or records[0].get("kind") != "device"
            or [r.get("ordinal") for r in records[1:]] != list(range(len(fixtures)))
        ):
            raise ValueError("missing, duplicated, or reordered GPU fixture results")
        evidence = new_evidence(
            tier="gpu-numerical",
            subject=f"f-shell/{name}",
            inputs_hash=canonical_hash([f.inputs_hash for f in fixtures]),
        )
        evidence.update(
            revision=revision,
            device=records[0],
            backend_selected="cuda",
            hardware=outcome("pass"),
            toolchain={
                **report["toolchain"],
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyscf": pyscf.__version__,
            },
            settings={
                "dirty": dirty,
                "device": "cuda",
                "fast_compile": False,
                "seed": 135,
                "screening": 0.0,
                "slurm_command": command,
                "reference": "independent libcint values/all-center derivatives and explicit ERI orbit loops",
                "fixture_hashes": [f.inputs_hash for f in fixtures],
            },
            compilation={
                "seconds": compilation["seconds"] + link_seconds,
                "reason": None,
            },
        )
        evidence["hashes"].update(
            equation=row["source"]["ir_hash"],
            ir=row["source"]["ir_hash"],
            source=canonical_hash(
                {"kernel": row["source"]["source_hash"], "driver": file_hash(driver)}
            ),
            schedule=canonical_hash(row["source"]["schedule"]),
        )
        evidence["stages"]["representation"] = outcome("pass")
        evidence["stages"]["source"] = outcome("pass")
        evidence["stages"]["compilation"] = outcome("pass")
        values = []
        for ordinal, (fixture, record) in enumerate(zip(fixtures, records[1:])):
            decoded = decoded_outputs(fixture, record)
            expected_keys = {
                f"{spin}_{consumer}{suffix}"
                for spin in ("rhf", "uhf")
                for consumer in ("fock", "force")
                for suffix in ("", "_persistent")
            }
            if set(decoded) != expected_keys:
                raise ValueError("GPU fixture omitted a supported generated consumer")
            values.append(decoded)
            # Displaced cases are already covered by the same independent
            # analytic oracle; keep their full errors in the evidence too.
            for key, actual in decoded.items():
                evidence["block_errors"][f"fixture_{ordinal}/{key}"] = numerical_error(
                    actual,
                    fixture.reference[key.removesuffix("_persistent")],
                    atol=2e-10,
                    rtol=2e-10,
                )
                if "force" in key:
                    evidence["block_errors"][f"fixture_{ordinal}/{key}/translation"] = (
                        numerical_error(
                            actual.sum(axis=0),
                            np.zeros(3),
                            atol=2e-10,
                            rtol=0,
                        )
                    )
        finite_differences = []
        for step, indices in fd_indices:
            errors = {}
            for spin in ("rhf", "uhf"):
                gradient = np.zeros((4, 3))
                for center, xyz in product(range(4), range(3)):
                    energies = []
                    for sign in (-1, 1):
                        index = indices[(center, xyz, sign)]
                        density = (
                            fixtures[index].density
                            if spin == "rhf"
                            else fixtures[index].spin_density
                        )
                        energies.append(
                            0.5 * float(np.sum(density * values[index][f"{spin}_fock"]))
                        )
                    gradient[center, xyz] = (energies[1] - energies[0]) / (2 * step)
                errors[spin] = numerical_error(
                    gradient, -values[0][f"{spin}_force"], atol=1e-6, rtol=1e-6
                )
            finite_differences.append({"step_bohr": step, "errors": errors})
        passed = all(error["passed"] for error in evidence["block_errors"].values())
        passed &= all(
            error["passed"]
            for sample in finite_differences
            for error in sample["errors"].values()
        )
        evidence["stages"]["numerical"] = outcome(
            "pass" if passed else "fail",
            None if passed else "independent GPU numerical/FD gate failed",
        )
        validate_evidence(evidence)
        row["numerical"] = {
            **evidence["stages"]["numerical"],
            "fixture_count": len(fixtures),
            "variants": list(FIXTURE_VARIANTS),
            "finite_differences": finite_differences,
            "evidence": evidence,
            "raw_output_hash": file_hash(directory / "numerical.jsonl"),
            "driver_source_hash": file_hash(driver),
            "executable_hash": file_hash(executable),
        }
        row["resources"]["occupancy"] = outcome("pass", kernels=records[0]["kernels"])
        # Persist each class as it finishes so long manual matrices can be
        # inspected or resumed without discarding already verified evidence.
        (directory / "numerical-evidence.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )
        if progress is not None:
            progress(row)
    return report
