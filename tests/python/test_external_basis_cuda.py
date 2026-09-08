"""Opt-in public imported-basis GPU gates; run within a finite Slurm job."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator, import_bse

from tools.validate_external_basis import (
    DATA,
    basis_for,
    case_endpoints,
    fixtures,
    ragged_endpoints,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_BASIS_CUDA_TEST") != "1", reason="opt-in Slurm CUDA gate"
)


@pytest.mark.parametrize("index", range(6))
def test_public_imported_energy_forces_and_prepared_replays(index):
    manifest, arrays = fixtures()
    case_endpoints(manifest["cases"][index], arrays, ("cuda",), 1)


def test_imported_ragged_order_changed_geometry_and_failure_isolation():
    manifest, arrays = fixtures()
    ragged_endpoints(manifest, arrays, "cuda", 2)


def test_unsupported_data_and_changed_basis_never_execute_as_old_model():
    for filename, symbol, diagnostic in (
        ("cc-pvtz-fe.json", "Fe", "l=4"),
        ("def2-tzvp-au.json", "Au", "ECP"),
    ):
        basis = import_bse(
            DATA / filename,
            source="local BSE fixture",
            source_version="pinned",
            license="BSD-3-Clause",
        )
        with pytest.raises(NotImplementedError, match=diagnostic):
            Calculator(basis=basis, device="cuda").singlepoint([(symbol, (0, 0, 0))])
    manifest, _ = fixtures()
    spec = manifest["cases"][0]
    basis = basis_for(spec)
    calculator = Calculator(basis=basis, device="cuda")
    with calculator.prepare_batch([spec["atoms"]]) as prepared:
        prepared.execute(strict=True)
        calculator._basis = replace(basis, representation="spherical")
        with pytest.raises(RuntimeError, match="identity changed"):
            prepared.execute()


def test_numpy_integer_metadata_preserves_native_gpu_occupation_rejection():
    manifest, _ = fixtures()
    calculator = Calculator(basis=basis_for(manifest["cases"][0]), device="cuda")
    atoms = [[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], [("H", (0, 0, 0))]]
    metadata = calculator.basis_metadata(
        atoms[1], charge=np.int64(0), multiplicity=np.int64(1)
    )
    assert "occupation_error" in metadata["orbital"]["electrons"]
    json.dumps(metadata)
    # Native preparation rejects an invalid RHF occupation before execution.
    # NumPy scalars must preserve that existing
    # behavior, rather than failing earlier in metadata JSON serialization.
    for charges in ([0, 0], np.array([0, 0])):
        with pytest.raises(RuntimeError, match="invalid argument"):
            calculator.batch_singlepoint(atoms, charges=charges)
    result = calculator.batch_singlepoint(
        [atoms[0], atoms[0]],
        charges=np.array([0, 0]),
        multiplicities=np.array([1, 1]),
        strict=True,
    )
    assert all(item.executed_backend == "cuda" for item in result.items)
