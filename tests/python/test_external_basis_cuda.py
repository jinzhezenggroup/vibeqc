"""Opt-in public imported-basis GPU gates; run within a finite Slurm job."""

import os
from dataclasses import replace

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
