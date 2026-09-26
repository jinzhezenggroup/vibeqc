"""Output selection must skip response work and survive later force replays."""

import json
import os
import typing

import numpy as np
import pytest
from vibeqc import Calculator

SYSTEMS = [
    [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))],
    [(1, (0.0, 0.0, -0.8)), (1, (0.0, 0.0, 0.8))],
]


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize(
    "route", ["cpu-direct", "cpu-df", "cuda-direct", "cuda-df", "cuda-source"]
)
def test_energy_only_batch_preserves_replay_and_force_recovery(
    method: typing.Any, route: typing.Any, monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    device, provider = route.split("-")
    if device == "cuda":
        if os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1":
            pytest.skip("explicit allocated-GPU opt-in")
        assert os.environ.get("SLURM_JOB_ID")
    options = {
        "method": method,
        "device": device,
        "density_fitting": "none" if provider == "direct" else device,
        "density_fitting_memory_budget_bytes": 8 << 20 if provider == "source" else 0,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    actual = Calculator(**options)
    reference = Calculator(**options)
    # Include an empty beta occupied space and a closed-shell neighbor in the
    # UHF fleet, so output selection also crosses independent spin buckets.
    preparation = {"multiplicities": [3, 1]} if method == "uhf" else {}
    with (
        actual.prepare_batch(SYSTEMS, **preparation) as batch,
        reference.prepare_batch(SYSTEMS, **preparation) as full,
    ):
        for replay in range(3):
            expected = full.execute(strict=True)
            trace = tmp_path / f"energy-{replay}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            result = batch.execute(properties=("energy",), strict=True)
            monkeypatch.delenv("VIBEQC_DF_TRACE")
            np.testing.assert_allclose(
                result.energies, expected.energies, atol=1e-10, rtol=0
            )
            assert all(item.forces is None for item in result.items)
            if device == "cuda" and provider != "direct":
                # Reading only output buffers would pass numerical parity too.
                # Executed traces must prove response was never launched.
                records = [json.loads(line) for line in trace.read_text().splitlines()]
                assert records
                assert not {
                    "force_response",
                    "one_electron_response",
                    "one_electron_derivative_export",
                    "nuclear_derivative_export",
                } & {record["operation"] for record in records}
            expected = full.execute(strict=True)
            recovered = batch.execute(strict=True)
            np.testing.assert_allclose(
                recovered.energies, expected.energies, atol=1e-10, rtol=0
            )
            for item, target in zip(recovered.items, expected.items, strict=True):
                np.testing.assert_allclose(
                    item.forces, target.forces, atol=1e-9, rtol=0
                )
        failed = batch.execute([np.zeros((1, 3)), None], properties=("energy",))
        assert failed.failure_indices == (0,)
        assert failed.items[1].converged and failed.items[1].forces is None


@pytest.mark.parametrize(
    "properties,error",
    [
        ("energy", TypeError),
        ([], ValueError),
        (["forces"], ValueError),
        (["energy", "dipole"], ValueError),
        ([[]], TypeError),
    ],
)
def test_invalid_batch_properties_reject_before_execution(
    properties: typing.Any, error: typing.Any, monkeypatch: typing.Any
) -> None:
    with Calculator().prepare_batch(SYSTEMS) as batch:

        def forbidden(*args: typing.Any) -> None:
            pytest.fail("invalid output request reached native execution")

        monkeypatch.setattr(batch._library, "vibeqc_batch_execute", forbidden)
        with pytest.raises(error):
            batch.execute(properties=properties)


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_energy_only_dft_batch_rejects_forces_before_execution(
    method: typing.Any, monkeypatch: typing.Any
) -> None:
    preparation = (
        {"charges": [-1], "multiplicities": [2]} if method.endswith("uks") else {}
    )
    with Calculator(method=method).prepare_batch(SYSTEMS[:1], **preparation) as batch:

        def forbidden(*args: typing.Any) -> None:
            pytest.fail("unsupported force request reached native execution")

        monkeypatch.setattr(batch._library, "vibeqc_batch_execute", forbidden)
        with pytest.raises(ValueError, match="does not support properties: forces"):
            batch.execute(properties=("energy", "forces"))
