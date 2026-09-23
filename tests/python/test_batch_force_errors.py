"""Force adapter failures retain their cause without poisoning batch neighbors."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions, _native


@pytest.mark.parametrize(
    "error,status",
    [
        (
            FileNotFoundError("missing packaged stationary CUDA artifact for pbe_rks"),
            _native.STATUS_INTERNAL_ERROR,
        ),
        (
            OSError("cannot load stationary shared library"),
            _native.STATUS_INTERNAL_ERROR,
        ),
        (
            NotImplementedError("unsupported force source"),
            _native.STATUS_NOT_IMPLEMENTED,
        ),
        (MemoryError("force arena budget exceeded"), _native.STATUS_OUT_OF_MEMORY),
        (TypeError("invalid force layout type"), _native.STATUS_INVALID_ARGUMENT),
        (
            ValueError("force artifact identity mismatch"),
            _native.STATUS_INVALID_ARGUMENT,
        ),
        (
            ArithmeticError("nonfinite force coefficient"),
            _native.STATUS_NUMERICAL_FAILURE,
        ),
        (RuntimeError("force execution failed"), _native.STATUS_NUMERICAL_FAILURE),
    ],
)
def test_generated_force_error_retains_cause_and_neighbor_results(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status: int
) -> None:
    # The shared Python error boundary needs no GPU or broken installation.
    # Native CPU SCF supplies real input-ordered results. Admit only the injected
    # force consumer below; this does not claim a new production CPU capability.
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calculator = Calculator(
        method="pbe-rks",
        device="cpu",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
        ),
    )
    monkeypatch.setattr(
        calculator,
        "_capabilities",
        replace(
            calculator._capabilities,
            supported_properties=frozenset(("energy", "forces")),
        ),
    )
    with calculator.prepare_batch([atoms] * 3) as batch:
        visited = []
        fail = True

        def force(index: int, geometry: object) -> tuple[np.ndarray, dict]:
            visited.append(index)
            if fail and index == 1:
                raise error
            return np.full((2, 3), index + 1.0), {}

        monkeypatch.setattr(batch, "_public_dft_cpu_force", force)
        result = batch.execute()
        assert visited == [0, 1, 2]
        assert tuple(item.index for item in result.items) == (0, 1, 2)
        assert result.failure_indices == (1,)
        failed = result.items[1]
        assert failed.status == status
        assert failed.forces is None
        assert failed.converged
        assert type(error).__name__ in failed.status_message
        assert str(error) in failed.status_message
        for index in (0, 2):
            np.testing.assert_array_equal(result.items[index].forces, index + 1.0)
            assert str(error) not in result.items[index].status_message

        visited.clear()
        with pytest.raises(RuntimeError) as caught:
            batch.execute(strict=True)
        assert visited == [0, 1, 2]
        assert f"1: {failed.status_message}" in str(caught.value)

        # An explicit energy-only replay neither invokes the failed consumer nor
        # inherits its status; a repaired force replay also clears the old cause.
        visited.clear()
        energy = batch.execute(strict=True, properties=("energy",))
        assert energy.succeeded and not visited
        fail = False
        recovered = batch.execute(strict=True)
        assert visited == [0, 1, 2]
        assert recovered.succeeded
        assert all(str(error) not in item.status_message for item in recovered.items)
