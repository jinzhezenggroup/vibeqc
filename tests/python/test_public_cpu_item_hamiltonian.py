"""Mixed ECP fleets select Hamiltonian identity per item, not per basis set."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
from vibeqc import _cpu_force_resources, _dft_gradient, _stationary_cpu, ecp
from vibeqc.batch import PreparedBatch
from vibeqc_compiler import dft

if TYPE_CHECKING:
    from typing_extensions import Self


@pytest.mark.parametrize(
    "cores,hamiltonian,accepted",
    (
        ((0, 0), "all-electron", True),
        ((2, 0), "scalar-semilocal-ecp", True),
        ((0, 0), "scalar-semilocal-ecp", False),
        ((2, 0), "all-electron", False),
    ),
)
def test_cpu_force_checks_current_item_ecp_inventory(
    monkeypatch: pytest.MonkeyPatch,
    cores: tuple[int, ...],
    hamiltonian: str,
    accepted: bool,
) -> None:
    class Basis:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(dft, "NativeAO", lambda *args, **kwargs: Basis())
    monkeypatch.setattr(_cpu_force_resources, "qualified_basis", lambda basis: True)
    monkeypatch.setattr(
        _cpu_force_resources, "cpu_force_inventory", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(ecp, "resolve_ecp", lambda *args: (cores, ()))
    source = SimpleNamespace(backend="cpu", hamiltonian=hamiltonian, close=Mock())
    state = SimpleNamespace(_source=source)
    monkeypatch.setattr(
        _dft_gradient.StationaryKsState, "from_native", lambda *args, **kwargs: state
    )
    gradient = np.arange(6.0).reshape(2, 3)
    consumer = Mock(return_value=SimpleNamespace(gradient=gradient, work={}))
    monkeypatch.setattr(_stationary_cpu, "complete_rks_gradient_diagnostic", consumer)
    calculator = SimpleNamespace(
        _basis=object(),
        _method_name="pbe-rks",
        _device_name="cpu",
        _representation_name="cartesian",
        _ks_options=SimpleNamespace(
            grid=SimpleNamespace(radial_points=1, angular_polar=1, angular_azimuth=1)
        ),
    )
    batch = SimpleNamespace(_calculator=calculator, _charges=[0], _multiplicities=[1])
    atoms = (object(), object())
    if accepted:
        forces, _ = PreparedBatch._public_dft_cpu_force(batch, 0, atoms)
        np.testing.assert_array_equal(forces, -gradient)
        consumer.assert_called_once()
    else:
        with pytest.raises(ValueError, match="Hamiltonian identity mismatch"):
            PreparedBatch._public_dft_cpu_force(batch, 0, atoms)
        consumer.assert_not_called()
    source.close.assert_called_once()
