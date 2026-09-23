"""Host-side public admission tests; no GPU numerical qualification is implied."""

from __future__ import annotations

import typing
from dataclasses import replace

import pytest
from test_ecp import fixture
from test_ecp_stationary_cpu import GRID
from vibeqc import Calculator, KsOptions, _native
from vibeqc._dft_gradient import StationaryKsState


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize(
    "representation,d_shell,f_shell",
    [
        ("cartesian", False, False),
        ("spherical", False, False),
        ("cartesian", True, False),
        ("spherical", True, False),
        ("cartesian", False, True),
        ("spherical", False, True),
    ],
)
def test_cuda_ecp_public_force_capability_has_an_explicit_basis_domain(
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
    method: typing.Any,
    serialized: typing.Any,
    representation: typing.Any,
    d_shell: typing.Any,
    f_shell: typing.Any,
) -> None:
    atoms, record, _ = fixture(
        representation=representation, d_shell=d_shell, f_shell=f_shell
    )
    basis = record
    if serialized:
        path = tmp_path / "ecp-basis.json"
        record.write(path)
        basis = path
    # Only capability/admission is exercised. Do not require or probe a GPU.
    library = _native.load_library(device="cpu")
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: library)
    calculator = Calculator(basis=basis, method=method, device="cuda")
    admitted = not f_shell
    assert ("forces" in calculator._capabilities.supported_properties) == admitted
    cpu = Calculator(basis=basis, method=method, device="cpu")
    # The bounded public CPU consumer also stops at d, even though separate
    # integral and HF routes accept f shells.
    assert ("forces" in cpu._capabilities.supported_properties) == admitted

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("unqualified public ECP forces reached native preparation")

    monkeypatch.setattr(calculator, "prepare_batch", forbidden)
    spin = int(method.endswith("uks"))
    if not admitted:
        with pytest.raises(ValueError, match="forces"):
            calculator.singlepoint(
                atoms,
                charge=spin,
                multiplicity=spin + 1,
                properties=("energy", "forces"),
            )
    # An all-electron basis still receives the existing C2 public capability.
    all_electron = replace(record, elements=(record.by_element[1],))
    qualified = Calculator(basis=all_electron, method=method, device="cuda")
    assert qualified._capabilities.supported_properties == frozenset(
        {"energy", "forces"}
    )


def test_public_force_wrapper_rejects_cpu_owner_before_compilation(
    monkeypatch: typing.Any,
) -> None:
    from vibeqc import _stationary_cuda

    atoms, record, _ = fixture(representation="cartesian")
    calculator = Calculator(
        basis=record,
        method="lda-rks",
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    sources = []
    original = StationaryKsState.from_native

    def track(batch: typing.Any, basis: typing.Any, *, index: int = 0) -> typing.Any:
        state = original(batch, basis, index=index)
        sources.append(state._source)
        return state

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("public ECP forces reached a compiler or diagnostic consumer")

    monkeypatch.setattr(StationaryKsState, "from_native", staticmethod(track))
    monkeypatch.setattr(
        _stationary_cuda, "complete_rks_cuda_gradient_diagnostic", forbidden
    )
    with calculator.prepare_batch([atoms]) as batch:
        batch.execute(strict=True, properties=("energy",))
        monkeypatch.setattr(batch, "_stationary_cuda_compiler", forbidden)
        with pytest.raises(NotImplementedError, match="qualified CUDA owner"):
            batch._public_dft_cuda_force(0, batch._systems[0])
        assert len(sources) == 1
        assert not sources[0]._handle
