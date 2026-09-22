"""Real Stuttgart RLC Na/K complete public DFT force qualification."""

import hashlib
import json
import os
import typing
from dataclasses import replace
from pathlib import Path

import pytest
import test_ecp_public_cpu as cpu_gates
import test_ecp_public_cuda as cuda_gates
from test_ecp_stuttgart import PARAMETER_SHA256, stuttgart_fixture

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            os.environ.get("VIBEQC_ECP_CUDA_TEST") != "1",
            reason="explicit real-device ECP gate",
        ),
    ),
]


@pytest.fixture(autouse=True)
def retain_stuttgart_evidence(request: typing.Any) -> typing.Iterator[None]:
    yield
    folder = Path(".artifacts/ecp-stuttgart-dft")
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]
    (folder / f"{key}.json").write_text(
        json.dumps(
            {
                "test": request.node.nodeid,
                "measurements": dict(request.node.user_properties),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def select_fixture(
    symbol: str,
    device: str,
    monkeypatch: typing.Any,
    record_property: typing.Any,
) -> typing.Any:
    """Reuse identical endpoint gates with independently pinned physical records."""
    gates = cpu_gates if device == "cpu" else cuda_gates

    def selected(
        *,
        spin: int = 0,
        representation: str = "spherical",
        d_shell: bool = False,
    ) -> typing.Any:
        assert not d_shell
        atoms, basis, mol = stuttgart_fixture(symbol, spin=spin)
        assert all(
            s.angular_momentum <= 1
            for element in basis.elements
            for s in element.shells
        )
        mol.cart = representation == "cartesian"
        basis = replace(basis, representation=representation)
        record_property("parameter_sha256", PARAMETER_SHA256[symbol])
        record_property("effective_charges", [1, 1])
        record_property("nao", mol.nao_nr())
        return atoms, basis, mol

    monkeypatch.setattr(gates, "fixture", selected)
    return gates


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symbol", ["Na", "K"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_stuttgart_complete_force_analytic_and_fd(
    device: str,
    symbol: str,
    representation: str,
    method: str,
    monkeypatch: typing.Any,
    record_property: typing.Any,
    tmp_path: Path,
) -> None:
    gates = select_fixture(symbol, device, monkeypatch, record_property)
    gates.test_public_ecp_force_analytic_and_reconverged_fd(
        method,
        representation,
        record_property,
        tmp_path,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symbol", ["Na", "K"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("method", ["pbe-rks", "pbe-uks"])
def test_stuttgart_budgeted_replay_and_recovery(
    device: str,
    symbol: str,
    representation: str,
    method: str,
    monkeypatch: typing.Any,
    record_property: typing.Any,
) -> None:
    gates = select_fixture(symbol, device, monkeypatch, record_property)
    gates.test_public_ecp_budgeted_ragged_replay_and_failure_recovery(
        method,
        representation,
        record_property,
    )
