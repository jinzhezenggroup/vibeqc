"""Real LANL2DZ Br/I complete public DFT force qualification."""

import hashlib
import json
import os
import typing
from dataclasses import replace
from pathlib import Path

import pytest
import test_ecp_public_cpu as cpu_gates
import test_ecp_public_cuda as cuda_gates
from test_ecp_heavy import PARAMETER_SHA256, VALENCE_CHARGES, heavy_fixture

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
def retain_halogen_evidence(request: typing.Any) -> typing.Iterator[None]:
    yield
    folder = Path(".artifacts/ecp-halogen-dft")
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
    """Reuse the public force gates with independently pinned Br/I records."""
    gates = cpu_gates if device == "cpu" else cuda_gates

    def selected(
        *,
        spin: int = 0,
        representation: str = "spherical",
        d_shell: bool = False,
    ) -> typing.Any:
        assert not d_shell
        atoms, basis, mol = heavy_fixture(symbol, spin=0)
        if spin:
            # The cationic doublet has a symmetry-degenerate pi hole and
            # oscillates between equivalent integer occupations in both
            # VibeQC and an independent PySCF solve. The anionic doublet has
            # a nondegenerate sigma* frontier while retaining the same pinned
            # two-center LANL2DZ/STO-3G ECP fixture.
            mol.charge = -1
            mol.spin = 1
            mol.build(False, False)
        assert all(
            shell.angular_momentum <= 1
            for element in basis.elements
            for shell in element.shells
        )
        mol.cart = representation == "cartesian"
        basis = replace(basis, representation=representation)
        record_property("parameter_sha256", PARAMETER_SHA256[symbol])
        record_property("effective_charges", [VALENCE_CHARGES[symbol], 1])
        record_property("qualification_charge", mol.charge)
        record_property("qualification_multiplicity", mol.spin + 1)
        record_property("nao", mol.nao_nr())
        return atoms, basis, mol

    original_reference = gates.reference

    def selected_reference(
        mol: typing.Any, state: typing.Any, method: str
    ) -> typing.Any:
        if not method.endswith("uks"):
            return original_reference(mol, state, method)
        # The independent atomic guess avoids the slowly rotating open-shell
        # basin reached from the one-electron guess. Do not seed from VibeQC.
        record_property("independent_initial_guess", "atom")
        record_property("independent_maximum_cycles", 600)
        return original_reference(
            mol, state, method, initial_guess="atom", maximum_cycles=600
        )

    monkeypatch.setattr(gates, "fixture", selected)
    monkeypatch.setattr(gates, "reference", selected_reference)
    return gates


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symbol", ["Br", "I"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_halogen_complete_force_analytic_and_fd(
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
@pytest.mark.parametrize("symbol", ["Br", "I"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("method", ["pbe-rks", "pbe-uks"])
def test_halogen_budgeted_replay_and_recovery(
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


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_halogen_recovery_energy_certifies_exportable_physical_projector(
    representation: str,
) -> None:
    """A recovered energy success must certify the same state used by forces."""
    from vibeqc._ks_snapshot import NativeKsSnapshot

    atoms, basis, mol = heavy_fixture("I", spin=1)
    basis = replace(basis, representation=representation)
    xyz = mol.atom_coords()
    moved = xyz.copy()
    moved[1] += [0.03, -0.02, 0.09]
    calc = cpu_gates.calculator(basis, "pbe-uks")
    with calc.prepare_batch(
        [atoms, [("H", (0.0, 0.0, 0.0))]],
        charges=[1, 0],
        multiplicities=[2, 2],
    ) as batch:
        for coordinates in (None, None, [moved, None]):
            batch.execute(coordinates=coordinates, properties=("energy",), strict=True)
        failed = batch.execute(coordinates=[[0.0], None], properties=("energy",))
        assert not failed.items[0].succeeded and failed.items[1].succeeded
        recovered = batch.execute(
            coordinates=[xyz, None], properties=("energy",), strict=True
        )
        assert recovered.items[0].converged
        snapshot = NativeKsSnapshot(batch, 0)
        snapshot.close()
