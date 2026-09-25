import inspect
import json

import numpy as np
import pytest
from vibeqc_compiler.method.gfn1_es3 import (
    GFN1_ES3_VERSION,
    build_gfn1_es3_kernel,
    build_gfn1_es3_primal,
)
from vibeqc_compiler.method.gfn2_es3_runtime import build_gfn2_es3_primal
from vibeqc_compiler.method.xtb import GFN1_PARAMETER_SET
from vibeqc_compiler.tensor import execute

from tools import source_registry
from tools.parameters import generate_gfn1


@pytest.mark.parametrize(
    ("atomic_number", "charge"),
    (
        (6, -1.3),
        (8, 0.45),
        (17, 1.7),
    ),
)
def test_gfn1_es3_matches_pinned_atom_parameters(
    atomic_number: int, charge: float
) -> None:
    source_bytes, _ = generate_gfn1.load_registered_inputs()
    parameters = json.loads(source_bytes)
    assert parameters["thirdorder"] == {"mode": "atom", "shell_resolved": False}
    gamma3 = float(parameters["elements"][atomic_number - 1]["gam3"])

    values = execute(
        build_gfn1_es3_kernel(),
        {
            "gamma3": np.asarray(gamma3, dtype=np.float64),
            "charge": np.asarray(charge, dtype=np.float64),
            "d_charge": np.asarray(1.0, dtype=np.float64),
        },
    ).outputs

    charge_squared = charge * charge
    expected_potential = charge_squared * gamma3
    expected_energy = charge_squared * charge * gamma3 / 3.0
    assert values["energy"] == pytest.approx(expected_energy, rel=2e-15, abs=1e-18)
    assert values["potential"] == pytest.approx(
        expected_potential, rel=2e-15, abs=1e-18
    )
    assert values["energy_charge_derivative"] == pytest.approx(
        values["potential"], rel=2e-15, abs=1e-18
    )


def test_gfn1_es3_identity_binds_atom_resolution_and_parameter_set() -> None:
    primal = build_gfn1_es3_primal()
    provenance = primal.provenance

    assert provenance["kind"] == "gfn1-es3-atom-primal"
    assert provenance["version"] == GFN1_ES3_VERSION
    assert provenance["state_resolution"] == "atom"
    assert provenance["third_order_mode"] == "atom"
    assert provenance["parameter_set_identity"] == GFN1_PARAMETER_SET.identity
    assert provenance["parameter_revision"] == GFN1_PARAMETER_SET.revision


def test_gfn2_es3_wrapper_preserves_existing_semantic_provenance() -> None:
    provenance = build_gfn2_es3_primal().provenance

    assert provenance == {
        "kind": "gfn2-es3-shell-primal",
        "version": "gfn2-es3-runtime-ir-v1",
        "source": "#560",
        "fp64_order": "q2=q*q; potential=q2*gamma3; energy=(q2*q*gamma3)/3",
    }


def test_gfn1_and_gfn2_delegate_third_order_math_to_shared_owner() -> None:
    gfn1_source = inspect.getsource(build_gfn1_es3_primal)
    gfn2_source = inspect.getsource(build_gfn2_es3_primal)

    for source in (gfn1_source, gfn2_source):
        assert "build_onsite_third_order_primal" in source
        assert "multiply(" not in source


@pytest.mark.parametrize("message", ("missing registered input", "source hash mismatch"))
def test_gfn1_es3_propagates_reference_verification_failure(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    def reject_reference() -> tuple[bytes, dict]:
        raise source_registry.SourceRegistryError(message)

    monkeypatch.setattr(generate_gfn1, "load_registered_inputs", reject_reference)
    with pytest.raises(source_registry.SourceRegistryError, match=message):
        test_gfn1_es3_matches_pinned_atom_parameters(6, -1.3)
