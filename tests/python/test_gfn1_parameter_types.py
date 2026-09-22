"""Discrete GFN1 metadata must be representable without implicit coercion."""

import json
from pathlib import Path

import pytest

from tools.parameters.generate_gfn1 import ParameterError, validate

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_valence", "false"),
        ("is_valence", 0),
        ("index", 0.0),
        ("angular_momentum", 0.0),
        ("principal_quantum_number", 1.5),
        ("principal_quantum_number", 256),
        ("principal_quantum_number", -1),
        ("ngauss", 4.0),
        ("ngauss", 256),
        ("ngauss", -1),
    ],
)
def test_gfn1_shell_discrete_types_fail_before_rendering(
    field: str, value: object
) -> None:
    parameters = json.loads(
        (
            ROOT / "upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1.json"
        ).read_text()
    )
    parameters["elements"][0]["shells"][0][field] = value
    with pytest.raises(ParameterError):
        validate(parameters)
