"""Discrete GFN1 metadata must be representable without implicit coercion."""

import pytest

from tools.parameters.generate_gfn1 import ParameterError, validate


def _valid_parameters() -> dict:
    shell_counts = [3] * 65 + [2] * 21
    elements = []
    for atomic_number, count in enumerate(shell_counts, 1):
        shells = [
            {
                "angular_momentum": index,
                "coordination_number_scale": 0.0,
                "index": index,
                "is_valence": True,
                "level": -1.0,
                "ngauss": 3,
                "principal_quantum_number": 1,
                "reference_occupation": 1.0,
                "shell_hubbard_scale": 0.0,
                "shell_polynomial": 0.0,
                "slater": 1.0,
            }
            for index in range(count)
        ]
        elements.append(
            {
                "arep": 1.0,
                "atomic_number": atomic_number,
                "atomic_radius_bohr": 1.0,
                "covalent_radius_bohr": 1.0,
                "dkernel": 1.0,
                "en": 1.0,
                "gam": 1.0,
                "gam3": 1.0,
                "mprad": 1.0,
                "mpvcn": 1.0,
                "qkernel": 1.0,
                "shells": shells,
                "symbol": f"E{atomic_number}",
                "xbond": 0.0,
                "zeff": 1.0,
            }
        )
    shell_pairs = [
        {"angular_momenta": [first, second], "value": 1.0}
        for first in range(3)
        for second in range(first, 3)
    ]
    return {
        "charge": {"average": "harmonic", "gexp": 2.0},
        "coordination_number": {
            "coincident_cutoff_inclusive": True,
            "coincident_distance_squared_cutoff_bohr2": 1.0e-12,
            "count_expression": "1 / (1 + exp(-k * ((r_cov_i + r_cov_j) / r - 1)))",
            "covalent_radius_scale": 1.0,
            "cutoff_bohr": 25.0,
            "cutoff_inclusive": True,
            "derivative_expression": "(-k * (r_cov_i + r_cov_j) * expterm) / (r^2 * (expterm + 1)^2)",
            "directed_factor": 1.0,
            "maximum_cn_cutoff": None,
            "model": "exp",
            "pair_loop": "ordered",
            "steepness": 16.0,
        },
        "dispersion": {
            "a1": 1.0,
            "a2": 1.0,
            "model": "d3",
            "s6": 1.0,
            "s8": 1.0,
            "s9": 0.0,
            "self_consistent": False,
        },
        "elements": elements,
        "halogen": {"damping": 0.44, "radius_scale": 1.3},
        "hamiltonian": {
            "coordination_number": "exp",
            "enscale": 1.0,
            "kpol": 1.0,
            "pair_scale_default": 1.0,
            "pair_scale_overrides": [],
            "shell_pair_scale": shell_pairs,
            "wexp": 1.0,
        },
        "meta": {},
        "method": "gfn1-xtb",
        "repulsion": {"kexp": 1.5, "klight": 1.5},
        "schema_version": 2,
        "thirdorder": {"mode": "atom", "shell_resolved": False},
    }


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
    parameters = _valid_parameters()
    parameters["elements"][0]["shells"][0][field] = value
    with pytest.raises(ParameterError):
        validate(parameters)
