"""Scalar ECP preflight, independent of native libraries and reference packages."""

import json

import pytest
from vibeqc import Atom, BasisProvenance, BasisSet, BasisShell, ElementBasis
from vibeqc.ecp import resolve_ecp


def potential(**changes):
    return {
        "ecp_type": "scalar_ecp",
        "angular_momentum": [0],
        "r_exponents": [2],
        "gaussian_exponents": ["0.8"],
        "coefficients": [["-2"]],
    } | changes


def element(records):
    return ElementBasis(
        11,
        (BasisShell(0, ("0.7",), (("1",),)),),
        ecp_core_electrons=10,
        ecp_data=json.dumps(records),
    )


def resolve(record):
    basis = BasisSet(
        "synthetic scalar ECP",
        (record,),
        BasisProvenance("synthetic test", "1", "CC0", "0" * 64),
    )
    return resolve_ecp(basis, (Atom(11, (0.0, 0.0, 0.0)),))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"ecp_type": "spin_orbit_ecp"}, "scalar_ecp"),
        ({"ecp_type": "unknown"}, "scalar_ecp"),
        ({"ecp_type": None}, "scalar_ecp"),
        ({"angular_momentum": [0, 1]}, "one angular channel"),
        ({"angular_momentum": []}, "one angular channel"),
        ({"angular_momentum": 0}, "one angular channel"),
    ],
)
def test_scalar_contract_at_construction_and_resolution(changes, message):
    malformed = [potential(**changes)]
    with pytest.raises(ValueError, match=message):
        element(malformed)

    # Simulate a stale/corrupted owned record to exercise the resolver's own
    # boundary independently of the normal immutable constructor's validation.
    record = element([potential()])
    object.__setattr__(record, "ecp_data", json.dumps(malformed))
    with pytest.raises(ValueError, match=message):
        resolve(record)


def test_highest_singleton_channel_is_local():
    cores, terms = resolve(element([potential(angular_momentum=[1]), potential()]))
    assert cores == (10,)
    assert terms == ((0, -1, 2, 0.8, -2.0), (0, 0, 2, 0.8, -2.0))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"angular_momentum": [4]}, "local channel"),
        ({"spin_orbit": True}, "unknown ECP parameter fields"),
    ],
)
def test_unsupported_execution_conventions_remain_rejected(changes, message):
    with pytest.raises(NotImplementedError, match=message):
        resolve(element([potential(**changes)]))
