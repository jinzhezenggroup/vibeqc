"""Content hashes must not admit malformed execution ABI metadata."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest
from vibeqc_compiler.xc import production_domain_evidence as evidence

SPINS = ("polarized", "unpolarized")


def outputs(size: int) -> tuple[tuple[int, ...], ...]:
    return (
        (),
        *((i,) for i in range(size)),
        *combinations_with_replacement(range(size), 2),
    )


@pytest.fixture
def capability(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    profile = SimpleNamespace(
        spin_layouts=SPINS,
        outputs=("energy", "vxc", "fxc"),
        case_ids_for_spin=lambda _: ("density/near-zero",),
        identity="c" * 64,
        eligible=True,
    )
    cap = SimpleNamespace(
        name="schema-fixture",
        identity="a" * 64,
        required_ingredients=("rho", "sigma"),
        production_domain_profile=profile,
    )
    monkeypatch.setattr(evidence, "functional_capability", lambda _: cap)
    return cap


def programs(capability: SimpleNamespace) -> dict[str, SimpleNamespace]:
    result = {}
    for spin in SPINS:
        features = ("rho_a", "rho_b") if spin == "polarized" else ("rho",)
        if "sigma" in capability.required_ingredients:
            features += (
                ("sigma_aa", "sigma_ab", "sigma_bb")
                if spin == "polarized"
                else ("sigma",)
            )
        if "tau" in capability.required_ingredients:
            features += ("tau_a", "tau_b") if spin == "polarized" else ("tau",)
        result[spin] = SimpleNamespace(
            spec=SimpleNamespace(
                identifier=capability.name,
                capability_identity=capability.identity,
                spin=spin,
                domain="qualification-domain/v1",
                source_identity="d" * 64,
                features=features,
            ),
            expression_hash="e" * 64,
            optimization="after",
            order=2,
            outputs=outputs(len(features)),
        )
    return result


def receipt(capability: SimpleNamespace) -> dict:
    execution = evidence.build_execution_binding(capability.name, programs(capability))
    cases = [
        {
            "spin": spin,
            "case_id": "density/near-zero",
            "status": "pass",
            "outputs": ["energy", "vxc", "fxc"],
            "reason": None,
        }
        for spin in SPINS
    ]
    return evidence.build_result(
        capability.name, cases, evidence="test://schema-only", execution=execution
    )


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("value", (False, 0.0))
def test_type_changed_output_cannot_keep_original_receipt_hash(
    capability: SimpleNamespace, spin: str, value: object
) -> None:
    original = receipt(capability)
    changed = deepcopy(original)
    record = next(
        row for row in changed["execution"]["programs"] if row["spin"] == spin
    )
    record["outputs"][1][0] = value
    assert changed["identity"] == original["identity"]
    with pytest.raises(ValueError, match="outputs"):
        evidence.validate_result(capability.name, changed)


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("value", (False, 0.0))
def test_builder_rejects_noninteger_output_indices(
    capability: SimpleNamespace, spin: str, value: object
) -> None:
    candidates = programs(capability)
    candidates[spin].outputs = ((), (value,), *candidates[spin].outputs[2:])
    with pytest.raises(ValueError, match="outputs"):
        evidence.build_execution_binding(capability.name, candidates)


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("defect", ("missing", "extra", "reordered", "unknown"))
def test_execution_features_must_match_registration_and_spin(
    capability: SimpleNamespace, spin: str, defect: str
) -> None:
    original = receipt(capability)
    execution = deepcopy(original["execution"])
    record = next(row for row in execution["programs"] if row["spin"] == spin)
    features = record["features"]
    record["features"] = {
        "missing": features[:-1],
        "extra": [*features, "unregistered"],
        "reordered": list(reversed(features)),
        "unknown": ["unregistered", *features[1:]],
    }[defect]
    record["outputs"] = [list(row) for row in outputs(len(record["features"]))]
    execution["identity"] = evidence.canonical_hash(
        {key: value for key, value in execution.items() if key != "identity"}
    )
    with pytest.raises(ValueError, match="feature"):
        evidence.build_result(
            capability.name,
            original["cases"],
            evidence="test://bad-abi",
            execution=execution,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("optimization", "unknown"),
        ("source_identity", "not-a-digest"),
        ("expression_hash", "g" * 64),
    ),
)
def test_binding_factory_rejects_invalid_execution_metadata(
    capability: SimpleNamespace, field: str, value: str
) -> None:
    candidates = programs(capability)
    candidate = candidates["polarized"]
    target = candidate.spec if field == "source_identity" else candidate
    setattr(target, field, value)
    with pytest.raises(ValueError, match="execution"):
        evidence.build_execution_binding(capability.name, candidates)


@pytest.mark.parametrize(
    "ingredients",
    (("rho",), ("rho", "sigma"), ("rho", "tau"), ("rho", "sigma", "tau")),
)
@pytest.mark.parametrize("optimization", ("none", "before", "after"))
def test_supported_abis_and_recorded_domains_remain_byte_identical(
    capability: SimpleNamespace, ingredients: tuple[str, ...], optimization: str
) -> None:
    capability.required_ingredients = ingredients
    candidates = programs(capability)
    for candidate in candidates.values():
        candidate.optimization = optimization
    binding = evidence.build_execution_binding(capability.name, candidates)
    assert evidence._normalize_execution(binding, capability) == binding
    assert all(
        row["domain"] == "qualification-domain/v1" for row in binding["programs"]
    )
    assert binding["identity"] == evidence.canonical_hash(
        {key: value for key, value in binding.items() if key != "identity"}
    )
