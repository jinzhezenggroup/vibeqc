from __future__ import annotations

from types import SimpleNamespace

import pytest
from vibeqc_compiler.method import bulk_ks
from vibeqc_compiler.method.spec import UnsupportedMethod
from vibeqc_compiler.xc.capability_resolution import (
    CapabilityNotQualified,
    CapabilityResolution,
)
from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability


def _qualified_resolution(name: str) -> tuple[object, CapabilityResolution]:
    capability = functional_capability(name)
    qualified = CapabilityResolution(
        name=capability.name,
        identity=capability.identity,
        required_stages=(
            "compiled-cpu",
            "production-domain",
            "molecular-scf",
        ),
        qualified_stages=(
            "graph-imported",
            "pointwise-validated",
            "compiled-cpu",
            "production-domain",
            "molecular-scf",
        ),
        public_dft=False,
    )
    return capability, qualified


def _candidate_resolution(name: str) -> tuple[object, CapabilityResolution]:
    capability = functional_capability(name)
    qualified = CapabilityResolution(
        name=capability.name,
        identity=capability.identity,
        required_stages=("compiled-cpu", "production-domain"),
        qualified_stages=(
            "graph-imported",
            "pointwise-validated",
            "compiled-cpu",
            "production-domain",
        ),
        public_dft=False,
    )
    return capability, qualified


def test_bulk_ks_fails_closed_without_molecular_evidence() -> None:
    with pytest.raises(CapabilityNotQualified) as exc:
        bulk_ks.resolve_bulk_ks("GGA_X_PBE_SOL")

    assert "compiled-cpu" in exc.value.missing_stages
    assert "production-domain" in exc.value.missing_stages
    assert "molecular-scf" in exc.value.missing_stages


def test_bulk_ks_candidate_breaks_molecular_scf_evidence_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, qualified = _candidate_resolution("GGA_X_PBE_SOL")
    requests: list[tuple[str, ...]] = []

    def fake_resolve(
        name: str,
        *,
        required_stages: tuple[str, ...],
        evidence: dict[str, object] | None = None,
    ) -> CapabilityResolution:
        assert name == capability.name
        requests.append(tuple(required_stages))
        return qualified

    monkeypatch.setattr(bulk_ks, "resolve_capability", fake_resolve)
    result = bulk_ks.resolve_bulk_ks_candidate(capability.name)

    assert requests == [("compiled-cpu", "production-domain")]
    assert result.capability.required_stages == (
        "compiled-cpu",
        "production-domain",
    )
    assert "molecular-scf" not in result.capability.qualified_stages
    assert result.plan.required_lowerers == ("semilocal-xc",)
    assert not result.to_payload()["public_dft"]


def test_bulk_ks_requires_exact_cpu_stages_and_builds_pure_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, qualified = _qualified_resolution("GGA_X_PBE_SOL")
    requests: list[tuple[object, ...]] = []
    evidence = {"compiled-cpu": {"sentinel": True}}

    def fake_capability(
        name: str,
        *,
        evidence: dict[str, object] | None = None,
    ) -> object:
        requests.append(("capability", name, evidence))
        return capability

    def fake_resolve(
        name: str,
        *,
        required_stages: tuple[str, ...],
        evidence: dict[str, object] | None = None,
    ) -> CapabilityResolution:
        requests.append(("resolve", name, tuple(required_stages), evidence))
        return qualified

    monkeypatch.setattr(bulk_ks, "functional_capability", fake_capability)
    monkeypatch.setattr(bulk_ks, "resolve_capability", fake_resolve)

    result = bulk_ks.resolve_bulk_ks(
        capability.name,
        spin="polarized",
        evidence=evidence,
    )

    assert requests[0] == ("capability", capability.name, evidence)
    assert requests[1] == (
        "resolve",
        capability.name,
        (
            "compiled-cpu",
            "production-domain",
            "molecular-scf",
        ),
        evidence,
    )
    assert result.capability is qualified
    assert result.required_ingredients == ("rho", "sigma")
    assert result.method.reference == "unrestricted"
    assert result.plan.exchange == ()
    assert result.plan.nonlocal_correlation is None
    assert result.plan.post_scf == ()
    assert result.plan.required_lowerers == ("semilocal-xc",)
    assert not result.to_payload()["public_dft"]


def test_bulk_ks_descriptive_identifier_does_not_change_semantic_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, qualified = _qualified_resolution("LDA_C_VWN_4")
    monkeypatch.setattr(
        bulk_ks, "resolve_capability", lambda *args, **kwargs: qualified
    )

    first = bulk_ks.resolve_bulk_ks(capability.name, identifier="candidate-a")
    second = bulk_ks.resolve_bulk_ks(capability.name, identifier="candidate-b")

    assert first.method.identifier != second.method.identifier
    assert first.method.identity == second.method.identity
    assert first.plan.identity == second.plan.identity
    assert first.capability.identity == second.capability.identity


def test_bulk_ks_rejects_non_cpu_backend_before_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bulk_ks,
        "functional_capability",
        lambda *args, **kwargs: pytest.fail("capability lookup must stay inactive"),
    )

    with pytest.raises(UnsupportedMethod, match="currently qualified only for CPU"):
        bulk_ks.resolve_bulk_ks("GGA_X_PBE_SOL", backend="cuda")


def test_bulk_ks_rejects_unsupported_ingredient_before_stage_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = functional_capability("GGA_X_PBE_SOL")
    capability = SimpleNamespace(
        name=base.name,
        identity=base.identity,
        required_ingredients=("rho", "sigma", "laplacian"),
    )
    monkeypatch.setattr(
        bulk_ks, "functional_capability", lambda *args, **kwargs: capability
    )
    monkeypatch.setattr(
        bulk_ks,
        "resolve_capability",
        lambda *args, **kwargs: pytest.fail("stage resolution must stay inactive"),
    )

    with pytest.raises(UnsupportedMethod, match="laplacian"):
        bulk_ks.resolve_bulk_ks(base.name)


def test_bulk_ks_detects_capability_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, qualified = _qualified_resolution("GGA_X_PBE_SOL")
    drifted = CapabilityResolution(
        name=qualified.name,
        identity="0" * 64,
        required_stages=qualified.required_stages,
        qualified_stages=qualified.qualified_stages,
        public_dft=False,
    )
    monkeypatch.setattr(bulk_ks, "resolve_capability", lambda *args, **kwargs: drifted)

    with pytest.raises(RuntimeError, match="identity changed"):
        bulk_ks.resolve_bulk_ks(capability.name)
