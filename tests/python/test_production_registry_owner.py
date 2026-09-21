"""Ownership regressions for production registry serialization."""

from __future__ import annotations

from vibeqc_compiler.integral import (
    production,
    production_profile,
    production_registry,
    production_selection,
)


def test_registry_serializers_have_one_canonical_owner() -> None:
    """Keep legacy production imports as identity-preserving re-exports."""

    for name in (
        "emit_registry_header",
        "emit_registry_source",
        "emit_multi_registry_header",
        "emit_multi_registry_source",
    ):
        canonical = getattr(production_registry, name)
        assert canonical.__module__ == production_registry.__name__
        assert getattr(production, name) is canonical


def test_shared_registry_naming_and_ordering_live_below_orchestration() -> None:
    """Keep registry/emission helpers out of the orchestration god module."""

    assert production._profile_identifier is production_profile._profile_identifier
    assert production._as_selection is production_selection._as_selection
    assert (
        production._stable_selection_order
        is production_selection._stable_selection_order
    )
