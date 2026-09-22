"""Upstream-owned DFT alias generation and resolution gates."""

from __future__ import annotations

import pytest
from vibeqc_compiler.method import (
    METHOD_ALIASES,
    METHOD_CATALOG,
    UnsupportedMethod,
    resolve_method,
)
from vibeqc_compiler.method._generated_xc_aliases import (
    UPSTREAM_XC_ALIAS_PROVENANCE,
)


def _signature(
    libxc: object, name: str
) -> tuple[tuple[float, ...], tuple[tuple[int, float], ...]]:
    hybrid, functionals = libxc.parse_xc(name)
    return (
        tuple(float(value) for value in hybrid),
        tuple(
            (int(identifier), float(coefficient))
            for identifier, coefficient in functionals
        ),
    )


def test_aliases_are_not_duplicated_as_canonical_method_specs() -> None:
    assert set(METHOD_ALIASES).isdisjoint(METHOD_CATALOG)
    for alias, canonical in METHOD_ALIASES.items():
        assert canonical in METHOD_CATALOG
        reference = resolve_method(canonical)
        resolved = resolve_method(alias)
        assert resolved.identifier == alias
        assert resolved.identity == reference.identity
        assert resolved.manifest_identity != reference.manifest_identity


@pytest.mark.parametrize("name", ["PW91PW91", "BHHLYP"])
def test_names_without_pinned_upstream_alias_evidence_fail_closed(name: str) -> None:
    with pytest.raises(UnsupportedMethod, match="unknown DFT method"):
        resolve_method(name)


def test_alias_provenance_matches_repository_reference_versions() -> None:
    assert UPSTREAM_XC_ALIAS_PROVENANCE["pyscf_version"] == "2.14.0"
    assert UPSTREAM_XC_ALIAS_PROVENANCE["libxc_version"] == "7.0.0"
    assert UPSTREAM_XC_ALIAS_PROVENANCE["module"] == "pyscf.dft.libxc"
    assert len(UPSTREAM_XC_ALIAS_PROVENANCE["module_sha256"]) == 64


def test_alias_semantics_match_pinned_pyscf_libxc_when_available() -> None:
    pyscf = pytest.importorskip("pyscf")
    from pyscf.dft import libxc

    assert pyscf.__version__ == UPSTREAM_XC_ALIAS_PROVENANCE["pyscf_version"]
    assert libxc.__version__ == UPSTREAM_XC_ALIAS_PROVENANCE["libxc_version"]
    for alias, canonical in METHOD_ALIASES.items():
        assert _signature(libxc, alias) == _signature(libxc, canonical)
