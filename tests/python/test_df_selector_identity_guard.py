"""Regression guard against benchmark identities in production DF auto selectors."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

DF_AUTO_SELECTOR_SOURCES = (
    "src/scf/df_exchange_policy.hpp",
    "src/scf/cuda/df_scf_factor.cpp",
    "src/scf/cuda/df_force_response.cpp",
    "src/scf/cuda/df_shell_derivatives.cu",
    "src/scf/cuda/df_gradient_bridge.cu",
)

# Historical benchmark identities belong in evidence/tests, not production
# dispatch. Keep this list deliberately literal so general workload features
# (including shell statistics and architecture capabilities) remain legal.
FORBIDDEN_WORKLOAD_LITERALS = (
    "rtx 5090",
    "def2-svp",
    "def2_svp",
    "water-octamer",
    "water_octamer",
    "water-hexadecamer",
    "water_hexadecamer",
)

# Exact endpoint equality is the brittle policy shape retired by #444. General
# thresholds/profile ranges remain legal and should not match this expression.
_ENDPOINT_VARIABLE = (
    r"(?:n(?:bf|ao|aux|occ)|ao_count|aux(?:iliary)?_count|"
    r"occupied(?:_rank|_count)?|rank)"
)
# C++ unsigned/long suffixes do not change the benchmark identity. Keep the
# final word boundary so larger values and identifier continuations stay legal.
_ENDPOINT_VALUE = r"(?:80|160|384|768)(?:u(?:ll?)?|ll?u?)?"
EXACT_ENDPOINT_IDENTITY = re.compile(
    rf"\b(?:{_ENDPOINT_VARIABLE}\s*==\s*{_ENDPOINT_VALUE}|"
    rf"{_ENDPOINT_VALUE}\s*==\s*{_ENDPOINT_VARIABLE})\b",
    re.IGNORECASE,
)


def _benchmark_identity_hits(source: str) -> tuple[str, ...]:
    lowered = source.lower()
    hits = [
        f"literal:{literal}"
        for literal in FORBIDDEN_WORKLOAD_LITERALS
        if literal in lowered
    ]
    hits.extend(
        f"endpoint:{match.group(0)}"
        for match in EXACT_ENDPOINT_IDENTITY.finditer(source)
    )
    return tuple(hits)


def test_benchmark_identity_guard_recognizes_retired_policy_shapes() -> None:
    source = """
    if (nbf == 384 || nocc == 80 || 160 == rank) use_fast_path();
    if (device_name == \"NVIDIA GeForce RTX 5090\") use_fast_path();
    // water-octamer / def2-SVP was a measurement fixture, not a policy identity.
    """

    hits = _benchmark_identity_hits(source)

    assert "endpoint:nbf == 384" in hits
    assert "endpoint:nocc == 80" in hits
    assert "endpoint:160 == rank" in hits
    assert "literal:rtx 5090" in hits
    assert "literal:def2-svp" in hits
    assert "literal:water-octamer" in hits


def test_df_auto_selectors_do_not_encode_benchmark_identity() -> None:
    violations: dict[str, tuple[str, ...]] = {}
    for relative in DF_AUTO_SELECTOR_SOURCES:
        path = ROOT / relative
        assert path.is_file(), (
            f"tracked DF selector moved without updating guard: {relative}"
        )
        hits = _benchmark_identity_hits(path.read_text(encoding="utf-8"))
        if hits:
            violations[relative] = hits

    assert not violations, (
        "production DF auto-selection must use workload/profile capabilities, "
        f"not benchmark endpoint identities: {violations}"
    )


@pytest.mark.parametrize(
    "suffix",
    (
        "",
        "u",
        "U",
        "l",
        "L",
        "ul",
        "UL",
        "lu",
        "LU",
        "ll",
        "LL",
        "ull",
        "ULL",
        "llu",
        "LLU",
    ),
)
@pytest.mark.parametrize(
    "variable,value", (("nbf", 384), ("nao", 768), ("nocc", 80), ("rank", 160))
)
@pytest.mark.parametrize("reversed_order", (False, True))
def test_identity_guard_recognizes_cpp_integer_suffixes(
    suffix: str, variable: str, value: int, reversed_order: bool
) -> None:
    literal = f"{value}{suffix}"
    expression = (
        f"{literal} == {variable}" if reversed_order else f"{variable} == {literal}"
    )
    assert _benchmark_identity_hits(expression) == (f"endpoint:{expression}",)


@pytest.mark.parametrize(
    "source",
    (
        "nbf >= 384U",
        "768ULL > nao",
        "nocc < 80",
        "rank <= 160L",
        "nbf == 3840U",
        "7681ULL == nao",
        "nocc == 801",
        "rank == 1600",
        "nbf == 384_units",
        "rank == 160bytes",
    ),
)
def test_identity_guard_preserves_general_thresholds_and_other_values(
    source: str,
) -> None:
    assert _benchmark_identity_hits(source) == ()
