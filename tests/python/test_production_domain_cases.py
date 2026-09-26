"""Profile-driven physical inputs for bulk Libxc production qualification."""

from __future__ import annotations

import math

import pytest
from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability
from vibeqc_compiler.xc.production_domain_cases import (
    control_case_ids,
    numerical_cases,
)


@pytest.mark.parametrize(
    "name",
    ("LDA_C_VWN_4", "GGA_X_PBE_SOL", "MGGA_X_R2SCAN01"),
)
@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_numeric_and_control_cases_cover_exact_profile(name: str, spin: str) -> None:
    profile = functional_capability(name).production_domain_profile
    numeric = numerical_cases(profile, spin=spin)
    controls = control_case_ids(profile, spin=spin)

    numeric_ids = tuple(case.case_id for case in numeric)
    assert len(set(numeric_ids)) == len(numeric_ids)
    assert not (set(numeric_ids) & set(controls))
    assert set(numeric_ids) | set(controls) == set(profile.case_ids_for_spin(spin))
    assert controls == (
        "control/lazy-inactive-branch",
        "control/invalid-nonfinite",
    )

    for case in numeric:
        names, values = case.runtime_features(profile)
        assert len(names) == len(values)
        assert len(set(names)) == len(names)
        assert all(math.isfinite(value) for value in values)
        assert all(math.isfinite(value) for row in case.pyscf_rho() for value in row)


def test_polarized_sigma_cases_are_physical_gram_coordinates() -> None:
    profile = functional_capability("GGA_X_PBE_SOL").production_domain_profile

    for case in numerical_cases(profile, spin="polarized"):
        names, values = case.runtime_features(profile)
        features = dict(zip(names, values, strict=True))
        aa = features["sigma_aa"]
        ab = features["sigma_ab"]
        bb = features["sigma_bb"]

        assert aa >= 0.0
        assert bb >= 0.0
        assert ab * ab <= aa * bb + 64.0 * math.ulp(max(aa * bb, 1.0))


def test_unpolarized_matrix_does_not_reintroduce_alpha_beta_cases() -> None:
    profile = functional_capability("MGGA_X_R2SCAN01").production_domain_profile
    numeric = numerical_cases(profile, spin="unpolarized")

    assert numeric
    assert all(not case.case_id.startswith("spin/") for case in numeric)
    assert all(len(case.rho) == 1 for case in numeric)
    assert all(len(case.gradient) == 1 for case in numeric)
    assert all(len(case.tau) == 1 for case in numeric)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("LDA_C_VWN_4", ("rho_a", "rho_b")),
        (
            "GGA_X_PBE_SOL",
            ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"),
        ),
        (
            "MGGA_X_R2SCAN01",
            (
                "rho_a",
                "rho_b",
                "sigma_aa",
                "sigma_ab",
                "sigma_bb",
                "tau_a",
                "tau_b",
            ),
        ),
    ],
)
def test_runtime_feature_layout_is_ingredient_driven(
    name: str, expected: tuple[str, ...]
) -> None:
    profile = functional_capability(name).production_domain_profile
    case = numerical_cases(profile, spin="polarized")[0]
    names, _ = case.runtime_features(profile)

    assert names == expected


def test_vacuum_case_is_exact_zero_coordinate() -> None:
    profile = functional_capability("MGGA_X_R2SCAN01").production_domain_profile
    case = next(
        item
        for item in numerical_cases(profile, spin="polarized")
        if item.case_id == "density/vacuum"
    )
    _, values = case.runtime_features(profile)

    assert values == (0.0,) * len(values)
