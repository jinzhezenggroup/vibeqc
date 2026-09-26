"""Executable binding tests for generic bulk Libxc semilocal point programs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.xc import bulk_point_program
from vibeqc_compiler.xc.bulk_runtime import (
    PRODUCTION_CANDIDATE_DOMAIN,
    build_bulk_runtime_program,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("name", "features", "ingredient_mask"),
    [
        ("LDA_C_VWN_4", ("rho_a", "rho_b"), 1),
        (
            "GGA_X_PBE_SOL",
            ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"),
            7,
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
            15,
        ),
    ],
)
def test_runtime_aot_binding_uses_compact_native_feature_abi(
    name: str, features: tuple[str, ...], ingredient_mask: int
) -> None:
    program = build_bulk_runtime_program(name, spin="polarized", order=1)
    variant = bulk_point_program.inspect_runtime_program(program)
    binding = bulk_point_program.bind_runtime_semilocal_point_program(program)

    assert variant.features == features
    assert binding.ingredient_mask == ingredient_mask
    assert binding.capability_identity == program.spec.capability_identity
    assert binding.point_expression_identity == program.expression_hash
    assert binding.variant.emission_identity == variant.emission_identity
    assert "lapl_a" not in variant.features and "lapl_b" not in variant.features
    payload = binding.to_payload()
    assert payload["artifact_emission_identity"] == variant.emission_identity
    assert payload["point_expression_identity"] == program.expression_hash
    assert payload["capability_identity"] == program.spec.capability_identity
    assert "production" not in payload and "public" not in payload


def test_native_domain_version_is_derived_from_runtime_domain() -> None:
    interior = build_bulk_runtime_program("GGA_X_PBE_SOL", spin="polarized", order=1)
    candidate = build_bulk_runtime_program(
        "GGA_X_PBE_SOL",
        spin="polarized",
        order=1,
        domain=PRODUCTION_CANDIDATE_DOMAIN,
    )

    interior_binding = bulk_point_program.bind_runtime_semilocal_point_program(interior)
    candidate_binding = bulk_point_program.bind_runtime_semilocal_point_program(
        candidate
    )

    assert interior_binding.domain_version == 1
    assert candidate_binding.domain_version == 2
    assert (
        interior_binding.to_payload()["domain"]
        != candidate_binding.to_payload()["domain"]
    )
    assert interior_binding.identity != candidate_binding.identity

    with pytest.raises(ValueError, match="unsupported native XC domain"):
        bulk_point_program.native_domain_version("unknown-domain")


def test_adapter_source_binds_all_identities_and_tau_convention() -> None:
    program = build_bulk_runtime_program("MGGA_X_R2SCAN01", spin="polarized", order=1)
    binding = bulk_point_program.bind_runtime_semilocal_point_program(program)
    source = binding.emit_source()

    assert binding.identity in source
    assert program.spec.capability_identity in source
    assert program.expression_hash in source
    assert binding.variant.emission_identity in source
    assert "double features[7]" in source
    assert "tau[0], tau[1]" in source
    assert "out.kinetic[0] = 0.5 * outputs[6];" in source
    assert "out.kinetic[1] = 0.5 * outputs[7];" in source
    assert "SemilocalPointProgram kPointProgram" in source
    assert ", 15U, 1U, evaluate_point};" in source


def test_binding_rejects_nonpolarized_or_partial_point_contract() -> None:
    unpolarized = build_bulk_runtime_program(
        "GGA_X_PBE_SOL", spin="unpolarized", order=1
    )
    with pytest.raises(ValueError, match="polarized"):
        bulk_point_program.bind_runtime_semilocal_point_program(unpolarized)

    partial = build_bulk_runtime_program(
        "GGA_X_PBE_SOL", spin="polarized", order=1, outputs=((), (0,))
    )
    with pytest.raises(ValueError, match="complete E/vxc"):
        bulk_point_program.bind_runtime_semilocal_point_program(partial)


def test_noncurated_gga_adapter_executes_projected_graph_exactly(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")

    program = build_bulk_runtime_program("GGA_X_PBE_SOL", spin="polarized", order=1)
    binding = bulk_point_program.bind_runtime_semilocal_point_program(program)

    rho = np.array([0.7, 0.4], dtype=np.float64)
    gradient = np.array([[0.1, 0.2, 0.05], [0.05, -0.1, 0.15]], dtype=np.float64)
    sigma = np.array(
        [
            np.dot(gradient[0], gradient[0]),
            np.dot(gradient[0], gradient[1]),
            np.dot(gradient[1], gradient[1]),
        ]
    )
    packed = np.concatenate((rho, sigma)).reshape(-1, 1)
    raw = program.evaluate(packed)[:, 0]
    expected = [
        raw[0],
        raw[1],
        raw[2],
        *(2.0 * raw[3] * gradient[0] + raw[4] * gradient[1]),
        *(raw[4] * gradient[0] + 2.0 * raw[5] * gradient[1]),
        0.0,
        0.0,
    ]

    source = (
        binding.emit_source()
        + r"""
#include <iomanip>
#include <iostream>

int main() {
  const double rho[2]{0.7, 0.4};
  const double gradient[2][3]{{0.1, 0.2, 0.05}, {0.05, -0.1, 0.15}};
  const double tau[2]{0.0, 0.0};
  const auto value = vibeqc::dft::bulk_generated::evaluate_point(rho, gradient, tau);
  std::cout << std::setprecision(17)
            << value.energy << ' ' << value.rho[0] << ' ' << value.rho[1];
  for (const auto& spin : value.gradient)
    for (double item : spin) std::cout << ' ' << item;
  std::cout << ' ' << value.kinetic[0] << ' ' << value.kinetic[1] << '\n';
}
"""
    )
    path = tmp_path / "bulk_point.cpp"
    executable = tmp_path / "bulk_point"
    path.write_text(source, encoding="utf-8")
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            str(path),
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run(
        [str(executable)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = [float(value) for value in executed.stdout.split()]
    assert observed == pytest.approx(expected, rel=2e-13, abs=2e-13)
