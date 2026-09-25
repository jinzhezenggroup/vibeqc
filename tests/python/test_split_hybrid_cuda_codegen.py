"""CUDA source generation for split Libxc global hybrids."""

import re

import pytest

from tools.generate_xc_split_hybrid_cuda import emit_split_hybrid_device


@pytest.mark.parametrize(
    ("name", "symbol", "exact_hex"),
    (
        ("M06-2X", "M062X", float(27 / 50).hex()),
        ("MN15", "MN15", float(11 / 25).hex()),
    ),
)
def test_split_hybrid_cuda_codegen_uses_rho_sigma_tau_without_laplacian(
    name: str, symbol: str, exact_hex: str
) -> None:
    source = emit_split_hybrid_device(name)
    assert f"struct {symbol}DeviceValue" in source
    assert f"k{symbol}ExactExchange = {exact_hex}" in source
    assert "__device__ inline" in source
    assert "lapl_a" not in source
    assert "lapl_b" not in source
    assert "double tau_a" in source and "double tau_b" in source
    assert "feature_derivative[7]" in source
    assert "exchange.energy_density + correlation.energy_density" in source
    assert not re.search(r"\b(?:torch|pyscf|libxc_eval)\b", source, re.IGNORECASE)


def test_m062x_cuda_codegen_reproduces_component_work_mgga_thresholds() -> None:
    source = emit_split_hybrid_device("M06-2X")
    assert f"if (total_density < {float(1.0e-15).hex()}) return out;" in source
    assert f"if (total_density < {float(1.0e-12).hex()}) return out;" in source
    assert source.count(f"fmax({float(1.0e-20).hex()}, tau_a)") == 2
    assert source.count("raw.energy_density * total_density / work_density") == 2
    assert source.count(
        "fmax(-sigma_average, fmin(sigma_average, sigma_ab))"
    ) == 2


def test_split_hybrid_cuda_codegen_is_deterministic() -> None:
    assert emit_split_hybrid_device("M06-2X") == emit_split_hybrid_device("M06-2X")
