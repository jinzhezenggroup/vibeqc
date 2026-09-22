"""Compile the real RI reverse preflight without linking a scientific runtime."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def reverse_preflight(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the resource preflight contract")
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/posthf/mp2_gradient.cpp").read_text()
    reverse = source.split(
        "DensityFittedLagrangianWeights density_fitted_lagrangian_weights(", 1
    )[1]
    begin = reverse.index("  auto result_elements =")
    end = reverse.index("  const auto& c = reference.coefficients;", begin)
    # Exercise the production arithmetic and rejection condition, not a second
    # Python planner. Provider metadata and the checked helpers are isolated.
    preflight = reverse[begin:end]
    directory = tmp_path_factory.mktemp("ri_reverse_preflight")
    cpp = directory / "preflight.cpp"
    cpp.write_text(
        r"""
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
namespace posthf {
std::size_t checked_add(std::size_t a, std::size_t b) {
  constexpr auto limit = static_cast<std::size_t>(INT64_MAX);
  if (a > limit || b > limit - a) throw std::overflow_error("overflow");
  return a + b;
}
std::size_t checked_mul(std::size_t a, std::size_t b) {
  constexpr auto limit = static_cast<std::size_t>(INT64_MAX);
  if (a && b > limit / a) throw std::overflow_error("overflow");
  return a * b;
}
}
int main(int argc, char** argv) {
  if (argc != 4) return 4;
  try {
    const auto n = std::strtoull(argv[1], nullptr, 10);
    const auto na = std::strtoull(argv[2], nullptr, 10);
    const auto maximum_bytes = std::strtoull(argv[3], nullptr, 10);
    const auto n2 = posthf::checked_mul(n, n);
    const auto n4 = posthf::checked_mul(n2, n2);
    const auto a2 = posthf::checked_mul(na, na);
    const auto three = posthf::checked_mul(n2, na);
    struct Provider { std::size_t provider_bytes() const { return 4096; } } provider;
"""
        + preflight
        + r"""
    std::cout << required << '\n';
    return 0;
  } catch (const std::length_error&) { return 2; }
    catch (const std::overflow_error&) { return 3; }
}
"""
    )
    binary = directory / "preflight"
    subprocess.run(
        [compiler, "-std=c++17", str(cpp), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize(("n", "na"), [(2, 2), (2, 6), (8, 48), (64, 256)])
def test_reverse_peak_covers_live_metric_pullback(
    reverse_preflight: Path, n: int, na: int
) -> None:
    def run(budget: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(reverse_preflight), str(n), str(na), str(budget)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    admitted = run((1 << 63) - 1)
    assert admitted.returncode == 0, admitted.stderr
    planned = int(admitted.stdout)
    # Independent live-buffer inventory at symmetric_matrix_function_vjp:
    # input H/S/ERI weights; result H/S/A; bar_B/bar_A; bar_X; eigenvectors;
    # four VJP matrices (including returned M); eigenvalues and uint8 rank mask.
    live_bytes = 4096 + 8 * (4 * n**2 + n**4 + 3 * n**2 * na + 6 * na**2) + 9 * na
    assert planned >= live_bytes
    assert run(planned).returncode == 0
    assert run(planned - 1).returncode == 2


def test_reverse_peak_rejects_overflow(reverse_preflight: Path) -> None:
    result = subprocess.run(
        [str(reverse_preflight), str(1 << 32), "2", str((1 << 63) - 1)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 3
