"""Regression guard for the CPU exact restricted J-only BLAS route."""

from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2] / "src" / "scf" / "fock_build.cpp"
).read_text(encoding="utf-8")


def test_restricted_j_only_uses_shared_cpu_gemv() -> None:
    assert '#include "tensor/cpu_linalg.hpp"' in SOURCE
    assert (
        "strategy.spec.coulomb.present && !strategy.spec.exchange.present && "
        "!unrestricted"
    ) in SOURCE
    assert "tensor::cpu_gemv('N', count, count, eri.data(), density.data()" in SOURCE


def test_generic_jk_loop_remains_for_exchange_and_uhf() -> None:
    gemv = SOURCE.index("tensor::cpu_gemv")
    generic = SOURCE.index("for (std::size_t i = 0; i < nbf; ++i)", gemv)
    assert gemv < generic
    assert "if (strategy.spec.exchange.present)" in SOURCE[:generic]
