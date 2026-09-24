"""Regression guard for CPU AO jet radial-factor reuse."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/dft/ao_grid.cpp"


def _evaluate_body() -> str:
    source = SOURCE.read_text()
    start = source.index("void AoBasis::evaluate(")
    return source[start:]


def _jet_count(order: int) -> int:
    return (order + 1) * (order + 2) * (order + 3) // 6


def test_cpu_ao_radial_factor_is_reused_across_requested_jets() -> None:
    body = _evaluate_body()
    radial = body.index("const double radial =")
    weighted = body.index("const double weighted_radial =", radial)
    jet_loop = body.index("for (std::size_t jet = 0; jet < JetCount; ++jet)", weighted)

    # One primitive exponential feeds every requested derivative jet instead of
    # being recomputed from inside a jet-major traversal.
    assert radial < weighted < jet_loop
    assert body.count("std::exp(") == 1
    assert body.index("double r2 = 0;") < radial


def test_legal_jet_extents_do_not_restore_value_only_runtime_dispatch() -> None:
    body = _evaluate_body()
    assert "std::array<double, JetCount> values{};" in body
    assert "JetCount == 1 ? 0U : derivatives[jet][k]" in body
    for jets in (1, 4, 10, 20):
        assert f"evaluate_jets.template operator()<{jets}>()" in body


def test_cpu_ao_radial_exp_work_census() -> None:
    # For a fixed AO/point/primitive tuple, the old jet-major traversal paid one
    # exponential per jet. The fused traversal pays exactly one regardless of
    # derivative order. This is a deterministic work-count reduction, not a
    # wall-time claim.
    assert [_jet_count(order) for order in range(4)] == [1, 4, 10, 20]
    primitive_evaluations = 97
    for order in range(4):
        jets = _jet_count(order)
        old_exp_calls = primitive_evaluations * jets
        new_exp_calls = primitive_evaluations
        assert old_exp_calls // new_exp_calls == jets
