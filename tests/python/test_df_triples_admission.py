"""DF triples must admit bounded scratch before work and publish finite energies."""

import weakref

import numpy as np
import pytest

from tools.vibeqc_cc import df_triples as module


def inputs(o: int = 2, v: int = 3, q: int = 2) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(896)
    raw = rng.normal(size=(q, v, v))
    return (
        rng.normal(size=(q, o, v)),
        0.5 * (raw + raw.transpose(0, 2, 1)),
        rng.normal(size=(o, v, o, o)),
        rng.normal(size=(o, v, o, v)),
        rng.normal(size=(o, v)),
        rng.normal(size=(o, v)),
        rng.normal(size=(o, o, v, v)),
        -np.arange(1, o + 1, dtype=float),
        np.arange(1, v + 1, dtype=float),
    )


def test_tiny_budget_rejects_before_numeric_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = inputs()

    def unexpected(value: object) -> object:
        raise AssertionError("numeric scan occurred before scratch admission")

    monkeypatch.setattr(module.np, "isfinite", unexpected)
    with pytest.raises(MemoryError, match="temporary numeric bytes"):
        module.factorized_triples_energy(*values, max_bytes=1)


def test_validation_is_chunked_with_occupied_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = inputs(o=1, v=17)
    original = np.isfinite

    class Validated(Exception):
        pass

    def bounded(value: object) -> object:
        limit = module.factorized_triples_workspace_bytes(1) // 24
        assert np.asarray(value).size <= limit, "virtual-sized validation scratch"
        return original(value)

    def stop(*args: object) -> None:
        raise Validated

    monkeypatch.setattr(module.np, "isfinite", bounded)
    monkeypatch.setattr(module, "_check_denominators", stop)
    with pytest.raises(Validated):
        module.factorized_triples_energy(
            *values, max_bytes=module.factorized_triples_workspace_bytes(1)
        )


@pytest.mark.parametrize("which", ("factors", "orbital_energies"))
def test_finite_inputs_cannot_publish_nonfinite_arithmetic(which: str) -> None:
    values = list(inputs())
    if which == "factors":
        values[0] *= 1e200
        values[1] *= 1e200
    else:
        values[-2][:] = -1e308
    with np.errstate(all="ignore"), pytest.raises(FloatingPointError):
        module.factorized_triples_energy(*values)


def test_previous_virtual_triple_does_not_remain_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_w, original_einsum = module._w_factorized, np.einsum
    previous_z: list[weakref.ReferenceType] = []
    calls = 0

    def checked_w(*args: object) -> np.ndarray:
        nonlocal calls
        if calls and calls % 6 == 0:
            assert all(ref() is None for ref in previous_z), (
                "previous triple is still resident"
            )
            previous_z.clear()
        calls += 1
        return original_w(*args)

    def collect(equation: str, *args: object, **kwargs: object) -> object:
        if equation == "ijk,ijk":
            previous_z.append(weakref.ref(args[1]))
        return original_einsum(equation, *args, **kwargs)

    monkeypatch.setattr(module, "_w_factorized", checked_w)
    monkeypatch.setattr(module.np, "einsum", collect)
    assert np.isfinite(module.factorized_triples_energy(*inputs()))


@pytest.mark.parametrize("empty", ("occupied", "virtual"))
def test_empty_orbital_domains_reject_explicitly(empty: str) -> None:
    values = inputs(o=0 if empty == "occupied" else 1, v=0 if empty == "virtual" else 1)
    with pytest.raises(ValueError, match="nonempty"):
        module.factorized_triples_energy(*values)


def test_composed_total_cannot_overflow_to_a_converged_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from test_df_ccsd_factorized import FixtureDFProvider, _method_problem

    snapshot, contract, factors, arrays = _method_problem("h2")
    provider = FixtureDFProvider(snapshot, factors, arrays["g"])
    original = module._run
    large = float(np.finfo(np.float64).max)

    def huge_cc(prepared: object) -> object:
        return replace(original(prepared), correlation_energy=large)

    monkeypatch.setattr(module, "_run", huge_cc)
    monkeypatch.setattr(module, "factorized_triples_energy", lambda *a, **k: large)
    with pytest.raises(FloatingPointError, match="total energy"):
        module.solve_df_ccsdt(snapshot, provider, contract)
    assert provider.statistics["external_reserved_bytes"] == 0
