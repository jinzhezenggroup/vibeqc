"""Production D3(BJ) runtime qualification against independent goldens."""

import json
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc import D3CorrectionBatch, evaluate_d3_correction

_DATA = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "d3_bj_reference.json").read_text()
)
_DFT_FIXTURES = [
    fixture
    for fixture in _DATA["fixtures"]
    if fixture["name"].startswith(("pbe-bj-two-body/", "pbe0-bj-two-body/"))
]


def _method(fixture: typing.Any) -> typing.Any:
    return (
        "PBE0-D3(BJ)"
        if fixture["name"].startswith("pbe0-bj-two-body/")
        else "PBE-D3(BJ)"
    )


def _system(fixture: typing.Any) -> typing.Any:
    return fixture["numbers"], fixture["positions"]


@pytest.mark.parametrize(
    "fixture", _DFT_FIXTURES, ids=[fixture["name"] for fixture in _DFT_FIXTURES]
)
def test_production_cpu_matches_independent_simple_dftd3(
    fixture: typing.Any,
) -> None:
    result = evaluate_d3_correction(
        _method(fixture), fixture["numbers"], fixture["positions"], device="cpu"
    )
    assert result.ok
    assert result.backend == "cpu"
    assert result.energy == pytest.approx(fixture["energy"], abs=2.0e-15)
    np.testing.assert_allclose(
        result.gradient, np.asarray(fixture["gradient"]), atol=2.0e-14, rtol=0.0
    )


def test_production_ragged_replay_diagnostics_budget_and_lifetime() -> None:
    first, second = _DFT_FIXTURES[0], _DFT_FIXTURES[1]
    systems = [_system(first), _system(second)]
    batch = D3CorrectionBatch("PBE-D3(BJ)", systems, device="cpu")
    diagnostic = batch.diagnostic()
    assert diagnostic.backend == "cpu"
    assert diagnostic.system_count == 2
    assert diagnostic.total_atoms == len(first["numbers"]) + len(second["numbers"])
    assert diagnostic.maximum_atoms == max(
        len(first["numbers"]), len(second["numbers"])
    )
    assert diagnostic.workspace_bytes > 0
    assert diagnostic.plan_host_bytes > 0
    initial = batch.execute()
    for result, fixture in zip(initial, (first, second), strict=True):
        assert result.ok
        assert result.energy == pytest.approx(fixture["energy"], abs=2.0e-15)
        np.testing.assert_allclose(
            result.gradient,
            np.asarray(fixture["gradient"]),
            atol=2.0e-14,
            rtol=0.0,
        )

    moved = np.asarray(first["positions"], dtype=np.float64).copy()
    moved[1, 0] += 0.013
    replay = batch.execute([moved, None])
    reference = evaluate_d3_correction(
        "PBE-D3(BJ)", first["numbers"], moved, device="cpu"
    )
    assert replay[0].energy == pytest.approx(reference.energy, abs=2.0e-15)
    np.testing.assert_allclose(replay[0].gradient, reference.gradient, atol=2.0e-14)
    assert replay[1].energy == pytest.approx(second["energy"], abs=2.0e-15)

    energy_only = batch.execute(gradients=False)
    assert all(result.gradient is None for result in energy_only)
    batch.close()
    batch.close()
    with pytest.raises(RuntimeError, match="closed"):
        batch.execute()
    with pytest.raises(RuntimeError, match="closed"):
        batch.diagnostic()


def test_production_budget_is_explicitly_bounded() -> None:
    first = _DFT_FIXTURES[0]
    with pytest.raises(RuntimeError, match="maximum_bytes"):
        D3CorrectionBatch("PBE-D3(BJ)", [_system(first)], maximum_bytes=1)


def test_production_cuda_ragged_replay_matches_independent_goldens() -> None:
    first, second = _DFT_FIXTURES[0], _DFT_FIXTURES[1]
    try:
        batch = D3CorrectionBatch(
            "PBE-D3(BJ)", [_system(first), _system(second)], device="cuda"
        )
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(f"CUDA D3 runtime unavailable: {error}")

    with batch:
        assert batch.diagnostic().backend == "cuda"
        initial = batch.execute()
        for result, fixture in zip(initial, (first, second), strict=True):
            assert result.ok
            assert result.backend == "cuda"
            assert result.energy == pytest.approx(fixture["energy"], abs=2.0e-15)
            np.testing.assert_allclose(
                result.gradient,
                np.asarray(fixture["gradient"]),
                atol=2.0e-14,
                rtol=0.0,
            )

        moved = np.asarray(first["positions"], dtype=np.float64).copy()
        moved[2, 1] -= 0.017
        replay = batch.execute([moved, None])
        cpu = evaluate_d3_correction("PBE-D3(BJ)", first["numbers"], moved)
        assert replay[0].energy == pytest.approx(cpu.energy, abs=2.0e-15)
        np.testing.assert_allclose(replay[0].gradient, cpu.gradient, atol=2.0e-14)
        assert replay[1].energy == pytest.approx(second["energy"], abs=2.0e-15)


@pytest.mark.parametrize("number", [2**32 + 6, -(2**32) + 6, 2**63 + 6])
def test_atomic_number_cannot_wrap_to_a_supported_element(
    number: typing.Any,
) -> None:
    from vibeqc.dispersion import _normalize_system

    dtype = np.uint64 if number >= 2**63 else np.int64
    with pytest.raises(NotImplementedError, match="H through Rn"):
        _normalize_system((np.array([number], dtype=dtype), np.zeros((1, 3))))


@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_prepare_rejects_complex_coordinates_before_narrowing(
    imaginary: typing.Any,
) -> None:
    from vibeqc.dispersion import _normalize_system

    xyz = np.ones((1, 3), dtype=complex) * (1 + imaginary * 1j)
    with pytest.raises(TypeError, match="real"):
        _normalize_system(([1], xyz))


@pytest.mark.parametrize("budget", [True, 2**64])
def test_budget_cannot_wrap_at_the_native_abi(
    budget: typing.Any, monkeypatch: typing.Any
) -> None:
    from vibeqc import dispersion

    monkeypatch.setattr(
        dispersion,
        "_method_ir",
        lambda *_: pytest.fail("invalid budget reached method resolution"),
    )
    with pytest.raises(ValueError, match="maximum_bytes"):
        D3CorrectionBatch("PBE-D3(BJ)", [([1], np.zeros((1, 3)))], maximum_bytes=budget)


@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_replay_rejects_complex_coordinates_without_changing_state(
    imaginary: typing.Any,
) -> None:
    fixture = _DFT_FIXTURES[0]
    with D3CorrectionBatch("PBE-D3(BJ)", [_system(fixture)]) as batch:
        before = batch.execute()[0]
        coordinates = np.asarray(fixture["positions"], dtype=complex) + imaginary * 1j
        with pytest.raises(TypeError, match="real"):
            batch.execute([coordinates])
        after = batch.execute()[0]
        assert after.energy == before.energy
        np.testing.assert_array_equal(after.gradient, before.gradient)
