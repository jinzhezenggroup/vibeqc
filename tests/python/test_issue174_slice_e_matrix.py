"""Hardware-free checks of the precision benchmark's sampling and evidence protocol."""

import typing
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import issue174_slice_e_matrix as matrix

BASE = (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))
MOVED = matrix._displaced_atoms(BASE, 0.05)
CASE = SimpleNamespace(charge=0, multiplicity=1)


def arguments() -> typing.Any:
    return SimpleNamespace(
        modes=("fp64", "auto"),
        tolerances=(1e-6,),
        repeats=3,
        batch_sizes=(1, 4),
        properties=("energy", "forces"),
        force_target=1e-6,
        reference_energy_tolerance=1e-13,
        reference_density_tolerance=1e-11,
        reference_screening_tolerance=1e-14,
        reference_max_iterations=200,
    )


def result(**updates: typing.Any) -> typing.Any:
    values = {
        "energy": -1.0,
        "forces": np.zeros((2, 3)),
        "converged": True,
        "iterations": 2,
        "energy_change": 1e-12,
        "density_rms": 1e-10,
        "precision": {"effective_bits": 64},
        "executed_backend": "cuda",
    }
    return SimpleNamespace(**(values | updates))


def references() -> typing.Any:
    return {
        label: {"result": result(), "model": object()} for label in ("base", "moved")
    }


@pytest.fixture
def measurement(monkeypatch: typing.Any) -> typing.Any:
    """Keep tests on the measurement protocol without initializing CUDA."""
    monkeypatch.setattr(matrix, "_synchronize", lambda backend: None)
    monkeypatch.setattr(matrix, "_evidence", lambda *args: {"status": "observed_met"})
    return arguments()


def test_singlepoint_samples_are_cold_and_balanced(
    measurement: typing.Any, monkeypatch: typing.Any
) -> None:
    refs = references()
    calls = []
    monkeypatch.setattr(
        matrix,
        "_strict_reference",
        lambda case, args, atoms, **kw: refs["base" if atoms == BASE else "moved"],
    )

    def calculator(
        case: typing.Any, args: typing.Any, *, precision: typing.Any, **kw: typing.Any
    ) -> typing.Any:
        def singlepoint(atoms: typing.Any, **kw: typing.Any) -> typing.Any:
            calls.append((precision, atoms))
            return result()

        return SimpleNamespace(singlepoint=singlepoint)

    monkeypatch.setattr(matrix, "_calculator", calculator)
    rows, _ = matrix._tolerance_matrix("h2", CASE, BASE, MOVED, measurement, None)
    assert Counter((row["mode"], row["geometry"], row["kind"]) for row in rows) == {
        (mode, geometry, "cold"): measurement.repeats
        for mode in measurement.modes
        for geometry in ("base", "moved")
    }
    for index in range(0, len(calls), 2):
        assert {mode for mode, atoms in calls[index : index + 2]} == {"fp64", "auto"}
        assert calls[index][1] == calls[index + 1][1]
    assert {row["samples"] for row in matrix._summarize(rows)} == {measurement.repeats}


def install_batches(
    monkeypatch: typing.Any, *, fail_warm: typing.Any = False
) -> typing.Any:
    """Expose persistent state transitions and closure of each prepared plan."""
    plans, calls = [], []

    class Batch:
        def __init__(self, systems: typing.Any, mode: typing.Any) -> None:
            self.systems, self.mode, self.step, self.closed = systems, mode, 0, False
            plans.append(self)

        def __enter__(self) -> typing.Any:
            return self

        def __exit__(self, *exc: object) -> None:
            self.closed = True

        def execute(self, coordinates: typing.Any, *, strict: typing.Any) -> typing.Any:
            assert strict is False
            calls.append((len(self.systems), self.step, self.mode))
            if self.step == 2:
                expected = [
                    [position for _, position in (MOVED if atoms == BASE else BASE)]
                    for atoms in self.systems
                ]
                assert coordinates == expected
            else:
                assert coordinates is None
            if fail_warm and self.step == 1:
                raise RuntimeError("warm failure")
            items = [
                result(
                    index=index,
                    status_message="success",
                    succeeded=True,
                    warm_start_used=self.step > 0,
                    warm_start_fallback=False,
                    restart_origin="warm" if self.step else "cold",
                    fock_builds=None,
                )
                for index in range(len(self.systems))
            ]
            self.step += 1
            return SimpleNamespace(items=items)

    def calculator(
        case: typing.Any, args: typing.Any, *, precision: typing.Any, **kw: typing.Any
    ) -> typing.Any:
        def prepare(
            systems: typing.Any, *, warm_start: typing.Any, **kw: typing.Any
        ) -> typing.Any:
            assert warm_start is True
            return Batch(systems, precision)

        return SimpleNamespace(prepare_batch=prepare)

    monkeypatch.setattr(matrix, "_calculator", calculator)
    return plans, calls


def test_batch_states_repeat_and_interleave_each_policy(
    measurement: typing.Any, monkeypatch: typing.Any
) -> None:
    plans, calls = install_batches(monkeypatch)
    rows = matrix._batch_matrix(
        "h2", CASE, BASE, MOVED, references(), measurement, None
    )
    assert Counter((row["batch_size"], row["mode"]) for row in rows) == {
        (size, mode): measurement.repeats
        for size in (1, 4)
        for mode in measurement.modes
    }
    assert len(plans) == len(rows)
    assert all(plan.closed and plan.step == 3 for plan in plans)
    for row in rows:
        for state in ("cold", "warm", "changed"):
            assert len(row[f"{state}_items"]) == row["batch_size"]
            assert all(
                item["warm_start_used"] == (state != "cold")
                for item in row[f"{state}_items"]
            )
    for index in range(0, len(calls), 2):
        pair = calls[index : index + 2]
        assert pair[0][:2] == pair[1][:2]
        assert {call[2] for call in pair} == {"fp64", "auto"}
    for size in (1, 4):
        # Consecutive repeat pairs alternate which policy is timed first.
        first_modes = [
            mode for batch_size, step, mode in calls if batch_size == size and step == 0
        ][::2]
        assert first_modes == ["fp64", "auto", "fp64"]
    assert {row["batch_size"] for row in matrix._summarize_batch(rows)} == {1, 4}


def test_later_batch_failure_preserves_cold_measurement(
    measurement: typing.Any, monkeypatch: typing.Any
) -> None:
    plans, _ = install_batches(monkeypatch, fail_warm=True)
    rows = matrix._batch_matrix(
        "h2", CASE, BASE, MOVED, references(), measurement, None
    )
    assert all(plan.closed for plan in plans)
    assert all("cold_items" in row and "warm failure" in row["failure"] for row in rows)
    assert len(matrix._summarize_batch(rows)) == len(rows)


@pytest.mark.parametrize("reference_result", [None, result(converged=False)])
def test_invalid_reference_cannot_produce_batch_accuracy(
    measurement: typing.Any, reference_result: typing.Any
) -> None:
    item = result(
        index=0,
        status_message="success",
        succeeded=True,
        warm_start_used=False,
        warm_start_fallback=False,
        restart_origin="cold",
        fock_builds=None,
    )
    rows = matrix._batch_items(
        SimpleNamespace(items=[item]), [{"result": reference_result}], object()
    )
    assert rows[0]["energy_absolute_error"] is None
    assert rows[0]["force_max_abs_error"] is None
    assert "accuracy" not in rows[0]


def test_failed_strict_reference_is_retained(
    measurement: typing.Any, monkeypatch: typing.Any
) -> None:
    def singlepoint(*args: typing.Any, **kw: typing.Any) -> typing.Any:
        raise RuntimeError("reference did not converge")

    monkeypatch.setattr(
        matrix,
        "_calculator",
        lambda *args, **kw: SimpleNamespace(
            singlepoint=singlepoint,
            resolved_model=lambda *args, **kw: object(),
        ),
    )
    ref = matrix._strict_reference(
        CASE, measurement, BASE, measurement.properties, None
    )
    assert ref["result"] is None
    assert ref["record"]["converged"] is False
    assert "reference did not converge" in ref["record"]["failure"]
