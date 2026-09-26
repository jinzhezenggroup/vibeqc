"""Control construction failures must remain explicit campaign evidence."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from vibeqc_compiler.xc import production_domain_controls as controls

LAZY = "control/lazy-inactive-branch"
NONFINITE = "control/invalid-nonfinite"


@pytest.fixture(autouse=True)
def control_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = SimpleNamespace(
        spin_layouts=("polarized", "unpolarized"),
        outputs=("energy", "vxc", "fxc"),
        case_ids_for_spin=lambda _: (LAZY, NONFINITE),
    )
    monkeypatch.setattr(
        controls,
        "functional_capability",
        lambda _: SimpleNamespace(
            name="control-fixture", production_domain_profile=profile
        ),
    )


def inject_failure(
    monkeypatch: pytest.MonkeyPatch, stage: str, error: type[BaseException]
) -> str:
    def fail() -> None:
        raise error("injected control construction failure")

    if stage == "program":

        def build(*args: object, **kwargs: object) -> None:
            fail()

        monkeypatch.setattr(controls, "build_bulk_runtime_program", build)
        return NONFINITE

    class Graph:
        def __init__(self) -> None:
            self.derivatives = 0
            if stage == "graph":
                fail()

        def variable(self, name: str) -> float:
            if stage == "variable":
                fail()
            return 1.0

        def select_le(self, *args: object) -> float:
            if stage == "select":
                fail()
            return 1.0

        def differentiate(self, *args: object) -> float:
            self.derivatives += 1
            if stage == f"derivative-{self.derivatives}":
                fail()
            return 1.0

        def evaluate(self, *args: object) -> float:
            pytest.fail("construction failure should precede graph evaluation")

    monkeypatch.setattr(controls, "Graph", Graph)
    return LAZY


STAGES = ("program", "graph", "variable", "select", "derivative-1", "derivative-2")


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(
    "error", (ValueError, RuntimeError, TypeError, ArithmeticError)
)
def test_setup_failure_is_a_serializable_failed_row(
    monkeypatch: pytest.MonkeyPatch, stage: str, error: type[Exception]
) -> None:
    case_id = inject_failure(monkeypatch, stage, error)
    row, detail = controls.run_control_case(
        "control-fixture", spin="polarized", case_id=case_id
    )
    assert row["status"] == detail["status"] == "fail"
    assert row["case_id"] == detail["case_id"] == case_id
    assert row["outputs"] == ["energy", "vxc", "fxc"]
    assert error.__name__ in row["reason"]
    assert "injected control construction failure" in row["reason"]
    assert row["reason"] == detail["reason"]
    json.dumps({"row": row, "detail": detail}, allow_nan=False)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("error", (KeyboardInterrupt, SystemExit))
def test_process_interruptions_are_not_converted_to_control_evidence(
    monkeypatch: pytest.MonkeyPatch, stage: str, error: type[BaseException]
) -> None:
    case_id = inject_failure(monkeypatch, stage, error)
    with pytest.raises(error, match="injected control construction failure"):
        controls.run_control_case("control-fixture", spin="polarized", case_id=case_id)


@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_valid_nonfinite_rejection_counts_are_preserved(
    monkeypatch: pytest.MonkeyPatch, spin: str
) -> None:
    seen = []
    features = ("rho_a", "rho_b") if spin == "polarized" else ("rho",)

    def evaluate(values: object) -> None:
        seen.append(values)
        raise controls.UnsupportedXC("nonfinite bulk XC input")

    monkeypatch.setattr(
        controls,
        "build_bulk_runtime_program",
        lambda *args, **kwargs: SimpleNamespace(
            spec=SimpleNamespace(features=features), evaluate=evaluate
        ),
    )
    row, detail = controls.run_control_case(
        "control-fixture", spin=spin, case_id=NONFINITE
    )
    assert row["status"] == "pass"
    assert detail["rejected"] == detail["expected"] == len(seen) == 3 * len(features)
