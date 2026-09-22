"""Identical first-derivative capability evidence must not be emitted twice."""

from collections import Counter

import pytest
from vibeqc_compiler.integral import capabilities
from vibeqc_compiler.integral.shell_spec import PSSS_SPEC, SSSS_SPEC, ShellClassSpec


@pytest.mark.parametrize("spec", [SSSS_SPEC, PSSS_SPEC])
def test_report_reuses_actual_first_derivative_evidence(
    spec: ShellClassSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = capabilities._check_recurrence
    calls: Counter[str] = Counter()

    def count(*args: object, **kwargs: object) -> object:
        calls[args[1]] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(capabilities, "_check_recurrence", count)
    report = capabilities.build_capability_report(
        architecture="sm_120", specifications=(spec,)
    )
    assert calls == Counter(
        {name: 1 for name in ("subset_wick", "rys2", "rys3", "rys4", "rys5")}
    )
    row = report["shell_classes"][0]
    assert row["force_derivative_orders"]["1"] == row["recurrences"]["subset_wick"]
    assert not row["force_derivative_orders"]["2"]["supported"]
