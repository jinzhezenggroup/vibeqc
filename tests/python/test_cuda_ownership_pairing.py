"""Preserve all workloads/samples when reducing time between paired versions."""

import json
from types import SimpleNamespace

import pytest
from vibeqc_compiler.common.timing import interleaved_selection_order

from tools import benchmark_cuda_ownership as benchmark


def arguments(tmp_path, scope):
    """Use the complete public DF matrix; no GPU is touched by these driver tests."""
    return SimpleNamespace(
        output=tmp_path, process_scope=scope, samples=5, case=None, domain="df"
    )


@pytest.mark.parametrize("scope", ["inventory", "case"])
def test_all_cases_have_the_same_five_sample_abba_order(tmp_path, monkeypatch, scope):
    observed = []
    saved = {}

    def measure(args, label, cases, path):
        observed.append((label, tuple(cases)))
        payload = {
            "source": label,
            "process_scope": scope,
            "endpoints": [{"case": case, "sample": path.name} for case in cases],
        }
        benchmark.write(path, payload)
        saved[path] = json.loads(path.read_text())
        return payload

    monkeypatch.setattr(benchmark, "measure_worker", measure)
    runs, measured = benchmark.collect_runs(arguments(tmp_path, scope))
    cases = [benchmark.case_id(row) for row in benchmark.endpoint_inventory("df")]
    assert len(cases) == len(set(cases)) == 18
    order = list(interleaved_selection_order(5))
    assert [label for label, _ in measured] == order
    for case in cases:
        assert [label for label, unit in observed if case in unit] == order
    assert len(observed) == (180 if scope == "case" else 10)
    for label, group in runs.items():
        assert len(group) == 5
        for sample, run in enumerate(group):
            assert [row["case"] for row in run["endpoints"]] == cases
            assert all(
                row["sample"] == f"{label}-{sample}.json" for row in run["endpoints"]
            )
    # The original process records remain byte-for-byte reconstructible from
    # the lossless aggregates, including the complete shared metadata.
    for path, process in saved.items():
        assert json.loads(path.read_text()) == process
        label, sample = path.stem.split("-")
        aggregate = runs[label][int(sample)]
        wanted = {row["case"] for row in process["endpoints"]}
        reconstructed = {
            **aggregate,
            "endpoints": [
                row for row in aggregate["endpoints"] if row["case"] in wanted
            ],
        }
        assert reconstructed == process
    with pytest.raises(FileExistsError, match="reuse"):
        benchmark.collect_runs(arguments(tmp_path, scope))


def test_case_aggregation_rejects_changed_source_metadata(tmp_path, monkeypatch):
    def measure(args, label, cases, path):
        return {
            "source": label + path.parent.name,
            "endpoints": [{"case": cases[0]}],
        }

    monkeypatch.setattr(benchmark, "measure_worker", measure)
    with pytest.raises(ValueError, match="provenance changed"):
        benchmark.collect_runs(arguments(tmp_path, "case"))


@pytest.mark.parametrize("cases", [["unknown"], ["spf/rhf/cartesian/df/batch1"] * 2])
def test_bad_case_requests_fail_before_launch(tmp_path, monkeypatch, cases):
    args = arguments(tmp_path, "case")
    args.case = cases
    monkeypatch.setattr(
        benchmark, "measure_worker", lambda *unused: pytest.fail("must not launch")
    )
    with pytest.raises(ValueError, match="duplicate or unknown"):
        benchmark.collect_runs(args)
