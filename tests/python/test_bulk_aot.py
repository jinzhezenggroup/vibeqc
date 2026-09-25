"""Build identity/budget/measurement tests without importing the QC runtime."""

from __future__ import annotations

import gc
import json
import shutil
import sys
import weakref
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from vibeqc_compiler.common.compiler_process import CompileResult
from vibeqc_compiler.common.compiler_work import charge_symbolic_intern
from vibeqc_compiler.xc import bulk_aot

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def variant(**changes: Any) -> bulk_aot.SourceVariant:
    return replace(
        bulk_aot.SourceVariant(
            name="ALIAS_A",
            family="lda",
            spin="unpolarized",
            import_identity="source-owner-A",
            domain="interior-test/v1",
            features=("rho",),
            derivative_order=1,
            backend="cpu",
            source="void bulk_xc_point(const double *x, double *y) { y[0] = x[0] * x[0]; y[1] = 2 * x[0]; }\n",
            energy_nodes=2,
            ssa={"root_count": 2, "reachable_node_count": 4},
        ),
        **changes,
    )


def test_exact_alias_reuses_build_not_provenance() -> None:
    first = variant()
    second = variant(name="ALIAS_B", import_identity="source-owner-B")
    assert first.emission_identity == second.emission_identity
    plan = bulk_aot.plan_package([first, second], bulk_aot.PackageBudget(1, 10000))
    assert plan["summary"]["reused_build_tasks"] == 1
    assert plan["summary"]["selected_artifacts"] == 1
    assert {
        row["import_identity"] for row in plan["artifacts"][0]["registrations"]
    } == {
        "source-owner-A",
        "source-owner-B",
    }
    assert plan["capability_claims"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"source": "void bulk_xc_point(const double *x, double *y) { y[0] = 0; }"},
        {"features": ("rho_a",)},
        {"spin": "polarized"},
        {"derivative_order": 2},
        {"domain": "another-domain/v1"},
        {"backend": "cuda"},
    ],
)
def test_emission_identity_invalidates_on_code_or_abi(change: dict) -> None:
    assert variant().emission_identity != variant(**change).emission_identity


@pytest.mark.parametrize(
    "change",
    [
        {"derivative_order": True},
        {"derivative_order": 3},
        {"backend": "metal"},
        {"source": ""},
        {"features": ()},
        {"domain": ""},
    ],
)
def test_invalid_variant_is_rejected(change: dict) -> None:
    with pytest.raises(ValueError):
        variant(**change)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_positive_package_limits(bad: int) -> None:
    with pytest.raises(ValueError):
        bulk_aot.PackageBudget(bad, 100)
    with pytest.raises(ValueError):
        bulk_aot.PackageBudget(1, bad)


def test_plan_is_order_independent_and_has_hard_byte_and_count_bounds() -> None:
    sources = [variant(), variant(name="B", source=variant().source + "\n")]
    budget = bulk_aot.PackageBudget(1, 10000)
    assert bulk_aot.plan_package(sources, budget) == bulk_aot.plan_package(
        reversed(sources), budget
    )
    plan = bulk_aot.plan_package(sources, budget)
    assert plan["summary"]["selected_artifacts"] == 1
    assert [a["blocker"] for a in plan["artifacts"]].count("artifact-count-budget") == 1
    tiny = bulk_aot.plan_package(sources, bulk_aot.PackageBudget(2, 1))
    assert tiny["summary"]["selected_source_bytes"] == 0
    assert all(a["blocker"] == "source-byte-budget" for a in tiny["artifacts"])
    exact = bulk_aot.plan_package(
        [sources[0]], bulk_aot.PackageBudget(1, len(sources[0].source.encode()))
    )
    assert exact["summary"]["selected_artifacts"] == 1


def test_duplicate_registration_is_not_silently_deduplicated() -> None:
    with pytest.raises(ValueError, match="duplicate census variant"):
        bulk_aot.plan_package([variant(), variant()])


def test_streaming_plan_does_not_retain_catalog_sources() -> None:
    refs: list[weakref.ReferenceType] = []

    def stream() -> Iterator[bulk_aot.SourceVariant]:
        for index in range(20):
            item = variant(name=f"ALIAS_{index}")
            refs.append(weakref.ref(item))
            gc.collect()
            assert sum(ref() is not None for ref in refs) <= 2
            yield item

    plan = bulk_aot.plan_package(stream())
    assert plan["summary"]["registration_variants"] == 20
    assert plan["summary"]["unique_artifacts"] == 1
    assert all(ref() is None for ref in refs)
    assert '"source":' not in json.dumps(plan)


def test_compile_probe_uses_real_cc_without_runtime_execution() -> None:
    if shutil.which("cc") is None:
        pytest.skip("C compiler unavailable")
    result = bulk_aot.compile_probe(variant())
    assert result["status"] == "compiled"
    assert result["object_bytes"] > 0
    assert len(result["object_sha256"]) == 64
    assert result["compile_seconds"] >= 0
    assert result["runtime_validation"] == "not-run"
    assert result["resources"] is None
    assert bulk_aot.compile_probe(variant(source="not valid C"))["status"] == "failed"


def test_missing_compiler_and_unknown_resources_are_not_success() -> None:
    assert (
        bulk_aot.compile_probe(variant(), compiler="/missing/xc-compiler")["status"]
        == "unavailable"
    )
    assert all(value is None for value in bulk_aot.ptxas_resources("").values())


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        bulk_aot.compile_probe(variant(), timeout=timeout)


def test_cuda_probe_retains_function_and_parses_observed_resources() -> None:
    source = variant(backend="cuda").source
    probe = bulk_aot.cuda_probe_source(source)
    assert probe.startswith(source)
    assert "__global__ void bulk_xc_census_probe" in probe
    assert "bulk_xc_point(features, outputs)" in probe
    resources = bulk_aot.ptxas_resources(
        "ptxas info : Function properties for bulk_xc_census_probe\n"
        "    24 bytes stack frame, 0 bytes spill stores, 8 bytes spill loads\n"
        "ptxas info : Used 42 registers, 16 bytes smem, 4 bytes lmem\n"
        "ptxas info : Function properties for bulk_xc_point\n"
        "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        "ptxas info : Used 60 registers, 2 bytes lmem\n"
    )
    assert resources == {
        "registers": 60,
        "shared_bytes": 16,
        "stack_bytes": 24,
        "spill_store_bytes": 0,
        "spill_load_bytes": 8,
        "local_bytes": 4,
    }

    zero_shared = bulk_aot.ptxas_resources(
        "ptxas info : Function properties for bulk_xc_census_probe\n"
        "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        "ptxas info : Used 32 registers, 0 bytes lmem\n"
    )
    assert zero_shared["shared_bytes"] == 0
    with pytest.raises(ValueError, match="cuda_arch"):
        bulk_aot.compile_probe(variant(), cuda_arch="native")


def test_cuda_resource_gate_fails_closed_on_unknown_observations() -> None:
    limits = bulk_aot.CudaResourceLimits(128, 64, 0, 0, 0)
    evidence = {
        "status": "compiled",
        "resources": {
            "registers": 64,
            "stack_bytes": 0,
            "spill_store_bytes": 0,
            "spill_load_bytes": 0,
            "shared_bytes": 0,
            "local_bytes": None,
        },
    }

    gate = bulk_aot.gate_cuda_resources(evidence, limits)
    assert gate["status"] == "unavailable"
    assert not gate["package_eligible"]
    assert gate["reasons"] == ["unknown-resource:local_bytes"]


def test_cuda_resource_gate_enforces_register_stack_local_shared_and_spills() -> None:
    limits = bulk_aot.CudaResourceLimits(64, 32, 8, 16, 0)
    passing = {
        "status": "compiled",
        "resources": {
            "registers": 64,
            "stack_bytes": 32,
            "spill_store_bytes": 0,
            "spill_load_bytes": 0,
            "shared_bytes": 16,
            "local_bytes": 8,
        },
    }
    assert bulk_aot.gate_cuda_resources(passing, limits) == {
        "status": "passed",
        "package_eligible": True,
        "reasons": [],
        "limits": limits.to_payload(),
        "observed": passing["resources"],
    }

    failing = {
        "status": "compiled",
        "resources": {
            "registers": 65,
            "stack_bytes": 33,
            "spill_store_bytes": 4,
            "spill_load_bytes": 4,
            "shared_bytes": 17,
            "local_bytes": 9,
        },
    }
    gate = bulk_aot.gate_cuda_resources(failing, limits)
    assert gate["status"] == "rejected"
    assert not gate["package_eligible"]
    assert gate["reasons"] == [
        "register-limit",
        "stack-limit",
        "local-memory-limit",
        "shared-memory-limit",
        "spill-limit",
    ]


@pytest.mark.parametrize(
    "args",
    [
        (0, 0, 0, 0, 0),
        (64, -1, 0, 0, 0),
        (64, 0, -1, 0, 0),
        (64, 0, 0, -1, 0),
        (64, 0, 0, 0, -1),
    ],
)
def test_cuda_resource_limits_reject_invalid_bounds(
    args: tuple[int, int, int, int, int],
) -> None:
    with pytest.raises(ValueError):
        bulk_aot.CudaResourceLimits(*args)


def test_measurement_marks_cuda_ineligible_without_resource_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = variant(backend="cuda")
    plan = bulk_aot.plan_package([item])

    monkeypatch.setattr(
        bulk_aot,
        "compile_probe",
        lambda *args, **kwargs: {
            "status": "compiled",
            "resources": {
                "registers": 32,
                "stack_bytes": 0,
                "spill_store_bytes": 0,
                "spill_load_bytes": 0,
                "shared_bytes": 0,
                "local_bytes": 0,
            },
        },
    )
    measured = bulk_aot.measure_plan(plan, lambda _: item)
    gate = next(iter(measured.values()))["resource_gate"]
    assert gate == {
        "status": "not-run",
        "package_eligible": False,
        "reasons": ["resource-policy-not-supplied"],
    }

    measured = bulk_aot.measure_plan(
        plan,
        lambda _: item,
        cuda_resource_limits=bulk_aot.CudaResourceLimits(64, 0, 0, 0, 0),
    )
    assert next(iter(measured.values()))["resource_gate"]["status"] == "passed"


def test_compilation_timeout_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def run(command: list[str], timeout: float, *, label: str) -> CompileResult:
        nonlocal calls
        calls += 1
        assert timeout == 1
        assert label.startswith("XC census")
        if calls == 1:
            assert command[-1] == "--version"
            return CompileResult(0, False, 0.01, "test compiler", "")
        return CompileResult(124, True, 1.0, "", "injected timeout")

    monkeypatch.setattr(bulk_aot.shutil, "which", lambda _: sys.executable)
    monkeypatch.setattr(bulk_aot, "run_compiler", run)
    assert bulk_aot.compile_probe(variant(), timeout=1)["status"] == "timed-out"
    assert calls == 2


def test_measurement_rebuilds_once_per_selected_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = variant()
    alias = variant(name="ALIAS_B", import_identity="B")
    plan = bulk_aot.plan_package([first, alias], bulk_aot.PackageBudget(1, 10000))
    loaded: list[str] = []
    compiled: list[str] = []

    def load(record: dict) -> bulk_aot.SourceVariant:
        loaded.append(record["name"])
        return first

    def compile_one(item: bulk_aot.SourceVariant, **kwargs: Any) -> dict:
        compiled.append(item.name)
        return {"status": "compiled"}

    monkeypatch.setattr(bulk_aot, "compile_probe", compile_one)
    evidence = bulk_aot.measure_plan(plan, load)
    assert loaded == compiled == ["ALIAS_A"]
    assert len(evidence) == 1
    with pytest.raises(ValueError, match="identity changed"):
        bulk_aot.measure_plan(plan, lambda _: replace(first, import_identity="changed"))
    with pytest.raises(ValueError, match="identity changed"):
        bulk_aot.measure_plan(
            plan, lambda _: replace(first, source=first.source + "\n")
        )


def test_deferred_groups_are_never_loaded() -> None:
    plan = bulk_aot.plan_package([variant()], bulk_aot.PackageBudget(1, 1))

    def forbidden(record: dict) -> bulk_aot.SourceVariant:
        raise AssertionError("unselected source was rebuilt")

    assert (
        next(iter(bulk_aot.measure_plan(plan, forbidden).values()))["status"]
        == "not-selected"
    )


@pytest.fixture
def synthetic_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Control importer inputs; no alternative scientific formula implementation."""
    from vibeqc_compiler import xc

    catalog = {
        "registrations": [
            {"name": "A", "family": "lda", "graph_status": "imported"},
            {
                "name": "B",
                "family": "gga",
                "graph_status": "blocked",
                "reason": "unsupported-test-input",
            },
        ],
        "source_files": {},
    }
    fake = ModuleType("vibeqc_compiler.xc.libxc_bulk")
    fake.CATALOG_PATH = tmp_path / "catalog.json"
    fake.BULK_SEMANTICS = "interior-test/v1"
    fake.SOURCE_ASSET = "unused"
    fake.read_catalog = lambda _: catalog

    def build(record: dict, files: dict, root: Path, *, spin: str) -> SimpleNamespace:
        charge_symbolic_intern()
        return SimpleNamespace(name=record["name"], spin=spin)

    def inspect(
        program: SimpleNamespace, order: int, backend: str, *, domain: str
    ) -> bulk_aot.SourceVariant:
        return variant(
            name=program.name,
            spin=program.spin,
            derivative_order=order,
            backend=backend,
            domain=domain,
        )

    fake.build_record = build
    maple = ModuleType("vibeqc_compiler.xc.libxc_maple")
    maple.MapleImportError = ValueError
    paths = ModuleType("vibeqc_compiler.common.paths")
    paths.asset_path = lambda _: tmp_path
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setitem(sys.modules, maple.__name__, maple)
    monkeypatch.setitem(sys.modules, paths.__name__, paths)
    monkeypatch.setattr(xc, "libxc_bulk", fake, raising=False)
    monkeypatch.setattr(bulk_aot, "inspect_program", inspect)
    return catalog


def test_census_has_no_eager_derivative_product(synthetic_catalog: dict) -> None:
    plan = bulk_aot.census_catalog()
    assert len(plan["observations"]) == 4  # 1 imported x 2 spins x 2 backends
    assert plan["request"]["derivative_orders"] == [1]
    assert plan["blocked_imports"] == [
        {"name": "B", "reason": "unsupported-test-input"}
    ]
    assert all(row["symbolic_work"] == 1 for row in plan["observations"])
    assert "measurements" not in plan


@pytest.mark.parametrize(
    "options",
    [
        {"names": []},
        {"names": ["UNKNOWN"]},
        {"spins": []},
        {"spins": ["wrong"]},
        {"spins": ["polarized", "polarized"]},
        {"backends": ["gpu"]},
        {"derivative_orders": [True]},
        {"derivative_orders": [1, 1]},
        {"work_limit": 0},
    ],
)
def test_invalid_census_selection(synthetic_catalog: dict, options: dict) -> None:
    with pytest.raises(ValueError):
        bulk_aot.census_catalog(**options)


def test_work_limit_is_recorded_without_source_or_admission(
    synthetic_catalog: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def expensive(*args: Any, **kwargs: Any) -> bulk_aot.SourceVariant:
        charge_symbolic_intern()
        raise AssertionError("work limit must stop before reaching here")

    monkeypatch.setattr(bulk_aot, "inspect_program", expensive)
    plan = bulk_aot.census_catalog(work_limit=1)
    assert plan["summary"]["unique_artifacts"] == 0
    assert all(row["status"] == "work-budget-exhausted" for row in plan["observations"])
    assert plan["capability_claims"] == []
