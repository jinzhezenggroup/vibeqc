"""CPU-only guards for the OMol25 benchmark's scientific and plotting contract."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from generativeqc import GridSpec, import_bse, load_basis

from benchmarks.readme_omol25 import (
    SCHEMA,
    check_record,
    main,
    native_fock_observation,
    protocol,
    require_force_endpoint,
    source_hashes,
)
from benchmarks.readme_wb97mv import reference_vv10_domain
from tools.render_omol25_benchmarks import collect, figure, validate


@pytest.mark.parametrize("converged", (False, True))
@pytest.mark.parametrize("count", (0, 17))
def test_native_fock_observation_uses_ks_counter(converged: bool, count: int) -> None:
    """The KS measurement takes precedence even on failure or a zero-build path."""
    item = SimpleNamespace(
        ks_diagnostic=SimpleNamespace(fock_builds=count),
        fock_builds=91,
        iterations=5,
        converged=converged,
    )
    assert native_fock_observation(item) == {
        "fock_builds": count,
        "fock_builds_source": "ks_diagnostic.fock_builds",
    }


@pytest.mark.parametrize("diagnostic", (None, SimpleNamespace(fock_builds=None)))
@pytest.mark.parametrize("count", (None, 0, 13))
def test_native_fock_observation_preserves_legacy_or_missing_counts(
    diagnostic: SimpleNamespace | None, count: int | None
) -> None:
    """An absent counter stays unavailable, never zero or the iteration count."""
    item = SimpleNamespace(ks_diagnostic=diagnostic, fock_builds=count, iterations=5)
    assert native_fock_observation(item) == {
        "fock_builds": count,
        "fock_builds_source": "fock_builds" if count is not None else None,
    }


def sample(phase: str, geometry: int, repeat: int = 0) -> dict:
    """Make a complete shape-correct endpoint with explicit semantic work."""
    return {
        "phase": phase,
        "geometry": geometry,
        "repeat": repeat,
        "energy": -76.0,
        "forces": [[0.0, 0.0, 0.0]] * 3,
        "converged": True,
        "status": 0,
        "gate": True,
        "iterations": 1,
        "complete_seconds": 1.0,
    }


def run() -> dict:
    """Keep cold, moved and every requested replay in a deterministic fixture."""
    return {
        "schema": SCHEMA,
        "protocol": {
            "repeats": 5,
            "aos": 58,
            "force_return": "host_array_in_timed_endpoint",
        },
        "status": "measured",
        "stage": "complete",
        "records": [
            sample("cold", 0),
            *[sample("warm", 0, i) for i in range(5)],
            sample("moved", 1),
            *[sample("moved-warm", 1, i) for i in range(5)],
        ],
    }


def test_protocol_preserves_diffuse_basis_and_hf_geometries() -> None:
    source = Path(__file__).parents[1] / "data/external_basis/sto-3g-ho.json"
    basis = import_bse(
        source,
        source="test",
        source_version="1",
        license="BSD-3-Clause",
        representation="spherical",
    )
    original = protocol(3, basis, GridSpec(), 5)
    assert original["aos"] == 7
    assert original["basis_identity"] == basis.identity
    assert original["energy_tolerance"] == 1e-12
    assert (
        original["density_tolerance"]
        == original["reference_gradient_tolerance"]
        == 1e-10
    )
    assert original["properties"] == ["energy", "forces"]
    assert original["force_return"] == "host_array_in_timed_endpoint"
    assert original["nonlocal_density_threshold"] == 1e-8
    assert original["reference_nonlocal_weight_threshold"] == 1e-14
    first, second = original["geometries_bohr"]
    assert first[0] == second[0] and first[2] == second[2]
    assert second[1][1][2] - first[1][1][2] == pytest.approx(0.001)
    oxygen = basis.by_element[8]
    extra = replace(oxygen.shells[-1], exponents=("0.01",), coefficients=(("1",),))
    changed = replace(
        basis,
        elements=tuple(
            replace(element, shells=(*element.shells, extra))
            if element.atomic_number == 8
            else element
            for element in basis.elements
        ),
    )
    assert protocol(3, changed, GridSpec(), 5)["aos"] == 10
    assert changed.identity != basis.identity


def test_reference_force_export_precedes_endpoint_clock_stop() -> None:
    """HF includes GPU4PySCF's force D2H transfer, not just gradient kernels."""
    reference = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(main)))
        if isinstance(node, ast.FunctionDef) and node.name == "execute_reference"
    )
    assignments = {
        target.id: node
        for node in ast.walk(reference)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    export = assignments["forces"]
    assert isinstance(export.value, ast.Call)
    assert isinstance(export.value.func, ast.Attribute)
    assert export.value.func.attr == "asnumpy"
    assert export.lineno < assignments["seconds"].lineno


def test_source_identity_includes_public_and_geometry_consumers() -> None:
    """A native binary digest cannot identify an uncommitted Python force fix."""
    hashes = source_hashes()
    root = Path(__file__).resolve().parents[2]
    for path in (
        "python/generativeqc/calculator.py",
        "python/generativeqc/ks.py",
        "python/generativeqc/_stationary_cuda.py",
        "python/generativeqc/_stationary_composite_cuda.py",
        "src/scf/cuda/direct_jk.cpp",
        "src/scf/cuda/direct_jk_kernels.cu",
        "src/scf/cuda/direct_jk_kernels.hpp",
        "src/scf/cuda/direct_jk_plan.hpp",
    ):
        assert hashes[path] == hashlib.sha256((root / path).read_bytes()).hexdigest()


@pytest.mark.parametrize("fail", [False, True])
def test_reference_vv10_screening_matches_force_module_and_restores_defaults(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """SCF-only configuration leaves the independently imported force mask stale."""
    scf = SimpleNamespace(NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD=1e-10)
    forces = SimpleNamespace(NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD=2e-10)
    monkeypatch.setitem(sys.modules, "gpu4pyscf.dft", SimpleNamespace(numint=scf))
    monkeypatch.setitem(sys.modules, "gpu4pyscf.grad", SimpleNamespace(rks=forces))
    try:
        with reference_vv10_domain(1e-8) as domain:
            assert domain == {"scf": 1e-8, "forces": 1e-8}
            assert scf.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == 1e-8
            assert forces.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == 1e-8
            if fail:
                raise RuntimeError("reference SCF failed")
    except RuntimeError:
        assert fail
    assert scf.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == 1e-10
    assert forces.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD == 2e-10


@pytest.mark.parametrize(
    "mutation",
    [
        "force",
        "shape",
        "nan",
        "convergence",
        "timing",
        "duplicate",
        "missing",
        "old-schema",
        "no-host-export",
    ],
)
def test_validate_checks_every_repeat(mutation: str) -> None:
    reference = run()
    native = copy.deepcopy(reference)
    native["native_schedule_policy"] = "automatic-generated-SPD/canonical-through-f"
    row = native["records"][3]
    if mutation == "force":
        row["forces"][0][0] = 1e-4
    elif mutation == "shape":
        row["forces"] = [[0.0, 0.0, 0.0]]
    elif mutation == "nan":
        row["energy"] = float("nan")
    elif mutation == "convergence":
        row["converged"] = False
    elif mutation == "timing":
        row["complete_seconds"] = float("inf")
    elif mutation == "duplicate":
        row["repeat"] = 0
    elif mutation == "old-schema":
        native["schema"] = "generativeqc.readme-omol25.v2"
    elif mutation == "no-host-export":
        native["protocol"].pop("force_return")
    else:
        native["records"].pop()
    with pytest.raises(ValueError):
        validate(native, reference)


def test_collect_binds_independent_reference_and_preserves_timeouts(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "3"
    directory.mkdir()
    reference = run()
    reference_path = directory / "reference.json"
    reference_path.write_text(json.dumps(reference))
    native = copy.deepcopy(reference)
    native["native_schedule_policy"] = "automatic-generated-SPD/canonical-through-f"
    native["reference_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    (directory / "native.json").write_text(json.dumps(native))
    for engine in ("native", "reference"):
        (directory / f"{engine}.outcome").write_text(
            '{"exit_code":0,"time_limit_seconds":900}'
        )
    point = collect(tmp_path, 3)
    assert point["engines"]["native"]["medians"]["warm"] == 1.0
    assert (
        point["engines"]["native"]["native_schedule_policy"]
        == native["native_schedule_policy"]
    )
    assert "forces" not in point["engines"]["native"]["records"][0]
    assert "forces" in point["independent_references"][0]
    native["reference_sha256"] = "wrong"
    (directory / "native.json").write_text(json.dumps(native))
    with pytest.raises(ValueError, match="different independent"):
        collect(tmp_path, 3)
    (directory / "native.outcome").write_text(
        '{"exit_code":124,"time_limit_seconds":900}'
    )
    native.update(status="running", stage="cold/0")
    (directory / "native.json").write_text(json.dumps(native))
    point = collect(tmp_path, 3)
    assert point["engines"]["native"]["status"] == "timeout"
    assert "medians" not in point["engines"]["native"]
    (directory / "native.outcome").write_text(
        '{"exit_code":null,"time_limit_seconds":900,"reason":"reference_unavailable"}'
    )
    assert (
        collect(tmp_path, 3)["engines"]["native"]["status"] == "reference_unavailable"
    )
    native.update(
        status="unsupported",
        stage="native/capability",
        records=[],
        native_capabilities={
            "supported_properties": ["energy"],
            "maximum_basis_angular_momentum": 3,
        },
    )
    (directory / "native.json").write_text(json.dumps(native))
    (directory / "native.outcome").write_text(
        '{"exit_code":1,"time_limit_seconds":900}'
    )
    assert collect(tmp_path, 3)["engines"]["native"]["status"] == "unsupported"


@pytest.mark.parametrize("exit_code", [124, 137])
def test_collect_preserves_timeout_before_first_journal(
    tmp_path: Path, exit_code: int
) -> None:
    """Setup can exhaust the process limit before the first raw JSON exists."""
    directory = tmp_path / "3"
    directory.mkdir()
    (directory / "native.outcome").write_text(
        json.dumps({"exit_code": exit_code, "time_limit_seconds": 900})
    )
    point = collect(tmp_path, 3)
    assert point["engines"]["native"]["status"] == "timeout"
    assert "medians" not in point["engines"]["native"]
    assert point["engines"]["reference"]["status"] == "not_run"


def test_plot_keeps_variable_iteration_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    original = Axes.errorbar
    observed = []

    def capture(self: Axes, values: list, times: list, **kwargs: object) -> object:
        observed.append((values, times, kwargs["yerr"]))
        return original(self, values, times, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture)
    rows = [
        dict(
            sample("warm", 0, index),
            complete_seconds=float(index + 1),
            iterations=index + 1,
        )
        for index in range(5)
    ]
    figure(
        [
            {
                "atoms": 3,
                "protocol": {"aos": 58},
                "engines": {
                    engine: {"status": "measured", "records": rows}
                    for engine in ("native", "reference")
                },
            }
        ],
        tmp_path,
    )
    assert observed == [([58], [3.0], [[2.0], [2.0]])] * 2
    assert (tmp_path / "omol25.svg").is_file()


def test_check_record_rejects_missing_force() -> None:
    row = sample("warm", 0)
    row["forces"] = None
    assert not check_record(row, sample("cold", 0))["gate"]


@pytest.mark.parametrize("properties", [["energy"], []])
def test_preflight_rejects_energy_only_and_allows_qualified_f_shells(
    properties: list[str],
) -> None:
    with pytest.raises(NotImplementedError, match="analytic forces"):
        require_force_endpoint(
            {
                "supported_properties": properties,
                "maximum_basis_angular_momentum": 3,
            }
        )
    require_force_endpoint(
        {
            "supported_properties": ["energy", "forces"],
            "maximum_basis_angular_momentum": 3,
        }
    )


def test_retained_omol25_basis_preserves_f_and_diffuse_shells() -> None:
    root = Path(__file__).parents[2] / "benchmarks/results/omol25-wb97mv-20261001"
    basis = load_basis(root / "def2-tzvpd-ho.json")
    restored = import_bse(
        root / "def2-tzvpd-ho.bse.json",
        source=basis.provenance.source,
        source_version=basis.provenance.version.partition("; basis-data=")[0],
        license=basis.provenance.license,
        representation="spherical",
    )
    assert restored.identity == basis.identity
    assert max(shell.angular_momentum for shell in basis.by_element[8].shells) == 3
    assert [
        protocol(atoms, basis, GridSpec(), 5)["aos"] for atoms in (3, 6, 12, 24, 48, 96)
    ] == [58, 116, 232, 464, 928, 1856]


def test_unsupported_native_plot_has_no_native_latency(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    figure(
        [
            {
                "atoms": 3,
                "protocol": {"aos": 58},
                "engines": {
                    "native": {"status": "unsupported", "records": []},
                    "reference": {
                        "status": "measured",
                        "records": [sample("warm", 0, i) for i in range(5)],
                    },
                },
            }
        ],
        tmp_path,
    )
    assert (
        "f-shell force API unavailable (no native timings)"
        in (tmp_path / "omol25.svg").read_text()
    )


@pytest.mark.parametrize("failure", ["SCF did not converge", "stationary force failed"])
def test_native_failure_is_journaled_without_a_force_work_record(failure: str) -> None:
    """Execute the real row/retention code with a failed strict=False endpoint."""
    from types import CodeType, FunctionType

    from benchmarks.dft_force_components import normalize_force_work

    tree = ast.parse(inspect.getsource(main))
    row = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == "native_force_components"
            for key in node.value.keys
        )
    )
    retain = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "retain"
    )
    wrapper = ast.parse("def exercise():\n    pass").body[0]
    wrapper.body = [retain, row] + ast.parse("retain(row, baseline)").body
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
        "<native benchmark retention>",
        "exec",
    )
    compiled = next(value for value in code.co_consts if isinstance(value, CodeType))
    record = {"records": [], "stage": "cold/0"}
    saved = []
    scope = {
        "Any": object,
        "record": record,
        "save": saved.append,
        "check_record": check_record,
        "normalize_force_work": normalize_force_work,
        "native_fock_observation": native_fock_observation,
        "force_work": None,
        "phase": "cold",
        "geometry_index": 0,
        "repeat": 0,
        "seconds": 2.5,
        "prepare": 0.2,
        "baseline": sample("cold", 0),
        "item": SimpleNamespace(
            energy=-76.0,
            forces=None,
            converged=failure != "SCF did not converge",
            status=5,
            status_message=failure,
            iterations=100,
            fock_builds=101,
            energy_change=0.1,
            density_rms=0.2,
            physical_residual_rms=0.3,
            warm_start_used=False,
            warm_start_fallback=False,
        ),
    }
    with pytest.raises(RuntimeError, match="independent energy/force gate failed"):
        FunctionType(compiled, scope)()
    assert saved == ["cold/0"]
    assert len(record["records"]) == 1
    failed = record["records"][0]
    assert failed["detail"] == failure and failed["status"] == 5
    assert failed["iterations"] == 100 and failed["seconds"] == 2.5
    assert failed["native_force_components"] is None
    assert failed["gate"] is False
