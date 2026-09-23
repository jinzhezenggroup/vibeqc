"""Compact runtime features preserve the imported graph, not zero a live input."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.xc import bulk_runtime, libxc_bulk
from vibeqc_compiler.xc.spec import UnsupportedXC


@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_compact_runtime_preserves_full_mgga_roots_and_source(
    monkeypatch: pytest.MonkeyPatch, spin: str, order: int
) -> None:
    full = libxc_bulk.build_bulk_program("MGGA_X_R2SCAN01", spin=spin)
    capability = SimpleNamespace(
        name=full.name,
        family=full.family,
        spin_layouts=("polarized", "unpolarized"),
        required_ingredients=("rho", "sigma", "tau"),
        identity="projection-contract",
    )
    monkeypatch.setattr(bulk_runtime, "functional_capability", lambda _: capability)
    before = full.emit_source(order)
    program = bulk_runtime.build_bulk_runtime_program(full.name, spin=spin, order=order)
    assert program.spec.source_identity == full.identity
    assert program.spec.features == tuple(
        n for n in full.features if not n.startswith("lapl")
    )
    assert full.emit_source(order) == before
    data = np.full((len(full.features), 2), 0.7)
    for i, name in enumerate(full.features):
        if name.startswith("sigma"):
            data[i] = 0.02 if name != "sigma_ab" else 0.001
        elif name.startswith("lapl"):
            data[i] = [-0.9, 1.3]
    rows = dict(zip(full.features, data, strict=True))
    variables = [full.variables[full.features.index(n)] for n in program.spec.features]
    derivatives = {(): full.energy}
    for output in program.outputs:
        for depth in range(1, len(output) + 1):
            key = output[:depth]
            if key not in derivatives:
                derivatives[key] = full.graph.differentiate(
                    derivatives[key[:-1]], variables[key[-1]]
                )
    expected = evaluate_array_graph(
        full.graph, tuple(derivatives[k] for k in program.outputs), rows
    )
    expected = np.stack([np.broadcast_to(v, (2,)) for v in expected])
    actual = program.evaluate(np.stack([rows[n] for n in program.spec.features]))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
@pytest.mark.parametrize("live_feature", ["lapl", "tau"])
def test_flags_cannot_silently_zero_a_reachable_input(
    monkeypatch: pytest.MonkeyPatch, spin: str, live_feature: str
) -> None:
    # Inject a source/flag disagreement while keeping the actual importer and
    # Graph. The mock is not a scientific oracle; it isolates admission policy.
    record = {
        "name": "SYNTHETIC_MGGA",
        "family": "mgga",
        "flags": "XC_FLAGS_NEEDS_TAU" if live_feature == "lapl" else "",
    }
    ingredients = (
        ("rho", "sigma", "tau") if live_feature == "lapl" else ("rho", "sigma")
    )
    slot = 5 if live_feature == "lapl" else 7
    module = SimpleNamespace(
        transitive_sha256="synthetic-source",
        call=lambda graph, name, *arguments: arguments[slot],
    )
    monkeypatch.setattr(libxc_bulk, "import_record", lambda *a, **k: module)
    monkeypatch.setattr(
        libxc_bulk,
        "build_bulk_program",
        lambda name, **kwargs: libxc_bulk.build_record(record, {}, Path("."), **kwargs),
    )
    monkeypatch.setattr(
        bulk_runtime,
        "functional_capability",
        lambda _: SimpleNamespace(
            name=record["name"],
            family="mgga",
            spin_layouts=("polarized", "unpolarized"),
            required_ingredients=ingredients,
            identity="synthetic-capability",
        ),
    )
    with pytest.raises(UnsupportedXC, match="cannot discard reachable feature"):
        bulk_runtime.build_bulk_runtime_program(record["name"], spin=spin)
