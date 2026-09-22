"""Keep shared native infrastructure independent of concrete GFN/xTB runtime."""

from __future__ import annotations

from pathlib import Path

from tools.check_electronic_structure_boundaries import (
    SHARED_CONTRACTS,
    SHARED_OWNERS,
    _native_dependency_edges,
)

ROOT = Path(__file__).resolve().parents[2]


def _shared_xtb_edges(root: Path) -> list[str]:
    """Return shared-native dependency edges that cross into src/xtb."""
    violations: list[str] = []
    for edge in _native_dependency_edges(root):
        source = str(edge["source"])
        target = str(edge["target"])
        owner = source.split("/", 1)[0]
        if owner not in SHARED_OWNERS and source not in SHARED_CONTRACTS:
            continue
        if target.startswith("xtb/"):
            violations.append(f"{source}:{edge['line']} -> {target}")
    return violations


def test_shared_native_layers_do_not_depend_on_xtb_runtime() -> None:
    """GFN/xTB stays a consumer of shared infrastructure, never its owner."""
    assert _shared_xtb_edges(ROOT) == []


def test_shared_xtb_guard_rejects_direct_and_relative_includes(tmp_path: Path) -> None:
    target = tmp_path / "src/xtb/native/runtime.hpp"
    target.parent.mkdir(parents=True)
    target.write_text("// concrete GFN/xTB runtime\n", encoding="utf-8")

    shared = tmp_path / "src/runtime/shared.cpp"
    shared.parent.mkdir(parents=True)
    shared.write_text('#include "xtb/native/runtime.hpp"\n', encoding="utf-8")
    assert _shared_xtb_edges(tmp_path) == [
        "runtime/shared.cpp:1 -> xtb/native/runtime.hpp"
    ]

    shared.write_text('#include "../xtb/native/runtime.hpp"\n', encoding="utf-8")
    assert _shared_xtb_edges(tmp_path) == [
        "runtime/shared.cpp:1 -> xtb/native/runtime.hpp"
    ]


def test_provider_contract_cannot_depend_on_xtb_runtime(tmp_path: Path) -> None:
    target = tmp_path / "src/xtb/native/runtime.hpp"
    target.parent.mkdir(parents=True)
    target.write_text("// concrete GFN/xTB runtime\n", encoding="utf-8")

    contract = tmp_path / "src/integrals/electron_interaction_source.hpp"
    contract.parent.mkdir(parents=True)
    contract.write_text('#include "xtb/native/runtime.hpp"\n', encoding="utf-8")
    assert _shared_xtb_edges(tmp_path) == [
        "integrals/electron_interaction_source.hpp:1 -> xtb/native/runtime.hpp"
    ]
