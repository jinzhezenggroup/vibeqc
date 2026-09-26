"""Regression tests for the Issue #490 public-facade ownership split."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import vibeqc._generated_methods as method_manifest
from vibeqc import Atom
from vibeqc._api_types import Result
from vibeqc._model_resolution import ModelResolutionInput, resolve_model_identity
from vibeqc._result_translation import backend_name, copy_force_array
from vibeqc._warm_state import WarmStartState


def test_model_resolution_is_native_free_and_preserves_identity() -> None:
    atoms = (
        Atom(1, (0.0, 0.0, 0.0)),
        Atom(1, (0.0, 0.0, 1.4)),
    )
    orbital = {
        "mathematical_identity": "basis-hash",
        "electrons": {"electron_count": 2},
    }
    resolved = resolve_model_identity(
        ModelResolutionInput(
            method_id=method_manifest.METHOD_RHF,
            representation="cartesian",
            basis_metadata={"orbital": orbital},
            geometry=atoms,
            charge=0,
            multiplicity=1,
            density_fitting=False,
            density_fitting_relative_threshold=None,
        )
    )
    assert resolved.basis_hash == "basis-hash"
    assert resolved.multiplicity == 1
    assert resolved.approximation == "conventional"


def test_result_translation_has_no_calculator_dependency() -> None:
    force_storage = np.ctypeslib.as_ctypes(np.arange(6, dtype=np.float64))
    forces = copy_force_array(force_storage, 2)
    assert np.array_equal(forces, np.arange(6, dtype=np.float64).reshape(2, 3))
    assert backend_name(0) == "cpu_reference"


def test_warm_state_owns_restart_metadata_and_origin() -> None:
    state = WarmStartState(2)
    state.restart_indices.add(1)
    assert state.origin(1, warm_start_used=True, warm_start_fallback=False) == (
        "persistent_restart"
    )
    state.record_success(1, controls={"max_iterations": 5}, backend="cpu")
    assert state.metadata[1] == {
        "controls": {"max_iterations": 5},
        "backend": "cpu",
    }
    assert not state.restart_indices
    state.clear()
    assert state.metadata == [None, None]


def test_public_record_types_are_reexported_from_legacy_module() -> None:
    from vibeqc.calculator import Atom as LegacyAtom
    from vibeqc.calculator import Result as LegacyResult

    assert LegacyAtom is Atom
    assert LegacyResult is Result


def test_internal_owners_do_not_import_execution_facades() -> None:
    """Keep model/result/state owners one-way and independent of the facades."""

    root = Path(__file__).parents[2] / "python" / "vibeqc"
    forbidden = {".calculator", ".batch"}
    for name in ("_model_resolution.py", "_result_translation.py", "_warm_state.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1
        }
        assert not imports & forbidden, (name, imports & forbidden)
