"""Geometry regeneration is an audit, never permission to rewrite frozen v1."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def audit_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    pytest.importorskip("rdkit")
    from tools.dft_mp_v1 import generate_inputs

    source = {"kind": "fixture", "license": "test", "attribution": "test"}
    data = {
        "water": {
            "atoms": [["O", [0.0, 0.0, 0.0]]],
            "charge": 0,
            "multiplicity": 1,
            "source": source,
        },
    }
    target = tmp_path / "inputs"
    target.mkdir()
    value = {"schema_version": 1, "id": "water", "units": "bohr", **data["water"]}
    original = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    changed = json.loads(json.dumps(value))
    changed["id"] = "water-changed"
    changed["atoms"][0][1][0] += 0.01
    changed["source"] = {
        "kind": "fixed_displacement",
        "parent": "water",
        "atom_index": 0,
        "axis": "x",
        "delta_bohr": 0.01,
        "license": "test",
        "attribution": "test",
    }
    displaced = (json.dumps(changed, indent=2, sort_keys=True) + "\n").encode()
    (target / "water.json").write_bytes(original)
    (target / "water-changed.json").write_bytes(displaced)
    monkeypatch.setattr(generate_inputs, "ROOT", tmp_path)
    monkeypatch.setattr(generate_inputs, "build", lambda: data)
    return generate_inputs, target, data


def test_mismatched_regeneration_preserves_all_frozen_bytes(
    audit_fixture: tuple,
) -> None:
    generator, target, data = audit_fixture
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    data["water"]["atoms"][0][1][0] = 1e-8
    with pytest.raises(RuntimeError, match="no files changed"):
        generator.main()
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before


def test_missing_frozen_file_does_not_create_a_replacement(
    audit_fixture: tuple,
) -> None:
    generator, target, _ = audit_fixture
    (target / "water-changed.json").unlink()
    before = (target / "water.json").read_bytes()
    with pytest.raises(RuntimeError, match="cannot audit frozen input"):
        generator.main()
    assert not (target / "water-changed.json").exists()
    assert (target / "water.json").read_bytes() == before


@pytest.mark.parametrize("crlf", [False, True])
def test_successful_audit_is_readonly(
    audit_fixture: tuple, monkeypatch: pytest.MonkeyPatch, crlf: bool
) -> None:
    generator, target, _ = audit_fixture
    if crlf:
        for path in target.iterdir():
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    monkeypatch.setattr(
        type(target), "write_text", lambda *a, **k: pytest.fail("audit wrote text")
    )
    monkeypatch.setattr(
        type(target), "write_bytes", lambda *a, **k: pytest.fail("audit wrote bytes")
    )
    generator.main()
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
