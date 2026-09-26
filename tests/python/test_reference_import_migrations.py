"""Retained reference records accept only exact reviewed import migrations."""

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from vibeqc_compiler.common.reference_sources import reference_source_matches

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = Path("tests/reference_data/reference_source_import_migrations.json")


def _shape(node: Any) -> Any:
    if isinstance(node, ast.AST):
        return [
            type(node).__name__,
            [
                [k, _shape(v)]
                for k, v in ast.iter_fields(node)
                if not (k == "type_params" and not v)
            ],
        ]
    if isinstance(node, list):
        return [
            _shape(x) for x in node if not isinstance(x, (ast.Import, ast.ImportFrom))
        ]
    return node


def test_import_migration_receipts_bind_exact_current_sources() -> None:
    payload = json.loads((ROOT / RECEIPT).read_text())
    assert "no new reference execution" in payload["scope"]
    assert len(payload["sources"]) == 6
    for record in payload["sources"]:
        data = (ROOT / record["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["current_sha256"]
        shape = json.dumps(
            _shape(ast.parse(data)), separators=(",", ":"), ensure_ascii=True
        )
        assert (
            hashlib.sha256(shape.encode()).hexdigest()
            == record["non_import_ast_sha256"]
        )
        assert record["removed_imports"] and record["added_imports"]
        assert reference_source_matches(
            ROOT, record["path"], record["historical_sha256"]
        )
        assert reference_source_matches(ROOT, record["path"], record["current_sha256"])
        assert not reference_source_matches(ROOT, record["path"], "0" * 64)


@pytest.mark.parametrize("mutation", ["formula", "import", "receipt", "path", "absent"])
def test_unreviewed_source_or_receipt_is_rejected(
    tmp_path: Path, mutation: str
) -> None:
    payload = json.loads((ROOT / RECEIPT).read_text())
    record = payload["sources"][0]
    source = tmp_path / record["path"]
    source.parent.mkdir(parents=True)
    source.write_bytes((ROOT / record["path"]).read_bytes())
    receipt = tmp_path / RECEIPT
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(payload))
    assert reference_source_matches(
        tmp_path, record["path"], record["historical_sha256"]
    )
    if mutation in {"formula", "import"}:
        with source.open("a") as stream:
            stream.write("\nx = 2.0\n" if mutation == "formula" else "\nimport math\n")
    elif mutation == "receipt":
        receipt.write_text('{"schema": "unapproved", "sources": []}')
    elif mutation == "path":
        record["path"] = "tools/other.py"
        receipt.write_text(json.dumps(payload))
    else:
        receipt.unlink()
    assert not reference_source_matches(
        tmp_path, source.relative_to(tmp_path).as_posix(), record["historical_sha256"]
    )


def test_current_exact_bytes_need_no_migration_receipt(tmp_path: Path) -> None:
    source = tmp_path / "export.py"
    source.write_text("value = 1\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert reference_source_matches(tmp_path, "export.py", digest)
    assert not reference_source_matches(tmp_path, "../export.py", digest)
