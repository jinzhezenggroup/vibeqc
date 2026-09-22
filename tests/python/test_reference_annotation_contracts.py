"""Concrete public helper types and explicitly audited source-only rebinding."""

import ast
import json
from pathlib import Path

from vibeqc_compiler.common.reference_sources import reference_source_matches

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "tests/reference_data/reference_source_annotation_audit.json"


def test_reference_source_rebinding_is_exact_and_does_not_claim_reexecution() -> None:
    audit = json.loads(AUDIT.read_text())
    assert "no new reference execution" in audit["scope"]
    assert len(audit["sources"]) == 11
    for record in audit["sources"]:
        assert reference_source_matches(
            ROOT, record["path"], record["annotated_source_sha256"]
        )
        assert len(record["original_source_sha256"]) == 64
        assert len(record["executable_ast_sha256"]) == 64


def test_reference_cli_contracts_are_not_blanket_any_annotations() -> None:
    audit = json.loads(AUDIT.read_text())
    count = 0
    for record in audit["sources"]:
        tree = ast.parse((ROOT / record["path"]).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name == "main":
                assert node.returns is not None and ast.unparse(node.returns) == "None"
                count += 1
            if node.name == "generate":
                for parameter in node.args.args:
                    if parameter.arg in {"destination", "directory", "path", "output"}:
                        assert parameter.annotation is not None
                        assert ast.unparse(parameter.annotation) == "Path"
                        count += 1
            if node.name in {
                "_triples_feeds",
                "_mo_fock",
                "_endpoint_inputs",
                "_inputs_hash",
            }:
                assert ast.unparse(node.args.args[0].annotation) == "str"
                assert ast.unparse(node.returns) not in {"Any", "typing.Any"}
                count += 1
    assert count >= 15
