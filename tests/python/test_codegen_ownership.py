"""Structural ownership guards for the legacy codegen compatibility suite."""

from pathlib import Path

LEGACY_CODEGEN_TEST = Path(__file__).with_name("test_codegen.py")
LEGACY_CODEGEN_MAX_LINES = 4518
OWNERSHIP_GUIDANCE = "New domain-focused tests should live in focused sibling modules"


def test_legacy_codegen_suite_does_not_regrow() -> None:
    """Keep new compiler coverage out of the legacy catch-all module."""
    text = LEGACY_CODEGEN_TEST.read_text(encoding="utf-8")
    line_count = len(text.splitlines())

    assert line_count <= LEGACY_CODEGEN_MAX_LINES, (
        "tests/python/test_codegen.py is a legacy compatibility suite and must not "
        f"grow beyond {LEGACY_CODEGEN_MAX_LINES} lines (found {line_count}). Move new "
        "domain-focused coverage into a focused sibling test module."
    )


def test_legacy_codegen_suite_keeps_ownership_guidance() -> None:
    """Preserve the in-file pointer that tells contributors where new tests belong."""
    text = LEGACY_CODEGEN_TEST.read_text(encoding="utf-8")

    assert OWNERSHIP_GUIDANCE in text, (
        "Keep the legacy ownership guidance in test_codegen.py so new codegen coverage "
        "is directed to focused sibling modules."
    )
