"""Retired/commented declarations must not certify a current tuning constant."""

from pathlib import Path

import pytest

from tools import check_direct_jk_tuning_inventory as checker


@pytest.mark.parametrize(
    "source",
    [
        "// constexpr unsigned kTest = 16;\nconstexpr unsigned kTest = 24;\n",
        "/* constexpr unsigned kTest = 16; */\nconstexpr unsigned kTest = 24;\n",
        "/*\nconstexpr unsigned kTest = 16;\n*/\n",
        'const char* text = "kTest = 16;";\n',
        "kTest = 16;\n",
        "constexpr unsigned kTest = 16;\nconstexpr unsigned kTest = 24;\n",
    ],
)
def test_noncurrent_or_ambiguous_constant_is_rejected(
    tmp_path: Path, source: str
) -> None:
    (tmp_path / "constants.hpp").write_text(source)
    errors: list[str] = []
    checker._require_constexpr(
        tmp_path,
        {"symbol": "kTest", "source": "constants.hpp", "expected_expression": "16"},
        "fixture",
        errors,
    )
    assert errors, "noncurrent text satisfied the constant declaration gate"


@pytest.mark.parametrize(
    "source",
    [
        "inline constexpr unsigned kTest = 16;\n",
        "constexpr\nunsigned kTest = 16;\n",
        "// constexpr unsigned kTest = 24;\nconstexpr unsigned kTest = 16;\n",
    ],
)
def test_current_constant_declaration_remains_valid(
    tmp_path: Path, source: str
) -> None:
    (tmp_path / "constants.hpp").write_text(source)
    errors: list[str] = []
    checker._require_constexpr(
        tmp_path,
        {"symbol": "kTest", "source": "constants.hpp", "expected_expression": "16"},
        "fixture",
        errors,
    )
    assert errors == []
