"""Later finite samples must not hide an earlier invalid seed measurement."""

from copy import deepcopy

import pytest
from test_xtb_guess_validation import _validate_measurements


def _rows() -> list[dict[str, str]]:
    return [
        {
            "case": "water",
            "repeat": str(index),
            "baseline_seconds": "1.0",
            "candidate_total_seconds": "1.25",
            "seed_seconds": "0.125",
            "energy_difference": "0.0",
            "candidate_residual": "1e-10",
        }
        for index in range(9)
    ]


@pytest.mark.parametrize("row_index", range(9))
@pytest.mark.parametrize("invalid", ["nan", "inf", "-inf"])
def test_every_sample_is_checked(row_index: int, invalid: str) -> None:
    rows = _rows()
    rows[row_index]["seed_seconds"] = invalid
    before = deepcopy(rows)
    with pytest.raises(AssertionError, match=f"row {row_index}, field seed_seconds"):
        _validate_measurements(rows)
    assert rows == before


def test_finite_rows_are_preserved() -> None:
    rows = _rows()
    before = deepcopy(rows)
    _validate_measurements(rows)
    assert rows == before
