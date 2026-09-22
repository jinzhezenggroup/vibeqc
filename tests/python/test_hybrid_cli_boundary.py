"""Do not advertise hybrid dry runs before the CLI exposes explicit grids."""

import pytest
from vibeqc import _generated_methods
from vibeqc.__main__ import _public_method_rows, parser


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks"))
def test_resource_cli_does_not_advertise_unusable_hybrid(method: str) -> None:
    # Hybrid resource planning is available through KsOptions(grid=...) in
    # Python. The CLI has no grid control, so accepting these names would only
    # defer an unavoidable failure until scientific model resolution.
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["resources", "unused.xyz", "--method", method])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "method", ("rhf", "uhf", "lda-rks", "pbe-rks", "lda-uks", "pbe-uks")
)
def test_resource_cli_preserves_qualified_method_choices(method: str) -> None:
    arguments = parser().parse_args(["resources", "unused.xyz", "--method", method])
    assert arguments.method == method


def test_method_catalog_is_generated_from_public_manifest() -> None:
    rows = _public_method_rows()
    assert [row["name"] for row in rows] == list(_generated_methods.METHOD_METADATA)
    by_name = {row["name"]: row for row in rows}
    assert by_name["pbe0-rks"]["properties"] == ("energy",)
    assert by_name["b3lyp-uks"]["status"] == "available"
    assert by_name["wb97m-v"]["status"] == "unavailable"


def test_methods_command_is_publicly_parseable() -> None:
    assert parser().parse_args(["methods"]).command == "methods"
    args = parser().parse_args(["methods", "--json"])
    assert args.command == "methods"
    assert args.json
