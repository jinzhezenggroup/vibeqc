"""Do not advertise hybrid dry runs before the CLI exposes explicit grids."""

import pytest
from vibeqc.__main__ import parser


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
