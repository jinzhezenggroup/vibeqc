"""Production B97M composition admission; no numerical oracle is stubbed."""

from fractions import Fraction

import pytest
from vibeqc_compiler.xc import wb97mv_maple
from vibeqc_compiler.xc.spec import FunctionalSpec

X = "MGGA_X_WB97M_V"
C = "MGGA_C_WB97M_V"


class _ReachedMaple(Exception):
    """Sentinel proving whether source loading was reached."""


def _stop_at_source(_omega: Fraction) -> None:
    raise _ReachedMaple


@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
@pytest.mark.parametrize(
    "components",
    (
        ((X, Fraction(1)),),
        ((C, Fraction(1)),),
        ((X, Fraction(2)), (C, Fraction(1))),
        ((X, Fraction(1)), (C, Fraction(-1))),
    ),
)
def test_production_rejects_noncanonical_b97m_before_source_loading(
    monkeypatch: pytest.MonkeyPatch,
    spin: str,
    components: tuple[tuple[str, Fraction], ...],
) -> None:
    spec = FunctionalSpec("test", components, spin=spin, range_omega=Fraction(3, 10))
    monkeypatch.setattr(wb97mv_maple, "_module", _stop_at_source)
    with pytest.raises(ValueError, match="canonical unit-weight"):
        wb97mv_maple.energy_expression(spec)


@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_canonical_b97m_reaches_unchanged_maple_lowering(
    monkeypatch: pytest.MonkeyPatch, spin: str
) -> None:
    spec = FunctionalSpec(
        "test",
        ((X, Fraction(1)), (C, Fraction(1))),
        spin=spin,
        range_omega=Fraction(3, 10),
    )
    monkeypatch.setattr(wb97mv_maple, "_module", _stop_at_source)
    with pytest.raises(_ReachedMaple):
        wb97mv_maple.energy_expression(spec)
