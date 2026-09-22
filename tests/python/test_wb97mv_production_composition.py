"""Production B97M composition ownership across the master/#935 cutover."""

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
def test_production_accepts_separated_and_weighted_b97m_lowering(
    monkeypatch: pytest.MonkeyPatch,
    spin: str,
    components: tuple[tuple[str, Fraction], ...],
) -> None:
    spec = FunctionalSpec("test", components, spin=spin, range_omega=Fraction(3, 10))
    monkeypatch.setattr(wb97mv_maple, "_wb97mv_module", _stop_at_source)
    with pytest.raises(_ReachedMaple):
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
    monkeypatch.setattr(wb97mv_maple, "_wb97mv_module", _stop_at_source)
    with pytest.raises(_ReachedMaple):
        wb97mv_maple.energy_expression(spec)


@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_host_codegen_has_one_maple_owner(spin: str) -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    program = """
import runpy
import sys
from fractions import Fraction
namespace = runpy.run_path('tools/generate_xc_cpu.py')
from vibeqc_compiler.xc.spec import FunctionalSpec
spec = FunctionalSpec('WB97M-V-test', (
    ('MGGA_X_WB97M_V', Fraction(1)), ('MGGA_C_WB97M_V', Fraction(1))),
    spin=sys.argv[1], range_omega=Fraction(3, 10))
outputs = ((), *((i,) for i in range(len(spec.features))))
build = namespace['build_roots']
assert build(spec, outputs)[2] == build(spec, outputs, production=True)[2]
"""
    subprocess.run(
        [sys.executable, "-S", "-c", program, spin],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
