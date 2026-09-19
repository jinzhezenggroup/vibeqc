# MB16-43/06 coordinates: grimme-lab/mstore a9070de0..., Apache-2.0.
import math

import pytest

from tools.vibeqc_gcp.reference import evaluate_r2scan3c_gcp

ZS = (5, 7, 1, 8, 5, 1, 13, 1, 5, 12, 1, 1, 1, 1, 6, 1)
XYZ = (
    (0.10912945825730, 1.64180252123600, 0.27838149792131),
    (-2.30085163837888, 0.87765138232225, -0.60457694150897),
    (2.78083551168063, 4.95421363506113, 0.40788634984219),
    (-5.36229602768251, -7.29510945515334, 0.06097106408867),
    (2.13846114572058, -0.99012126457352, 0.93647189687052),
    (0.09330150731888, -2.75648066796634, -3.70294675694565),
    (-1.52684105316140, -2.44981814860506, -1.02727325811774),
    (-0.45240334635443, 5.86105501765814, 0.30815308772432),
    (-3.95419048213910, -5.52061943693205, -0.31702321028260),
    (2.68706169520082, -0.13577304635533, -3.57041492458512),
    (-3.79914135008731, 2.06429808651079, -0.77285245656187),
    (0.89693752015341, 4.58640300917890, 3.09718012019731),
    (2.76317093138142, -0.62928000132252, 3.08807601371151),
    (1.00075543259914, -3.11885279872042, 1.08659460804098),
    (0.86969979951508, 4.43363816376984, 1.02355776570620),
    (4.05637089597643, -1.52300699610852, -0.29218485610105),
)


def test_upstream_mb16_43_06_energy_fixture():
    result = evaluate_r2scan3c_gcp(ZS, XYZ)
    assert result.status == "ok"
    assert result.energy == pytest.approx(0.0113040952, abs=5.0e-8)
    assert result.provenance.source == "dftd3/simple-dftd3"
    assert max(abs(sum(row[k] for row in result.gradient)) for k in range(3)) < 2.0e-14
    assert result.forces[0][0] == -result.gradient[0][0]


def test_gcp_gradient_matches_all_cartesian_finite_differences():
    result = evaluate_r2scan3c_gcp(ZS, XYZ)
    assert math.isfinite(result.energy)
    for h in (1.0e-4, 3.0e-5):
        for atom in range(len(ZS)):
            for axis in range(3):
                plus = [list(r) for r in XYZ]
                minus = [list(r) for r in XYZ]
                plus[atom][axis] += h
                minus[atom][axis] -= h
                ep = evaluate_r2scan3c_gcp(ZS, plus).energy
                em = evaluate_r2scan3c_gcp(ZS, minus).energy
                fd = (ep - em) / (2 * h)
                assert fd == pytest.approx(result.gradient[atom][axis], abs=2.0e-8)


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
def test_single_atom_gcp_rejects_nonfinite_geometry(coordinate):
    with pytest.raises(ValueError, match="finite"):
        evaluate_r2scan3c_gcp((1,), ((coordinate, 0.0, 0.0),))
