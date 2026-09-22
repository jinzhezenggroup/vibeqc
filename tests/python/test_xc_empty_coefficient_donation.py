"""Empty XC tiles keep no-op semantics across packed owner donation."""

from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.coefficients import coefficient_program
from vibeqc_compiler.xc.native import NativeContractionProgram
from vibeqc_compiler.xc.potential import assemble_coefficients


@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
@pytest.mark.parametrize("family", ["lda", "gga", "mgga"])
@pytest.mark.parametrize("npoint", [0, 1, 5])
def test_packed_views_preserve_empty_and_nonempty_assembly(
    spin: str, family: str, npoint: int
) -> None:
    program = coefficient_program(spin, family)
    rng = np.random.default_rng(1056)
    packed = rng.normal(size=(len(program.roots), npoint))
    jets = rng.normal(size=(4, npoint, 3))
    weights = np.ones(npoint)
    expected = assemble_coefficients(jets, program.unpack(packed, npoint), weights)
    views = program.unpack_views(packed, npoint)
    actual = assemble_coefficients(jets, views, weights)
    np.testing.assert_array_equal(actual, expected)
    assert not views.owner.flags.writeable
    if npoint == 0:
        np.testing.assert_array_equal(actual, np.zeros_like(expected))


@pytest.fixture(scope="module")
def native_pbe(tmp_path_factory: pytest.TempPathFactory) -> NativeContractionProgram:
    return NativeContractionProgram(
        functional("PBE", spin="polarized"),
        compiler=CppCompilerAdapter(Path("c++")),
        cache=tmp_path_factory.mktemp("empty-coefficient-native"),
    )


@pytest.mark.parametrize("donate", [False, True])
@pytest.mark.parametrize("npoint", [0, 1, 13])
def test_native_packed_coefficients_accept_zero_point_donation(
    native_pbe: NativeContractionProgram, donate: bool, npoint: int
) -> None:
    features = np.zeros((7, npoint))
    features[0], features[1] = 0.7, 0.3
    features[2], features[3], features[4] = 0.02, 0.005, 0.03
    rows = native_pbe.scalar_values_packed(features, donate_coefficients=donate)
    v = rows.feature_gradient
    gradient = np.full((2, npoint, 3), 0.01)
    expected = coefficient_program("polarized", "gga").evaluate(gradient, v)
    owner = rows.coefficient_output_owner
    actual = native_pbe.coefficients.evaluate(gradient, v, output_owner=owner)
    for name, expected_value in expected.items():
        np.testing.assert_array_equal(actual[name], expected_value)
    result = assemble_coefficients(np.ones((4, npoint, 3)), actual, np.ones(npoint))
    assert result.shape == (2, 3, 3)
    if donate:
        assert actual.owner is owner
    if npoint == 0:
        np.testing.assert_array_equal(result, np.zeros((2, 3, 3)))


@pytest.mark.parametrize("npoint", [1, 7])
def test_nonempty_foreign_coefficient_views_remain_rejected(npoint: int) -> None:
    program = coefficient_program("polarized", "gga")
    views = program.unpack_views(np.ones((8, npoint)), npoint)
    foreign = views["rho"].copy()
    foreign.setflags(write=False)
    views["rho"] = foreign
    with pytest.raises(ValueError, match="borrowed XC coefficient"):
        assemble_coefficients(np.ones((4, npoint, 3)), views, np.ones(npoint))


@pytest.mark.parametrize("npoint", [1, 7])
def test_nonempty_disjoint_donation_still_fails(
    native_pbe: NativeContractionProgram, npoint: int
) -> None:
    features = np.zeros((7, npoint))
    features[0], features[1] = 0.7, 0.3
    rows = native_pbe.scalar_values_packed(features, donate_coefficients=True)
    with pytest.raises(ValueError, match="donation owner"):
        native_pbe.coefficients.evaluate(
            np.zeros((2, npoint, 3)),
            rows.feature_gradient,
            output_owner=np.empty((8, npoint)),
        )
