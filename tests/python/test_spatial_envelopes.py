"""Pre-evaluation bounds checked against the independent native AO evaluator."""

import numpy as np
import pytest
from vibeqc import Primitive, Shell
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.ao import jet_indices
from vibeqc_compiler.dft.envelopes import ao_region_envelopes, derivative_domain


@pytest.fixture(params=["cartesian", "spherical"])
def basis(request):
    shells = tuple(
        Shell(atom, angular, (Primitive(exponent, 1.0), Primitive(2 * exponent, -0.2)))
        for atom, exponent in ((0, 0.03), (1, 40.0))
        for angular in range(4)
    )
    with NativeAO(
        [("H", (0.1, -0.2, 0.3)), ("H", (-0.7, 0.4, 0.0))],
        basis=shells,
        representation=request.param,
    ) as result:
        yield result


def test_contracted_derivative_bounds_cover_corners_and_interior(basis):
    rng = np.random.default_rng(234)
    corners = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    for center, width in (
        ([0.1, -0.2, 0.3], 0.0),
        ([-0.7, 0.4, 0], 0.03),
        ([0, 0, 0], 2.0),
        ([17, -21, 12], 0.2),
    ):
        bounds = np.array([np.array(center) - width, np.array(center) + width])
        fractions = np.concatenate((corners, rng.random((31, 3))))
        points = bounds[0] + fractions * (bounds[1] - bounds[0])
        envelope = ao_region_envelopes(basis, bounds, jet_indices(3))
        actual = basis.evaluate(points, order=3)
        assert np.all(np.max(np.abs(actual), axis=1) <= envelope)
        assert not np.isnan(envelope).any()
        with pytest.raises(ValueError):
            envelope.setflags(write=True)


def test_value_node_does_not_hide_requested_derivative():
    with NativeAO(
        [("He", (0, 0, 0))], basis=(Shell(0, 1, (Primitive(0.5, 1.0),)),)
    ) as basis:
        domain = ((0, 0, 0), (1, 0, 0))
        bound = ao_region_envelopes(basis, np.zeros((2, 3)), domain)
        actual = basis.evaluate(np.zeros((1, 3)), order=1)
        assert actual[0, 0, 0] == 0
        assert abs(actual[1, 0, 0]) > 0.1
        assert bound[0, 0] < 1e-12
        assert bound[1, 0] >= abs(actual[1, 0, 0])
        assert np.max(bound[:, 0]) > 1e-12


def test_requested_domain_and_nested_regions(basis):
    domain = ((0, 1, 1), (0, 0, 0))
    assert derivative_domain(domain) == ((0, 0, 0), (0, 1, 1))
    inner = ao_region_envelopes(basis, [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], domain)
    outer = ao_region_envelopes(basis, [[-0.1, 0.0, 0.1], [0.4, 0.5, 0.6]], domain)
    assert np.all(outer >= inner)
    full = ao_region_envelopes(
        basis, [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], jet_indices(3)
    )
    np.testing.assert_array_equal(inner, full[[0, jet_indices(3).index((0, 1, 1))]])


def test_extreme_boxes_fail_conservatively_without_nan(basis):
    bound = ao_region_envelopes(basis, [[-1e308] * 3, [1e308] * 3], jet_indices(3))
    assert not np.isnan(bound).any()
    assert np.all(np.max(bound, axis=0) > 1e-12)


@pytest.mark.parametrize(
    "domain",
    [
        (),
        ((True, 0, 0),),
        ((-1, 0, 0),),
        ((4, 0, 0),),
        ((0, 0),),
        ((0, 0, 0), (0, 0, 0)),
    ],
)
def test_unsupported_derivative_requests_fail(domain):
    with pytest.raises(ValueError):
        derivative_domain(domain)


@pytest.mark.parametrize(
    "bounds",
    [[[1, 0, 0], [0, 1, 1]], [[0, 0, 0], [np.inf, 1, 1]], [[0, 0, 0], [np.nan, 1, 1]]],
)
def test_invalid_regions_fail(basis, bounds):
    with pytest.raises(ValueError):
        ao_region_envelopes(basis, bounds)
