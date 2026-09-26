"""Optional feature derivatives must not narrow the native energy domain."""

import numpy as np
import pytest
from vibeqc.nonlocal_runtime import NonlocalFixedGridPlan
from vibeqc_compiler.dft import nonlocal_energy_reference
from vibeqc_compiler.method import original_nonlocal_correlation


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
def test_energy_only_does_not_evaluate_unrequested_singular_feature_scales(
    variant: str,
) -> None:
    coordinates = np.zeros((1, 3))
    weights = np.ones(1)
    density = np.array([1e-70])
    gradient = np.zeros((1, 3))
    spec = original_nonlocal_correlation(variant)
    expected = nonlocal_energy_reference(coordinates, weights, density, gradient, spec)
    assert np.isfinite(expected) and expected != 0.0
    with NonlocalFixedGridPlan(spec, 1) as plan:
        result = plan.execute(coordinates, weights, density, gradient, features=False)
    assert result.energy == pytest.approx(expected, rel=2e-14, abs=0.0)
    assert result.vrho is None and result.vsigma is None
