"""Independent Libxc oracle checks for split global-hybrid production E/vxc."""

import numpy as np
import pytest
from vibeqc_compiler.xc.boundary import (
    bulk_feature_names,
    semilocal_boundary_probes,
)

from tools.generate_libxc_boundary_reference import _configure, evaluate_reference
from tools.libxc_split_hybrid import evaluate_split_global_hybrid_production


_METHODS = {
    "M06-2X": (450, 236),
    "MN15": (268, 269),
}


@pytest.mark.parametrize("name", tuple(_METHODS))
def test_split_hybrid_production_matches_independent_libxc_7_boundary(
    name: str,
) -> None:
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("split-hybrid production oracle requires pinned Libxc 7.0.0")

    library = libxc._itrf
    _configure(library)
    features = bulk_feature_names("mgga", "polarized")
    exchange_id, correlation_id = _METHODS[name]
    records = (
        {"name": f"{name}-X", "id": exchange_id, "family": "mgga"},
        {"name": f"{name}-C", "id": correlation_id, "family": "mgga"},
    )

    for probe in semilocal_boundary_probes(features, spin="polarized"):
        expected = np.zeros(1 + len(features), dtype=np.float64)
        for record in records:
            reference = evaluate_reference(
                library, record, "polarized", probe.feature_names, probe.values
            )
            assert reference["oracle_finite"], (name, probe.label, reference)
            expected += np.asarray(reference["expected"], dtype=np.float64)

        actual_features, actual = evaluate_split_global_hybrid_production(
            name, probe.values
        )
        assert actual_features == features
        assert np.isfinite(actual).all(), (name, probe.label, actual)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=3.0e-8,
            atol=3.0e-10,
            err_msg=f"{name} production mismatch at {probe.label}",
        )
