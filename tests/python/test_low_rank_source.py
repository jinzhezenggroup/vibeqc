"""Raw column and target recovery gates against pinned independent AO fixtures."""

import numpy as np
import pytest

from tools.vibeqc_posthf.coulomb_columns import CoulombColumns
from tools.vibeqc_posthf.fixtures import load_fixture, source_arguments
from tools.vibeqc_posthf.low_rank import IncrementalCholesky
from tools.vibeqc_posthf.sources import NativeSource


def source_for(name):
    metadata, arrays = load_fixture(name)
    try:
        source = NativeSource(**source_arguments(metadata))
    except (OSError, FileNotFoundError, AttributeError) as error:
        pytest.skip(f"native post-HF library unavailable: {error}")
    return source, arrays


def packed_reference(space, ao):
    """Independent explicit four-index projection for these tiny fixtures only."""
    out = np.empty((space.size, space.size))
    for i in range(space.size):
        mu, nu = space.pair(i)
        for j in range(space.size):
            rho, sigma = space.pair(j)
            multiplicity = (1 if mu == nu else 2) * (1 if rho == sigma else 2)
            out[i, j] = np.sqrt(multiplicity) * ao[mu, nu, rho, sigma]
    return out


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
def test_native_columns_and_full_target_recovery(name):
    source, arrays = source_for(name)
    with source:
        columns = CoulombColumns(source)
        reference = packed_reference(columns.space, arrays["ao"])
        size = columns.space.size
        for begin in range(0, size, 5):
            count = min(5, size - begin)
            np.testing.assert_allclose(
                columns.diagonal(begin, count),
                reference.diagonal()[begin : begin + count],
                atol=1e-11,
                rtol=1e-10,
            )
            for pivot in (0, size // 2, size - 1):
                np.testing.assert_allclose(
                    columns.column(pivot, begin, count),
                    reference[begin : begin + count, pivot],
                    atol=1e-11,
                    rtol=1e-10,
                )
        with IncrementalCholesky(columns, rank_capacity=size, pair_tile=5) as plan:
            plan.refine(1e-11, maximum_rank=min(2, size))
            old = plan.factor_tile(0, 1)
            outcome = plan.refine(1e-11)
            assert outcome.status == "threshold_met"
            np.testing.assert_array_equal(old, plan.factor_tile(0, 1))
            factors = np.concatenate([plan.factor_tile(i, 1) for i in range(plan.rank)])
            np.testing.assert_allclose(
                factors.T @ factors, reference, atol=3e-11, rtol=1e-10
            )
        assert columns.column(0, size, 0).shape == (0,)
        with pytest.raises(ValueError):
            columns.column(0, size, 1)
    with pytest.raises(RuntimeError, match="closed"):
        columns.diagonal(0, 1)


def test_spherical_f_partial_pair_rows():
    source, arrays = source_for("f_heh")
    with source:
        columns = CoulombColumns(source)
        reference = packed_reference(columns.space, arrays["ao"])
        size = columns.space.size
        for pivot in (0, size - 1):
            for begin, count in ((0, 3), (size // 2, 4), (size - 2, 2)):
                np.testing.assert_allclose(
                    columns.column(pivot, begin, count),
                    reference[begin : begin + count, pivot],
                    atol=1e-11,
                    rtol=1e-10,
                )
