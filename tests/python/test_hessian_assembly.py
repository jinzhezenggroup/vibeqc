"""Checks for the explicit frozen-density Hessian component boundary."""

import typing

import numpy as np
import pytest

from tools.vibeqc_hessian import assemble_frozen_skeleton


def _component(seed: typing.Any, n: typing.Any = 2) -> typing.Any:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3, n, 3))


def test_skeleton_keeps_components_and_sums_without_extra_prefactors() -> None:
    components = [_component(seed) for seed in range(4)]
    report = assemble_frozen_skeleton(
        nuclear_repulsion=components[0],
        one_electron_skeleton=components[1],
        overlap_pulay_skeleton=components[2],
        two_electron_skeleton=components[3],
    )
    expected = sum(components)
    np.testing.assert_array_equal(report["skeleton"], expected)
    assert set(report["components"]) == {
        "nuclear_repulsion",
        "one_electron_skeleton",
        "overlap_pulay_skeleton",
        "two_electron_skeleton",
    }
    assert report["includes_response"] is False
    for value in report["components"].values():
        value[...] = 0.0
    assert np.any(report["skeleton"] != 0.0)


def test_skeleton_rejects_mismatched_or_nonfinite_components() -> None:
    base = _component(10)
    with pytest.raises(ValueError, match="identical shapes"):
        assemble_frozen_skeleton(
            nuclear_repulsion=base,
            one_electron_skeleton=_component(11, n=3),
            overlap_pulay_skeleton=base,
            two_electron_skeleton=base,
        )
    bad = base.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        assemble_frozen_skeleton(
            nuclear_repulsion=bad,
            one_electron_skeleton=base,
            overlap_pulay_skeleton=base,
            two_electron_skeleton=base,
        )


def test_skeleton_rejects_transposed_layout_instead_of_broadcasting() -> None:
    base = _component(12)
    with pytest.raises(ValueError, match="shape"):
        assemble_frozen_skeleton(
            nuclear_repulsion=base.reshape(6, 6),
            one_electron_skeleton=base,
            overlap_pulay_skeleton=base,
            two_electron_skeleton=base,
        )
