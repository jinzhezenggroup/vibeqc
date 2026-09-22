"""Qualification for compiler-owned GFN2 primitive S/D/Q algebra."""

import math

import numpy as np
import pytest
from vibeqc_compiler.integral.gfn2_sdq import (
    GFN2_SDQ_COMPONENTS,
    build_gfn2_sdq_primitive_kernel,
    evaluate_gfn2_sdq_primitive,
)


def test_ss_sdq_matches_closed_form_ket_origin_moments() -> None:
    alpha, beta = 0.7, 1.3
    a = np.array([0.2, -0.3, 0.5])
    b = np.array([-0.4, 0.1, -0.2])
    kernel = build_gfn2_sdq_primitive_kernel((0, 0), ("", ""))
    actual = np.array(
        evaluate_gfn2_sdq_primitive(
            kernel,
            (alpha, beta),
            (tuple(a), tuple(b)),
        ).values
    )

    p = alpha + beta
    ab = a - b
    overlap = (math.pi / p) ** 1.5 * math.exp(-alpha * beta / p * np.dot(ab, ab))
    pb = alpha / p * ab
    raw = np.outer(pb, pb)
    raw[np.diag_indices(3)] += 1.0 / (2.0 * p)
    trace = np.trace(raw)
    quadrupole = np.array(
        [
            1.5 * raw[0, 0] - 0.5 * trace,
            1.5 * raw[0, 1],
            1.5 * raw[1, 1] - 0.5 * trace,
            1.5 * raw[0, 2],
            1.5 * raw[1, 2],
            1.5 * raw[2, 2] - 0.5 * trace,
        ]
    )
    expected = np.concatenate(([overlap], pb * overlap, quadrupole * overlap))
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-15)
    assert len(actual) == len(GFN2_SDQ_COMPONENTS)


@pytest.mark.parametrize(
    ("angular", "components"),
    [
        ((0, 0), ("", "")),
        ((1, 0), ("x", "")),
        ((0, 1), ("", "z")),
        ((1, 2), ("y", "xz")),
        ((2, 2), ("xy", "zz")),
    ],
)
def test_sdq_coordinate_gradients_match_finite_difference_and_translate(
    angular: tuple[int, int],
    components: tuple[str, str],
) -> None:
    exponents = (0.83, 1.17)
    centers = (
        (0.21, -0.34, 0.49),
        (-0.38, 0.16, -0.27),
    )
    kernel = build_gfn2_sdq_primitive_kernel(angular, components)
    evaluated = evaluate_gfn2_sdq_primitive(kernel, exponents, centers)

    for axis in range(3):
        np.testing.assert_allclose(
            np.array(evaluated.gradients[0][axis])
            + np.array(evaluated.gradients[1][axis]),
            0.0,
            rtol=0,
            atol=2e-12,
        )

    step = 2e-6
    for center in range(2):
        for axis in range(3):
            plus = [list(position) for position in centers]
            minus = [list(position) for position in centers]
            plus[center][axis] += step
            minus[center][axis] -= step
            plus_values = np.array(
                evaluate_gfn2_sdq_primitive(
                    kernel,
                    exponents,
                    (tuple(plus[0]), tuple(plus[1])),
                ).values
            )
            minus_values = np.array(
                evaluate_gfn2_sdq_primitive(
                    kernel,
                    exponents,
                    (tuple(minus[0]), tuple(minus[1])),
                ).values
            )
            finite_difference = (plus_values - minus_values) / (2.0 * step)
            np.testing.assert_allclose(
                np.array(evaluated.gradients[center][axis]),
                finite_difference,
                rtol=2e-8,
                atol=2e-9,
            )


def test_sdq_rejects_non_gfn2_public_shell_domain() -> None:
    with pytest.raises(ValueError, match="s/p/d"):
        build_gfn2_sdq_primitive_kernel((0, 3), ("", "xxx"))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_sdq_interpreter_rejects_nonphysical_exponents(bad: float) -> None:
    kernel = build_gfn2_sdq_primitive_kernel((0, 0), ("", ""))
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_gfn2_sdq_primitive(
            kernel, (bad, 1.0), ((0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
        )


def test_sdq_interpreter_rejects_nonfinite_centers() -> None:
    kernel = build_gfn2_sdq_primitive_kernel((0, 0), ("", ""))
    with pytest.raises(ValueError, match="centers must be finite"):
        evaluate_gfn2_sdq_primitive(
            kernel, (1.0, 1.0), ((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0))
        )
