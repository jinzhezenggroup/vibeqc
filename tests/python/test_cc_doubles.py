"""Complete physical residuals, independent projections and expanded DAG."""

import numpy as np
import pytest

from tools.vibeqc_cc.doubles import build_ccsd_program
from tools.vibeqc_cc.oracle import DeterminantOracle, dense_feeds, random_case
from tools.vibeqc_tensor import Program, add, execute


@pytest.mark.parametrize(
    "o,v,seed", [(1, 1, 148), (1, 3, 149), (2, 2, 150), (2, 3, 151), (3, 2, 152)]
)
def test_full_residuals_against_determinants_and_expanded_shared(o, v, seed):
    arrays = random_case(o, v, seed)
    f, g, x, y = arrays
    reference = DeterminantOracle(f, g, o).evaluate_full(x, y)
    outputs = []
    for form in ("expanded", "shared", "optimized"):
        p = build_ccsd_program(o, v, form=form)
        value = execute(Program.loads(p.dumps()), dense_feeds(*arrays)).outputs
        for key, ref in zip(
            ("correlation_energy", "singles_residual", "doubles_residual"), reference
        ):
            np.testing.assert_allclose(value[key], ref, atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(
            value["doubles_residual"],
            value["doubles_residual"].transpose(1, 0, 3, 2),
            atol=1e-12,
            rtol=0,
        )
        outputs.append(value)
    for key in outputs[0]:
        for value in outputs[1:]:
            np.testing.assert_allclose(
                value[key], outputs[0][key], atol=1e-11, rtol=1e-10, err_msg=key
            )


def test_zero_amplitudes_and_mutant_pair_and_ladder_terms():
    f, g, x, y = random_case()
    p = build_ccsd_program(2, 2)
    zero = execute(p, dense_feeds(f, g, x * 0, y * 0)).outputs
    np.testing.assert_array_equal(
        zero["doubles_residual"], g[:2, 2:, :2, 2:].transpose(0, 2, 1, 3)
    )
    ref = DeterminantOracle(f, g, 2).evaluate_full(x, y)[2]
    for term in (
        "D04_oo_ladder",
        "D05_vv_ladder",
        "D08_ring",
        "D09_exchange",
        "D10_cross",
    ):
        mutant = Program(
            {
                "bad": add(
                    p.outputs["doubles_residual"], p.outputs[term], coefficients=(1, -1)
                )
            }
        )
        bad = execute(mutant, dense_feeds(f, g, x, y)).outputs["bad"]
        assert np.max(np.abs(bad - ref)) > 1e-7


def test_fock_doubles_has_all_diagonal_and_offdiagonal_terms():
    f, g, x, y = random_case()
    expected = DeterminantOracle(f, g * 0, 2).evaluate_full(x, y)[2]
    result = execute(build_ccsd_program(2, 2), dense_feeds(f, g * 0, x, y)).outputs[
        "doubles_residual"
    ]
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0)
    assert np.max(np.abs(result)) > 1e-3
