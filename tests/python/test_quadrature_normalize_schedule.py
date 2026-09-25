"""Source/work-count gates for CUDA molecular-grid normalization."""

from vibeqc_compiler.xc.quadrature_cuda import emit_quadrature_cuda


def test_normalize_reuses_owner_exponential() -> None:
    """Reuse the owner's denominator term instead of evaluating exp twice."""
    source = emit_quadrature_cuda()
    normalize = source.split("__global__ void normalize_kernel", 1)[1].split(
        "inline void launch_partition", 1
    )[0]

    assert "const size_t owner = (begin + p) / per_atom;" in normalize
    assert "double total = 0.0, owner_term = 0.0;" in normalize
    assert "const double term = exp(logs[a * count + p] - maximum);" in normalize
    assert "total += term;" in normalize
    assert "if (a == owner) owner_term = term;" in normalize
    assert "weights[p] * (owner_term / total)" in normalize
    assert normalize.count("exp(") == 1

    # The stable log-sum-exp denominator still evaluates one exp per atom and
    # preserves the same accumulation order. The only removed work is the old
    # second owner exp, exactly once per molecular point.
    for atoms, points, old_exp, new_exp in (
        (48, 1_327_104, 65_028_096, 63_700_992),
        (96, 2_654_208, 257_458_176, 254_803_968),
    ):
        assert points * (atoms + 1) == old_exp
        assert points * atoms == new_exp
        assert old_exp - new_exp == points
