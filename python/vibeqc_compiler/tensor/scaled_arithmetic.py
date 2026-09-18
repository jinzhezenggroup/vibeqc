"""Range-safe evaluation of (a*b - c*d)/(e*f), TensorIR arithmetic v1.

Products are formed only from frexp mantissas, and exponents remain integers
until the final ldexp. Error-free product residuals preserve cancellation that
would be lost by rounding both products before subtracting them. This is not
an arbitrary-precision oracle or a correctly-rounded rational implementation.
"""

from __future__ import annotations

import numpy as np


def _product(left, right):
    """Dekker product/residual on normalized FP32/FP64 mantissas."""
    precision = np.finfo(left.dtype).nmant + 1
    splitter = left.dtype.type((1 << ((precision + 1) // 2)) + 1)
    product = left * right
    split_left, split_right = splitter * left, splitter * right
    high_left = split_left - (split_left - left)
    high_right = split_right - (split_right - right)
    low_left, low_right = left - high_left, right - high_right
    error = (high_left * high_right - product) + high_left * low_right
    error = (error + low_left * high_right) + low_left * low_right
    return product, error


def scaled_bilinear_value(a, b, c, d, e, f):
    """Evaluate one fused bilinear quotient in the operands' real dtype.

    Inputs are finite, equally shaped FP32 or FP64 arrays. Zero denominators
    always fail, even for zero numerators. True final overflow is left to the
    caller's existing finite-value/error boundary. Final subnormals/underflow
    follow the dtype's ordinary round-to-nearest semantics.
    """
    if np.any(e == 0) or np.any(f == 0):
        raise ValueError("tensor division by zero (scaled_bilinear)")
    (ma, ea), (mb, eb), (mc, ec), (md, ed), (me, ee), (mf, ef) = (
        np.frexp(value) for value in (a, b, c, d, e, f)
    )
    p, pe = _product(ma, mb)
    q, qe = _product(mc, md)
    ep, eq = ea + eb, ec + ed
    # A zero product must not dominate the alignment exponent of a nonzero
    # product (including a subnormal times a subnormal).
    exponent = np.where(p == 0, eq, np.where(q == 0, ep, np.maximum(ep, eq)))
    # Beyond 2p+4 bits the smaller product cannot participate in cancellation
    # or affect the stated few-ulp accuracy. Drop it explicitly rather than
    # underflowing intermediate ldexp operations under a caller's errstate.
    limit = 2 * (np.finfo(ma.dtype).nmant + 1) + 4
    dp, dq = ep - exponent, eq - exponent
    p = np.where(dp < -limit, 0, np.ldexp(p, np.maximum(dp, -limit)))
    pe = np.where(dp < -limit, 0, np.ldexp(pe, np.maximum(dp, -limit)))
    q = np.where(dq < -limit, 0, np.ldexp(q, np.maximum(dq, -limit)))
    qe = np.where(dq < -limit, 0, np.ldexp(qe, np.maximum(dq, -limit)))
    # TwoSum(p, -q), including the exact product residuals.
    difference = p - q
    tail = difference - p
    error = (p - (difference - tail)) - (q + tail)
    numerator = difference + ((pe - qe) + error)
    return np.ldexp(numerator / (me * mf), exponent - ee - ef)


def emit_scaled_bilinear(prefix: str) -> str:
    """Emit the same FP64 arithmetic contract, with exact FMA residuals.

    These mathematical helpers belong to the compiler, not runtime resource
    templates. Explicit round-to-nearest intrinsics prohibit reassociation;
    the primitive is the error boundary, not its normalized intermediates.
    """
    return f"""
__device__ inline double {prefix}scaled_bilinear(double a, double b, double c,
    double d, double e, double f, int* error, int node) {{
    if (e == 0.0 || f == 0.0) {{
        atomicCAS(error, 0, -(node + 1));
        return 0.0;
    }}
    int ea, eb, ec, ed, ee, ef;
    double ma = frexp(a, &ea), mb = frexp(b, &eb);
    double mc = frexp(c, &ec), md = frexp(d, &ed);
    double me = frexp(e, &ee), mf = frexp(f, &ef);
    double p = __dmul_rn(ma, mb), q = __dmul_rn(mc, md);
    double pe = __fma_rn(ma, mb, -p), qe = __fma_rn(mc, md, -q);
    int ep = ea + eb, eq = ec + ed;
    int exponent = p == 0.0 ? eq : (q == 0.0 ? ep : (ep > eq ? ep : eq));
    int dp = ep - exponent, dq = eq - exponent;
    if (dp < -110) {{ p = 0.0; pe = 0.0; }}
    else {{ p = scalbn(p, dp); pe = scalbn(pe, dp); }}
    if (dq < -110) {{ q = 0.0; qe = 0.0; }}
    else {{ q = scalbn(q, dq); qe = scalbn(qe, dq); }}
    double difference = __dsub_rn(p, q);
    double tail = __dsub_rn(difference, p);
    double residual = __dsub_rn(__dsub_rn(p, __dsub_rn(difference, tail)),
                               __dadd_rn(q, tail));
    double numerator = __dadd_rn(difference, __dadd_rn(__dsub_rn(pe, qe), residual));
    double ratio = __ddiv_rn(numerator, __dmul_rn(me, mf));
    return finite(scalbn(ratio, exponent - ee - ef), error, node);
}}
"""
