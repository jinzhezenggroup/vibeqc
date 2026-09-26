"""Range-safe evaluation of (a*b - c*d)/(e*f), TensorIR arithmetic v1.

Products are formed only from frexp mantissas, and exponents remain integers
until the final ldexp. Error-free product residuals preserve cancellation that
would be lost by rounding both products before subtracting them. This is not
an arbitrary-precision oracle or a correctly-rounded rational implementation.
"""

from __future__ import annotations

import typing

import numpy as np

from .cuda_dtype import scalar_type


def _product(left: typing.Any, right: typing.Any) -> typing.Any:
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


def scaled_bilinear_value(
    a: typing.Any,
    b: typing.Any,
    c: typing.Any,
    d: typing.Any,
    e: typing.Any,
    f: typing.Any,
) -> typing.Any:
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


def emit_scaled_bilinear(prefix: str, *, dtype: str = "float64") -> str:
    """Emit the selected dtype arithmetic contract, with exact FMA residuals.

    These mathematical helpers belong to the compiler, not runtime resource
    templates. Explicit round-to-nearest intrinsics prohibit reassociation;
    the primitive is the error boundary, not its normalized intermediates.
    """
    scalar = scalar_type(dtype)
    ty, suffix = scalar.ctype, scalar.suffix
    add, sub, mul, div = (scalar.intrinsic(op) for op in ("add", "sub", "mul", "div"))
    fma = "__fmaf_rn" if dtype == "float32" else "__fma_rn"
    limit = 2 * scalar.precision + 4
    return f"""
__device__ inline {ty} {prefix}scaled_bilinear({ty} a, {ty} b, {ty} c,
    {ty} d, {ty} e, {ty} f, int* error, int node) {{
    if (e == {scalar.zero} || f == {scalar.zero}) {{
        atomicCAS(error, 0, -(node + 1));
        return {scalar.zero};
    }}
    int ea, eb, ec, ed, ee, ef;
    {ty} ma = frexp{suffix}(a, &ea), mb = frexp{suffix}(b, &eb);
    {ty} mc = frexp{suffix}(c, &ec), md = frexp{suffix}(d, &ed);
    {ty} me = frexp{suffix}(e, &ee), mf = frexp{suffix}(f, &ef);
    {ty} p = {mul}(ma, mb), q = {mul}(mc, md);
    {ty} pe = {fma}(ma, mb, -p), qe = {fma}(mc, md, -q);
    int ep = ea + eb, eq = ec + ed;
    int exponent = p == {scalar.zero} ? eq : (q == {scalar.zero} ? ep : (ep > eq ? ep : eq));
    int dp = ep - exponent, dq = eq - exponent;
    if (dp < -{limit}) {{ p = {scalar.zero}; pe = {scalar.zero}; }}
    else {{ p = scalbn{suffix}(p, dp); pe = scalbn{suffix}(pe, dp); }}
    if (dq < -{limit}) {{ q = {scalar.zero}; qe = {scalar.zero}; }}
    else {{ q = scalbn{suffix}(q, dq); qe = scalbn{suffix}(qe, dq); }}
    {ty} difference = {sub}(p, q);
    {ty} tail = {sub}(difference, p);
    {ty} residual = {sub}({sub}(p, {sub}(difference, tail)),
                               {add}(q, tail));
    {ty} numerator = {add}(difference, {add}({sub}(pe, qe), residual));
    {ty} ratio = {div}(numerator, {mul}(me, mf));
    return finite(scalbn{suffix}(ratio, exponent - ee - ef), error, node);
}}
"""
