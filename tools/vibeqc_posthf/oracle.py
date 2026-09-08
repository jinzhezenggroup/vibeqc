"""Small-only dense AO-to-MO oracle; never used by streaming providers."""

import numpy as np


def dense_ao_to_mo(eri, coefficients, *, maximum_n=12):
    """Transform all four chemists' slots in four independently visible stages.

    Full N**4 AO/MO tensors are allowed only behind this explicit small-system
    guard. No optimized path selection or provider implementation is called.
    """
    c = np.asarray(coefficients)
    if np.iscomplexobj(c) or c.ndim != 2 or max(c.shape) > min(maximum_n, 12):
        raise ValueError("dense oracle supports real systems with at most 12 AOs/MOs")
    g = np.asarray(eri)
    n = c.shape[0]
    if g.shape != (n,) * 4 or not np.isfinite(g).all():
        raise ValueError("invalid dense chemists' AO tensor")
    one = np.einsum("uvwx,up->pvwx", g, c, optimize=False)
    two = np.einsum("pvwx,vq->pqwx", one, c, optimize=False)
    three = np.einsum("pqwx,wr->pqrx", two, c, optimize=False)
    return np.einsum("pqrx,xs->pqrs", three, c, optimize=False)
