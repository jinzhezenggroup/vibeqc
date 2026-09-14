"""Audited reference for the closed-shell CCSD(T) triples correction (CG11 slice A).

This module transcribes the non-iterative (T) energy of PySCF 2.14.0
``pyscf/cc/ccsd_t_slow.py`` (``kernel`` + ``r3``), Apache-2.0, Copyright
2014-2020 The PySCF Developers; author Qiming Sun.  It is a pure-CPU /
NumPy-only reference: no PySCF runtime import, no CUDA, no public Calculator
registration.  The pinned upstream files, version and SHA-256 identities are
recorded in :data:`tools.vibeqc_cc.source_manifest` and ``NOTICE``.

Mathematical contract (faithful transcription, nothing re-derived)
------------------------------------------------------------------

Inputs are all-MO, real, Hartree, occupied then virtual (``docs/posthf.md``):

    ovvv = (i, a, b, c)          eris.get_ovvv()             (i occupied; a,b,c virtual)
    ovoo = (i, a, j, m)          eris.ovoo
    ovov = (i, a, j, b)          eris.ovov    == (ia|jb)
    fov  = (i, a)                Fock[o, v];  fvo = fov.T == F[v, o]
    t1   = (i, a)                singles
    t2   = (i, j, a, b)          pair-symmetric doubles
    eps_o, eps_v                 orbital energies

With ``t2T = t2.transpose(2,3,0,1)``, ``eris_vvov = ovvv.transpose(1,3,0,2)``,
``eris_vooo = ovoo.transpose(1,0,2,3)``, ``eris_vvoo = ovov.transpose(1,3,0,2)``
and ``fvo = fov.T``, the label seeds for a fixed virtual triple (a,b,c) are::

    W(a,b,c)[i,j,k] = sum_f vvov[a,b,i,f] * t2T[c,f,k,j]
                    - sum_m vooo[a,i,j,m] * t2T[b,c,m,k]          (2 terms)
    V(a,b,c)[i,j,k] = vvoo[a,b,i,j] * t1T[c,k]
                    + t2T[a,b,i,j] * fvo[c,k]                     (2 terms)

The angular-momentum projection (note the six signed permutations of the three
occupied indices; this is the ``r3`` of ccsd_t_slow, JCP 94, 442 (1991)) is::

    r3(w) = 4*w + w.transpose(1,2,0) + w.transpose(2,0,1)
             - 2*w.transpose(2,1,0) - 2*w.transpose(0,2,1) - 2*w.transpose(1,0,2)

The (T) amplitude energy correction is the triangular virtual sum (a >= b >= c)
with the exact degeneracy 6 for a==b==c, 2 for a==b or b==c (applied to the
*distributed* denominator, not the whole block), times the overall factor 2::

    E_T = 2 * sum_{a>=b>=c} [ 36-term contraction of the six W/V virtual
          permutations against r3(W + 0.5 V)/d3 ]
    d3[i,j,k] = eps_o[i]+eps_o[j]+eps_o[k] - eps_v[a]-eps_v[b]-eps_v[c]
    d3 *= 6 if a==c else (2 if a==b or b==c else 1)

The 36-term contraction is the literal ``SLOW_TABLE`` below: six rows (one per
virtual permutation ``z`` of r3(W+0.5V)) and six columns per row pairing a
virtual permutation ``w`` of the left factor with an occupied permutation of
``w`` (``ijk``/``ikj``/... = index strings copied from ccsd_t_slow.py).

Two independent numerical engines live here:

* :func:`triples_energy` -- the auditable triangular reference above.
* :func:`triples_fullsum` -- an independent full-sum oracle: no triangular
  shortcut, no 6/2 pre-factor, nested explicit loops over *all* ordered
  (a,b,c).  It replaces the hand-written 6x6 table by the S3 pair rule
  (see that function's docstring) and is proven equal to the reference in
  ``tests/python/test_cc_triples.py`` to ~1e-13 on random inputs.

A tiny-TensorIR lowering (:func:`build_triples_program`) expresses the same
inventory as ``einsum``/``transpose``/``gather``/``reduce_sum``/``divide``/
``broadcast``/``add`` nodes so that future AD over t1/t2/eps/g is possible
(issue steps 3 and 11); it is numerically identical to :func:`triples_energy`.
"""

from fractions import Fraction
from itertools import permutations

import numpy as np
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    divide,
    einsum,
    execute,
    gather,
    input_tensor,
    reduce_sum,
    transpose,
)

from tools.vibeqc_validation.schema import canonical_hash

# ---------------------------------------------------------------------------
# Auditable rational inventory in doubles.DEFINITIONS style:
# (factor, einsum, operands), with fixed virtual indices a,b,c. Operand names
# refer to the transposed views in _views, exactly as in the pinned slow code.
# Tests compare each inventory term with explicit source-index loops and the
# numerical seeds so its provenance hash cannot silently certify a typo.
# ---------------------------------------------------------------------------

# W = W1 + W2  (exact rational coefficients, no implicit sign absorption)
W_TERMS = (
    (1, "if,fkj->ijk", ("vvov", "t2T")),  # + sum_f vvov[a,b,i,f] t2T[c,f,k,j]
    (-1, "ijm,mk->ijk", ("vooo", "t2T")),  # - sum_m vooo[a,i,j,m] t2T[b,c,m,k]
)

# V = V1 + V2 ; the r3 argument is W + (1/2) V
V_TERMS = (
    (1, "ij,k->ijk", ("vvoo", "t1T")),  # + vvoo[a,b,i,j] t1T[c,k]
    (1, "ij,k->ijk", ("t2T", "fvo")),  # + t2T[a,b,i,j] fvo[c,k]
)

# r3(w) = sum_k coeff[k] * w.transpose(perm[k]); coefficients (4,1,1,-2,-2,-2)
R3 = (
    (4, (0, 1, 2)),
    (1, (1, 2, 0)),
    (1, (2, 0, 1)),
    (-2, (2, 1, 0)),
    (-2, (0, 2, 1)),
    (-2, (1, 0, 2)),
)

# Degeneracy of the triangular virtual domain (factor into the denominator).
DEGENERACY_DESCRIPTION = "6 for a==b==c; 2 for a==b or b==c; else 1 (a>=b>=c)"

# Virtual-permutation labels map a base triple (a,b,c) to the re-ordered triple.
VP = {
    "abc": (0, 1, 2),
    "acb": (0, 2, 1),
    "bac": (1, 0, 2),
    "bca": (1, 2, 0),
    "cab": (2, 0, 1),
    "cba": (2, 1, 0),
}

# Occupied-permutation index strings map to numpy transpose axes so that
# "w[wlbl].transpose(OP[ost])" equals the source einsum "ost,ijk" reading of w
# against z at (i,j,k).  For a string s, OP[s] = (position of 'i', position of
# 'j', position of 'k') in s.
OP = {
    "ijk": (0, 1, 2),
    "ikj": (0, 2, 1),
    "jik": (1, 0, 2),
    "jki": (2, 0, 1),
    "kij": (1, 2, 0),
    "kji": (2, 1, 0),
}

# Literal 36-entry contraction table of ccsd_t_slow.py: one row per virtual
# permutation z of r3(W+0.5V); each entry is (w-virtual-permutation,
# occupied-permutation of w).  "ijk" is the identity occupied pairing.
SLOW_TABLE = {
    "abc": [
        ("abc", "ijk"),
        ("acb", "ikj"),
        ("bac", "jik"),
        ("bca", "jki"),
        ("cab", "kij"),
        ("cba", "kji"),
    ],
    "acb": [
        ("acb", "ijk"),
        ("abc", "ikj"),
        ("cab", "jik"),
        ("cba", "jki"),
        ("bac", "kij"),
        ("bca", "kji"),
    ],
    "bac": [
        ("bac", "ijk"),
        ("bca", "ikj"),
        ("abc", "jik"),
        ("acb", "jki"),
        ("cba", "kij"),
        ("cab", "kji"),
    ],
    "bca": [
        ("bca", "ijk"),
        ("bac", "ikj"),
        ("cba", "jik"),
        ("cab", "jki"),
        ("abc", "kij"),
        ("acb", "kji"),
    ],
    "cab": [
        ("cab", "ijk"),
        ("cba", "ikj"),
        ("acb", "jik"),
        ("abc", "jki"),
        ("bca", "kij"),
        ("bac", "kji"),
    ],
    "cba": [
        ("cba", "ijk"),
        ("cab", "ikj"),
        ("bca", "jik"),
        ("bac", "jki"),
        ("acb", "kij"),
        ("abc", "kji"),
    ],
}

_LABELS = ("abc", "acb", "bac", "bca", "cab", "cba")

VERSION = 1


def _inventory():
    """Canonical record of the audited coefficient/symmetry content."""
    return {
        "method": "RCCSD(T)",
        "slice": "A",
        "inventory_version": VERSION,
        "upstream": "PySCF 2.14.0 ccsd_t_slow.py kernel + r3",
        "w_terms": sorted(W_TERMS),
        "v_terms": sorted(V_TERMS),
        "r3": sorted(R3),
        "degeneracy": DEGENERACY_DESCRIPTION,
        "virtual_domain": "triangular a>=b>=c",
        "overall_factor": 2,
        "contraction_table": {k: SLOW_TABLE[k] for k in _LABELS},
    }


INVENTORY_HASH = canonical_hash(_inventory())


def r3(w):
    """Angular projection r3(w); six signed occupied-axis permutations."""
    return (
        4 * w
        + w.transpose(1, 2, 0)
        + w.transpose(2, 0, 1)
        - 2 * w.transpose(2, 1, 0)
        - 2 * w.transpose(0, 2, 1)
        - 2 * w.transpose(1, 0, 2)
    )


def _validate(nocc, nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v):
    arrays = {
        "ovvv": ovvv,
        "ovoo": ovoo,
        "ovov": ovov,
        "fov": fov,
        "t1": t1,
        "t2": t2,
        "eps_o": eps_o,
        "eps_v": eps_v,
    }
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype != np.float64:
            raise ValueError(f"{name} must be a float64 numpy array")
        # NaN comparisons bypass the denominator guard; infinities can hide
        # invalid reference data behind an apparently finite zero correction.
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
    if any(type(n) is not int or n < 1 for n in (nocc, nvir)):
        raise ValueError("triples require nonempty occupied and virtual spaces")
    expected = {
        "ovvv": (nocc, nvir, nvir, nvir),
        "ovoo": (nocc, nvir, nocc, nocc),
        "ovov": (nocc, nvir, nocc, nvir),
        "fov": (nocc, nvir),
        "t1": (nocc, nvir),
        "t2": (nocc, nocc, nvir, nvir),
        "eps_o": (nocc,),
        "eps_v": (nvir,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} shape {arrays[name].shape} != expected {shape}")


def _views(ovvv, ovoo, ovov, fov, t1, t2):
    """Slow-code transposed views; mirrors ccsd_t_slow.kernel exactly."""
    t1T = t1.T
    t2T = t2.transpose(2, 3, 0, 1)
    eris_vvov = ovvv.transpose(1, 3, 0, 2)
    eris_vooo = ovoo.transpose(1, 0, 2, 3)
    eris_vvoo = ovov.transpose(1, 3, 0, 2)
    fvo = fov.T
    return t1T, t2T, eris_vvov, eris_vooo, eris_vvoo, fvo


def _w(np_views, a, b, c):
    _, t2T, vvov, vooo, _, _ = np_views
    w = np.einsum("if,fkj->ijk", vvov[a, b], t2T[c, :])
    w -= np.einsum("ijm,mk->ijk", vooo[a, :], t2T[b, c])
    return w


def _v(np_views, a, b, c):
    t1T, t2T, _, _, vvoo, fvo = np_views
    v = np.einsum("ij,k->ijk", vvoo[a, b], t1T[c])
    v += np.einsum("ij,k->ijk", t2T[a, b], fvo[c])
    return v


def _permuted(triple, axes):
    return (triple[axes[0]], triple[axes[1]], triple[axes[2]])


def _degeneracy(a, b, c):
    if a == c:  # a == b == c
        return 6
    if a == b or b == c:
        return 2
    return 1


def _check_denominators(eps_o, eps_v, threshold):
    """Reject noncanonical or near-zero (T) denominators before evaluation.

    A canonical RHF reference has occupied energies below virtual energies, so
    every ``eijk - (ev[a]+ev[b]+ev[c])`` is negative.  A nonnegative value
    signals an invalid/crossing reference, and a near-zero magnitude signals
    near-degeneracy; both must fail explicitly (issue #150 step 7) rather than
    divide into silence or NaN.
    """
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError(
            "triples denominator_threshold must be a positive finite number"
        )
    eijk = eps_o[:, None, None] + eps_o[None, :, None] + eps_o[None, None, :]
    closest = np.inf
    for a in range(len(eps_v)):
        for b in range(a + 1):
            for c in range(b + 1):
                d3 = eijk - eps_v[a] - eps_v[b] - eps_v[c]
                if np.any(d3 >= 0):
                    raise ValueError(
                        "noncanonical (T) denominator: an orbital energy "
                        "denominator is nonnegative"
                    )
                closest = min(closest, float(np.min(np.abs(d3))))
    if closest <= threshold:
        raise ValueError(
            f"near-zero (T) denominator (min|d3|={closest:.3e}); "
            "canonical-reference degeneracy"
        )


def triples_energy(
    nocc,
    nvir,
    ovvv,
    ovoo,
    ovov,
    fov,
    t1,
    t2,
    eps_o,
    eps_v,
    *,
    denominator_threshold=1e-10,
):
    """Non-iterative closed-shell (T) correction in Hartree (triangle + 6/2).

    Faithful transcription of ``ccsd_t_slow.kernel`` including the triangular
    virtual domain a>=b>=c, the 6/2 degeneracy folded into the denominator,
    and the literal 36-entry contraction table.
    """
    _validate(nocc, nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    _check_denominators(eps_o, eps_v, denominator_threshold)
    views = _views(ovvv, ovoo, ovov, fov, t1, t2)
    eijk = eps_o[:, None, None] + eps_o[None, :, None] + eps_o[None, None, :]
    et = 0.0
    for a in range(nvir):
        for b in range(a + 1):
            for c in range(b + 1):
                d3 = eijk - eps_v[a] - eps_v[b] - eps_v[c]
                d3 = d3 * _degeneracy(a, b, c)
                ws = {lbl: _w(views, *_permuted((a, b, c), VP[lbl])) for lbl in _LABELS}
                vs = {lbl: _v(views, *_permuted((a, b, c), VP[lbl])) for lbl in _LABELS}
                zs = {lbl: r3(ws[lbl] + 0.5 * vs[lbl]) / d3 for lbl in _LABELS}
                for zlbl, row in SLOW_TABLE.items():
                    for wlbl, ost in row:
                        et += np.einsum(
                            "ijk,ijk", ws[wlbl].transpose(OP[ost]), zs[zlbl]
                        )
    return et * 2


def triples_fullsum(
    nocc,
    nvir,
    ovvv,
    ovoo,
    ovov,
    fov,
    t1,
    t2,
    eps_o,
    eps_v,
    *,
    denominator_threshold=1e-10,
):
    """Independent full-sum oracle of (T): all ordered (a,b,c), no 6/2, /6.

    The triangular reference sums six virtual permutations of W against six
    of ``r3(W+0.5 V)/d3`` with an explicit hand-written occupied pairing (the
    SLOW_TABLE).  This oracle instead loops over *all* ordered virtual triples
    and all 36 = 6x6 (z-perm, w-perm) pairs, deriving the occupied pairing
    from the same spin-summation structure: for a z built from the seed at
    virtual permutation p and a left factor at virtual permutation q, the
    matched occupied axes are ``compose(inv(q), p)`` (an identity that the
    tests verify reproduces SLOW_TABLE's 36 entries exactly).  The (T) energy
    equals ``2 * (1/6) *`` this sum because the 36-term contraction is
    symmetric under simultaneous (a,b,c) permutation and each distinct triple
    therefore occurs with multiplicity 6 / degeneracy in the ordered sum.

    Besides the S3-derived pairing this function shares only the label seeds
    (W/V/r3) with the reference; it never touches SLOW_TABLE, the triangular
    domain, or the 6/2 pre-factor.
    """
    _validate(nocc, nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    _check_denominators(eps_o, eps_v, denominator_threshold)
    views = _views(ovvv, ovoo, ovov, fov, t1, t2)
    eijk = eps_o[:, None, None] + eps_o[None, :, None] + eps_o[None, None, :]
    perms = list(permutations((0, 1, 2)))

    def inv(p):
        out = [0, 0, 0]
        for i, x in enumerate(p):
            out[x] = i
        return tuple(out)

    def compose(p, q):
        return tuple(p[i] for i in q)

    total = 0.0
    for a in range(nvir):
        for b in range(nvir):
            for c in range(nvir):
                d3 = eijk - eps_v[a] - eps_v[b] - eps_v[c]
                triple = (a, b, c)
                for p in perms:
                    za, zb, zc = _permuted(triple, p)
                    z = r3(_w(views, za, zb, zc) + 0.5 * _v(views, za, zb, zc)) / d3
                    for pp in perms:
                        wa, wb, wc = _permuted(triple, pp)
                        q = compose(inv(pp), p)
                        w = _w(views, wa, wb, wc)
                        total += np.einsum("ijk,ijk", w.transpose(q), z)
    return 2.0 * total / 6.0


# ---------------------------------------------------------------------------
# Tiny-TensorIR lowering (same inventory; differentiable under the AD rules).
# ---------------------------------------------------------------------------


def _t_views(nodes):
    return {
        "t1T": transpose(nodes["t1"], (1, 0)),
        "t2T": transpose(nodes["t2"], (2, 3, 0, 1)),
        "vvov": transpose(nodes["ovvv"], (1, 3, 0, 2)),
        "vooo": transpose(nodes["ovoo"], (1, 0, 2, 3)),
        "vvoo": transpose(nodes["ovov"], (1, 3, 0, 2)),
        "fvo": transpose(nodes["fov"], (1, 0)),
    }


def _fix(node, *selections):
    """Select single virtual coordinates and reduce the singleton axes away."""
    axes = []
    for axis, pos in selections:
        node = gather(node, axis, [pos])
        axes.append(axis)
    return reduce_sum(node, tuple(sorted(axes)))


def _w_node(v, a, b, c):
    ab = _fix(v["vvov"], (0, a), (1, b))  # (i, f)
    cc = _fix(v["t2T"], (0, c))  # (f, k, j)
    w1 = einsum("if,fkj->ijk", ab, cc)
    a0 = _fix(v["vooo"], (0, a))  # (i, j, m)
    bc = _fix(v["t2T"], (0, b), (1, c))  # (m, k)
    w2 = einsum("ijm,mk->ijk", a0, bc)
    return add(w1, w2, coefficients=(1, -1))


def _v_node(v, a, b, c):
    ab = _fix(v["vvoo"], (0, a), (1, b))  # (i, j)
    cc = _fix(v["t1T"], (0, c))  # (k,)
    v1 = einsum("ij,k->ijk", ab, cc)
    ab2 = _fix(v["t2T"], (0, a), (1, b))  # (i, j)
    c2 = _fix(v["fvo"], (0, c))  # (k,)
    v2 = einsum("ij,k->ijk", ab2, c2)
    return add(v1, v2, coefficients=(1, 1))


def _r3_node(w):
    return add(
        *(transpose(w, perm) for _, perm in R3), coefficients=tuple(c for c, _ in R3)
    )


def _d3_node(nodes, ijk, a, b, c, fac):
    e0 = broadcast(nodes["eps_o"], ijk, (0,))
    e1 = broadcast(nodes["eps_o"], ijk, (1,))
    e2 = broadcast(nodes["eps_o"], ijk, (2,))
    eijk = add(e0, e1, e2)
    ev = add(
        reduce_sum(gather(nodes["eps_v"], 0, [a]), (0,)),
        reduce_sum(gather(nodes["eps_v"], 0, [b]), (0,)),
        reduce_sum(gather(nodes["eps_v"], 0, [c]), (0,)),
    )
    return add(eijk, broadcast(ev, ijk, ()), coefficients=(fac, -fac))


def build_triples_program(nocc, nvir):
    """Lower the (T) inventory to unshared TensorIR; differentiable inputs.

    Inputs are declared ``restricted_spatial`` parameters *without* symmetry
    constraints so dense tangent/cotangent spaces are legal under the current
    AD rules (``_require_general_inputs`` in the autodiff module rejects
    symmetric inputs as slice B of #151).  The graph is an unrolled sum of
    36 scalar ``einsum`` contractions per triangular virtual triple.
    """
    if any(type(n) is not int or n < 1 for n in (nocc, nvir)):
        raise ValueError("triples require nonempty occupied and virtual spaces")
    occ = IndexSpace("occupied", "occupied", nocc)
    vir = IndexSpace("virtual", "virtual", nvir)

    def O(name):
        return Index(name, occ)

    def V(name):
        return Index(name, vir)

    common = {
        "role": "parameter",
        "differentiable": True,
        "representation": "restricted_spatial",
    }
    nodes = {
        "ovvv": input_tensor(
            "ovvv", TensorSpec((O("i0"), V("a0"), V("b0"), V("f0")), **common)
        ),
        "ovoo": input_tensor(
            "ovoo", TensorSpec((O("i1"), V("a1"), O("j1"), O("m1")), **common)
        ),
        "ovov": input_tensor(
            "ovov", TensorSpec((O("i2"), V("a2"), O("j2"), V("b2")), **common)
        ),
        "fov": input_tensor("fov", TensorSpec((O("k3"), V("c3")), **common)),
        "t1": input_tensor("t1", TensorSpec((O("i4"), V("a4")), **common)),
        "t2": input_tensor(
            "t2", TensorSpec((O("i5"), O("j5"), V("a5"), V("b5")), **common)
        ),
        "eps_o": input_tensor("eps_o", TensorSpec((O("i6"),), **common)),
        "eps_v": input_tensor("eps_v", TensorSpec((V("a7"),), **common)),
    }
    ijk = (O("io"), O("jo"), O("ko"))
    views = _t_views(nodes)

    scalar_terms = []
    for a in range(nvir):
        for b in range(a + 1):
            for c in range(b + 1):
                d3 = _d3_node(nodes, ijk, a, b, c, _degeneracy(a, b, c))
                ws = {
                    lbl: _w_node(views, *_permuted((a, b, c), VP[lbl]))
                    for lbl in _LABELS
                }
                vs = {
                    lbl: _v_node(views, *_permuted((a, b, c), VP[lbl]))
                    for lbl in _LABELS
                }
                halves = Fraction(1, 2)
                zs = {
                    lbl: divide(
                        _r3_node(add(ws[lbl], vs[lbl], coefficients=(1, halves))), d3
                    )
                    for lbl in _LABELS
                }
                for zlbl, row in SLOW_TABLE.items():
                    for wlbl, ost in row:
                        scalar_terms.append(
                            einsum("ijk,ijk->", transpose(ws[wlbl], OP[ost]), zs[zlbl])
                        )
    et = add(*scalar_terms)
    energy = add(et, coefficients=(2,))
    return Program(
        {"triples_energy": energy},
        provenance={
            "method": "RCCSD(T)",
            "slice": "A",
            "inventory_version": VERSION,
            "inventory_hash": INVENTORY_HASH,
            "source": "PySCF 2.14.0 ccsd_t_slow.py kernel + r3; see source_manifest.json",
            "note": "unshared tiny-TensorIR reference; dense general inputs for AD",
        },
    )


def triples_energy_tensorir(
    nocc,
    nvir,
    ovvv,
    ovoo,
    ovov,
    fov,
    t1,
    t2,
    eps_o,
    eps_v,
    *,
    denominator_threshold=1e-10,
):
    """Build and execute the TensorIR lowering; returns the E_T scalar."""
    _validate(nocc, nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    _check_denominators(eps_o, eps_v, denominator_threshold)
    feeds = {
        "ovvv": ovvv,
        "ovoo": ovoo,
        "ovov": ovov,
        "fov": fov,
        "t1": t1,
        "t2": t2,
        "eps_o": eps_o,
        "eps_v": eps_v,
    }
    result = execute(build_triples_program(nocc, nvir), feeds)
    return float(result.outputs["triples_energy"])
