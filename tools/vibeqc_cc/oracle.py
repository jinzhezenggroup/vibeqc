"""Tiny determinant-space validation oracle, independent of tensor contractions.

This deliberately exponential test/reference utility is never called by the
CC facade. It constructs fermionic operators with explicit bit-string signs,
then applies terminating exponential series. No PySCF or equation inventory.
"""

from itertools import combinations, product
from math import comb

import numpy as np


def _excite(bits, p, q):
    if not bits & (1 << q):
        return None
    sign = (-1) ** ((bits & ((1 << q) - 1)).bit_count())
    bits ^= 1 << q
    if bits & (1 << p):
        return None
    sign *= (-1) ** ((bits & ((1 << p) - 1)).bit_count())
    return bits | (1 << p), sign


class DeterminantOracle:
    """Fixed Nalpha=Nbeta sector, alpha spatial orbitals before beta orbitals."""

    def __init__(self, fock, eri, nocc, *, max_determinants=128):
        n = len(fock)
        if not 0 < nocc < n or comb(n, nocc) ** 2 > max_determinants:
            raise ValueError("determinant oracle is restricted to tiny test systems")
        self.n, self.o = n, nocc
        spin_strings = [sum(1 << k for k in c) for c in combinations(range(n), nocc)]
        basis = [a | (b << n) for a in spin_strings for b in spin_strings]
        positions = {b: k for k, b in enumerate(basis)}
        dim = len(basis)
        self.ref = positions[(2**nocc - 1) * (1 + 2**n)]
        self.ket = np.eye(dim)[:, self.ref]
        operators = np.zeros((n, n, dim, dim))
        alpha = np.zeros_like(operators)
        for p, q in product(range(n), repeat=2):
            for col, bits in enumerate(basis):
                for spin in (0, 1):
                    change = _excite(bits, p + spin * n, q + spin * n)
                    if change is not None:
                        new, sign = change
                        operators[p, q, positions[new], col] += sign
                        if spin == 0:
                            alpha[p, q, positions[new], col] += sign
        self.excitation = operators[nocc:, :nocc]
        self.bras = np.array(
            [[alpha[a, i] @ self.ket for a in range(nocc, n)] for i in range(nocc)]
        )
        self.double_bras = np.array(
            [
                alpha[a, i] @ (operators[b, j] - alpha[b, j]) @ self.ket
                for i, j, a, b in product(
                    range(nocc), range(nocc), range(nocc, n), range(nocc, n)
                )
            ]
        ).reshape(nocc, nocc, n - nocc, n - nocc, dim)
        # Recover h from the chosen full Fock and reference contraction. This
        # allows off-diagonal F and independent symmetric random ERIs.
        h = np.array(fock, copy=True)
        for p, q, i in product(range(n), range(n), range(nocc)):
            h[p, q] -= 2 * eri[p, q, i, i] - eri[p, i, i, q]
        H = np.zeros((dim, dim))
        for p, q in product(range(n), repeat=2):
            H += h[p, q] * operators[p, q]
        for p, q, r, s in product(range(n), repeat=4):
            pair = operators[p, q] @ operators[r, s]
            if q == r:
                pair = pair - operators[p, s]
            H += 0.5 * eri[p, q, r, s] * pair
        self.reference_energy = H[self.ref, self.ref]
        self.hnormal = H - self.reference_energy * np.eye(dim)

    def cluster(self, t1, t2):
        """Construct the cluster operator, also usable for excitation-norm tests."""
        o, v = t1.shape
        T = np.zeros_like(self.hnormal)
        for i, a in product(range(o), range(v)):
            T += t1[i, a] * self.excitation[a, i]
        for i, j, a, b in product(range(o), range(o), range(v), range(v)):
            T += 0.5 * t2[i, j, a, b] * (self.excitation[a, i] @ self.excitation[b, j])

        return T

    def transformed(self, t1, t2):
        """Apply Hbar_N to the reference, without assuming a residual formula."""
        T = self.cluster(t1, t2)

        def exponential(vector, sign):
            term = vector.copy()
            result = term.copy()
            # T strictly raises the number of virtual electrons. Nilpotence
            # terminates the series after at most the number of electrons.
            for k in range(1, 2 * self.o + 1):
                term = sign * (T @ term) / k
                result += term
            return result

        return exponential(self.hnormal @ exponential(self.ket, 1), -1)

    def evaluate(self, t1, t2):
        """Return <Phi|Hbar_N|Phi> and each alpha single projection."""
        vector = self.transformed(t1, t2)
        return float(vector[self.ref]), self.bras @ vector

    def evaluate_full(self, t1, t2):
        """Also return the normalized opposite-spin doubles projections."""
        vector = self.transformed(t1, t2)
        return float(vector[self.ref]), self.bras @ vector, self.double_bras @ vector


def random_case(nocc=2, nvir=2, seed=148):
    """Eightfold symmetric, signed ERIs; non-diagonal F; pair-symmetric t2."""
    rng = np.random.default_rng(seed)
    n = nocc + nvir
    f = rng.normal(scale=0.2, size=(n, n))
    f = (f + f.T) / 2 + np.diag(np.linspace(-1, 1, n))
    g = rng.normal(scale=0.1, size=(n, n, n, n))
    g = (g + g.transpose(1, 0, 2, 3)) / 2
    g = (g + g.transpose(0, 1, 3, 2)) / 2
    g = (g + g.transpose(2, 3, 0, 1)) / 2
    t1 = rng.normal(scale=0.15, size=(nocc, nvir))
    t2 = rng.normal(scale=0.15, size=(nocc, nocc, nvir, nvir))
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    return f, g, t1, t2


def dense_feeds(f, g, t1, t2):
    """Test-only dense-to-block adapter; provider execution uses explicit MOBlock."""
    o = t1.shape[0]
    slices = {"o": slice(0, o), "v": slice(o, None)}
    feeds = {"foo": f[:o, :o], "fov": f[:o, o:], "fvv": f[o:, o:], "t1": t1, "t2": t2}
    for block in ("ovov", "ovvo", "oovv", "ovvv", "ovoo", "oooo", "vvvv"):
        feeds[block] = g[tuple(slices[k] for k in block)]
    return feeds


def homogeneous_groups(f, g, x, y):
    """Independently isolate amplitude polynomial degrees at finite points.

    This is exact polynomial interpolation for CC singles (degree <=3), not
    a small-step finite-difference approximation or a shared equation oracle.
    """
    f_oracle = DeterminantOracle(f, g * 0, len(x))
    g_oracle = DeterminantOracle(f * 0, g, len(x))
    ef, rf = f_oracle.evaluate(x, y)
    e1, r1 = g_oracle.evaluate(x, y * 0)
    em1, rm1 = g_oracle.evaluate(-x, y * 0)
    _, r2 = g_oracle.evaluate(2 * x, y * 0)
    _, rm2 = g_oracle.evaluate(-2 * x, y * 0)
    e_y, r_y = g_oracle.evaluate(x * 0, y)
    _, r_xy = g_oracle.evaluate(x, y)
    odd1, odd2 = (r1 - rm1) / 2, (r2 - rm2) / 2
    cubic = (odd2 - 2 * odd1) / 6
    energy, residual = DeterminantOracle(f, g, len(x)).evaluate(x, y)
    return {
        "energy_t1": ef,
        "energy_t2": e_y,
        "energy_t1t1": (e1 + em1) / 2,
        "singles_fock": rf,
        "singles_g_t1": odd1 - cubic,
        "singles_g_t2": r_y,
        "singles_g_t1t1": (r1 + rm1) / 2,
        "singles_g_t1t2": r_xy - r1 - r_y,
        "singles_g_t1t1t1": cubic,
        "correlation_energy": energy,
        "singles_residual": residual,
    }
