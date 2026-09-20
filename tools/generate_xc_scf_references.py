"""Independent Libxc/interior and mpmath/boundary E/V fixtures for native SCF.

The high precision oracle evaluates the original PW92/PBE formulas in rs, t2
and A, before the native scaled-coordinate and reciprocal rewrites. Numerical
differentiation uses 450 decimal digits. This supplements, and never rewrites,
the existing #214 identical-grid Libxc fixtures.
"""

import typing
from pathlib import Path

import mpmath as mp
import numpy as np
from pyscf.dft import libxc


def energy(pbe: typing.Any, inputs: typing.Any) -> typing.Any:
    """Original per-volume formula; explicit C2 extension only for PBE phi."""
    a, b, *gradient = inputs
    n = a + b
    if not n:
        return mp.mpf(0)
    ga, gb = gradient[:3], gradient[3:]
    rs = (3 / (4 * mp.pi * n)) ** (mp.mpf(1) / 3)
    up, down, z = 2 * a / n, 2 * b / n, (a - b) / n
    parameters = (
        (
            "0.0310907" if pbe else "0.031091",
            "0.21370",
            "7.5957",
            "3.5876",
            "1.6382",
            "0.49294",
        ),
        (
            "0.01554535" if pbe else "0.015545",
            "0.20548",
            "14.1189",
            "6.1977",
            "3.3662",
            "0.62517",
        ),
        (
            "0.0168869" if pbe else "0.016887",
            "0.11125",
            "10.357",
            "3.6231",
            "0.88026",
            "0.49671",
        ),
    )
    pw = []
    for row in parameters:
        aa, alpha, b1, b2, b3, b4 = map(mp.mpf, row)
        aux = b1 * mp.sqrt(rs) + b2 * rs + b3 * rs ** mp.mpf("1.5") + b4 * rs**2
        pw.append(-2 * aa * (1 + alpha * rs) * mp.log1p(1 / (2 * aa * aux)))
    fz = (up ** (mp.mpf(4) / 3) + down ** (mp.mpf(4) / 3) - 2) / (
        2 ** (mp.mpf(4) / 3) - 2
    )
    fzz = mp.mpf("1.709920934161365617563962776245" if pbe else "1.709921")
    e0, e1, em = pw
    eps = e0 + z**4 * fz * (e1 - e0 + em / fzz) - fz * em / fzz
    beta, kappa = mp.mpf("0.06672455060314922"), mp.mpf("0.804")
    gamma = (1 - mp.log(2)) / mp.pi**2
    exchange = mp.mpf(0)
    cx = mp.mpf(3) / 8 * (3 / mp.pi) ** (mp.mpf(1) / 3) * 4 ** (mp.mpf(2) / 3)
    for density, grad in ((a, ga), (b, gb)):
        if not density:
            continue
        enhancement = 1
        if pbe:
            s2 = sum(g * g for g in grad) / (
                4 * (6 * mp.pi**2) ** (mp.mpf(2) / 3) * density ** (mp.mpf(8) / 3)
            )
            enhancement += kappa * (1 - kappa / (kappa + beta * mp.pi**2 / 3 * s2))
        exchange -= cx * density ** (mp.mpf(4) / 3) * enhancement
    if pbe:

        def spin_power(u: typing.Any) -> typing.Any:
            cutoff = mp.mpf("1e-18")
            if u >= cutoff:
                return u ** (mp.mpf(2) / 3)
            t = u / cutoff
            return cutoff ** (mp.mpf(2) / 3) * (14 * t - 7 * t**2 + 2 * t**3) / 9

        phi = (spin_power(up) + spin_power(down)) / 2
        sigma = sum((x + y) ** 2 for x, y in zip(ga, gb))
        t2 = sigma * n ** (-mp.mpf(8) / 3) / (16 * 2 ** (mp.mpf(2) / 3) * phi**2 * rs)
        aa = beta / (gamma * mp.expm1(-eps / (gamma * phi**3)))
        f1 = t2 + aa * t2**2
        eps += gamma * phi**3 * mp.log1p(beta * f1 / (gamma * (1 + aa * f1)))
    return exchange + n * eps


def boundary_reference(pbe: typing.Any, values: typing.Any) -> typing.Any:
    """Differentiate original formula in normalized coordinates, at high precision."""
    values = [mp.mpf(float(x)) for x in values]
    scale = values[0] + values[1]
    result = [energy(pbe, values)]
    for i in range(8):

        def displaced(t: typing.Any, index: typing.Any = i) -> typing.Any:
            inputs = values.copy()
            inputs[index] += t * scale
            return energy(pbe, inputs) / scale

        # Large extra precision removes the h^(1/3) exchange error from a
        # one-sided derivative at an exactly empty spin, even in deep tails.
        result.append(mp.diff(displaced, 0, direction=1, addprec=1500))
    return np.array([float(x) for x in result])


def main() -> None:
    mp.mp.dps = 450
    rng = np.random.default_rng(162)
    rows = []
    for pbe in (False, True):
        rho = 10 ** rng.uniform(-8, 2, (1, 12)) * rng.uniform(0.15, 1, (2, 12))
        grad = rng.normal(size=(2, 3, 12)) * rho[:, None, :] ** (4 / 3)
        features = np.concatenate([rho[:, None, :], grad], axis=1)
        exc, v, _, _ = libxc.eval_xc(
            "PBE" if pbe else "LDA_X,LDA_C_PW",
            features if pbe else rho[:, None, :],
            spin=1,
            deriv=1,
        )
        for i in range(12):
            coefficients = np.zeros((2, 3))
            if pbe:
                coefficients[0] = (
                    2 * v[1][i, 0] * grad[0, :, i] + v[1][i, 1] * grad[1, :, i]
                )
                coefficients[1] = (
                    2 * v[1][i, 2] * grad[1, :, i] + v[1][i, 1] * grad[0, :, i]
                )
            inputs = np.r_[rho[:, i], grad[:, :, i].ravel() if pbe else np.zeros(6)]
            reference = np.r_[exc[i] * rho[:, i].sum(), v[0][i], coefficients.ravel()]
            rows.append((int(pbe), 0, *inputs, *reference))
        for scale in (1.0, 1e-12, 1e-30, 1e-100, 1e-240, 1e-300):
            for fraction in (0.0, 0.3, 1.0):
                rho = scale * np.array([fraction, 1 - fraction])
                grad = (
                    rho[:, None] * np.array([[0.2, -0.3, 0.4], [-0.3, 0.5, -0.1]])
                    if pbe
                    else np.zeros((2, 3))
                )
                inputs = np.r_[rho, grad.ravel()]
                rows.append((int(pbe), 1, *inputs, *boundary_reference(pbe, inputs)))
        # Probe both sides of the explicit spin extension, with zero minority
        # gradient and nonzero total gradient to expose phi's rho derivative.
        for fraction in (0.0, 1e-21, 0.49e-18, 0.5e-18, 0.51e-18, 1e-15):
            inputs = np.array([fraction, 1 - fraction, 0, 0, 0, 0.2, -0.1, 0.3])
            if not pbe:
                inputs[2:] = 0
            rows.append((int(pbe), 1, *inputs, *boundary_reference(pbe, inputs)))
    # Unbalanced spins can underflow exchange intermediates even when total
    # density is ordinary. Include zero/finite majority gradients so correlation
    # cannot conceal an incorrect tiny-spin exchange gradient coefficient.
    for minority in (1e-100, 1e-200, 1e-300):
        for majority_gradient, minority_gradient in (
            (0.0, 0.0),
            (0.0, 0.2 * minority),
            (0.2, 0.0),
            (0.2, 0.2 * minority),
        ):
            inputs = np.array(
                [0.5, minority, majority_gradient, 0, 0, minority_gradient, 0, 0]
            )
            rows.append((1, 1, *inputs, *boundary_reference(True, inputs)))
    inputs = np.array([0.5, np.nextafter(0.0, 1.0), 0.2, 0, 0, 0, 0, 0])
    rows.append((1, 1, *inputs, *boundary_reference(True, inputs)))
    for factor in (0.999, 1.001):
        inputs = np.array([0.5, 0.3, 0.1, 0, 0, factor * 0.3 ** (4 / 3), 0, 0])
        rows.append((1, 1, *inputs, *boundary_reference(True, inputs)))
    # Finite input gradients can overflow division by a tiny density, their
    # squares, or their sum. Opposite huge spin gradients must still cancel.
    # Tiny gradients also retain first derivatives when their square is zero.
    for values in (
        [1e-300, 5e-301, 1e-3, 0, 0, 0, 0, 0],
        [1e-100, 5e-101, 1e100, 0, 0, 0, 0, 0],
        [0.5, 0.3, 1e308, 0, 0, 1e308, 0, 0],
        [0.5, 0.3, 1e308, 0, 0, -1e308, 0, 0],
        [0.5, 0.3, 1e308, 0.2, 0, -1e308, 0, 0],
        [0.5, 0.3, 1e-180, 0, 0, 0, 0, 0],
        [1e100, 5e99, 0, 0, 0, 0, 0, 0],
        [1e100, 5e99, 1e99, 0, 0, 0, 0, 0],
    ):
        inputs = np.array(values)
        rows.append((1, 1, *inputs, *boundary_reference(True, inputs)))
    # Locate A*t2=1 from the independent original equations, then sample
    # both sides of the native correlation's equivalent numerical forms.
    a, b = mp.mpf("0.5"), mp.mpf("0.3")
    n = a + b
    cx = mp.mpf(3) / 8 * (3 / mp.pi) ** (mp.mpf(1) / 3) * 4 ** (mp.mpf(2) / 3)
    eps = (
        energy(True, [a, b, *([mp.mpf(0)] * 6)])
        + cx * (a ** (mp.mpf(4) / 3) + b ** (mp.mpf(4) / 3))
    ) / n
    phi = ((2 * a / n) ** (mp.mpf(2) / 3) + (2 * b / n) ** (mp.mpf(2) / 3)) / 2
    gamma = (1 - mp.log(2)) / mp.pi**2
    aa = mp.mpf("0.06672455060314922") / (gamma * mp.expm1(-eps / (gamma * phi**3)))
    connection = mp.sqrt(
        16
        * 2 ** (mp.mpf(2) / 3)
        * (3 / (4 * mp.pi)) ** (mp.mpf(1) / 3)
        * n ** (mp.mpf(7) / 3)
        * phi**2
        / aa
    )
    for factor in (0.999, 1.001):
        inputs = np.array([0.5, 0.3, factor * float(connection), 0, 0, 0, 0, 0])
        rows.append((1, 1, *inputs, *boundary_reference(True, inputs)))
    path = Path(__file__).resolve().parents[1] / "tests/data/xc/scf_domain.tsv"
    np.savetxt(
        path,
        rows,
        fmt="%.17g",
        header="pbe oracle(0=Libxc7,1=mpmath450dps) rho[2] gradient[2,3] E vrho[2] vgradient[2,3]\nsemilocal-scaled-v1/pbe-spin-c2-1e-18; generated by tools/generate_xc_scf_references.py",
    )
    print(f"Wrote {len(rows)} independent E/V points to {path}")


if __name__ == "__main__":
    main()
