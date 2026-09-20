"""Independent 450-digit mixed derivatives of original spin PW92/PBE energy.

Runtime acceptance reads this committed table, never the production point jet.
Empty-spin references use a one-sided derivative of the energy normal to that
boundary and a tangent direction that preserves the empty spin.
"""

from pathlib import Path

import mpmath as mp
import numpy as np

from tools.generate_xc_scf_references import energy


def exchange_energy(pbe: bool, values: list[mp.mpf]) -> mp.mpf:
    """Original spin-scaled exchange; used to quantify X/C cancellation only."""
    a, b, *gradient = values
    cx = mp.mpf(3) / 8 * (3 / mp.pi) ** (mp.mpf(1) / 3) * 4 ** (mp.mpf(2) / 3)
    beta, kappa = mp.mpf("0.06672455060314922"), mp.mpf("0.804")
    result = mp.mpf(0)
    for rho, grad in ((a, gradient[:3]), (b, gradient[3:])):
        if not rho:
            continue
        enhancement = mp.mpf(1)
        if pbe:
            s2 = sum(g * g for g in grad) / (
                4 * (6 * mp.pi**2) ** (mp.mpf(2) / 3) * rho ** (mp.mpf(8) / 3)
            )
            enhancement += kappa * (1 - kappa / (kappa + beta * mp.pi**2 / 3 * s2))
        result -= cx * rho ** (mp.mpf(4) / 3) * enhancement
    return result


def reference(pbe: bool, point: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Mixed energy derivative and independent |delta_X|+|delta_C| per output."""
    x, d = [[mp.mpf(float(v)) for v in row] for row in (point, direction)]
    scale = x[0] + x[1]
    result, magnitudes = [], []
    for index in range(8):

        def displaced(
            s: mp.mpf, t: mp.mpf, exchange_only: bool = False, index: int = index
        ) -> mp.mpf:
            values = [v + t * delta for v, delta in zip(x, d, strict=True)]
            values[index] += s * scale
            return (
                exchange_energy(pbe, values) if exchange_only else energy(pbe, values)
            ) / scale

        # Avoid negative densities when differentiating the energy at an empty
        # spin. Extra precision suppresses its fractional-power truncation.
        options = (
            {"direction": 1, "addprec": 1500} if min(x[:2]) == 0 else {"addprec": 200}
        )
        total = mp.diff(displaced, (0, 0), (1, 1), **options)
        exchange = mp.diff(
            lambda s, t: displaced(s, t, exchange_only=True), (0, 0), (1, 1), **options
        )
        result.append(float(total))
        magnitudes.append(float(abs(exchange) + abs(total - exchange)))
    return np.array(result + magnitudes)


def main() -> None:
    mp.mp.dps = 450
    cases = []
    for scale in (1.0, 1e-20, 1e-80, 1e-180, 1e-280):
        for gradients in (
            (0.03, -0.02, 0.01, -0.01, 0.04, 0.02),
            (7.0, -2.0, 1.0, -3.0, 1.0, 2.0),
        ):
            cases.append(scale * np.array([0.65, 0.35, *gradients]))
    for minority in (
        0.0,
        1e-21,
        0.35e-18 * (1 - 1e-5),
        0.35e-18 * (1 + 1e-5),
        1e-100,
        1e-280,
    ):
        cases.append(
            np.array(
                [
                    0.7,
                    minority,
                    0.03,
                    -0.02,
                    0.01,
                    0.2 * minority,
                    -0.1 * minority,
                    0.03 * minority,
                ]
            )
        )
    cases.extend(
        (
            np.array([0.65, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.65, 0.35, 0.1, -0.2, 0.3, -0.1, 0.2, -0.3]),
        )
    )
    for factor in (1 - 1e-5, 1 + 1e-5):
        cases.append(
            np.r_[
                0.65,
                0.35,
                factor * 0.65 ** (4 / 3) * np.array([0.6, 0.8, 0.0]),
                [0.01, -0.03, 0.02],
            ]
        )
    # Connection between the two correlation formulas is A*t2=1 in original
    # coordinates. Derive it from the independent energy, not a native branch.
    a, b = mp.mpf(".65"), mp.mpf(".35")
    x = [a, b, *([mp.mpf(0)] * 6)]
    eps = (energy(True, x) - exchange_energy(True, x)) / (a + b)
    phi = (
        (2 * a / (a + b)) ** (mp.mpf(2) / 3) + (2 * b / (a + b)) ** (mp.mpf(2) / 3)
    ) / 2
    gamma = (1 - mp.log(2)) / mp.pi**2
    aa = mp.mpf("0.06672455060314922") / (gamma * mp.expm1(-eps / (gamma * phi**3)))
    connection = mp.sqrt(
        16
        * 2 ** (mp.mpf(2) / 3)
        * (3 / (4 * mp.pi)) ** (mp.mpf(1) / 3)
        * (a + b) ** (mp.mpf(7) / 3)
        * phi**2
        / aa
    )
    for factor in (1 - 1e-5, 1 + 1e-5):
        cases.append(
            np.array([0.65, 0.35, factor * float(connection), 0.0, 0.0, 0.0, 0.0, 0.0])
        )
    # rho^(4/3) underflows here, but the specified directional potential is
    # finite. Cover both density-only and gradient directions at exact grad=0.
    cases.extend([np.array([0.65e-280, 0.35e-280, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])] * 2)
    rows = []
    for pbe in (False, True):
        for case_index, point in enumerate(cases):
            a, b = point[:2]
            direction = np.r_[
                0.17 * a,
                -0.09 * b,
                a * np.array([0.07, 0.02, -0.03]),
                b * np.array([-0.03, 0.01, 0.05]),
            ]
            if case_index == len(cases) - 2:
                direction[2:] = 0.0
            rows.append(
                np.r_[int(pbe), point, direction, reference(pbe, point, direction)]
            )
    output = Path(__file__).resolve().parents[1] / "tests/data/xc/uks_response.tsv"
    np.savetxt(
        output,
        rows,
        fmt="%.17e",
        header="pbe rho[2] gradient[2,3] delta_rho[2] delta_gradient[2,3] delta_vrho[2] delta_vgradient[2,3] abs_delta_X_plus_abs_delta_C[8]\nOriginal PW92/PBE, mpmath 450 dps mixed derivatives; tools/generate_xc_uks_response_references.py",
    )
    print(f"Wrote {len(rows)} independent UKS directions to {output}")


if __name__ == "__main__":
    main()
