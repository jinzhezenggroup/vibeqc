"""Reproduce the FP64 DF two-root Chebyshev table from 90-digit Boys integrals.

This offline developer tool needs mpmath; source generation and execution do not.
Each two-unit interval uses a degree-17 interpolation at Chebyshev zeros. The
independent tests reconstruct F0--F3, including interval and asymptotic boundaries.
No GPU4PySCF data or scientific CUDA is copied.
"""

import argparse
import typing
from pathlib import Path

import mpmath as mp


def generate() -> typing.Any:
    """Return deterministic rounded coefficients for nodes u=t² and weights."""
    degree, intervals, width = 17, 24, 2
    rows = []
    with mp.workdps(90):
        n = degree + 1
        angles = [mp.pi * (j + mp.mpf("0.5")) / n for j in range(n)]
        for interval in range(intervals):
            samples = []
            for angle in angles:
                t = width * (interval + (1 + mp.cos(angle)) / 2)
                f = [
                    mp.gammainc(k + mp.mpf("0.5"), 0, t)
                    / (2 * t ** (k + mp.mpf("0.5")))
                    for k in range(4)
                ]
                determinant = f[0] * f[2] - f[1] ** 2
                a = (f[1] * f[2] - f[0] * f[3]) / determinant
                b = (f[1] * f[3] - f[2] ** 2) / determinant
                x0, x1 = (
                    (-a - mp.sqrt(a * a - 4 * b)) / 2,
                    (-a + mp.sqrt(a * a - 4 * b)) / 2,
                )
                w0 = (f[0] * x1 - f[1]) / (x1 - x0)
                samples.append((x0, x1, w0, f[0] - w0))
            for series in range(4):
                rows.append(
                    tuple(
                        float(
                            mp.fsum(
                                samples[j][series] * mp.cos(k * angles[j])
                                for j in range(n)
                            )
                            * (1 if k == 0 else 2)
                            / n
                        )
                        for k in range(n)
                    )
                )
    header = '''"""Generated DF Rys two-root coefficients; nodes are t² on [0,1].

Reproduce with tools/generate_df_rys2_table.py (mpmath, 90 decimal digits).
Four series per interval: ascending nodes, then corresponding weights.
The degree-17 Chebyshev interpolants cover [0,48] in two-unit intervals.
"""\n\n'''
    return (
        header
        + "DF_RYS2_COEFFICIENTS = (\n"
        + "".join(
            "    (\n" + "".join("        " + repr(c) + ",\n" for c in row) + "    ),\n"
            for row in rows
        )
        + ")\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("python/vibeqc_compiler/integral/df_rys2_data.py"),
    )
    args = parser.parse_args()
    args.output.write_text(generate())


if __name__ == "__main__":
    main()
