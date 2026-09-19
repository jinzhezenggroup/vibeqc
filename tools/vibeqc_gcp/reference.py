"""Independent CPU qualification provider for the pinned r2SCAN-3c gCP term.

SPDX-License-Identifier: LGPL-3.0-or-later
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from vibeqc_compiler.method.correction import CorrectionProvenance, CorrectionResult

ANGSTROM_TO_BOHR = 1.8897261254578281
_DATA = Path(__file__).resolve().parents[2] / "external/r2scan3c/gcp-r2scan3c-h-ar.json"


def _load():
    return json.loads(_DATA.read_text())


def _aaux(x, n=8):
    ex = math.exp(-x)
    rx = 1.0 / x
    out = [0.0] * (n + 1)
    out[0] = ex * rx
    for k in range(1, n + 1):
        out[k] = (k * out[k - 1] + ex) * rx
    return out


def _bint(x, n=8):
    out = [0.0] * (n + 1)
    if abs(x) < 1.0e-6:
        for k in range(0, n + 1, 2):
            out[k] = 2.0 / (k + 1)
        return out
    pw = [1.0]
    for i in range(1, 13):
        pw.append(pw[-1] * (-x) / i)
    for k in range(n + 1):
        out[k] = 2.0 * sum(pw[i] / (k + i + 1) for i in range(k % 2, 13, 2))
    return out


def _baux(x, n=8):
    ep, em, rx = math.exp(x), math.exp(-x), 1.0 / x
    out = []
    for k in range(n + 1):
        term = rx
        sgn = 1.0 if k % 2 == 0 else -1.0
        sp, sm = sgn * term, term
        for j in range(1, k + 1):
            term *= (k - j + 1) * rx
            sgn = -sgn
            sp += sgn * term
            sm += term
        out.append(ep * sp - em * sm)
    return out


def _overlap(r, shell_a, shell_b, za, zb):
    same = abs(za - zb) < 0.1
    key = shell_a * shell_b
    if key == 1:
        m, wt, pa, qb = 3, (1, -1), (2, 0), (0, 2)
        cnorm = 0.25 * math.sqrt((za * zb) ** 3)
    elif key == 2:
        if shell_a >= shell_b:
            za, zb = zb, za
        m, wt, pa, qb = 4, (1, -1, 1, -1), (3, 0, 2, 1), (0, 3, 1, 2)
        cnorm = math.sqrt(1 / 3) * math.sqrt(za**3 * zb**5) * 0.125
    elif key == 3:
        if shell_a >= shell_b:
            za, zb = zb, za
        m, wt, pa, qb = 5, (1, -1, 2, -2), (4, 0, 3, 1), (0, 4, 1, 3)
        cnorm = math.sqrt(za**3 * zb**7 / 7.5) * 0.0625 / math.sqrt(3)
    elif key == 4:
        m, wt, pa, qb = 5, (1, 1, -2), (4, 0, 2), (0, 4, 2)
        cnorm = math.sqrt((za * zb) ** 5) * 0.0625 / 3
    elif key == 6:
        if shell_a >= shell_b:
            za, zb = zb, za
        m = 6
        wt, pa, qb = (
            (1, 1, -2, -2, 1, 1),
            (5, 4, 3, 2, 1, 0),
            (0, 1, 2, 3, 4, 5),
        )
        cnorm = math.sqrt(za**5 * zb**7 / 7.5) * 0.03125 / 3
    elif key == 9:
        m, wt, pa, qb = 7, (1, -3, 3, -1), (6, 4, 2, 0), (0, 2, 4, 6)
        cnorm = math.sqrt((za * zb) ** 7) / 1440
    else:
        raise ValueError("invalid gCP shell combination")
    ha, hb = 0.5 * (za + zb), 0.5 * (zb - za)
    av = _aaux(ha * r)
    bv = _bint(hb * r) if same else _baux(hb * r)
    f0 = sum(w * av[p] * bv[q] for w, p, q in zip(wt, pa, qb))
    f1 = -sum(
        w * (ha * av[p + 1] * bv[q] + hb * av[p] * bv[q + 1])
        for w, p, q in zip(wt, pa, qb)
    )
    s = cnorm * r**m * f0
    ds = cnorm * (m * r ** (m - 1) * f0 + r**m * f1)
    return s, ds


def evaluate_r2scan3c_gcp(atomic_numbers, coordinates_bohr):
    """Return the pinned H-Ar gCP energy and Cartesian gradient in atomic units."""

    data = _load()
    domain = set(data["supported_atomic_numbers"])
    zs = tuple(atomic_numbers)
    xyz = tuple(tuple(float(v) for v in row) for row in coordinates_bohr)
    if len(zs) != len(xyz) or any(len(row) != 3 for row in xyz):
        raise ValueError("gCP geometry requires one xyz triplet per atom")
    if any(not math.isfinite(value) for row in xyz for value in row):
        raise ValueError("gCP geometry requires finite coordinates")
    bad = sorted(set(zs) - domain)
    if bad:
        raise ValueError(f"unsupported r2SCAN-3c gCP elements {tuple(bad)}")
    by_z = {row["atomic_number"]: row for row in data["elements"]}
    p = data["parameters"]
    radii = data["vdw_pair_radii_angstrom"]
    grad = [[0.0, 0.0, 0.0] for _ in zs]
    energy = 0.0
    for i in range(len(zs)):
        pi = by_z[zs[i]]
        for j in range(i):
            pj = by_z[zs[j]]
            vec = [xyz[i][k] - xyz[j][k] for k in range(3)]
            r = math.sqrt(sum(v * v for v in vec))
            if r <= 0.0:
                raise ValueError("coincident atoms are invalid for gCP")
            s, ds = _overlap(r, pi["shell"], pj["shell"], pi["slater"], pj["slater"])
            if s <= 0.0 or not math.isfinite(s):
                raise ValueError("invalid Slater overlap in gCP")
            bsse = math.exp(-p["alpha"] * r ** p["beta"]) / math.sqrt(s)
            m, n = max(zs[i], zs[j]), min(zs[i], zs[j])
            r0 = radii[m * (m - 1) // 2 + n - 1] * ANGSTROM_TO_BOHR
            x = r / r0
            t = p["damping_scale"] * x ** p["damping_exponent"]
            damp = 1.0 - 1.0 / (1.0 + t)
            ddamp = (
                p["damping_scale"]
                * p["damping_exponent"]
                * x ** (p["damping_exponent"] - 1)
                / r0
                / (1.0 + t) ** 2
            )
            xi = 1.0 / math.sqrt(pi["xv"]) if pi["xv"] >= 0.5 else 0.0
            xj = 1.0 / math.sqrt(pj["xv"]) if pj["xv"] >= 0.5 else 0.0
            pref = p["sigma"] * (pi["emiss"] * xj + pj["emiss"] * xi)
            pair = pref * bsse * damp
            energy += pair
            dbsse = bsse * (
                -p["alpha"] * p["beta"] * r ** (p["beta"] - 1) - 0.5 * ds / s
            )
            dedr = pref * (dbsse * damp + bsse * ddamp)
            for k in range(3):
                g = dedr * vec[k] / r
                grad[i][k] += g
                grad[j][k] -= g
    parameters = tuple(
        sorted({"basis": data["basis"], "profile": "r2scan3c", **p}.items())
    )
    return CorrectionResult(
        component="gcp",
        energy=energy,
        gradient=tuple(tuple(row) for row in grad),
        status="ok",
        parameters=parameters,
        provenance=CorrectionProvenance(
            source=data["upstream"]["repository"],
            source_revision=data["upstream"]["commit"],
            data_sha256="c1cede24b2527a2b688981b651d91da7206d8da1a223c3f217a86800c47eded2",
            license=data["upstream"]["license"],
            implementation="vibeqc-gcp-cpu-reference-v1",
        ),
    )
