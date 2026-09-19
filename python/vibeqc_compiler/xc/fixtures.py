"""Hash-checked pinned Libxc fixtures with explicit boundary-oracle selection."""

import json
from itertools import combinations_with_replacement

import numpy as np

from vibeqc_compiler.common.paths import source_root
from vibeqc_compiler.common.provenance import file_hash

from .reference import exchange_reference


def load_fixture(name, *, spin="polarized", domain="typical"):
    """Return metadata, physical features, checked oracle and raw Libxc output.

    Boundary exchange uses closed-form spin scaling because Libxc's internal
    reduced-variable cancellation corrupts a mathematically zero mixed partial.
    Combined functionals use this exchange plus the unmodified Libxc correlation.
    Raw library results always remain available for diagnostics and archival.
    """
    root = source_root() / "tests/data/xc"
    metadata = json.loads((root / "libxc.json").read_text())
    path = root / "libxc.npz"
    if file_hash(path) != metadata["arrays_sha256"]:
        raise ValueError("XC fixture hash mismatch")
    if spin not in ("polarized", "unpolarized") or domain not in (
        "typical",
        "boundary",
    ):
        raise ValueError("unknown XC fixture spin/domain")
    flag = int(spin == "polarized")
    tag = f"{domain}_{flag}"
    with np.load(path, allow_pickle=False) as arrays:
        features = arrays[f"{tag}_features"].copy()
        raw = arrays[f"{tag}_{name}"].copy()
        expected = raw.copy()
        oracle = "Libxc 7.0.0"
        if domain == "boundary" and name in ("LDA_X", "GGA_X_PBE", "LDA_XC_PW", "PBE"):
            e, v, h = exchange_reference(
                features, spin=flag, gga=name in ("GGA_X_PBE", "PBE")
            )
            expected = np.concatenate(
                (
                    e[None],
                    v,
                    np.stack(
                        [
                            h[i, j]
                            for i, j in combinations_with_replacement(range(len(v)), 2)
                        ]
                    ),
                )
            )
            oracle = "independent closed-form exchange"
            if name in ("PBE", "LDA_XC_PW"):
                correlation = "GGA_C_PBE" if name == "PBE" else "LDA_C_PW"
                expected += arrays[f"{tag}_{correlation}"]
                oracle += " + Libxc 7.0.0 correlation"
        if name == "MGGA_X_R2SCAN" and spin == "polarized":
            # Exchange is exactly spin separable: a-channel variables are
            # (rho_a, sigma_aa, tau_a), b-channel variables are
            # (rho_b, sigma_bb, tau_b), and sigma_ab is absent. Libxc's
            # reduced-variable algebra can lose these exact zeros near density
            # and polarization boundaries, so retain raw values but use the
            # structural zero oracle for those entries.
            groups = ({0, 2, 5}, {1, 4, 6})
            outputs = [
                (),
                *((i,) for i in range(7)),
                *combinations_with_replacement(range(7), 2),
            ]
            for row, output in enumerate(outputs):
                if len(output) == 1 and output[0] == 3:
                    expected[row] = 0
                elif len(output) == 2:
                    i, j = output
                    cross_spin = any(
                        (i in left and j in right) or (j in left and i in right)
                        for left, right in ((groups[0], groups[1]),)
                    )
                    if 3 in output or cross_spin:
                        expected[row] = 0
            oracle = "Libxc 7.0.0 with independent spin-separability zeros"
    return {**metadata, "oracle": oracle}, features, expected, raw
