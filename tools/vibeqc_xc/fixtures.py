"""Hash-checked pinned Libxc fixtures with explicit boundary-oracle selection."""

import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from vibeqc.profiles import file_hash

from .reference import exchange_reference


def load_fixture(name, *, spin="polarized", domain="typical"):
    """Return metadata, physical features, checked oracle and raw Libxc output.

    Boundary exchange uses closed-form spin scaling because Libxc's internal
    reduced-variable cancellation corrupts a mathematically zero mixed partial.
    Combined functionals use this exchange plus the unmodified Libxc correlation.
    Raw library results always remain available for diagnostics and archival.
    """
    root = Path(__file__).resolve().parents[2] / "tests/data/xc"
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
    return {**metadata, "oracle": oracle}, features, expected, raw
