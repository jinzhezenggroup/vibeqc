#!/usr/bin/env python3
"""Regenerate the independent r2SCAN tail fixture from pinned Libxc Maple C.

Requires GCC/libquadmath and the exact Libxc 7.0.0 source archive. This is an
offline qualification tool; it imports no VibeQC algebra, AD, or runtime. The
original upstream E/vxc formulas (including their cancelling spin coordinates)
are evaluated in 113-bit arithmetic, then rounded once to binary64. The narrow
driver preserves Libxc work_mgga thresholds and raw-work derivative semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/xc/r2scan-tail-reference.json"
TEMPLATE = ROOT / "tests/data/xc/r2scan-tail-oracle.cpp.in"
ARCHIVE_SHA256 = "8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77"
PREFIX = "libxc-7.0.0/src/"


def oracle_source(archive: Path) -> tuple[str, dict[str, str]]:
    """Extract only two E/vxc functions; never unpack archive paths to disk."""
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("expected the pinned Libxc 7.0.0 archive")
    with tarfile.open(archive) as stream:
        sources = {}
        for path in (
            "util.h",
            "maple2c/mgga_exc/mgga_c_r2scan.c",
            "maple2c/mgga_exc/mgga_x_r2scan.c",
        ):
            member = stream.extractfile(PREFIX + path)
            assert member is not None
            sources[path] = member.read().decode()
    constants = ["#undef M_PI"]
    for name in ("M_PI", "M_CBRT2", "M_CBRT3", "M_CBRT4", "M_CBRT6", "M_CBRTPI"):
        match = re.search(r"#\s*define " + name + r"\s+([\d.]+)", sources["util.h"])
        assert match is not None, name
        constants.append(f"#define {name} {match[1]}Q")
    formulas = []
    for component in ("c", "x"):
        text = sources[f"maple2c/mgga_exc/mgga_{component}_r2scan.c"]
        start = text.index("GPU_DEVICE_FUNCTION static inline void\nfunc_vxc_pol(")
        stop = text.index("\n}\n", start) + 3
        text = text[start:stop].replace("double", "real")
        text = text.replace("func_vxc_pol", f"{component}_vxc")
        # Preserve original decimal constants without first rounding to FP64.
        text = re.sub(r"\b(\d+\.\d*(?:[eE][+-]?\d+)?)", r"\1Q", text)
        text = re.sub(r"\b(sqrt|pow|exp|log)\(", r"\1q(", text)
        formulas.append(text)
    source = TEMPLATE.read_text().replace("@CONSTANTS@", "\n".join(constants))
    source = source.replace("@FORMULAS@", "\n".join(formulas))
    hashes = {
        name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in sources.items()
    }
    return source, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    points = fixture["points"]
    source, hashes = oracle_source(args.archive)
    with tempfile.TemporaryDirectory(prefix="libxc-r2scan-quad-") as directory:
        path = Path(directory)
        (path / "probe.cpp").write_text(source)
        data = struct.pack("=Q", len(points))
        data += b"".join(struct.pack("=7d", *point["inputs"]) for point in points)
        (path / "inputs.bin").write_bytes(data)
        subprocess.run(
            [
                args.cxx,
                "-std=gnu++17",
                "-O2",
                str(path / "probe.cpp"),
                "-lquadmath",
                "-o",
                str(path / "probe"),
            ],
            check=True,
            timeout=120,
        )
        subprocess.run(
            [str(path / "probe"), str(path / "inputs.bin"), str(path / "outputs.bin")],
            check=True,
            timeout=30,
        )
        output = (path / "outputs.bin").read_bytes()
        assert len(output) == 8 + len(points) * 64
        for i, point in enumerate(points):
            values = struct.unpack_from("=8d", output, 8 + 64 * i)
            assert all(math.isfinite(value) for value in values), point["label"]
            point["reference"] = values
    fixture["oracle"] = {
        "source": "Libxc 7.0.0 original Maple 2022 generated polarized E/vxc",
        "archive_sha256": ARCHIVE_SHA256,
        "source_sha256": hashes,
        "arithmetic": "GCC __float128/libquadmath, 113 significand bits",
        "generated_probe_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "regenerate": "python tools/generate_r2scan_tail_reference.py --archive /path/to/libxc-7.0.0.tar.gz",
        "channels": [
            "energy_density",
            "rho_a",
            "rho_b",
            "sigma_aa",
            "sigma_ab",
            "sigma_bb",
            "tau_a",
            "tau_b",
        ],
        "tolerance": {"rtol": 5e-12, "atol": 1e-12},
        "policy": "Unchanged work_mgga FP64 floors; physical-density energy, raw-work vxc",
    }
    args.fixture.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
