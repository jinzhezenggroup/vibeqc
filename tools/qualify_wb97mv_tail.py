"""Evaluate captured CUDA XC points with the original Libxc 7 Maple C in quad precision.

Requires the pinned Libxc source archive and GCC/libquadmath. This offline
diagnostic does not import the VibeQC compiler, runtime or generated formula.
The input is the opt-in binary capture from vibeqc_dft_cuda_tests; each output
row is energy density, two rho, three sigma and two tau derivatives.
"""

import argparse
import hashlib
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77"
PREFIX = "libxc-7.0.0/src/"


def oracle_source(archive: Path) -> str:
    """Widen only the upstream polarized E/vxc routine, preserving its branches."""
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("expected the pinned Libxc 7.0.0 archive")
    with tarfile.open(archive) as source:
        formula_file = source.extractfile(
            PREFIX + "maple2c/mgga_exc/hyb_mgga_xc_wb97mv.c"
        )
        utility_file = source.extractfile(PREFIX + "util.h")
        assert formula_file is not None and utility_file is not None
        formula, utilities = formula_file.read().decode(), utility_file.read().decode()

    start = formula.index("GPU_DEVICE_FUNCTION static inline void\nfunc_vxc_pol(")
    end = formula.index("\n}\n", start) + 3
    formula = formula[start:end].replace("double", "real")
    formula = re.sub(r"(?<![\w.])(\d+\.\d*(?:[eE][+-]?\d+)?)", r"\1Q", formula)
    formula = re.sub(r"\b(sqrt|pow|exp|log|log1p|erf)\(", r"\1q(", formula)
    for name in ("M_PI", "M_CBRT2", "M_CBRT3", "M_CBRT4", "M_CBRT6"):
        match = re.search(r"#\s*define\s+" + name + r"\s+([0-9.]+)L?", utilities)
        if match is None:
            raise ValueError(f"missing pinned Libxc constant {name}")
        formula = re.sub(r"\b" + name + r"\b", match[1] + "Q", formula)
    template = (ROOT / "tests/data/xc/wb97mv-tail-oracle.cpp.in").read_text()
    return template.replace("@FORMULA@", formula)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = oracle_source(args.archive)
    with tempfile.TemporaryDirectory(prefix="vibeqc-wb97mv-quad-") as directory:
        path = Path(directory)
        (path / "oracle.cpp").write_text(source)
        subprocess.run(
            [
                "c++",
                "-std=gnu++17",
                "-O2",
                str(path / "oracle.cpp"),
                "-lquadmath",
                "-o",
                str(path / "oracle"),
            ],
            check=True,
        )
        subprocess.run(
            [str(path / "oracle"), str(args.capture), str(args.output)], check=True
        )


if __name__ == "__main__":
    main()
