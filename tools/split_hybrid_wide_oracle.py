"""Independent 113-bit M06-2X boundary oracle from pinned Libxc Maple C.

Only the upstream E/vxc formulas are widened; public admission, work floors,
FHC clipping and source parameter values retain their binary64 semantics.
The production compiler, its AD, and its generated C++ are never imported.
"""

from __future__ import annotations

import hashlib
import re
import struct
import subprocess
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

TEMPLATE = Path(__file__).with_suffix(".cpp.in")
ARCHIVE_SHA256 = "8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77"
PREFIX = "libxc-7.0.0/src/"
SOURCES = (
    "util.h",
    "mgga_c_m06l.c",
    "hyb_mgga_x_m05.c",
    "maple2c/mgga_exc/mgga_c_m06l.c",
    "maple2c/mgga_exc/hyb_mgga_x_m05.c",
)


def oracle_source(archive: Path) -> tuple[str, dict[str, str]]:
    """Extract exact source routines and parameters from the authenticated archive."""
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("expected the pinned Libxc 7.0.0 archive")
    with tarfile.open(archive) as stream:
        sources = {}
        for path in SOURCES:
            member = stream.extractfile(PREFIX + path)
            if member is None:
                raise ValueError(f"missing Libxc source {path}")
            sources[path] = member.read().decode()
    constants = ["#undef M_PI"]
    for name in ("M_PI", "M_CBRT2", "M_CBRT3", "M_CBRT4", "M_CBRT6", "M_CBRTPI"):
        match = re.search(r"#\s*define " + name + r"\s+([\d.]+)", sources["util.h"])
        if match is None:
            raise ValueError(f"missing Libxc constant {name}")
        constants.append(f"#define {name} {match[1]}Q")
    parameters = []
    for owner, struct_name, array_name in (
        ("mgga_c_m06l.c", "mgga_c_m06l_params", "m062x_values"),
        ("hyb_mgga_x_m05.c", "mgga_x_m05_params", "par_m06_2x"),
    ):
        source = sources[owner]
        definition = re.search(
            r"typedef struct\s*\{.*?\}\s*" + struct_name + r";", source, re.DOTALL
        )
        values = re.search(
            r"static const double " + array_name + r"\[\w+\]\s*=\s*\{.*?\};",
            source,
            re.DOTALL,
        )
        if definition is None or values is None:
            raise ValueError(f"missing Libxc parameters: {owner}")
        parameters.extend(
            (
                definition[0],
                values[0].replace("[M06L_N_PAR]", "[]").replace("[N_PAR]", "[]"),
            )
        )
    formulas = []
    for component, path in (
        ("c", "maple2c/mgga_exc/mgga_c_m06l.c"),
        ("x", "maple2c/mgga_exc/hyb_mgga_x_m05.c"),
    ):
        text = sources[path]
        start = text.index("GPU_DEVICE_FUNCTION static inline void\nfunc_vxc_pol(")
        stop = text.index("\n}\n", start) + 3
        formula = text[start:stop].replace("double", "real")
        formula = formula.replace("func_vxc_pol", component + "_vxc")
        formula = re.sub(r"\b(\d+\.\d*(?:[eE][+-]?\d+)?)", r"\1Q", formula)
        formula = re.sub(r"\b(sqrt|pow|exp|log)\(", r"\1q(", formula)
        formulas.append(formula)
    result = TEMPLATE.read_text().replace("@CONSTANTS@", "\n".join(constants))
    result = result.replace("@PARAMETERS@", "\n".join(parameters))
    result = result.replace("@FORMULAS@", "\n".join(formulas))
    return result, {
        path: hashlib.sha256(source.encode()).hexdigest()
        for path, source in sources.items()
    }


def evaluate(
    archive: Path, compiler: str, points: NDArray[np.float64], work: Path
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Evaluate original Libxc E/vxc algebra at 113 bits for binary64 input rows."""
    source, hashes = oracle_source(archive)
    work.mkdir(parents=True, exist_ok=True)
    (work / "wide-probe.cpp").write_text(source, encoding="utf-8")
    executable = work / "wide-probe"
    subprocess.run(
        [
            compiler,
            "-std=gnu++17",
            "-O2",
            str(work / "wide-probe.cpp"),
            "-lquadmath",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    inputs = work / "wide-input.bin"
    outputs = work / "wide-output.bin"
    inputs.write_bytes(
        struct.pack("=Q", len(points))
        + np.ascontiguousarray(points, dtype=np.float64).tobytes()
    )
    subprocess.run([str(executable), str(inputs), str(outputs)], check=True, timeout=30)
    raw = outputs.read_bytes()
    if len(raw) != 8 + points.size // 7 * 64 or struct.unpack_from("=Q", raw)[0] != len(
        points
    ):
        raise RuntimeError("wide Libxc oracle returned a truncated result")
    return np.frombuffer(raw, dtype=np.float64, offset=8).reshape(-1, 8), {
        "archive_sha256": ARCHIVE_SHA256,
        "source_sha256": hashes,
        "probe_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "precision": "GCC __float128/libquadmath (113-bit significand)",
    }
