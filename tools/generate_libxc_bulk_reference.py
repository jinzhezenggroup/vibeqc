#!/usr/bin/env python3
"""Generate independent bulk E/vxc/fxc fixtures with the Libxc 7.0.0 C API.

This optional oracle tool is not imported by the source generator or compiler.
It reads IDs from the catalog, but no Maple expression, Graph, derivative or
parameter binding from VibeQC participates in the reference calculation.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "python/vibeqc_compiler/xc/libxc_bulk_catalog.json"
OUTPUT = ROOT / "tests/data/xc/libxc-bulk"
_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


def feature_groups(spin: str, family: str, point: int) -> list[np.ndarray]:
    """Two ordinary physical interior points, not threshold/endpoint fixtures."""
    if spin == "polarized":
        values = (
            ([0.55, 0.25], [0.025, 0.006, 0.018], [0.01, -0.005], [0.30, 0.16]),
            ([0.31, 0.62], [0.14, -0.03, 0.19], [-0.035, 0.04], [0.47, 0.56]),
        )[point]
    else:
        values = (([0.8], [0.061], [0.005], [0.46]), ([0.93], [0.27], [0.005], [1.03]))[
            point
        ]
    count = {"lda": 1, "gga": 2, "mgga": 4}[family]
    return [np.asarray(value, dtype=np.float64) for value in values[:count]]


def evaluate_reference(
    library: Any, record: dict[str, Any], spin: str, point: int
) -> tuple[list[float], list[float]]:
    inputs = feature_groups(spin, record["family"], point)
    sizes = [len(value) for value in inputs]
    blocks = [(i, j) for i in range(len(sizes)) for j in range(i, len(sizes))]
    second = [
        np.zeros(sizes[i] * (sizes[i] + 1) // 2 if i == j else sizes[i] * sizes[j])
        for i, j in blocks
    ]
    outputs = [np.zeros(1), *[np.zeros(size) for size in sizes], *second]
    function = library.xc_func_alloc()
    if not function:
        raise MemoryError("Libxc functional allocation failed")
    initialized = False
    try:
        if library.xc_func_init(
            function, record["id"], 2 if spin == "polarized" else 1
        ):
            raise ValueError(f"Libxc cannot initialize {record['name']}")
        initialized = True
        getattr(library, f"xc_{record['family']}_exc_vxc_fxc")(
            function,
            1,
            *[array.ctypes.data_as(_DOUBLE_POINTER) for array in inputs + outputs],
        )
    finally:
        if initialized:
            library.xc_func_end(function)
        library.xc_func_free(function)
    outputs[0] *= np.sum(inputs[0])  # Libxc zk is per particle, our E is per volume.
    offsets = np.cumsum([0, *sizes])
    hessian = np.zeros((sum(sizes), sum(sizes)))
    for (i, j), values in zip(blocks, second, strict=True):
        pairs = (
            [(a, b) for a in range(sizes[i]) for b in range(a, sizes[i])]
            if i == j
            else [(a, b) for a in range(sizes[i]) for b in range(sizes[j])]
        )
        for (a, b), value in zip(pairs, values, strict=True):
            x, y = offsets[i] + a, offsets[j] + b
            hessian[x, y] = hessian[y, x] = value
    packed = hessian[np.triu_indices(sum(sizes))]
    expected = np.concatenate([*outputs[: 1 + len(sizes)], packed])
    if not np.all(np.isfinite(expected)):
        raise ValueError(f"nonfinite independent reference for {record['name']}")
    return np.concatenate(inputs).tolist(), expected.tolist()


def main() -> int:
    import pyscf
    from pyscf.dft import libxc

    if libxc.__version__ != "7.0.0":
        raise RuntimeError("reference generation requires exactly Libxc 7.0.0")
    library = libxc._itrf
    library.xc_func_alloc.argtypes = []
    library.xc_func_alloc.restype = ctypes.c_void_p
    library.xc_func_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.xc_func_init.restype = ctypes.c_int
    for name in ("xc_func_end", "xc_func_free"):
        getattr(library, name).argtypes = [ctypes.c_void_p]
        getattr(library, name).restype = None
    for family, count in (("lda", 1), ("gga", 2), ("mgga", 4)):
        function = getattr(library, f"xc_{family}_exc_vxc_fxc")
        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t] + [_DOUBLE_POINTER] * (
            1 + 2 * count + count * (count + 1) // 2
        )
        function.restype = None
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for family in ("lda", "gga", "mgga"):
        cases = []
        for record in catalog["registrations"]:
            if record["graph_status"] != "imported" or record["family"] != family:
                continue
            for spin in ("polarized", "unpolarized"):
                samples = [
                    evaluate_reference(library, record, spin, point) for point in (0, 1)
                ]
                cases.append(
                    {
                        "name": record["name"],
                        "id": record["id"],
                        "spin": spin,
                        "features": [features for features, _ in samples],
                        "expected": [expected for _, expected in samples],
                    }
                )
        result = {
            "schema": "vibeqc.libxc-bulk-interior-reference/v1",
            "oracle": {
                "pyscf": pyscf.__version__,
                "libxc": libxc.__version__,
                "api": "xc_{lda,gga,mgga}_exc_vxc_fxc",
            },
            "domain": "two physical ordinary-interior points per spin; not production qualification",
            "output_order": "energy per volume, physical feature gradient, row-major upper-triangular feature Hessian",
            "family": family,
            "cases": cases,
        }
        (OUTPUT / f"{family}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"{family}: {len(cases)} spin cases, {2 * len(cases)} reference points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
