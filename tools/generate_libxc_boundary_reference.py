#!/usr/bin/env python3
"""Generate independent Libxc E/vxc boundary fixtures for bulk semilocal XC.

This optional oracle tool calls the Libxc 7.0.0 C API through PySCF.  It reads
only registration IDs plus deterministic physical probes; no VibeQC Graph,
Maple lowering, generated kernel or derivative participates in the reference.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any

import numpy as np
from vibeqc_compiler.xc.boundary import (
    BOUNDARY_SEMANTICS,
    bulk_feature_names,
    semilocal_boundary_probes,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "python/vibeqc_compiler/xc/libxc_bulk_catalog.json"
OUTPUT = ROOT / "tests/data/xc/libxc-boundary"
_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


def _groups(family: str, spin: str, mapping: dict[str, float]) -> list[np.ndarray]:
    if spin == "polarized":
        values: list[list[float]] = [[mapping["rho_a"], mapping["rho_b"]]]
        if family != "lda":
            values.append(
                [mapping["sigma_aa"], mapping["sigma_ab"], mapping["sigma_bb"]]
            )
        if family == "mgga":
            values.extend(
                (
                    [mapping["lapl_a"], mapping["lapl_b"]],
                    [mapping["tau_a"], mapping["tau_b"]],
                )
            )
    else:
        values = [[mapping["rho"]]]
        if family != "lda":
            values.append([mapping["sigma"]])
        if family == "mgga":
            values.extend(([mapping["lapl"]], [mapping["tau"]]))
    return [np.asarray(value, dtype=np.float64) for value in values]


def evaluate_reference(
    library: Any,
    record: dict[str, Any],
    spin: str,
    feature_names: tuple[str, ...],
    values: tuple[float, ...],
) -> dict[str, Any]:
    """Evaluate energy per volume plus first physical feature derivatives."""
    mapping = dict(zip(feature_names, values, strict=True))
    inputs = _groups(record["family"], spin, mapping)
    sizes = [len(value) for value in inputs]
    outputs = [np.zeros(1), *[np.zeros(size) for size in sizes]]
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
        getattr(library, f"xc_{record['family']}_exc_vxc")(
            function,
            1,
            *[array.ctypes.data_as(_DOUBLE_POINTER) for array in inputs + outputs],
        )
    finally:
        if initialized:
            library.xc_func_end(function)
        library.xc_func_free(function)
    outputs[0] *= np.sum(inputs[0])
    expected = np.concatenate(outputs)
    finite = bool(np.all(np.isfinite(expected)))
    return {
        "features": list(values),
        "oracle_finite": finite,
        "expected": expected.tolist() if finite else None,
        "nonfinite_outputs": (
            [] if finite else np.flatnonzero(~np.isfinite(expected)).tolist()
        ),
    }


def _configure(library: Any) -> None:
    library.xc_func_alloc.argtypes = []
    library.xc_func_alloc.restype = ctypes.c_void_p
    library.xc_func_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.xc_func_init.restype = ctypes.c_int
    for name in ("xc_func_end", "xc_func_free"):
        getattr(library, name).argtypes = [ctypes.c_void_p]
        getattr(library, name).restype = None
    for family, count in (("lda", 1), ("gga", 2), ("mgga", 4)):
        function = getattr(library, f"xc_{family}_exc_vxc")
        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t] + [_DOUBLE_POINTER] * (
            2 * count + 1
        )
        function.restype = None


def main() -> int:
    import pyscf
    from pyscf.dft import libxc

    if libxc.__version__ != "7.0.0":
        raise RuntimeError("boundary reference generation requires exactly Libxc 7.0.0")
    library = libxc._itrf
    _configure(library)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for family in ("lda", "gga", "mgga"):
        cases = []
        for record in catalog["registrations"]:
            if record["graph_status"] != "imported" or record["family"] != family:
                continue
            for spin in ("polarized", "unpolarized"):
                names = bulk_feature_names(family, spin)
                probes = []
                for probe in semilocal_boundary_probes(names, spin=spin):
                    result = evaluate_reference(
                        library, record, spin, probe.feature_names, probe.values
                    )
                    probes.append({"label": probe.label, **result})
                cases.append(
                    {
                        "name": record["name"],
                        "id": record["id"],
                        "spin": spin,
                        "feature_names": list(names),
                        "probes": probes,
                    }
                )
        payload = {
            "schema": "vibeqc.libxc-boundary-reference/v1",
            "boundary_semantics": BOUNDARY_SEMANTICS,
            "oracle": {
                "pyscf": pyscf.__version__,
                "libxc": libxc.__version__,
                "api": "xc_{lda,gga,mgga}_exc_vxc",
            },
            "output_order": (
                "energy per volume followed by first derivatives in feature_names order"
            ),
            "family": family,
            "cases": cases,
        }
        (OUTPUT / f"{family}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        finite = sum(
            probe["oracle_finite"] for case in cases for probe in case["probes"]
        )
        total = sum(len(case["probes"]) for case in cases)
        print(f"{family}: {finite}/{total} finite independent boundary probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
