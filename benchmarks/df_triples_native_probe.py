"""Qualify native DF (T) with supplied native-source/independent CC states.

Run under a finite Slurm GPU allocation. This validation adapter reads the DF
solver's binary replay format and pinned df_ccsdt_large_oracle.py states. It
does not run RHF, source construction, CCSD or forces. The loaded C adapter is
tests/native/df_triples_probe.cpp, built against the specified native library.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from benchmarks._retention import raw_output_path

COUNTS = (
    "virtual_triples",
    "occupied_tiles",
    "workspace_bytes",
    "arena_bytes",
    "provider_retained_bytes",
    "panel_capacity",
    "panel_gemms",
    "moment_gemms",
    "epilogue_kernels",
    "reduction_kernels",
    "epilogue_points",
    "contraction_summands",
    "h2d_bytes",
    "d2h_bytes",
)


def sha256(path: Path) -> str:
    """Hash large retained inputs without making a second resident copy."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--max-bytes", type=int, default=24 << 30)
    parser.add_argument("--panels", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    with args.input.open("rb") as stream:
        header = np.fromfile(stream, dtype="<u8", count=7)
    if len(header) != 7 or any(int(x) <= 0 for x in header[:3]):
        raise ValueError("invalid supplied DF solver-state header")
    o, v, q = map(int, header[:3])
    sizes = [
        o * o,
        o * v,
        v * v,
        o * v * o * v,
        o * v * v * o,
        o * o * v * v,
        0,
        o * v * o * o,
        o**4,
        0,
        o * v,
        o * o * v * v,
        o * v,
        o * o * v * v,
        q * o * v,
        q * v * v,
    ]
    if args.input.stat().st_size != 56 + 8 * sum(sizes):
        raise ValueError("supplied DF solver-state size mismatch")
    mapped = np.memmap(args.input, dtype="<f8", offset=56, mode="r")
    fields, offset = [], 0
    for size in sizes:
        fields.append(mapped[offset : offset + size])
        offset += size
    reference = json.loads(args.reference.read_text())
    if [reference[key] for key in ("nocc", "nvir", "naux")] != [o, v, q]:
        raise ValueError("reference dimensions do not match the supplied state")
    with np.load(args.state, allow_pickle=False) as state:
        eps = np.ascontiguousarray(state["orbital_energies"], dtype=np.float64)
    if eps.shape != (o + v,):
        raise ValueError("orbital-energy shape mismatch")
    arrays = [
        fields[14],
        fields[15],
        fields[7],
        fields[3],
        fields[1],
        fields[12],
        fields[13],
        eps[:o],
        eps[o:],
    ]
    pointers = (ct.POINTER(ct.c_double) * 9)(
        *(x.ctypes.data_as(ct.POINTER(ct.c_double)) for x in arrays)
    )
    # Preload the specified SONAME before the validation adapter resolves it.
    library = ct.CDLL(str(args.library.resolve()), mode=ct.RTLD_GLOBAL)
    probe = ct.CDLL(str(args.probe.resolve()))
    call = probe.df_triples_probe
    call.argtypes = [
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.POINTER(ct.c_double)),
        ct.c_double,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_size_t),
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    values = np.full(3, np.nan)
    counts = np.zeros(len(COUNTS), dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    started = time.perf_counter()
    status = call(
        o,
        v,
        q,
        pointers,
        1e-10,
        args.max_bytes,
        args.panels,
        values.ctypes.data_as(ct.POINTER(ct.c_double)),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    wall = time.perf_counter() - started
    energy_error = (
        float(abs(values[0] - reference["triples_energy"])) if status == 0 else None
    )
    record = {
        "scope": "Complete native DF (T) owner at supplied orbitals/amplitudes; excludes RHF/source construction/CCSD solve/forces",
        "shape": [o, v, q],
        "status": status,
        "error": error.value.decode(),
        "energy": float(values[0]) if status == 0 else None,
        "minimum_absolute_denominator": float(values[1]) if status == 0 else None,
        "native_seconds": float(values[2]) if status == 0 else None,
        "call_wall_seconds": wall,
        "counts": dict(zip(COUNTS, map(int, counts), strict=True)),
        "reference_energy": reference["triples_energy"],
        "absolute_energy_error": energy_error,
        "absolute_energy_gate": 3e-10,
        "sha256": {
            key: sha256(getattr(args, key))
            for key in ("input", "state", "reference", "probe", "library")
        },
    }
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2, allow_nan=False), flush=True)
    # Keep both dynamic-library handles and borrowed NumPy buffers live through
    # complete result publication, including exception-path stream cleanup.
    del library, probe
    if status or energy_error is None or energy_error >= 3e-10:
        raise RuntimeError("native DF triples qualification failed")


if __name__ == "__main__":
    main()
