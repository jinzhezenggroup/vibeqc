"""Small native D3(BJ) CPU/CUDA qualification provider derived from xTBloom.

This is a repository-only, synchronous, single-molecule baseline. It does not
register a production method, perform SCF, call xTBloom, or call simple-dftd3.
The expanded pair table is explicitly budgeted, not claimed to scale linearly.
"""

import ctypes
import hashlib
import json
import os
import shlex
import subprocess
import typing
from functools import lru_cache
from pathlib import Path

import numpy as np
from vibeqc_compiler.method.dispersion import D3Spec

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "external" / "xtbloom-d3"
_POINTER = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")


@lru_cache(maxsize=1)
def _tables() -> typing.Any:
    manifest = json.loads((_DATA / "manifest.json").read_text())
    values = {}
    for name, expected in manifest["data"].items():
        raw = (_DATA / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"D3 source data digest mismatch: {name}")
        values[name] = json.loads(raw)
    data = values["gfn1_d3.json"]
    radii = values["covalent_radii.json"]
    if len(data["elements"]) != 86 or len(radii) != 86:
        raise ValueError("D3 reference domain is exactly H through Rn")
    return data, radii, manifest["data"]


def make_spec(
    *,
    s6: typing.Any,
    s8: typing.Any,
    a1: typing.Any,
    a2: typing.Any,
    **kwargs: typing.Any,
) -> typing.Any:
    """Bind explicit parameters to the actual canonical reference-data digests."""
    _, _, identities = _tables()
    return D3Spec(
        s6=s6,
        s8=s8,
        a1=a1,
        a2=a2,
        table_sha256=identities["gfn1_d3.json"],
        radii_sha256=identities["covalent_radii.json"],
        **kwargs,
    )


def gfn1_compatibility() -> typing.Any:
    """The named GFN1 two-body profile, not a default for arbitrary DFT."""
    return make_spec(
        s6=1.0,
        s8=2.4,
        a1=0.63,
        a2=5.0,
        cn_cutoff=25.0,
        pair_cutoff=50.0,
        pair_switch_width=0.05,
    )


def build_reference(
    directory: typing.Any,
    *,
    backend: typing.Any = "cpu",
    nvcc: typing.Any = None,
    cuda_arch: typing.Any = None,
) -> typing.Any:
    """Explicit finite compiler call; never performed at import or evaluation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    native = Path(__file__).parent / "native"
    result = directory / f"d3_{backend}.so"
    if backend == "cpu":
        command = shlex.split(os.environ.get("CXX", "g++")) + [
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            str(native / "reference.cpp"),
            "-o",
            str(result),
        ]
    elif backend == "cuda":
        if nvcc is None or cuda_arch is None:
            raise ValueError(
                "CUDA qualification requires explicit NVCC and target architecture"
            )
        command = [
            str(nvcc),
            "-std=c++17",
            "-O2",
            "--fmad=false",
            f"-arch={cuda_arch}",
            "-shared",
            "-Xcompiler=-fPIC",
            str(native / "reference.cu"),
            "-o",
            str(result),
        ]
    else:
        raise ValueError("backend must be cpu or cuda")
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    return result


class NativeD3:
    """Explicit migrated reference implementation; energy in Eh, gradient in Eh/bohr."""

    def __init__(self, library: typing.Any) -> None:
        self._library = ctypes.CDLL(str(Path(library).resolve()))
        self._function = self._library.vibeqc_d3_reference
        self._function.argtypes = [ctypes.c_int64, _POINTER, _POINTER, _POINTER]
        self._function.restype = ctypes.c_int
        identity = self._library.vibeqc_d3_backend
        identity.argtypes = []
        identity.restype = ctypes.c_int
        self.backend = {0: "cpu", 1: "cuda"}[identity()]

    def evaluate(
        self,
        spec: typing.Any,
        numbers: typing.Any,
        positions: typing.Any,
        *,
        memory_budget_bytes: typing.Any = 64 * 1024**2,
    ) -> typing.Any:
        if not isinstance(spec, D3Spec):
            raise TypeError("D3 evaluation requires a D3Spec")
        numbers = np.asarray(numbers)
        if numbers.ndim != 1 or numbers.dtype.kind not in "iu":
            raise ValueError("atomic numbers must be a one-dimensional integer array")
        n = numbers.size
        if not 1 <= n <= 512 or np.any((numbers < 1) | (numbers > 86)):
            raise ValueError("reference scope is 1..512 atoms, each with Z=1..86")
        pairs = n * (n - 1) // 2
        size = 12 * n + 51 * pairs
        # Logical single native allocation: projected inputs + parameters +
        # candidates + scratch + error flag. Python/JSON overhead is excluded.
        required = (size + 5 + 1 + 3 * n + 16 * n) * 8 + 4
        if (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, int)
            or memory_budget_bytes < 0
        ):
            raise ValueError("memory budget must be a nonnegative integer")
        if required > memory_budget_bytes:
            raise MemoryError(f"D3 reference requires {required} native logical bytes")
        positions = np.asarray(positions)
        if positions.dtype.kind not in "fiu" or positions.shape != (n, 3):
            raise ValueError("positions require a real (natoms,3) array in bohr")
        xyz = np.array(positions, dtype=np.float64, order="C", copy=True)
        if not np.all(np.isfinite(xyz)):
            raise ValueError("positions must be finite")
        data, covalent, identities = _tables()
        if (
            spec.table_sha256 != identities["gfn1_d3.json"]
            or spec.radii_sha256 != identities["covalent_radii.json"]
        ):
            raise ValueError("specification is bound to different reference data")
        packed = np.zeros(size, dtype=np.float64)
        packed[: 3 * n] = xyz.ravel()
        packed[3 * n : 4 * n] = [covalent[int(z) - 1] for z in numbers]
        counts = [data["elements"][int(z) - 1]["reference_count"] for z in numbers]
        packed[4 * n : 5 * n] = counts
        cn = packed[5 * n : 12 * n].reshape(n, 7)
        for i, z in enumerate(numbers):
            entry = data["elements"][int(z) - 1]
            off = entry["reference_offset"]
            cn[i, : counts[i]] = data["coordination_numbers"][off : off + counts[i]]
        c6 = packed[12 * n : 12 * n + 49 * pairs].reshape(pairs, 7, 7)
        rr = packed[12 * n + 49 * pairs : 12 * n + 50 * pairs]
        damping = packed[12 * n + 50 * pairs :]
        for j in range(1, n):
            for i in range(j):
                p = j * (j - 1) // 2 + i
                zi, zj = int(numbers[i]), int(numbers[j])
                lo, hi = min(zi, zj), max(zi, zj)
                row = data["pair_records"][lo - 1 + hi * (hi - 1) // 2]
                off = row["c6_offset"]
                ni, nj = row["first_reference_count"], row["second_reference_count"]
                block = np.asarray(data["c6"][off : off + ni * nj]).reshape(nj, ni).T
                c6[p, : counts[i], : counts[j]] = block if zi <= zj else block.T
                rr[p] = 3 * data["r4r2"][zi - 1] * data["r4r2"][zj - 1]
                damping[p] = spec.a1 * np.sqrt(rr[p]) + spec.a2
        if not np.all(np.isfinite(packed)):
            raise ValueError("D3 projected parameters must be finite")
        parameters = np.array(
            [
                spec.s6,
                spec.s8,
                spec.cn_cutoff if spec.cn_cutoff is not None else np.inf,
                spec.pair_cutoff if spec.pair_cutoff is not None else np.inf,
                spec.pair_switch_width,
            ],
            dtype=np.float64,
        )
        output = np.zeros(1 + 3 * n, dtype=np.float64)
        status = self._function(n, packed, parameters, output)
        if status:
            raise RuntimeError(
                f"native {self.backend} D3 qualification failed: {status}"
            )
        return float(output[0]), output[1:].reshape(n, 3).copy()
