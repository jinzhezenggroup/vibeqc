"""Explicit scalar spelling and precision contracts for TensorIR CUDA.

This is a lowering type table, not a second mathematical IR. Each primitive
keeps its declared dtype; independently typed components need no implicit cast.
"""

from __future__ import annotations

import math
import os
import shlex
import struct
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class CudaScalar:
    dtype: str
    ctype: str
    itemsize: int
    prefix: str
    suffix: str
    precision: int

    def intrinsic(self, op: str) -> str:
        return f"__{self.prefix}{op}_rn"

    def coefficient(self, pair) -> float:
        """Match interpreter conversion: rational -> FP64 -> declared dtype."""
        try:
            value = float(Fraction(*pair))
            if self.itemsize == 4:
                value = struct.unpack("f", struct.pack("f", value))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError(
                f"tensor coefficient is outside finite FP{8 * self.itemsize}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(
                f"tensor coefficient is outside finite FP{8 * self.itemsize}"
            )
        return value

    def literal(self, pair) -> str:
        return self.coefficient(pair).hex() + self.suffix

    @property
    def zero(self) -> str:
        return "0.0" + self.suffix

    @property
    def one(self) -> str:
        return "1.0" + self.suffix


_SCALARS = {
    "float32": CudaScalar("float32", "float", 4, "f", "f", 24),
    "float64": CudaScalar("float64", "double", 8, "d", "", 53),
}


def scalar_type(dtype: str) -> CudaScalar:
    try:
        return _SCALARS[dtype]
    except KeyError as error:
        raise ValueError("CUDA tensors require float32 or float64") from error


def program_precision(program) -> str:
    dtypes = {node.spec.dtype for node in program.live_nodes}
    if len(dtypes) == 1:
        return "fp32" if dtypes == {"float32"} else "fp64"
    return "typed-fp32-fp64"


def compile_options(plan) -> tuple[str, ...]:
    """Keep the FP64 baseline; require gradual underflow for FP32 kernels."""
    if any(s.node.spec.dtype == "float32" for s in plan.steps):
        # NVCC_APPEND_FLAGS can override explicit command-line flags. Reject
        # arithmetic-changing overrides instead of caching a mislabeled binary.
        for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
            tokens = shlex.split(os.environ.get(name, ""))
            for i, token in enumerate(tokens):
                if token in ("--use_fast_math", "-use_fast_math"):
                    raise ValueError("FP32 TensorIR forbids fast-math overrides")
                key, separator, value = token.partition("=")
                if not separator and i + 1 < len(tokens):
                    value = tokens[i + 1]
                forbidden = {
                    "--ftz": "true",
                    "-ftz": "true",
                    "--prec-div": "false",
                    "-prec-div": "false",
                    "--prec-sqrt": "false",
                    "-prec-sqrt": "false",
                }
                if key in forbidden and value.lower() in (
                    forbidden[key],
                    "1" if forbidden[key] == "true" else "0",
                ):
                    raise ValueError(
                        "FP32 TensorIR requires strict arithmetic compiler flags"
                    )
        return ("--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    return ("--fmad=false",)


def symmetry_tolerance(dtype: str) -> tuple[float, float]:
    # Same atol/rtol as the CPU interpreter, evaluated in FP64 validation
    # scratch on both host and device. This is not tensor compute promotion.
    return (1e-6, 1e-5) if dtype == "float32" else (1e-11, 1e-10)
