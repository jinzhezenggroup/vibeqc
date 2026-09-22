"""Corrupt raw snapshots must never be reported as an exact match."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.issue434_fixed_density import chunked_maximum_difference


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("side", ("reference", "native"))
@pytest.mark.parametrize("chunk", (1, 8))
def test_raw_comparison_rejects_nonfinite_values(
    tmp_path: Path, bad: float, side: str, chunk: int
) -> None:
    reference = np.array([1.0, 2.0, 3.0])
    native = reference.copy()
    (reference if side == "reference" else native)[1] = bad
    left, right = tmp_path / "reference.npy", tmp_path / "native.bin"
    np.save(left, reference)
    native.tofile(right)
    with pytest.raises(ValueError, match="finite"):
        chunked_maximum_difference(left, right, chunk=chunk)


@pytest.mark.parametrize("extra_bytes", (-8, 1, 8))
def test_raw_comparison_requires_exact_payload_size(
    tmp_path: Path, extra_bytes: int
) -> None:
    array = np.array([1.0, 2.0, 3.0])
    left, right = tmp_path / "reference.npy", tmp_path / "native.bin"
    np.save(left, array)
    raw = array.tobytes()
    right.write_bytes(
        raw[:extra_bytes] if extra_bytes < 0 else raw + b"x" * extra_bytes
    )
    with pytest.raises(ValueError, match="size"):
        chunked_maximum_difference(left, right)


@pytest.mark.parametrize("chunk", (0, -1, True, 1.5))
def test_raw_comparison_rejects_invalid_chunk_size(
    tmp_path: Path, chunk: object
) -> None:
    left, right = tmp_path / "reference.npy", tmp_path / "native.bin"
    np.save(left, np.ones(3))
    np.ones(3).tofile(right)
    with pytest.raises(ValueError, match="positive integer"):
        chunked_maximum_difference(left, right, chunk=chunk)


@pytest.mark.parametrize("chunk", (1, 2, 64))
def test_raw_comparison_preserves_finite_maximum(tmp_path: Path, chunk: int) -> None:
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])
    native = np.array([[1.0, 1.75], [3.5, 4.0]])
    left, right = tmp_path / "reference.npy", tmp_path / "native.bin"
    np.save(left, reference)
    native.tofile(right)
    assert chunked_maximum_difference(left, right, chunk=chunk) == 0.5


@pytest.mark.parametrize("dtype", (np.float32, np.int64))
def test_raw_comparison_rejects_non_float64_reference(
    tmp_path: Path, dtype: object
) -> None:
    left, right = tmp_path / "reference.npy", tmp_path / "native.bin"
    np.save(left, np.ones(3, dtype=dtype))
    np.ones(3).tofile(right)
    with pytest.raises(ValueError, match="float64"):
        chunked_maximum_difference(left, right)
