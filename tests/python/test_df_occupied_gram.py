"""Compiler Gram admission agrees with its emitted native capacity contract."""

import ctypes
import subprocess
from typing import Any

import pytest
from generativeqc_compiler.method.df_exchange_schedule import native_header
from generativeqc_compiler.method.df_occupied_gram_cuda import occupied_gram_splits


@pytest.fixture(scope="module")
def native_gate(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("occupied-gram-gate")
    source = root / "gate.cpp"
    source.write_text(
        native_header()
        + r"""
extern "C" std::size_t query(std::size_t n,std::size_t k,std::size_t capacity) {
 return generativeqc::scf::generated::df_occupied_gram_splits(n,k,capacity);
}
"""
    )
    library = root / "gate.so"
    subprocess.run(
        [
            "ccache",
            "c++",
            "-std=c++20",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    query = ctypes.CDLL(str(library)).query
    query.argtypes = [ctypes.c_size_t] * 3
    query.restype = ctypes.c_size_t
    return query


def test_native_admission_respects_exact_capacity_and_integer_bounds(
    native_gate: Any,
) -> None:
    """One missing double, short work, or an excessive grid retains BLAS."""
    for n, k in [
        (384, 32768),
        (385, 32769),
        (768, 593920),
        (864, 751680),
        (900, 720000),
    ]:
        splits = (k + 32767) // 32768
        exact = n * n * splits
        for capacity, expected in [
            (exact - 1, 0),
            (exact, splits),
            (exact * 2, splits),
        ]:
            assert native_gate(n, k, capacity) == expected
            assert occupied_gram_splits(n, k, capacity) == expected
    for n, k in [
        (0, 0),
        (383, 593920),
        (768, 32767),
        (768, 32768 * 64 + 1),
        (46341, 65536),
        (2**63, 65536),
        (768, 2**31),
    ]:
        assert native_gate(n, k, 2**64 - 1) == 0
        assert occupied_gram_splits(n, k, 2**64 - 1) == 0
