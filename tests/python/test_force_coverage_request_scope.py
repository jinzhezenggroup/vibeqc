"""Compile the actual force coverage predicate; energy-only requests bypass it."""

import ctypes
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rejects(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., bool]:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    text = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    match = re.search(
        r"if \(([^{};]*unexpected_tuned_spd_fallback_mask != 0U)\) \{", text
    )
    assert match is not None
    root = tmp_path_factory.mktemp("force-coverage")
    source = root / "probe.cpp"
    source.write_text(
        "struct Options { bool compute_forces; };\n"
        "struct Profile { bool tuned, compatible; };\n"
        'extern "C" bool reject(bool compute, bool bounded_direct_streaming, bool tuned, '
        "bool compatible, bool override_filter, unsigned long long unexpected_tuned_spd_fallback_mask) {\n"
        "Options options{compute}; Profile selected_aot_profile{tuned,compatible};\n"
        "auto aot_shell_class_selection_override_requested = [&] { return override_filter; };\n"
        "return " + match.group(1) + "; }\n"
    )
    library = root / "probe.so"
    subprocess.run(
        [compiler, "-std=c++20", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    loaded = ctypes.CDLL(str(library))
    loaded.reject.argtypes = [ctypes.c_bool] * 5 + [ctypes.c_ulonglong]
    loaded.reject.restype = ctypes.c_bool
    return loaded.reject


@pytest.mark.parametrize(
    "args, expected",
    [
        ((False, True, True, True, False, 1), False),
        ((True, True, True, True, False, 1), True),
        ((True, True, True, True, True, 1), False),
        ((True, True, False, True, False, 1), False),
        ((True, True, True, False, False, 1), False),
        ((True, False, True, True, False, 1), False),
        ((True, True, True, True, False, 0), False),
    ],
)
def test_force_coverage_applies_only_to_force_requests(
    rejects: Callable[..., bool], args: tuple[bool | int, ...], expected: bool
) -> None:
    assert rejects(*args) is expected
