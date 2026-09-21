"""Feature histograms must not mistake a partial factorized panel for full weights."""

import ctypes
import shutil
import subprocess
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def admission(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the production admission predicate")
    source = (ROOT / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    begin = source.index("    const bool screening_features =")
    end = source.index("    const auto response_pair_stride =", begin)
    directory = tmp_path_factory.mktemp("df-feature-admission")
    unit = directory / "admission.cpp"
    unit.write_text(
        "#include <stdexcept>\n#include <string_view>\n"
        'extern "C" bool admitted(bool factorized_exchange, bool features) {\n'
        "  const bool shell_execution = true, full_shell_domain = true;\n"
        '  const std::string_view screening_feature_policy = features ? "1" : "off";\n'
        "  try {\n"
        + source[begin:end]
        + "  } catch (const std::invalid_argument&) { return false; }\n"
        "  return true;\n}\n"
    )
    library_path = directory / "admission.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-shared",
            "-fPIC",
            str(unit),
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    library = ctypes.CDLL(str(library_path))
    function = library.admitted
    function.argtypes = [ctypes.c_bool, ctypes.c_bool]
    function.restype = ctypes.c_bool
    return function


@pytest.mark.parametrize("factorized", (False, True))
@pytest.mark.parametrize("features", (False, True))
def test_factorized_screening_features_fail_closed(
    admission: typing.Any, factorized: bool, features: bool
) -> None:
    assert admission(factorized, features) is not (factorized and features)
