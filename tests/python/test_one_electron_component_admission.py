"""Execute the generated Cartesian mapper on legal and out-of-domain powers."""

import ctypes
import shutil
import subprocess

import pytest
from vibeqc_compiler.integral.one_electron_cuda import _emit_component_index
from vibeqc_compiler.integral.shell_spec import cartesian_components


@pytest.fixture(scope="module", params=range(5))
def mapper(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    maximum = request.param
    folder = tmp_path_factory.mktemp(f"component-map-{maximum}")
    source = folder / "mapper.cpp"
    source.write_text(
        "#define __device__\n#define __forceinline__ inline\n"
        + _emit_component_index(maximum)
        + '\nextern "C" unsigned map_component(unsigned x,unsigned y,unsigned z)'
        + " {return component_index(x,y,z);}\n"
    )
    library = folder / "mapper.so"
    subprocess.run(
        [compiler, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    module = ctypes.CDLL(str(library))
    module.map_component.argtypes = [ctypes.c_uint] * 3
    module.map_component.restype = ctypes.c_uint
    return maximum, module


def test_valid_components_keep_canonical_order(mapper: tuple) -> None:
    maximum, module = mapper
    components = [c for l in range(maximum + 1) for c in cartesian_components(l)]
    for index, component in enumerate(components):
        assert module.map_component(*(component.count(axis) for axis in "xyz")) == index


def test_out_of_domain_powers_cannot_alias_valid_components(mapper: tuple) -> None:
    maximum, module = mapper
    sentinel = sum(len(cartesian_components(l)) for l in range(maximum + 1))
    radix = maximum + 1
    cases = [
        (0, 0, radix),
        (0, radix, 0),
        (radix, 0, 0),
        (2**32 - 1, 1, 0),
        (0, 2**32 - 1, 1),
    ]
    for powers in cases:
        assert module.map_component(*powers) == sentinel, powers
