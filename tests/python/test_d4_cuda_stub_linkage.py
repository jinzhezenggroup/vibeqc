"""Compile real D4 kernel declarations as CUDA host stubs, without GPU math.

CuMetal registers these stub addresses from another translation unit. Empty
bodies and small dummy argument types isolate linkage, not layout or numerics.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_PREFIX = r"""
#include <cstddef>
#define __global__ __attribute__((global))
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
int cudaConfigureCall(dim3,dim3,std::size_t=0,void* =nullptr){return 0;}
extern "C" int cudaSetupArgument(const void*,std::size_t,std::size_t){return 0;}
extern "C" int cudaLaunch(const void*){return 0;}
namespace vibeqc::dft::dispersion {
struct D4CudaBatch{}; struct D4Parameters{}; struct D4Tables{}; struct D4CudaResult{};
"""


@pytest.mark.parametrize("negative_control", [False, True])
def test_d4_registration_resolves_host_stubs(
    tmp_path: Path, negative_control: bool
) -> None:
    compiler, nm = shutil.which("clang++"), shutil.which("nm")
    if sys.platform != "linux" or compiler is None or nm is None:
        pytest.skip("requires Linux CUDA-capable Clang and nm; no CUDA toolkit")
    source = (ROOT / "src/dft/dispersion/d4_cuda.cu").read_text(encoding="utf-8")
    namespace = re.search(
        r"namespace\s*(\w*)\s*\{\s*constexpr int kThreadsPerBlock", source
    )
    assert namespace is not None
    scope = "" if negative_control else namespace[1]
    declarations = re.findall(r"__global__ void (\w+)\(([^)]*)\)\s*\{", source)
    assert len(declarations) == 7
    definitions, calls = [], []
    arguments = {
        "D4CudaBatch": "D4CudaBatch{}",
        "D4Parameters": "D4Parameters{}",
        "D4Tables": "D4Tables{}",
        "D4CudaResult": "D4CudaResult{}",
        "double*": "nullptr",
        "bool": "false",
    }
    for name, parameters in declarations:
        definitions.append(f"__global__ void {name}({parameters}) {{}}")
        values = [
            arguments[p.strip().rsplit(None, 1)[0]] for p in parameters.split(",")
        ]
        calls.append(f"{name}<<<1,1>>>({','.join(values)});")
    unit = tmp_path / "stubs.cu"
    unit.write_text(
        _PREFIX
        + f"namespace {scope} {{\n"
        + "\n".join(definitions)
        + '\nextern "C" void retain_stubs() {\n'
        + "\n".join(calls)
        + "\n}\n}\n}\n",
        encoding="utf-8",
    )
    obj = tmp_path / "stubs.o"
    subprocess.run(
        [
            compiler,
            "-x",
            "cuda",
            "--cuda-host-only",
            "-nocudainc",
            "-nocudalib",
            "-std=c++20",
            "-O0",
            "-c",
            str(unit),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    symbols = subprocess.run(
        [nm, "--defined-only", str(obj)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    stubs = [
        line.split()[-1] for line in symbols.splitlines() if "__device_stub__" in line
    ]
    assert len(stubs) == len(declarations)
    registration = tmp_path / "registration.cpp"
    registration.write_text(
        "\n".join(
            f'extern "C" void stub_{i}() asm("{symbol}");'
            for i, symbol in enumerate(stubs)
        )
        + "\nvoid (*volatile registered[])() = {"
        + ",".join(f"&stub_{i}" for i in range(len(stubs)))
        + "};\nint main(){ return registered[0] == nullptr; }\n",
        encoding="utf-8",
    )
    linked = subprocess.run(
        [compiler, str(registration), str(obj), "-o", str(tmp_path / "probe")],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if negative_control:
        assert linked.returncode != 0 and "device_stub" in linked.stderr
    else:
        assert linked.returncode == 0, linked.stdout + linked.stderr
        subprocess.run([str(tmp_path / "probe")], check=True, timeout=10)
