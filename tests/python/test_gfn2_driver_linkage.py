"""Native GFN2 driver imports must not break CPU-only library loading."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_driver_imports_are_lazy_and_preserve_versioned_calls(tmp_path: Path) -> None:
    if platform.system() != "Linux" or shutil.which("cmake") is None:
        pytest.skip("ELF linkage test requires Linux and CMake")
    (tmp_path / "tools").symlink_to(ROOT / "tools", target_is_directory=True)
    (tmp_path / "cmake").symlink_to(ROOT / "cmake", target_is_directory=True)
    (tmp_path / "probe.cpp").write_text(r"""
#include <cstddef>
extern "C" int cuGetErrorString(int,const char**);
extern "C" int cuMemGetAddressRange_v2(unsigned long long*,std::size_t*,unsigned long long);
extern "C" int alive() {return 42;}
extern "C" int query() {const char* text=nullptr;unsigned long long base=0;std::size_t size=0;
  if(cuGetErrorString(0,&text) || !text) return -1;
  if(cuMemGetAddressRange_v2(&base,&size,4097)) return -2;
  return base==4096 && size==1024 ? 73 : -3;}
""")
    (tmp_path / "driver.cpp").write_text(r"""
#include <cstddef>
extern "C" int cuGetErrorString(int,const char** out) {*out="fake driver";return 0;}
extern "C" int cuMemGetAddressRange_v2(unsigned long long* base,std::size_t* size,unsigned long long ptr) {
  if(ptr!=4097) return 1;*base=4096;*size=1024;return 0;}
""")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(DriverProbe LANGUAGES CXX)\nenable_language(C ASM)\n"
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        f'include("{ROOT / "cmake/VibeQCCudaImplib.cmake"}")\n'
        "add_library(probe SHARED probe.cpp)\nvibeqc_attach_cuda_driver_implib(probe)\n"
        'target_link_options(probe PRIVATE "LINKER:-z,defs")\n'
        "add_library(cuda SHARED driver.cpp)\nset_target_properties(cuda PROPERTIES SOVERSION 1)\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "-j2"],
        check=True,
        capture_output=True,
        timeout=60,
    )
    dynamic = subprocess.check_output(
        ["readelf", "-d", str(build / "libprobe.so")], text=True
    )
    assert "Shared library: [libcuda" not in dynamic
    # No driver call is made during ordinary CPU loading/preflight.
    script = "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); assert lib.alive()==42"
    subprocess.run(
        [sys.executable, "-c", script, str(build / "libprobe.so")],
        check=True,
        timeout=15,
    )
    environment = {**os.environ, "LD_LIBRARY_PATH": str(build)}
    subprocess.run(
        [
            sys.executable,
            "-c",
            script + "; assert lib.query()==73",
            str(build / "libprobe.so"),
        ],
        env=environment,
        check=True,
        timeout=15,
    )
