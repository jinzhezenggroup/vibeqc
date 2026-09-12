"""Check actual CUDA build graphs for native direct ownership and portability."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "architectures,enabled,linked,global_link",
    [
        ("120-real", "ON", True, "OFF"),
        ("120-virtual", "ON", True, "OFF"),
        ("120", "ON", True, "OFF"),
        ("120-real", "OFF", False, "OFF"),
        ("120", "ON", True, "ON"),
    ],
)
def test_direct_device_link_preserves_architecture_request(
    tmp_path, architectures, enabled, linked, global_link
):
    """Keep architecture intent and the angular-force compilation boundary."""
    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to inspect native CUDA build graphs")
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(tmp_path),
            "-G",
            "Ninja",
            "-DVIBEQC_ENABLE_CUDA=ON",
            "-DVIBEQC_ENABLE_AOT_SHELLS=OFF",
            "-DVIBEQC_COMPILER_CACHE=off",
            f"-DVIBEQC_CUDA_SEPARABLE_COMPILATION={global_link}",
            f"-DCMAKE_CUDA_COMPILER={nvcc}",
            f"-DCMAKE_CUDA_ARCHITECTURES={architectures}",
            f"-DVIBEQC_CUDA_DIRECT_DEVICE_LINK={enabled}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    def command(target):
        return subprocess.check_output(
            ["ninja", "-C", str(tmp_path), "-t", "commands", target], text=True
        ).splitlines()[-1]

    owner = "vibeqc_direct_native" if linked else "vibeqc"
    direct = command(f"CMakeFiles/{owner}.dir/src/scf/cuda/direct_reference_force.cu.o")
    assert ("-rdc=true" in direct) == linked
    if architectures.endswith("-real"):
        assert "code=[sm_120]" in direct
    elif architectures.endswith("-virtual"):
        assert "code=[compute_120]" in direct
    else:
        assert "code=[compute_120,sm_120]" in direct
    # A direct archive must not drag an unrelated SCF kernel into device linking.
    assert (
        "-rdc=true"
        in command("CMakeFiles/vibeqc.dir/src/scf/cuda/scf_state_kernels.cu.o")
    ) == (global_link == "ON")
    assert "-rdc=true" not in command(
        "CMakeFiles/vibeqc_direct_angular_force.dir/"
        "src/scf/cuda/direct_angular_force.cu.o"
    )
    host = command("CMakeFiles/vibeqc.dir/src/scf/cuda_rhf.cpp.o")
    assert nvcc not in host
    assert "-rdc=true" not in host
