"""Native CC source generation must work in CMake's stdlib-only interpreter."""

import subprocess
import sys
from pathlib import Path


def test_native_cc_codegen_without_site_packages(tmp_path: Path) -> None:
    """Exercise all output modes with NumPy and runtime packages unavailable."""
    root = Path(__file__).resolve().parents[2]
    outputs = {
        "--cpu-header": tmp_path / "generated_rccsd_cpu.hpp",
        "--cuda-source": tmp_path / "generated_rccsd_cuda.cu",
        "--triples-cuda-source": tmp_path / "generated_triples_response_cuda.cu",
        "--triples-fock-cpu-header": tmp_path
        / "generated_triples_fock_response_cpu.hpp",
        "--triples-fock-cuda-source": tmp_path
        / "generated_triples_fock_response_cuda.cu",
    }
    command = [sys.executable, "-S", str(root / "tools/generate_rccsd_native.py")]
    for flag, path in outputs.items():
        command.extend((flag, str(path)))
    subprocess.run(command, cwd=root, check=True, capture_output=True, timeout=120)
    assert all(path.stat().st_size > 0 for path in outputs.values())
