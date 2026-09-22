"""The GFN2 CUDA archive must wait for its generated SDQ header."""

import re
from pathlib import Path


def test_cuda_archive_orders_sdq_header_generation() -> None:
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "cmake/VibeQCGfn2Runtime.cmake").read_text()
    dependencies = re.findall(r"add_dependencies\(vibeqc_gfn2_cuda\s+([^)]*)\)", cmake)
    assert any("vibeqc_gfn2_sdq_cuda_codegen" in row.split() for row in dependencies)
