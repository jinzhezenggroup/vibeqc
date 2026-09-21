"""Native response generation must work without runtime NumPy/site imports."""

import subprocess
import sys
from pathlib import Path


def test_complete_rccsd_response_header_without_site_packages(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "generated_rccsd_cpu.hpp"
    process = subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools/generate_rccsd_native.py"),
            "--cpu-header",
            str(output),
        ],
        check=False,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    text = output.read_text()
    assert "run_iteration_cpu" in text
    assert "triples" in text
    assert "lambda" in text
