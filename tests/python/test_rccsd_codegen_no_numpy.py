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


def test_generated_response_arena_expressions_have_bounded_depth(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "generated_rccsd_cpu.hpp"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools/generate_rccsd_native.py"),
            "--cpu-header",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=120,
    )
    # Clang's default parser rejects nesting above 256; do not raise that limit.
    depth = maximum = 0
    for character in output.read_text():
        if character == "(":
            depth += 1
            maximum = max(maximum, depth)
        elif character == ")":
            depth -= 1
    assert depth == 0
    assert maximum < 32, f"generated response expression nesting is {maximum}"


def test_complete_rccsd_cuda_lambda_codegen_without_site_packages(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "generated_rccsd_cuda.cu"
    process = subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools/generate_rccsd_native.py"),
            "--cuda-source",
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
    for entry in (
        "run_lambda_rhs_cuda",
        "run_lambda_transpose_cuda",
        "run_lambda_independent_rhs_cuda",
        "run_lambda_independent_transpose_cuda",
    ):
        assert entry in text
    assert text.count("auto* arena=s.response_arena;") == 4
    assert "s.bar_correlation_energy" in text
    assert "s.bar_singles_residual" in text
    assert "s.bar_doubles_residual" in text
