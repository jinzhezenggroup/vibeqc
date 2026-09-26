"""Response arena sizes compile with ordinary parser limits and checked sums."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.vibeqc_cc.lambda_equations import build_lambda_programs

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generated_header(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("rccsd-arena-depth")
    output = folder / "generated.hpp"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools/generate_rccsd_native.py"),
            "--cpu-header",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return output


@pytest.mark.parametrize("compiler_name", ["c++", "clang++"])
def test_generated_response_arena_with_default_compiler_depth(
    tmp_path: Path, generated_header: Path, compiler_name: str
) -> None:
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"{compiler_name} is unavailable")
    program = build_lambda_programs(2, 3).residual_vjp.program
    expected = sum(node.spec.size for node in program.live_nodes if node.op != "input")
    source = tmp_path / "check.cpp"
    source.write_text(
        '#include "generated.hpp"\n'
        "int main(){using namespace vibeqc::cc::generated;\n"
        f"if(lambda_transpose_arena_elements(2,3)!={expected})return 1;\n"
        "try{(void)lambda_transpose_arena_elements(std::numeric_limits<std::size_t>::max(),1);return 2;}"
        "catch(const std::length_error&){}return 0;}\n"
    )
    output = tmp_path / "check"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(generated_header.parent),
            str(source),
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run([str(output)], check=True, capture_output=True, timeout=10)


def test_response_arena_has_bounded_expression_nesting(generated_header: Path) -> None:
    header = generated_header.read_text()
    start = header.index("inline std::size_t lambda_transpose_arena_elements(")
    end = header.index("\ninline ", start)
    source = header[start:end]
    depth = maximum = 0
    for character in source:
        if character == "(":
            depth += 1
            maximum = max(maximum, depth)
        elif character == ")":
            depth -= 1
    assert depth == 0
    assert maximum <= 16, f"arena expression nesting grows with graph size: {maximum}"
