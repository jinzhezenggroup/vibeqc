"""The shared CUDA lifecycle must ship and participate in artifact identity."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.common import paths
from vibeqc_compiler.tensor.cuda_execute import tensor_source_identity

ROOT = Path(__file__).resolve().parents[2]
HEADER = "src/runtime/compiled_execution_region.hpp"


@pytest.mark.parametrize("compiler_name", ["c++", "clang++"])
def test_wheel_staged_lifecycle_header_compiles(
    tmp_path: Path, compiler_name: str
) -> None:
    """Compile against package destinations, not a source-tree include fallback."""
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"requires {compiler_name}")
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    entries = dict(re.findall(r'^"([^"]+)"\s*=\s*"([^"]+)"$', text, re.MULTILINE))
    destination = entries.get(HEADER)
    if destination is not None:
        target = tmp_path / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / HEADER, target)
    source = tmp_path / "lifecycle.cpp"
    source.write_text(
        '#include "runtime/compiled_execution_region.hpp"\n'
        "int main() {\n"
        "  vibeqc::runtime::CompiledExecutionRegion region;\n"
        '  region.bind({"installed-lifecycle"});\n'
        "  region.mark_success();\n"
        '  region.mark_failure("injected");\n'
        "  bool rejected = false;\n"
        "  try { region.mark_success(); }\n"
        "  catch (const std::logic_error&) { rejected = true; }\n"
        "  if (!rejected || !region.failed()) return 1;\n"
        "  region.recover();\n"
        "  region.mark_success();\n"
        "  region.invalidate();\n"
        "  return region.bound() || region.failed() || region.warmed();\n"
        "}\n",
        encoding="utf-8",
    )
    binary = tmp_path / "lifecycle"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I" + str(tmp_path / "vibeqc_compiler/assets/src"),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(binary)], check=True, timeout=10)


def test_lifecycle_header_change_invalidates_tensor_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real hash inventory with exactly one changed native input."""
    header = tmp_path / "compiled_execution_region.hpp"
    header.write_bytes((ROOT / HEADER).read_bytes())
    original = paths.asset_path

    def redirected(relative: str) -> Path:
        return header if relative == HEADER else original(relative)

    monkeypatch.setattr(paths, "asset_path", redirected)
    before = tensor_source_identity()
    header.write_bytes(header.read_bytes() + b"\n// independent input mutation\n")
    assert tensor_source_identity() != before
