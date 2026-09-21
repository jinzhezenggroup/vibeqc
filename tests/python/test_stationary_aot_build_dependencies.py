"""Post-link AOT identity dependencies cover every file hashed by its contract."""

import ast
import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.common.paths import source_hashes

ROOT = Path(__file__).resolve().parents[2]


def test_aot_manifest_dependency_filter_matches_compatibility_hashes(
    tmp_path: Path,
) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for dependency evaluation")
    tree = ast.parse(
        (ROOT / "python/vibeqc_compiler/method/stationary_cuda.py").read_text()
    )
    assets = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "STATIONARY_AOT_ASSETS"
            for t in node.targets
        )
    )
    expected = source_hashes("integral", "xc", "dft", assets=assets)
    workflow = (ROOT / "cmake/VibeQCCuda.cmake").read_text()
    selection = workflow.split("set(_vibeqc_stationary_contract_assets", 1)[1]
    selection = (
        "set(_vibeqc_stationary_contract_assets"
        + selection.split("endforeach()", 1)[0]
        + "endforeach()"
    )
    unrelated = ("python/vibeqc_compiler/tensor/ir.py", "tools/unrelated.py")
    entries = "\n".join(f'  "{path}"' for path in (*expected, *unrelated))
    output = tmp_path / "dependencies.txt"
    script = tmp_path / "evaluate.cmake"
    script.write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        f'set(CMAKE_CURRENT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f"set(_vibeqc_identity_inputs\n{entries}\n)\n"
        + selection
        + f'\nfile(WRITE "{output.as_posix()}" "${{_vibeqc_stationary_contract_inputs}}")\n'
    )
    subprocess.run(
        [cmake, "-P", str(script)], check=True, capture_output=True, timeout=20
    )
    actual = {
        Path(p).relative_to(ROOT).as_posix() for p in output.read_text().split(";")
    }
    assert actual == set(expected)
    assert 'OUTPUTS "${_vibeqc_stationary_manifest}"' in workflow
    assert (
        'GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/write_stationary_aot_manifest.py"'
        in workflow
    )
