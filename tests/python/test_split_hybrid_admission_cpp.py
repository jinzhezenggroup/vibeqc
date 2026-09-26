"""Exercise the real native descriptor gate without allocating a CUDA owner."""

import shutil
import subprocess
from pathlib import Path

from tools.generate_xc_split_hybrid_registry import emit_registry

ROOT = Path(__file__).resolve().parents[2]


def test_native_split_hybrid_descriptor_gate(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    assert compiler is not None, "native admission validation requires a C++ compiler"
    (tmp_path / "generated_split_hybrid_registry.cuh").write_text(
        emit_registry(), encoding="utf-8"
    )
    executable = tmp_path / "split-hybrid-admission"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-DVIBEQC_HAS_CUDA=1",
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "src"),
            "-I" + str(tmp_path),
            str(ROOT / "tests/native/split_hybrid_admission_probe.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=120,
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def test_single_and_batch_share_descriptor_admission() -> None:
    source = (ROOT / "src/methods/registry.cpp").read_text(encoding="utf-8")
    assert (
        "detail::validate_split_hybrid_descriptor(descriptor, backend, detail)"
        in source
    )
    assert (
        source.count(
            "validate_option_family(definition, descriptor, context.requested_backend);"
        )
        == 2
    )
