"""Shared pytest setup for cacheable generated C++ test fixtures."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_WRAPPER = r"""#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


def _generated_source(path: Path) -> bool:
    try:
        source = path.read_bytes()
    except OSError:
        return False
    return b"vibeqc::scf::generated_" in source


def main() -> int:
    real_cxx = os.environ["VIBEQC_TEST_REAL_CXX"]
    ccache = os.environ.get("VIBEQC_TEST_CCACHE")
    args = sys.argv[1:]
    sources = [
        (index, Path(value))
        for index, value in enumerate(args)
        if Path(value).suffix.lower() in {".cc", ".cpp", ".cxx"}
        and Path(value).is_file()
    ]
    if (
        ccache is None
        or "-shared" not in args
        or "-c" in args
        or "-o" not in args
        or len(sources) != 1
        or not _generated_source(sources[0][1])
    ):
        return subprocess.run([real_cxx, *args]).returncode

    source_index, source = sources[0]
    output_index = args.index("-o") + 1
    output = Path(args[output_index])
    repo = Path(os.environ["VIBEQC_TEST_REPO_ROOT"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    stable_dir = repo / "build" / "pytest-generated-cxx" / digest
    stable_dir.mkdir(parents=True, exist_ok=True)
    stable_source = stable_dir / source.name
    if not stable_source.exists():
        stable_source.write_bytes(source.read_bytes())
    object_path = output.with_name(output.name + ".o")

    compile_args: list[str] = []
    skip_output = False
    for index, value in enumerate(args):
        if skip_output:
            skip_output = False
            continue
        if value == "-shared":
            continue
        if value == "-o":
            compile_args.extend(("-o", str(object_path)))
            skip_output = True
            continue
        compile_args.append(str(stable_source) if index == source_index else value)
    compile_args.insert(0, "-c")

    environment = os.environ.copy()
    environment["CCACHE_BASEDIR"] = str(repo)
    environment["CCACHE_NOHASHDIR"] = "1"
    compile_result = subprocess.run(
        [ccache, real_cxx, *compile_args], env=environment
    )
    if compile_result.returncode:
        return compile_result.returncode

    link_args = [
        str(object_path) if index == source_index else value
        for index, value in enumerate(args)
    ]
    return subprocess.run([real_cxx, *link_args]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _install_generated_cpp_wrapper() -> None:
    real_cxx = os.environ.get("VIBEQC_TEST_REAL_CXX") or shutil.which("c++")
    ccache = os.environ.get("VIBEQC_TEST_CCACHE") or shutil.which("ccache")
    if real_cxx is None or ccache is None:
        return
    repo = Path(__file__).resolve().parents[2]
    directory = repo / "build" / "pytest-cxx-wrapper"
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = directory / "c++"
    if not wrapper.exists():
        temporary = directory / f"c++.tmp-{os.getpid()}"
        temporary.write_text(_WRAPPER)
        temporary.chmod(0o755)
        try:
            temporary.replace(wrapper)
        except FileNotFoundError:
            pass
    os.environ["VIBEQC_TEST_REAL_CXX"] = real_cxx
    os.environ["VIBEQC_TEST_CCACHE"] = ccache
    os.environ["VIBEQC_TEST_REPO_ROOT"] = str(repo)
    path = os.environ.get("PATH", "")
    entries = path.split(os.pathsep) if path else []
    if str(directory) not in entries:
        os.environ["PATH"] = str(directory) + (os.pathsep + path if path else "")


_install_generated_cpp_wrapper()
