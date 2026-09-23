"""Offline AOT probes must terminate their compiler children on timeout."""

from __future__ import annotations

import os
import shlex
import shutil
import time
from typing import TYPE_CHECKING

import pytest
from vibeqc_compiler.xc import bulk_aot

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX compiler process groups")
@pytest.mark.parametrize("phase", ("identification", "compilation"))
def test_aot_timeout_stops_real_compiler_children(tmp_path: Path, phase: str) -> None:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("requires a POSIX shell")
    started, escaped = tmp_path / "started", tmp_path / "escaped"
    compiler = tmp_path / "compiler"
    compiler.write_text(
        f"#!{shell}\n"
        f'if [ "$1" = "--version" ] && [ "{phase}" = "compilation" ]; then\n'
        '  echo "fake cc"; exit 0\n'
        "fi\n"
        f"(: > {shlex.quote(str(started))}; sleep 0.8; "
        f": > {shlex.quote(str(escaped))}) &\n"
        "exec sleep 30\n"
    )
    compiler.chmod(0o700)
    item = bulk_aot.SourceVariant(
        name="TIMEOUT_PROBE",
        family="lda",
        spin="unpolarized",
        import_identity="test-only-source",
        domain="process-test/v1",
        features=("rho",),
        derivative_order=0,
        backend="cpu",
        source="void bulk_xc_point(const double *x, double *y) { y[0] = x[0]; }\n",
        energy_nodes=1,
        ssa={"root_count": 1},
    )
    result = bulk_aot.compile_probe(item, compiler=str(compiler), timeout=0.3)
    # The child keeps inherited pipes and would write this marker if only its
    # parent were killed. It must have started, then be stopped as a group.
    time.sleep(1.0)
    assert started.is_file(), "fault fixture child never started"
    assert not escaped.exists(), "compiler child outlived the requested timeout"
    assert result["status"] == "timed-out"
