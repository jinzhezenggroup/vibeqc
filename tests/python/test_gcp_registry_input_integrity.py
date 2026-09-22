"""Exercise the real gCP CLI against an isolated canonical-input checkout."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "tools/parameters/r2scan3c_gcp.json"
GENERATOR = "tools/parameters/generate_gcp_r2scan3c.py"


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    for relative in (
        SOURCE,
        GENERATOR,
        "tools/__init__.py",
        "tools/source_registry.py",
        "upstream/manifest.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes().replace(b"\r\n", b"\n"))
    return tmp_path


@pytest.mark.parametrize("mutation", ("parameter", "invalid-json", "whitespace"))
def test_gcp_cli_rejects_changed_canonical_bytes_before_publication(
    checkout: Path, mutation: str
) -> None:
    source = checkout / SOURCE
    raw = source.read_bytes()
    if mutation == "parameter":
        data = json.loads(raw)
        data["elements"][0]["emiss"] += 1.0
        raw = json.dumps(data).encode()
    elif mutation == "invalid-json":
        raw = b"not JSON"
    else:
        raw += b"\n"
    source.write_bytes(raw)
    output = checkout / "existing.hpp"
    output.write_text("preserve existing output\n")
    result = subprocess.run(
        [sys.executable, str(checkout / GENERATOR), "--output", str(output)],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "canonical input digest mismatch" in result.stderr
    assert output.read_text() == "preserve existing output\n"


@pytest.mark.parametrize("crlf", (False, True))
def test_gcp_cli_valid_checkout_regenerates_pinned_header(
    checkout: Path, crlf: bool
) -> None:
    if crlf:
        for relative in (SOURCE, GENERATOR, "tools/source_registry.py"):
            path = checkout / relative
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    output = checkout / "generated.hpp"
    subprocess.run(
        [sys.executable, str(checkout / GENERATOR), "--output", str(output)],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    expected = ROOT / "src/dft/dispersion/gcp_r2scan3c_data.hpp"
    assert output.read_bytes() == expected.read_bytes().replace(b"\r\n", b"\n")


def test_gcp_cli_uses_its_checkout_with_foreign_pythonpath(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real package in another checkout must not replace the copied tool."""
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    output = checkout / "foreign-pythonpath.hpp"
    subprocess.run(
        [sys.executable, str(checkout / GENERATOR), "--output", str(output)],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    expected = ROOT / "src/dft/dispersion/gcp_r2scan3c_data.hpp"
    assert output.read_bytes() == expected.read_bytes().replace(b"\r\n", b"\n")
