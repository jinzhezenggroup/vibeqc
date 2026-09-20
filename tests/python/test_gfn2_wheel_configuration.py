"""Keep the private GFN2 provider available across wheel-build isolation."""

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")
ROOT = Path(__file__).resolve().parents[2]


def test_gfn2_provider_is_an_isolated_build_input() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    preparation = (ROOT / "python/ci/prepare-wheel-build.sh").read_text(
        encoding="utf-8"
    )
    provider_pins = set(re.findall(r"scipy-openblas32==[0-9.]+", preparation))
    assert len(provider_pins) == 1
    provider_pin = provider_pins.pop()
    assert provider_pin in config["build-system"]["requires"]
    assert not any(
        requirement.startswith("scipy-openblas32")
        for requirement in config["project"]["dependencies"]
    )


def test_gfn2_wheel_repair_has_a_persistent_provider() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    linux = config["tool"]["cibuildwheel"]["linux"]
    assert linux["before-build"] == "bash {package}/python/ci/prepare-wheel-build.sh"
    assert linux["repair-wheel-command"] == (
        "bash {package}/python/ci/repair-wheel.sh {dest_dir} {wheel}"
    )
