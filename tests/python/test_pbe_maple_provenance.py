"""PBE identities include the local adapter and importer that build its Graph."""

from pathlib import Path

import pytest
from vibeqc_compiler.xc import functional, libxc_maple, pbe_maple


@pytest.mark.parametrize("component", ("GGA_X_PBE", "GGA_C_PBE", "PBE"))
@pytest.mark.parametrize("owner", ("adapter", "importer"))
def test_pbe_identity_tracks_local_math_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str, owner: str
) -> None:
    module = pbe_maple if owner == "adapter" else libxc_maple
    snapshot = tmp_path / Path(module.__file__).name
    snapshot.write_bytes(Path(module.__file__).read_bytes())
    monkeypatch.setattr(module, "__file__", str(snapshot))
    selected = functional(component)
    before = selected.identity
    unrelated = functional("LDA_X").identity
    snapshot.write_bytes(
        snapshot.read_bytes() + b"\n# changed local mathematical implementation\n"
    )
    assert selected.identity != before
    assert functional("LDA_X").identity == unrelated
