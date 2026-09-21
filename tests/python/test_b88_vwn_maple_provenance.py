"""Production B88/VWN identities track imported math ownership."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from vibeqc_compiler.xc import b88_vwn_maple, libxc_maple
from vibeqc_compiler.xc.spec import FunctionalSpec, functional


def _spec(name: str) -> FunctionalSpec:
    return FunctionalSpec(f"{name}_PRODUCTION", ((name, Fraction(1)),))


@pytest.mark.parametrize("name", ("GGA_X_B88", "LDA_C_VWN", "LDA_C_VWN_RPA"))
@pytest.mark.parametrize("owner", ("adapter", "importer"))
def test_b88_vwn_identity_tracks_local_math_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, owner: str
) -> None:
    module = b88_vwn_maple if owner == "adapter" else libxc_maple
    snapshot = tmp_path / Path(module.__file__).name
    snapshot.write_bytes(Path(module.__file__).read_bytes())
    monkeypatch.setattr(module, "__file__", str(snapshot))

    before = _spec(name).identity
    unrelated = functional("LDA_X").identity
    snapshot.write_bytes(
        snapshot.read_bytes() + b"\n# changed local mathematical implementation\n"
    )

    assert _spec(name).identity != before
    assert functional("LDA_X").identity == unrelated


@pytest.mark.parametrize(
    ("name", "entry"),
    (
        ("GGA_X_B88", "gga_x_b88.mpl"),
        ("LDA_C_VWN", "lda_c_vwn.mpl"),
        ("LDA_C_VWN_RPA", "lda_c_vwn_rpa.mpl"),
    ),
)
def test_b88_vwn_identity_records_pinned_source_provenance(
    name: str, entry: str
) -> None:
    provenance = _spec(name).to_payload()["expression_provenance"]
    assert provenance["kind"] == "libxc-maple"
    assert provenance["importer_semantics"] == libxc_maple.IMPORTER_SEMANTICS
    component = provenance["components"][name]
    assert component["entry"] == entry
    assert len(component["source_sha256"]) == 64
    assert len(component["transitive_sha256"]) == 64
