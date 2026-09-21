"""Production P86/PZ identities track imported math ownership."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from vibeqc_compiler.xc import libxc_maple, p86_pz_maple
from vibeqc_compiler.xc.spec import FunctionalSpec, functional


def _spec(name: str) -> FunctionalSpec:
    return FunctionalSpec(f"{name}_PRODUCTION", ((name, Fraction(1)),))


@pytest.mark.parametrize("name", ("LDA_C_PZ", "GGA_C_P86"))
@pytest.mark.parametrize("owner", ("adapter", "importer"))
def test_p86_pz_identity_tracks_local_math_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, owner: str
) -> None:
    module = p86_pz_maple if owner == "adapter" else libxc_maple
    snapshot = tmp_path / Path(module.__file__).name
    snapshot.write_bytes(Path(module.__file__).read_bytes())
    monkeypatch.setattr(module, "__file__", str(snapshot))
    before = _spec(name).identity
    unrelated = functional("LDA_X").identity
    snapshot.write_bytes(snapshot.read_bytes() + b"\n# changed math\n")
    assert _spec(name).identity != before
    assert functional("LDA_X").identity == unrelated


@pytest.mark.parametrize(
    ("name", "entry"),
    (("LDA_C_PZ", "lda_c_pz.mpl"), ("GGA_C_P86", "gga_c_p86.mpl")),
)
def test_p86_pz_identity_records_pinned_source_provenance(
    name: str, entry: str
) -> None:
    provenance = _spec(name).to_payload()["expression_provenance"]
    assert provenance["kind"] == "libxc-maple"
    assert provenance["importer_semantics"] == libxc_maple.IMPORTER_SEMANTICS
    component = provenance["components"][name]
    module = (
        p86_pz_maple._pz_module() if name == "LDA_C_PZ" else p86_pz_maple._p86_module()
    )
    assert component == {
        "entry": entry,
        "source_sha256": module.source_sha256,
        "transitive_sha256": module.transitive_sha256,
    }
