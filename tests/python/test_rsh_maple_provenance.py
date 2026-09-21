"""Imported extended-GGA production identities track local and upstream math."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from vibeqc_compiler.xc import libxc_maple, rsh_maple
from vibeqc_compiler.xc.spec import FunctionalSpec, functional


def _lyp() -> FunctionalSpec:
    return FunctionalSpec(
        "GGA_C_LYP_PRODUCTION",
        (("GGA_C_LYP", Fraction(1)),),
    )


@pytest.mark.parametrize("owner", ("adapter", "importer"))
def test_lyp_identity_tracks_local_math_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str
) -> None:
    module = rsh_maple if owner == "adapter" else libxc_maple
    snapshot = tmp_path / Path(module.__file__).name
    snapshot.write_bytes(Path(module.__file__).read_bytes())
    monkeypatch.setattr(module, "__file__", str(snapshot))

    before = _lyp().identity
    unrelated = functional("LDA_X").identity
    snapshot.write_bytes(
        snapshot.read_bytes() + b"\n# changed local mathematical implementation\n"
    )

    assert _lyp().identity != before
    assert functional("LDA_X").identity == unrelated


def test_lyp_identity_records_pinned_importer_and_source_provenance() -> None:
    provenance = _lyp().to_payload()["expression_provenance"]

    assert provenance["kind"] == "libxc-maple"
    assert provenance["importer_semantics"] == libxc_maple.IMPORTER_SEMANTICS
    component = provenance["components"]["GGA_C_LYP"]
    module = rsh_maple._lyp_module()
    assert component == {
        "entry": "gga_c_lyp.mpl",
        "source_sha256": module.source_sha256,
        "transitive_sha256": module.transitive_sha256,
    }
