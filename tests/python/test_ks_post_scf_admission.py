"""An electronic KS projection must never drop a geometry correction."""

from dataclasses import replace

import pytest
from vibeqc import ks
from vibeqc_compiler.method import METHOD_CATALOG, resolve_method

from tools.vibeqc_d3.reference import gfn1_compatibility


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
@pytest.mark.parametrize("entry", ["coefficients", "options"])
def test_electronic_projection_rejects_unowned_post_scf(spin: str, entry: str) -> None:
    graph = resolve_method(
        replace(METHOD_CATALOG["PBE"], dispersion=gfn1_compatibility()), spin=spin
    )
    with pytest.raises(NotImplementedError, match="post-SCF|geometry-only"):
        if entry == "coefficients":
            ks.ks_coefficients(graph)
        else:
            method = "pbe-uks" if spin == "polarized" else "pbe-rks"
            ks.resolve_ks_options(method, ks.KsOptions(composition=graph))
