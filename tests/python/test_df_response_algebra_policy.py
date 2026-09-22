"""Regression guard for the production DF response algebra default."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/scf/cuda/df_gradient_bridge.cu"

_DEFAULT_ALGEBRA = re.compile(
    r"""const\s+std::string_view\s+algebra\s*=\s*
        algebra_control\s*\?\s*algebra_control\s*:\s*"(?P<default>[^"]+)"\s*;""",
    re.VERBOSE,
)


def test_df_response_algebra_defaults_to_blas_independent_of_owner() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    match = _DEFAULT_ALGEBRA.search(source)

    assert match, "DF response algebra default must remain explicit and auditable"
    assert match.group("default") == "blas", (
        "production DF response must default to compiler-qualified BLAS; "
        "scalar is an explicit diagnostic/ablation control only"
    )
