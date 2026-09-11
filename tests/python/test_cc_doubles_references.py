"""PySCF physical doubles reconstruction and independent shared intermediates."""

from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.tensor import execute

from tools.validate_cc import load_references
from tools.vibeqc_cc.doubles import build_ccsd_program
from tools.vibeqc_cc.oracle import dense_feeds

DATA = Path(__file__).resolve().parents[1] / "reference_data/cc/rccsd-b.json"


@pytest.mark.parametrize(
    "case", load_references(DATA)["cases"], ids=lambda c: c["name"]
)
def test_pinned_doubles_and_each_shared_intermediate(case):
    f, g, x, y = (np.array(case["inputs"][k]) for k in ("fock", "eri", "t1", "t2"))
    for form in ("expanded", "shared", "optimized"):
        result = execute(
            build_ccsd_program(*x.shape, form=form), dense_feeds(f, g, x, y)
        ).outputs
        for key, expected in case["intermediates"].items():
            np.testing.assert_allclose(
                result[key], expected, atol=1e-11, rtol=1e-10, err_msg=key
            )
        for shift in case["updates"]:
            expected = np.array(shift["denominator2"]) * (
                np.array(shift["updated_t2"]) - y
            )
            np.testing.assert_allclose(
                expected, shift["residual2"], atol=1e-12, rtol=1e-11
            )
            np.testing.assert_allclose(
                result["doubles_residual"], expected, atol=1e-11, rtol=1e-10
            )
            assert (
                np.max(np.abs(result["doubles_residual"] - shift["updated_t2"])) > 1e-5
            )


def test_full_evidence_fixes_oracle_and_execution_identity(tmp_path):
    from tools.validate_ccsd import run
    from tools.vibeqc_validation.schema import file_hash

    records = run(tmp_path)
    assert len(records) == 5
    for record in records:
        settings = record["settings"]
        assert settings["reference_file_sha256"] == file_hash(DATA)
        assert settings["reference_cases_hash"] == load_references(DATA)["cases_hash"]
        assert settings["upstream"]["version"] == "2.14.0"
        assert (
            "python/vibeqc_compiler/tensor/interpreter.py" in settings["source_files"]
        )
        assert "python/vibeqc_compiler/tensor/optimize.py" in settings["source_files"]
        assert isinstance(settings["dirty"], bool)
