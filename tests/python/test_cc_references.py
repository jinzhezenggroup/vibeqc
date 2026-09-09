"""Pinned off-shell PySCF outputs and the real #147 native provider boundary."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_cc import build_program, evaluate
from tools.vibeqc_cc.oracle import DeterminantOracle, dense_feeds, random_case
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_tensor import execute
from tools.vibeqc_validation.schema import canonical_hash

ROOT = Path(__file__).resolve().parents[1]


def references():
    data = json.loads((ROOT / "reference_data/cc/rccsd-a.json").read_text())
    assert data["pyscf"] == "2.14.0" and data["version"] == 1
    assert data["cases_hash"] == canonical_hash(data["cases"])
    assert data["stability"]["identical_cases"]
    return data["cases"]


@pytest.mark.parametrize("case", references(), ids=lambda c: c["name"])
def test_pinned_pyscf_updates_reconstruct_shift_independent_physical_residual(case):
    assert canonical_hash(case["inputs"]) == case["inputs_hash"]
    f, g, x, y = (np.array(case["inputs"][key]) for key in ("fock", "eri", "t1", "t2"))
    result = execute(build_program(*x.shape), dense_feeds(f, g, x, y)).outputs
    np.testing.assert_allclose(
        result["correlation_energy"], case["correlation_energy"], atol=1e-11, rtol=1e-10
    )
    for record in case["updates"]:
        r = np.array(record["denominator"]) * (np.array(record["updated_t1"]) - x)
        np.testing.assert_allclose(r, record["residual"], atol=1e-13, rtol=1e-12)
        np.testing.assert_allclose(
            result["singles_residual"], r, atol=1e-11, rtol=1e-10
        )
        assert not np.allclose(
            result["singles_residual"], record["updated_t1"], atol=1e-9
        )


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
def test_native_provider_same_C_and_new_vibeqc_HF(name):
    meta, arrays = load_fixture(name)
    try:
        source = NativeSource(**source_arguments(meta))
    except (OSError, FileNotFoundError) as error:
        pytest.skip(f"native post-HF library unavailable: {error}")
    except RuntimeError as error:
        if "VIBEQC native library was not found" not in str(error):
            raise
        pytest.skip(str(error))
    with source:
        snapshot = fixture_snapshot(meta, arrays)
        cc_ref = next(c for c in references() if c["name"] == name)
        x, y = (np.array(cc_ref["inputs"][k]) for k in ("t1", "t2"))
        with ConventionalProvider(snapshot, source) as provider:
            result = evaluate(snapshot, provider, x, y)
            np.testing.assert_allclose(
                result["correlation_energy"],
                cc_ref["correlation_energy"],
                atol=1e-11,
                rtol=1e-10,
            )
            np.testing.assert_allclose(
                result["singles_residual"],
                cc_ref["updates"][0]["residual"],
                atol=1e-11,
                rtol=1e-10,
            )
            count = provider.statistics["transformations"]
            zero = evaluate(snapshot, provider, x * 0, y * 0)
            assert zero["total_energy"] == snapshot.reference_energy
            assert zero["correlation_energy"] == 0
            assert provider.statistics["transformations"] == count == 5
            with pytest.raises(ValueError, match="identit"):
                evaluate(replace(snapshot, generation_id="other"), provider, x, y)
        # Newly converged VibeQC HF, its own C, native block provider. This is
        # a fixed-amplitude smoke check, not an end-to-end CCSD solve (slice C).
        fresh, _ = export_rhf(source)
        _, _, x, y = random_case(fresh.nocc, fresh.nmo - fresh.nocc)
        with ConventionalProvider(fresh, source) as provider:
            value = evaluate(fresh, provider, x, y)
            assert np.isfinite(value["total_energy"])
            assert value["reference_id"] == fresh.identity
            if name == "h2":
                c = fresh.coefficients
                independent_g = np.einsum(
                    "uvwx,up,vq,wr,xs->pqrs", arrays["ao"], c, c, c, c, optimize=True
                )
                expected_e, expected_r = DeterminantOracle(
                    c.T @ fresh.fock @ c, independent_g, fresh.nocc
                ).evaluate(x, y)
                np.testing.assert_allclose(
                    value["correlation_energy"], expected_e, atol=1e-11, rtol=1e-10
                )
                np.testing.assert_allclose(
                    value["singles_residual"], expected_r, atol=1e-11, rtol=1e-10
                )
        with pytest.raises(ValueError, match="conventional CPU"):
            evaluate(fresh, object(), x, y)
