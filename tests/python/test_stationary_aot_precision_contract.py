"""Packaged stationary artifacts must retain the strict floating-point policy."""

import json
from pathlib import Path

import pytest
from vibeqc_compiler.common.provenance import file_hash
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.stationary_cuda import (
    load_stationary_aot_artifact,
    stationary_aot_contract_identity,
)
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)


@pytest.mark.parametrize(
    "contract",
    (
        None,
        {},
        {"fp64": False, "fmad": False},
        {"fp64": True, "fmad": True},
        {"fp64": 1, "fmad": False},
        {"fp64": True, "fmad": 0},
    ),
)
def test_stationary_aot_rejects_unqualified_precision_contract(
    tmp_path: Path, contract: object
) -> None:
    plan = StationaryGradientPlan(
        resolve_method("PBE"), StationaryMeanField(SCF_POINT_MODEL)
    )
    library = tmp_path / "libvibeqc_stationary_pbe_rks.so"
    library.write_bytes(b"test artifact: never loaded as native code")
    metadata = {
        "schema": "vibeqc.stationary-cuda-aot.v2",
        "functional": 1,
        "spin": "unpolarized",
        "plan_identity": plan.identity,
        "partition_iterations": 3,
        "architectures": ["sm_80"],
        "code_objects": [{"architecture": "sm_80", "kind": "cubin"}],
        "contract_identity": stationary_aot_contract_identity(1, spin="unpolarized"),
        "source_identity": "test",
        "binary_sha256": file_hash(library),
        "binary_bytes": library.stat().st_size,
        "compile_contract": contract,
    }
    (tmp_path / "vibeqc_stationary_pbe_rks.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="precision contract"):
        load_stationary_aot_artifact(
            tmp_path, functional=1, spin="unpolarized", plan=plan, architecture="sm_80"
        )
