"""Prepared D3 contract admission and result publication remain transactional."""

import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_d3_generated_ragged import _PBE, make_spec
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.provenance import canonical_hash


@pytest.fixture
def runtime_case(monkeypatch: pytest.MonkeyPatch) -> tuple[typing.Any, ...]:
    import vibeqc_compiler.geometry.d3_cuda as runtime

    case = _PBE[0]
    spec = make_spec(**case["parameters"])
    systems = [(case["numbers"], np.asarray(case["positions"], dtype=np.float64))]
    owners = []
    mode = {"publication_failure": False}
    artifact = SimpleNamespace(
        metadata={
            "key": canonical_hash("d3-failure-artifact"),
            "binary_sha256": canonical_hash("d3-failure-binary"),
        }
    )

    class FakePrepared:
        def __init__(
            self, plan: typing.Any, artifact: typing.Any, *, device: int = 0
        ) -> None:
            self.plan = plan
            self.closed = False
            owners.append(self)

        def execute(self, feeds: typing.Any, *, profile: bool = False) -> typing.Any:
            outputs = {
                name: np.zeros(node.spec.shape, dtype=np.float64)
                for name, node in self.plan.program.outputs.items()
            }
            if mode["publication_failure"]:
                del outputs["gradient"]
            return SimpleNamespace(outputs=outputs, metrics={})

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(runtime, "compile_cuda", lambda *a, **k: artifact)
    monkeypatch.setattr(runtime, "PreparedCuda", FakePrepared)
    return (
        runtime,
        spec,
        systems,
        SimpleNamespace(target=cuda_target_info("sm_80")),
        owners,
        mode,
    )


def test_contract_rejection_precedes_device_owner_creation(
    runtime_case: tuple[typing.Any, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, spec, systems, compiler, owners, _ = runtime_case

    def reject(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise ValueError("injected request rejection")

    monkeypatch.setattr(runtime, "PreparedExecutionRequest", reject)
    with pytest.raises(ValueError, match="request rejection"):
        runtime.PreparedD3CudaBatch(spec, systems, compiler, tmp_path)
    assert not owners, "device owner allocated before metadata-only admission"


def test_publication_failure_does_not_publish_success_and_recovers(
    runtime_case: tuple[typing.Any, ...], tmp_path: Path
) -> None:
    runtime, spec, systems, compiler, owners, mode = runtime_case
    with runtime.PreparedD3CudaBatch(spec, systems, compiler, tmp_path) as batch:
        mode["publication_failure"] = True
        with pytest.raises(KeyError, match="gradient"):
            batch.execute()
        assert batch.diagnostic().prepared_executions == 0
        assert batch._lease.needs_refresh
        mode["publication_failure"] = False
        result = batch.execute()
        assert result.rebuilt
        assert owners[0].closed
        assert batch.diagnostic().prepared_executions == 1
    assert all(owner.closed for owner in owners)
