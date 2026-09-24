"""Ragged generated D3(BJ) PairIR/CUDA retirement-candidate qualification."""

from __future__ import annotations

import json
import os
import typing
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc.profiles import find_nvcc
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.geometry import (
    PreparedD3CudaBatch,
    compile_d3_bj_batch,
    execute_d3_bj_batch,
)

from tools.vibeqc_d3.reference import gfn1_compatibility, make_spec

_FIXTURES = json.loads(
    (Path(__file__).parents[1] / "data/d3_bj_reference.json").read_text()
)["fixtures"]
_PBE = [case for case in _FIXTURES if case["name"].startswith("pbe-bj-two-body/")]


def _ragged(
    cases: typing.Iterable[typing.Mapping[str, typing.Any]],
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    elements = []
    offsets = [0]
    coordinates = []
    for case in cases:
        elements.extend(case["numbers"])
        xyz = np.asarray(case["positions"], dtype=np.float64)
        coordinates.append(xyz)
        offsets.append(offsets[-1] + len(case["numbers"]))
    return tuple(elements), tuple(offsets), np.concatenate(coordinates, axis=0)


def test_ragged_d3_pair_ir_matches_independent_system_goldens() -> None:
    cases = _PBE[:2]
    assert len(cases) == 2
    assert cases[0]["parameters"] == cases[1]["parameters"]
    spec = make_spec(**cases[0]["parameters"])
    elements, offsets, coordinates = _ragged(cases)

    compiled = compile_d3_bj_batch(spec, elements, offsets, coordinates)
    output = execute_d3_bj_batch(compiled, coordinates, gradient=True)

    np.testing.assert_allclose(
        output["energy"],
        [case["energy"] for case in cases],
        atol=3e-13,
        rtol=0,
    )
    gradient = output["gradient"]
    for system, case in enumerate(cases):
        begin, end = offsets[system : system + 2]
        np.testing.assert_allclose(
            gradient[begin:end],
            case["gradient"],
            atol=8e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            gradient[begin:end].sum(axis=0),
            0.0,
            atol=2e-13,
            rtol=0,
        )

    for first, second in compiled.pair_state.topology.pairs:
        assert any(begin <= first < second < end for begin, end in pairwise(offsets))


def test_ragged_d3_batch_rebuild_boundary_is_explicit() -> None:
    spec = gfn1_compatibility()
    elements = (6, 8, 1, 1)
    offsets = (0, 2, 4)
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [49.975, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    compiled = compile_d3_bj_batch(spec, elements, offsets, coordinates)
    assert compiled.pair_state.energy_regions == ("switch", "inner")

    same_state = coordinates.copy()
    same_state[3, 1] += 0.01
    compiled.validate_coordinates(same_state)

    changed = coordinates.copy()
    changed[1, 0] = 50.0
    with pytest.raises(
        ValueError,
        match="stale D3 batch pair topology/switch state",
    ):
        compiled.validate_coordinates(changed)

    rebuilt = compile_d3_bj_batch(spec, elements, offsets, changed)
    assert rebuilt.identity != compiled.identity
    output = execute_d3_bj_batch(rebuilt, changed, gradient=True)
    assert output["energy"].shape == (2,)
    assert output["gradient"].shape == (4, 3)


def test_ragged_d3_primal_and_batch_vjp_lower_as_one_cuda_program() -> None:
    from vibeqc_compiler.common.cuda_target import CUDA_TARGETS
    from vibeqc_compiler.geometry.d3_cuda import _energy_gradient_program
    from vibeqc_compiler.tensor.cuda_emit import emit_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    cases = _PBE[:2]
    spec = make_spec(**cases[0]["parameters"])
    elements, offsets, coordinates = _ragged(cases)
    compiled = compile_d3_bj_batch(spec, elements, offsets, coordinates)
    program, seed_name = _energy_gradient_program(compiled)

    assert seed_name == "bar_energy"
    assert program.outputs["energy"].spec.shape == (2,)
    assert program.outputs["gradient"].spec.shape == coordinates.shape

    plan = plan_cuda(program, CUDA_TARGETS["sm_80"])
    source = emit_cuda(plan)
    assert "tensor_create" in source
    assert "tensor_run" in source
    assert program.provenance["kind"] == "d3-bj-generated-ragged-cuda-candidate"


def test_generated_d3_prepared_owner_uses_shared_compiled_execution_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vibeqc_compiler.geometry.d3_cuda as runtime

    cases = _PBE[:2]
    spec = make_spec(**cases[0]["parameters"])
    systems = [
        (case["numbers"], np.asarray(case["positions"], dtype=np.float64))
        for case in cases
    ]
    artifact = SimpleNamespace(
        metadata={
            "key": canonical_hash("d3-test-artifact"),
            "binary_sha256": canonical_hash("d3-test-binary"),
        }
    )

    class FakePrepared:
        def __init__(
            self, plan: typing.Any, unused_artifact: typing.Any, *, device: int = 0
        ) -> None:
            self.plan = plan
            self.closed = False

        def execute(self, feeds: typing.Any, *, profile: bool = False) -> typing.Any:
            outputs = {}
            for name, node in self.plan.program.outputs.items():
                outputs[name] = np.zeros(node.spec.shape, dtype=np.float64)
            return SimpleNamespace(outputs=outputs, metrics={"fake": True})

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(runtime, "compile_cuda", lambda *args, **kwargs: artifact)
    monkeypatch.setattr(runtime, "PreparedCuda", FakePrepared)
    compiler = SimpleNamespace(target=cuda_target_info("sm_80"))
    with runtime.PreparedD3CudaBatch(
        spec, systems, compiler, tmp_path / "cache"
    ) as batch:
        before = batch.diagnostic()
        assert len(before.prepared_identity) == 64
        assert before.prepared_executions == 0
        batch.execute()
        batch.execute([None, None])
        after = batch.diagnostic()
        assert after.prepared_identity == before.prepared_identity
        assert after.prepared_executions == 2
        assert after.rebuild_count == 0


@pytest.mark.skipif(
    os.environ.get("VIBEQC_D3_GENERATED_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
def test_generated_d3_cuda_ragged_batch_matches_independent_goldens(
    tmp_path: Path,
) -> None:
    cases = _PBE[:2]
    spec = make_spec(**cases[0]["parameters"])
    nvcc = find_nvcc()
    if nvcc is None:
        pytest.fail("VIBEQC_D3_GENERATED_CUDA_TEST requires a CUDA compiler")
    compiler = CudaCompilerAdapter(
        nvcc,
        cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120")),
    )
    systems = [
        (case["numbers"], np.asarray(case["positions"], dtype=np.float64))
        for case in cases
    ]
    with PreparedD3CudaBatch(
        spec,
        systems,
        compiler,
        tmp_path / "cache",
        max_bytes=256 * 1024**2,
    ) as batch:
        result = batch.execute()
        np.testing.assert_allclose(
            result.energies,
            [case["energy"] for case in cases],
            atol=3e-13,
            rtol=0,
        )
        assert result.gradients is not None
        for gradient, case in zip(result.gradients, cases, strict=True):
            np.testing.assert_allclose(
                gradient,
                case["gradient"],
                atol=8e-12,
                rtol=0,
            )
        diagnostic = batch.diagnostic()
        assert diagnostic.system_count == 2
        assert diagnostic.total_atoms == sum(len(case["numbers"]) for case in cases)
        assert diagnostic.peak_bytes <= 256 * 1024**2
