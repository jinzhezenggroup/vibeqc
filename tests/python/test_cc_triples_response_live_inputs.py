"""Selected triples VJPs must respect the resident input-name contract."""

from pathlib import Path

import numpy as np
import pytest
from test_cc_triples_response_cuda import (
    _cc_state,
    _fake_response_owner,
    _Resident,
    _triples_arrays,
)
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info

from tools.vibeqc_cc.lambda_solver import BoundCCSDLambda
from tools.vibeqc_cc.triples_response import accumulate_tile_triples_vjp


@pytest.mark.parametrize("selected", ("t1", "fov"))
def test_selected_triples_response_uploads_only_live_inputs(
    selected: str, tmp_path: Path
) -> None:
    snapshot, cc = _cc_state("h2o")
    bound = BoundCCSDLambda(snapshot, cc)
    arrays = _triples_arrays(bound)
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    owner, _ = _fake_response_owner(
        snapshot.nocc,
        snapshot.nmo - snapshot.nocc,
        compiler,
        tmp_path,
        chunk=1,
    )
    uploads: list[set[str]] = []

    class StrictResident(_Resident):
        def upload(self, feeds: dict[str, np.ndarray]) -> None:
            required = {
                node.attrs["name"]
                for node in self.plan.program.live_nodes
                if node.op == "input"
            }
            assert set(feeds) == required, (
                "resident upload includes dead/missing inputs"
            )
            uploads.append(set(feeds))
            super().upload(feeds)

    owner._PreparedResident = StrictResident
    actual = owner.run_tiles(arrays, inputs=(selected,))
    expected = accumulate_tile_triples_vjp(
        snapshot.nocc,
        snapshot.nmo - snapshot.nocc,
        arrays["ovvv"],
        arrays["ovoo"],
        arrays["ovov"],
        arrays["fov"],
        arrays["t1"],
        arrays["t2"],
        arrays["eps_o"],
        arrays["eps_v"],
        vir_chunk_size=1,
        inputs=(selected,),
    )
    assert uploads
    assert set(actual.sources) == {selected}
    np.testing.assert_allclose(
        actual.sources[selected], expected[selected], atol=1e-12, rtol=1e-12
    )
