"""A packaged all-electron force request must not discover an NVCC compiler."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vibeqc import _dft_gradient, _stationary_cuda
from vibeqc.batch import PreparedBatch
from vibeqc_compiler import dft
from vibeqc_compiler.common.cuda_target import cuda_target_info


@pytest.mark.parametrize("missing_artifact", (False, True))
def test_public_aot_force_does_not_probe_nvcc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_artifact: bool
) -> None:
    class Basis:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(dft, "NativeAO", lambda *args, **kwargs: Basis())
    source = SimpleNamespace(backend="cuda", hamiltonian="all-electron", close=Mock())
    state = SimpleNamespace(_source=source)
    monkeypatch.setattr(
        _dft_gradient.StationaryKsState, "from_native", lambda *args, **kwargs: state
    )
    target = cuda_target_info("sm_120")

    def calculate(*args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["compiler"] is None
        assert kwargs["target"] is target
        assert kwargs["aot_directory"] == tmp_path
        if missing_artifact:
            raise FileNotFoundError("missing packaged stationary CUDA artifact")
        return SimpleNamespace(gradient=np.ones((2, 3)), work={"tensor_executions": 0})

    monkeypatch.setattr(
        _stationary_cuda, "complete_rks_cuda_gradient_diagnostic", calculate
    )
    batch = SimpleNamespace(
        _calculator=SimpleNamespace(
            _basis=object(),
            _representation_name="cartesian",
            _capabilities=SimpleNamespace(supported_properties={"energy", "forces"}),
        ),
        _stationary_cuda_execution=object(),
        _charges=[0],
        _multiplicities=[1],
        _library=SimpleNamespace(_name=str(tmp_path / "libvibeqc.so")),
        _stationary_cuda_compiler=lambda: pytest.fail("NVCC discovery before AOT load"),
        _stationary_cuda_target=lambda: target,
    )
    if missing_artifact:
        with pytest.raises(FileNotFoundError, match="missing packaged"):
            PreparedBatch._public_dft_cuda_force(batch, 0, ())
    else:
        force, work = PreparedBatch._public_dft_cuda_force(batch, 0, ())
        np.testing.assert_array_equal(force, -np.ones((2, 3)))
        assert work["tensor_executions"] == 0
    source.close.assert_called_once()
