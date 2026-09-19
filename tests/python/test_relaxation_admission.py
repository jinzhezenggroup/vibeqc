"""Impossible CUDA relaxation budgets fail before response or compilation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.integral.first_gradient_execute import first_gradient_storage

from tools.vibeqc_hessian import first_order_cuda, hvp


@pytest.mark.parametrize("entry", ["hvp", "contraction"])
@pytest.mark.parametrize("budget_kind", ["one_byte", "one_byte_short"])
def test_relaxation_admission_precedes_provider_work(
    monkeypatch, tmp_path, entry, budget_kind
):
    class State:
        nbf, nat = 2, 2
        source = SimpleNamespace(shells=())
        offsets, primitives = (), ()
        coords, P0 = np.zeros((2, 3)), np.eye(2)
        cache = tmp_path

        def validate(self):
            pass

    monkeypatch.setattr(hvp, "NativeRHFState", State)
    monkeypatch.setattr(first_order_cuda, "NativeRHFState", State)
    compiler = CudaCompilerAdapter(Path("/bin/false"), cuda_target_info("sm_80"))

    def forbidden(*args, **kwargs):
        raise AssertionError("infeasible relaxation budget reached provider work")

    monkeypatch.setattr(hvp, "directional_rhf_response", forbidden)
    monkeypatch.setattr(first_order_cuda, "compile_first_gradient", forbidden)
    required = first_gradient_storage(2, 2, 3, 128)["numeric_peak_bytes"]
    budget = 1 if budget_kind == "one_byte" else required - 1
    with pytest.raises(MemoryError, match="relaxation numeric storage"):
        if entry == "hvp":
            hvp.rhf_hvp(
                State(),
                np.ones((2, 3)),
                relaxation_backend="cuda",
                relaxation_compiler=compiler,
                relaxation_budget_bytes=budget,
            )
        else:
            first_order_cuda.generated_rhf_relaxation_contraction_cuda(
                State(),
                np.eye(2),
                np.eye(2),
                compiler,
                budget_bytes=budget,
            )
    assert list(tmp_path.iterdir()) == []
