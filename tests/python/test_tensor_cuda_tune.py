"""Selection must reject noisy endpoints and incomplete resource information."""

import numpy as np
import pytest

from tools.vibeqc_tensor.cuda_resources import parse_resources
from tools.vibeqc_tensor.cuda_tune import endpoint_gate


def test_endpoint_gate_accepts_clear_gain_and_rejects_noise_or_regression():
    assert endpoint_gate([10] * 8, [8] * 8)["passed"]
    assert not endpoint_gate([10] * 8, [9.95] * 8)["passed"]
    assert not endpoint_gate([10] * 8, [11] * 8)["passed"]
    assert not endpoint_gate([1, 100, 1, 100, 1, 100], [80, 1, 80, 1, 80, 1])["passed"]
    with pytest.raises(ValueError, match="positive and finite"):
        endpoint_gate([10] * 5, [np.nan] * 5)


def test_tensor_resources_include_zero_shared_memory_and_spills():
    diagnostics = """ptxas info    : Function properties for _Zkernel
    16 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 42 registers, used 0 barriers
ptxas info    : Function properties for _Zshared
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers, used 1 barriers, 2048 bytes smem
"""
    a, b = parse_resources(diagnostics)
    assert (
        a.registers,
        a.stack_bytes,
        a.spill_store_bytes,
        a.spill_load_bytes,
        a.shared_bytes,
    ) == (42, 16, 8, 4, 0)
    assert b.shared_bytes == 2048
