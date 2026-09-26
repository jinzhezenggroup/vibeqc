"""Complete endpoint evidence cannot hide a failed or mismatched force sample."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.compare_df_direct_endpoint import check_endpoint


@pytest.mark.parametrize(
    "fault",
    ["shape", "nonfinite", "energy", "force", "convergence", "status", "oracle"],
)
def test_all_endpoint_samples_must_pass(fault: str) -> None:
    item = SimpleNamespace(
        energy=-76.0, forces=np.zeros((3, 3)), succeeded=True, converged=True
    )
    reference = {"energy": -76.0, "forces": np.zeros((3, 3)), "converged": True}
    assert check_endpoint(item, reference)["gate"]
    if fault == "shape":
        item.forces = np.zeros((1, 3))
    elif fault == "nonfinite":
        item.forces[0, 0] = np.nan
    elif fault == "energy":
        item.energy += 1e-7
    elif fault == "force":
        item.forces[0, 0] = 1e-6
    elif fault == "convergence":
        item.converged = False
    elif fault == "status":
        item.succeeded = False
        item.forces = None
    else:
        reference["converged"] = False
    assert not check_endpoint(item, reference)["gate"]
