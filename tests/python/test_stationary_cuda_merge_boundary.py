"""Qualified CUDA meta-GGA must survive parent-branch integration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_qualified_mgga_reaches_native_state_admission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vibeqc import _stationary_cuda as runtime

    contract = SimpleNamespace(family="mgga", validate=lambda state: None)
    monkeypatch.setattr(
        runtime, "StationaryDerivativeContract", lambda identity: contract
    )
    state = SimpleNamespace(identity=object(), _source=SimpleNamespace(backend="cpu"))
    with pytest.raises(NotImplementedError, match="requires a native CUDA KS state"):
        runtime.complete_rks_cuda_gradient_diagnostic(
            state, None, compiler=None, cache=tmp_path / "not-created"
        )
    assert not (tmp_path / "not-created").exists()
