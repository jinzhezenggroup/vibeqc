"""r2SCAN integration must not implicitly inherit LDA/PBE mixed-J promotion."""

import pytest
from vibeqc import Calculator, _native


@pytest.mark.parametrize("method", ["r2scan-rks", "r2scan-uks"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_r2scan_auto_rejected_before_runtime_loading(
    method: str, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(**kwargs: object) -> None:
        raise AssertionError("unsupported precision reached native runtime loading")

    monkeypatch.setattr(_native, "load_library", forbidden)
    with pytest.raises(NotImplementedError, match="r2SCAN.*strict FP64"):
        Calculator(method=method, device=device, precision="auto")
