"""Generated packet metadata must consume the compiler-owned schedule."""

from dataclasses import replace

import pytest
from vibeqc_compiler.common.homogeneous_schedule import HomogeneousExecution
from vibeqc_compiler.integral import df_shell_derivatives as emitter


@pytest.mark.parametrize("capacity", (7, 13))
def test_emission_reads_packet_capacity(
    monkeypatch: pytest.MonkeyPatch, capacity: int
) -> None:
    schedule = replace(emitter.DF_SIGNATURE_PACKET_SCHEDULE, packet_capacity=capacity)
    monkeypatch.setattr(emitter, "DF_SIGNATURE_PACKET_SCHEDULE", schedule)
    source = emitter.emit_df_shell_derivatives_cuda(classes=((0, 0, 0),))
    assert f"signature_packet_capacity={capacity};" in source


def test_emission_reads_execution_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    schedule = replace(
        emitter.DF_SIGNATURE_PACKET_SCHEDULE,
        execution=HomogeneousExecution.COOPERATIVE,
    )
    monkeypatch.setattr(emitter, "DF_SIGNATURE_PACKET_SCHEDULE", schedule)
    source = emitter.emit_df_shell_derivatives_cuda(classes=((0, 0, 0),))
    assert 'signature_packet_execution="cooperative";' in source
