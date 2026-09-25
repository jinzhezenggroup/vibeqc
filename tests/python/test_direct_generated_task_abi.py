"""Contracts for the compact generated Direct-J/K task ABI."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_generated_task_keeps_consumer_identity_in_existing_flag_word() -> None:
    """Pure-J identity must not grow every materialized shell-quartet record."""

    header = (REPOSITORY_ROOT / "src/scf/generated_shell_task.hpp").read_text()
    task = header.split("struct GeneratedShellTask", maxsplit=1)[1].split("};", maxsplit=1)[0]
    assert "GeneratedFockConsumer fock_consumer" not in task
    assert "kGeneratedShellTaskCoulombConsumerBit = 1U << 2U" in header
    assert "static_assert(sizeof(GeneratedShellTask) == 192U)" in header

    dispatch = (
        REPOSITORY_ROOT
        / "python/vibeqc_compiler/integral/lowering/dispatch.py"
    ).read_text()
    generated_task = dispatch.split(
        "struct GeneratedDpppShellTask", maxsplit=1
    )[1].split("};", maxsplit=1)[0]
    assert "fock_consumer" not in generated_task
    assert "kGeneratedDpppCoulombConsumerBit = 1U << 2U" in dispatch

    accumulation = (
        REPOSITORY_ROOT
        / "python/vibeqc_compiler/integral/lowering/fock_accumulation.py"
    ).read_text()
    assert (
        "(task.reversed_shell_pair_mask & kGeneratedDpppCoulombConsumerBit) != 0U"
        in accumulation
    )

    emission = (
        REPOSITORY_ROOT
        / "python/vibeqc_compiler/integral/production_emission.py"
    ).read_text()
    assert "GeneratedFockConsumer::Coulomb" in emission
    assert "kGenerated{class_name}CoulombConsumerBit" in emission
