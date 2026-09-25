"""Verify the actual trimmed inventories, not only synthetic recovery fixtures."""

import pytest

from tools import restore_retained_evidence as restore

_RETAINED_PROFILER_EXPORTS = {
    "benchmarks/results/df-derivatives-rtx5090/profile-generated-0.csv",
    "benchmarks/results/df-derivatives-rtx5090/profile-generated-1048576.csv",
    "benchmarks/results/df-derivatives-rtx5090/profile-reference-0.csv",
    "benchmarks/results/df-derivatives-rtx5090/profile-reference-1048576.csv",
    "benchmarks/results/one-electron-derivatives-rtx5090/profile-cooperative.csv",
    "benchmarks/results/one-electron-derivatives-rtx5090/profile-df-generated_thread.csv",
    "benchmarks/results/one-electron-derivatives-rtx5090/profile-df-scalar.csv",
    "benchmarks/results/one-electron-derivatives-rtx5090/profile-generated_shell_warp.csv",
    "benchmarks/results/one-electron-derivatives-rtx5090/profile-generated_thread.csv",
    "benchmarks/results/one-electron-derivatives-rtx5090/profile-scalar.csv",
}


@pytest.mark.parametrize(
    "name,count,size",
    [
        ("migration.json", 12, 3_974_035),
        ("bulk.manifest.json", 758, 27_343_545),
    ],
)
def test_trimmed_members_remain_recoverable_offline(
    name: str, count: int, size: int
) -> None:
    if not (restore.ROOT / ".git").exists():
        pytest.skip("history-only check requires a Git checkout, not a source archive")
    manifest = restore.ROOT / "benchmarks/results/retention-2026-09-25" / name
    records = restore._records(manifest)
    assert len(records) == count
    assert sum(entry["bytes"] for entry in records) == size
    for entry in records:
        assert entry["revision"] == "49a2e664aaeef0deeb3860aa7c0912c8d078068a"
        assert entry["path"].startswith("benchmarks/results/")
        path = restore.ROOT / entry["path"]
        # Checks bytes and every declared digest; network/lazy fetch is disabled.
        recovered = restore._read(entry)
        assert len(recovered) == entry["bytes"]
        if entry["path"] in _RETAINED_PROFILER_EXPORTS:
            assert path.is_file() and not path.is_symlink(), entry["path"]
            assert path.read_bytes() == recovered, entry["path"]
        else:
            assert not path.exists() and not path.is_symlink(), entry["path"]
