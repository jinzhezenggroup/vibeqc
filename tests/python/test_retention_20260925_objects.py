"""Verify the actual trimmed inventories, not only synthetic recovery fixtures."""

import pytest

from tools import restore_retained_evidence as restore


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
        removed = restore.ROOT / entry["path"]
        assert not removed.exists() and not removed.is_symlink(), entry["path"]
        # Checks bytes and every declared digest; network/lazy fetch is disabled.
        assert len(restore._read(entry)) == entry["bytes"]
