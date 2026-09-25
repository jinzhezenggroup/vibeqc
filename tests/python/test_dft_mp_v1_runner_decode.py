"""Decoder failures finalize attempts without certifying scientific evidence."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from tools.dft_mp_v1 import run as runner
from tools.dft_mp_v1.freeze_contract import digest

if TYPE_CHECKING:
    from pathlib import Path


def run_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> tuple[Path, dict]:
    contract = {
        "contract_sha256": "b" * 64,
        "rows": [
            {"id": "fixture/first", "case": "fixture"},
            {"id": "fixture/second", "case": "fixture"},
        ],
        "cases": {"fixture": {"input": "fixture.json"}},
    }
    monkeypatch.setattr(runner, "manifest", lambda: contract)
    monkeypatch.setattr(runner, "audit", lambda *_: None)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "_check_run", lambda *_: pytest.fail("unexpected scientific PASS")
    )
    raw = tmp_path / "payload.bin"
    raw.write_bytes(payload)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "row = sys.argv[sys.argv.index('--row') + 1]\n"
        "progress = Path(sys.argv[sys.argv.index('--progress') + 1])\n"
        "progress.write_text('{}\\n', encoding='utf-8')\n"
        "print('decode fixture', file=sys.stderr)\n"
        "if row == 'fixture/first':\n"
        "    sys.stdout.buffer.write(Path('payload.bin').read_bytes())\n"
        "else:\n"
        "    print('{\"status\":\"unsupported\",\"reason\":\"fixture only\"}')\n",
        encoding="utf-8",
    )
    record = {"path": str(raw), "sha256": digest(payload)}
    campaign = {
        "source_commit": "a" * 40,
        **{
            name: dict(record)
            for name in ("library", "artifact", "build_record", "conditions")
        },
        "adapter": {"path": str(adapter), "sha256": digest(adapter.read_bytes())},
    }
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "adapter_command": [sys.executable, "-S", str(adapter)],
                "adapter_command_file_index": 2,
                "timeout_seconds": 10,
                "campaign": campaign,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    receipt = runner.run(plan, out)
    result = json.loads(receipt.read_bytes())
    assert result["rows"][1]["status"] == "unsupported"
    for entry in result["rows"]:
        for capture in entry["capture"].values():
            assert digest((out / capture["path"]).read_bytes()) == capture["sha256"]
    first = result["rows"][0]
    assert (out / first["capture"]["stdout"]["path"]).read_bytes() == payload
    assert "evidence" not in first
    assert first["partial_progress"] == first["capture"]["progress"]
    retained = {path.name: path.read_bytes() for path in out.iterdir()}
    assert runner.run(plan, out) == receipt
    assert retained == {path.name: path.read_bytes() for path in out.iterdir()}
    return out, first


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"[" * 10_000 + b"0" + b"]" * 10_000, id="decoder-depth"),
        pytest.param(b'{"status":', id="syntax"),
        pytest.param(b"\xff", id="encoding"),
    ],
)
def test_decoder_failure_finishes_row_and_preserves_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    out, first = run_payload(tmp_path, monkeypatch, payload)
    assert first["status"] == "failed"
    assert first["reason"].startswith("invalid adapter JSON:")
    journal = [
        json.loads(line) for line in (out / "progress.jsonl").read_bytes().splitlines()
    ]
    assert [row["status"] for row in journal] == [
        "running",
        "failed",
        "running",
        "unsupported",
    ]


@pytest.mark.skipif(
    not hasattr(sys, "set_int_max_str_digits"), reason="Python has no integer digit cap"
)
def test_integer_digit_limit_is_not_a_stuck_running_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        payload = b'{"status":"pass","number":' + b"9" * 700 + b"}"
        _, first = run_payload(tmp_path, monkeypatch, payload)
    finally:
        sys.set_int_max_str_digits(previous)
    assert first["status"] == "failed"
    assert first["reason"].startswith("invalid adapter JSON:")
    assert "limit" in first["reason"]


def test_valid_nonpass_object_keeps_its_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, first = run_payload(
        tmp_path, monkeypatch, b'{"status":"unsupported","reason":"fixture only"}'
    )
    assert (first["status"], first["reason"]) == ("unsupported", "fixture only")
