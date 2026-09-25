"""Journal bounded production-adapter runs against the frozen DFT-MP-v1 matrix.

The adapter owns public VibeQC and independent-oracle calls. It receives the
manifest, row ID and immutable input path and writes one result JSON to stdout.
This runner never fabricates scientific or performance values. The adapter and
its build/artifact hashes are part of the retained campaign identity.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .freeze_contract import ROOT, digest
from .validate import (
    InvalidEvidence,
    _check_run,
    audit,
    checked_file,
    manifest,
    read_json,
    require,
)


def _write(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def _capture(out: Path, safe: str) -> dict:
    return {
        kind: {
            "path": f"{safe}.{suffix}",
            "sha256": digest((out / f"{safe}.{suffix}").read_bytes()),
        }
        for kind, suffix in (
            ("stdout", "stdout"),
            ("stderr", "stderr"),
            ("progress", "progress.jsonl"),
        )
    }


def _terminate_tree(process: subprocess.Popen[bytes]) -> bool:
    """Stop descendants as well as the direct adapter after a watchdog expiry."""

    try:
        if os.name == "nt":
            killed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=30,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if killed.returncode != 0:
                process.kill()
                process.wait(timeout=10)
                return False
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        return True
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
            process.wait(timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
        return False


def run(plan_path: Path, out: Path, selected: set[str] | None = None) -> Path:
    contract = manifest()
    plan = read_json(plan_path)
    command = plan.get("adapter_command")
    require(
        type(command) is list
        and command
        and all(type(arg) is str and arg for arg in command),
        "adapter_command must be an argv list",
    )
    timeout = plan.get("timeout_seconds")
    require(
        type(timeout) is int and 0 < timeout <= 86400,
        "bounded timeout_seconds required",
    )
    campaign = plan.get("campaign")
    require(type(campaign) is dict, "campaign identity required")
    adapter_index = plan.get("adapter_command_file_index")
    require(
        type(adapter_index) is int and 0 <= adapter_index < len(command),
        "adapter_command_file_index required",
    )
    for key in ("library", "artifact", "adapter", "build_record", "conditions"):
        path = checked_file(plan_path.resolve().parent, campaign.get(key))
        campaign[key]["path"] = str(path.resolve())
    adapter_path = Path(campaign["adapter"]["path"])
    require(
        Path(command[adapter_index]).resolve() == adapter_path.resolve(),
        "adapter command/provenance mismatch",
    )
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "contract_sha256": contract["contract_sha256"],
        "campaign": campaign,
        "rows": [
            {
                "id": row["id"],
                "status": "not-run",
                "reason": "not executed in this campaign",
            }
            for row in contract["rows"]
        ],
    }
    receipt_path = out / "receipt.json"
    if receipt_path.exists():
        prior = read_json(receipt_path)
        require(
            prior.get("contract_sha256") == receipt["contract_sha256"]
            and prior.get("campaign") == campaign,
            "resume campaign/contract mismatch",
        )
        require(
            [entry.get("id") for entry in prior.get("rows", [])]
            == [entry["id"] for entry in receipt["rows"]],
            "resume row inventory mismatch",
        )
        receipt = prior
    else:
        _write(receipt_path, receipt)
    journal = out / "progress.jsonl"
    require(
        not any(entry["status"] == "running" for entry in receipt["rows"]),
        "running adapter may still own child processes; stop it and start a new campaign",
    )
    audit(receipt_path)
    for index, row in enumerate(contract["rows"]):
        key = row["id"]
        if selected is not None and key not in selected:
            continue
        if receipt["rows"][index]["status"] != "not-run":
            continue
        safe = key.replace("/", "__")
        input_path = ROOT / contract["cases"][row["case"]]["input"]
        partial_path = out / f"{safe}.progress.jsonl"
        stdout_path = out / f"{safe}.stdout"
        stderr_path = out / f"{safe}.stderr"
        if any(path.exists() for path in (partial_path, stdout_path, stderr_path)):
            raise InvalidEvidence(
                "unclaimed adapter files may still be live; stop prior process and start a new campaign"
            )
        for path in (partial_path, stdout_path, stderr_path):
            path.touch()
        argv = [
            *command,
            "--manifest",
            str(ROOT / "manifest.json"),
            "--row",
            key,
            "--input",
            str(input_path),
            "--progress",
            str(partial_path),
        ]
        started = time.time()
        receipt["rows"][index] = {
            "id": key,
            "status": "running",
            "reason": "adapter attempt in progress",
        }
        _write(receipt_path, receipt)
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "row": key,
                        "status": "running",
                        "start_unix_s": started,
                        "argv": argv,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        stop_campaign = False
        try:
            with (
                stdout_path.open("wb") as stdout_handle,
                stderr_path.open("wb") as stderr_handle,
            ):
                process = subprocess.Popen(
                    argv,
                    cwd=plan_path.resolve().parent,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=os.name != "nt",
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.CREATE_NO_WINDOW
                    )
                    if os.name == "nt"
                    else 0,
                )
                try:
                    returncode = process.wait(timeout=timeout)
                    timed_out = False
                except subprocess.TimeoutExpired:
                    timed_out = True
                    stop_campaign = not _terminate_tree(process)
                    returncode = process.returncode
            if timed_out:
                entry = {
                    "id": key,
                    "status": "failed" if stop_campaign else "timed-out",
                    "reason": "watchdog could not terminate full process tree"
                    if stop_campaign
                    else f"watchdog {timeout}s",
                }
            elif returncode:
                entry = {
                    "id": key,
                    "status": "failed",
                    "reason": f"adapter exit {returncode}",
                }
            else:
                payload = json.loads(stdout_path.read_bytes())
                if type(payload) is not dict:
                    entry = {
                        "id": key,
                        "status": "failed",
                        "reason": "adapter JSON must be an object",
                    }
                elif payload.get("status") == "pass":
                    try:
                        _check_run(
                            payload,
                            campaign,
                            row,
                            contract,
                            out,
                            _capture(out, safe),
                        )
                    except (
                        InvalidEvidence,
                        AttributeError,
                        KeyError,
                        OverflowError,
                        TypeError,
                        ValueError,
                    ) as error:
                        entry = {
                            "id": key,
                            "status": "failed",
                            "reason": f"adapter pass rejected: {error}",
                        }
                    else:
                        entry = {
                            "id": key,
                            "status": "pass",
                        }
                else:
                    status = payload.get("status", "unknown")
                    if status not in (
                        "unknown",
                        "unsupported",
                        "not-run",
                        "failed",
                        "timed-out",
                    ):
                        status = "unknown"
                    entry = {
                        "id": key,
                        "status": status,
                        "reason": str(
                            payload.get("reason")
                            or "adapter did not provide a pass result"
                        ),
                    }
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            entry = {
                "id": key,
                "status": "failed",
                "reason": f"invalid adapter JSON: {error}",
            }
        except OSError as error:
            entry = {
                "id": key,
                "status": "failed",
                "reason": f"adapter launch failed: {error}",
            }
        entry["capture"] = _capture(out, safe)
        if entry["status"] == "pass":
            entry["evidence"] = entry["capture"]["stdout"]
        if entry["status"] != "pass":
            entry["partial_progress"] = entry["capture"]["progress"]
        receipt["rows"][index] = entry
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "row": key,
                        "status": entry["status"],
                        "start_unix_s": started,
                        "end_unix_s": time.time(),
                        "capture": entry["capture"],
                        "argv": argv,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        _write(receipt_path, receipt)
        if stop_campaign:
            break
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--row",
        action="append",
        help="run only this exact row ID; others remain not-run",
    )
    args = parser.parse_args()
    if args.row:
        known = {row["id"] for row in manifest()["rows"]}
        require(set(args.row) <= known, "unknown row selection")
    print(run(args.plan, args.out, set(args.row) if args.row else None))


if __name__ == "__main__":
    main()
