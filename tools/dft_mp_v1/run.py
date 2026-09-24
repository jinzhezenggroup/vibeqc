"""Journal bounded production-adapter runs against the frozen DFT-MP-v1 matrix.

The adapter owns public VibeQC and independent-oracle calls. It receives the
manifest, row ID and immutable input path and writes one result JSON to stdout.
This runner never fabricates scientific or performance values. The adapter and
its build/artifact hashes are part of the retained campaign identity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from .freeze_contract import ROOT, digest
from .validate import audit, checked_file, manifest, read_json, require


def _write(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp.replace(path)


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
    audit(receipt_path)
    journal = out / "progress.jsonl"
    for index, row in enumerate(contract["rows"]):
        key = row["id"]
        if selected is not None and key not in selected:
            continue
        if receipt["rows"][index]["status"] != "not-run":
            continue
        safe = key.replace("/", "__")
        input_path = ROOT / contract["cases"][row["case"]]["input"]
        partial_path = out / f"{safe}.progress.jsonl"
        partial_path.touch(exist_ok=True)
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
        try:
            completed = subprocess.run(
                argv,
                cwd=plan_path.resolve().parent,
                timeout=timeout,
                capture_output=True,
                check=False,
            )
            stdout, stderr = completed.stdout, completed.stderr
            if completed.returncode:
                entry = {
                    "id": key,
                    "status": "failed",
                    "reason": f"adapter exit {completed.returncode}",
                }
            else:
                payload = json.loads(stdout)
                if payload.get("status") == "pass":
                    evidence_path = out / f"{safe}.json"
                    evidence_path.write_bytes(stdout)
                    entry = {
                        "id": key,
                        "status": "pass",
                        "evidence": {
                            "path": evidence_path.name,
                            "sha256": digest(stdout),
                        },
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
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.stdout or b"", error.stderr or b""
            entry = {"id": key, "status": "timed-out", "reason": f"watchdog {timeout}s"}
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            entry = {
                "id": key,
                "status": "failed",
                "reason": f"invalid adapter JSON: {error}",
            }
            stdout, stderr = completed.stdout, completed.stderr
        except OSError as error:
            entry = {
                "id": key,
                "status": "failed",
                "reason": f"adapter launch failed: {error}",
            }
            stdout, stderr = b"", b""
        (out / f"{safe}.stdout").write_bytes(stdout)
        (out / f"{safe}.stderr").write_bytes(stderr)
        if entry["status"] != "pass":
            entry["partial_progress"] = {
                "path": partial_path.name,
                "sha256": digest(partial_path.read_bytes()),
            }
        receipt["rows"][index] = entry
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "row": key,
                        "status": entry["status"],
                        "start_unix_s": started,
                        "end_unix_s": time.time(),
                        "stdout_sha256": digest(stdout),
                        "stderr_sha256": digest(stderr),
                        "argv": argv,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        _write(receipt_path, receipt)
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
