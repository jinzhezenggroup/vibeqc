# Shard the CUDA ownership source

Date: 2026-09-20
Parent: #349

## Decision

Replace the single repository-tracked `docs/cuda_ownership.json` semantic ledger
with independently reviewable shards under `docs/cuda_ownership/`.

The previous monolith combined 199 native source records, 23 generated families
and 16 subsystem summaries. Unrelated CUDA/compiler PRs therefore edited the same
JSON file even when their scientific ownership did not overlap.

The current source is split as follows:

- one native record per `files/<source path>.json`;
- one generated family per `generated/<family>.json`;
- one semantic subsystem summary per `subsystems/<name>.json`;
- ledger-wide schema metadata only in `meta.json`.

`tools/report_cuda_ownership.py` discovers and sorts shards deterministically and
keeps the existing complete-inventory, stale-anchor, stale-evidence, duplicate-name
and role validation. Its `--ledger` option still accepts a monolithic JSON file for
historical benchmark/reproduction inputs.

There is intentionally no aggregate shard manifest: requiring every new CUDA file
or generated family to edit a central list would recreate the conflict hotspot this
change removes.

Agent: ChatGPT
Model: GPT-5.6 Sol
