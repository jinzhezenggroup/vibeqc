# CUDA ownership shards

This directory is the reviewed semantic source for CUDA ownership.

- `meta.json` carries ledger-wide schema metadata.
- `generated/<family>.json` owns one generated family per file.
- `subsystems/<name>.json` owns one subsystem summary per file.
- `files/<source path>.json` owns exactly one native CUDA/CUDA-bearing source record.

There is deliberately no central manifest of shard names. `tools/report_cuda_ownership.py`
discovers and sorts the shard files deterministically, then performs the same complete
source-inventory, stale-anchor, duplicate-owner and evidence checks as before. Adding
or changing one CUDA source should therefore touch only that source's shard plus any
semantic subsystem/generated-family shard whose meaning actually changed.

The reporter still accepts a monolithic JSON file through `--ledger <file>` for
historical/reproduction workflows, but the repository-tracked current source is this
directory.

Agent: ChatGPT
Model: GPT-5.6 Sol
