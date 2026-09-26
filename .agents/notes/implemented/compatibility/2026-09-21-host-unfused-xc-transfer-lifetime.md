# Decision: retain host-unfused XC upload controls in the plan

Status: implemented in follow-up branch; mainline integration pending
Date: 2026-09-21

The host-unfused path passed local totals/error values to asynchronous H2D
copies and returned before guaranteed completion. Its constructor also skipped
the former XC setup drain while uploading stack-based spin controls. Pageable
memory may be staged synchronously, but CUDA does not guarantee this for every
Async call, so incidental staging cannot define the ownership contract.

Retain these inputs as plan members until the existing stream completion and
cleanup boundaries. Count their numeric storage in retained host bytes. Reuse
the unchanged uploads, synchronization points, XC formulas and device buffers.
No extra synchronization or scientific fallback is introduced by this repair.

The regression compiles the actual stage_xc body with a deferred-copy contract.
It fails on the original source and passes all six functional/spin branches
with retained storage. A fresh CUDA 12.9/sm_120 library build passes. All 109
selected host profile/options/staging tests pass; four explicit GPU gates skip.
The requested Slurm GPU validation could not allocate a busy node, so no new
GPU numerical qualification is claimed. PR #798 merged concurrently before
publication; its merge does not include this follow-up repair.

Reference: NVIDIA CUDA Runtime API, API synchronization behavior.
Agent: ChatGPT
Model: GPT-6 Astra Pro
