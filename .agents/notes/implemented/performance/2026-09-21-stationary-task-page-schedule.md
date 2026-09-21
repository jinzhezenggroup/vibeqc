# Decision: promote bounded stationary CUDA task pages

Status: implementation candidate
Date: 2026-09-21
Refs: #660, #664, #667

## Decision

After compact AO-task batching (#664), use the existing maximum admitted task-page
capacity of 4096 for the production stationary CUDA diagnostic/public-force path.
This changes only the execution schedule. The mathematical AO tasks, primitive
work count, generated first-derivative programs, source weights, precision and
resource/failure contracts are unchanged. Callers can still select smaller
diagnostic pages explicitly.

The capacity remains finite and already passes the existing `[1,4096]` admission
contract. Relative to the previous 128-task default, the conservative formulas add
about 0.70 MiB device staging and 1.03 MiB host numeric staging at the maximum page,
well below the existing serialized 512 MiB device / 256 MiB host public-force caps.

## RTX 5090 evidence

The measured production implementation is PR #683. The benchmark checkout had one
test-only component-key repair ahead of that production tree; no production source
or native library input differed.

A PBE-RKS water sweep held `tile_points=4096` and `integral_terms=32` fixed:

| task page | task batches | launches | warm wall samples (s) |
| ---: | ---: | ---: | --- |
| 128 | 22 | 54 | 1.9411, 1.9474 |
| 256 | 12 | 34 | 1.1964, 1.1924 |
| 512 | 7 | 24 | 0.7701, 0.7625 |
| 1024 | 5 | 20 | 0.5652, 0.5649 |
| 2048 | 4 | 18 | 0.4472, 0.4436 |
| 4096 | 3 | 16 | 0.3668, 0.3403 |

Source H2D remained exactly 434,184 bytes in every case. The improvement therefore
comes from reducing bounded task-page execution boundaries, not from dropping work
or transfers.

A separate grid-tile sweep held the old 128-task page fixed. Increasing
`tile_points` from 256 to 4096 reduced launches from 120 to 54 and measured grid
kernel time from about 0.46 ms to 0.13 ms, but complete warm wall remained
approximately 1.94--1.96 s. Geometry tiling is therefore not the controlling
latency on this fixture.

The 4096-task water result is roughly 6.8--7.3x faster than the recorded #660
~2.49 s production-default PBE baseline. It also keeps launch count far below the
#660 10x-reduction threshold. This is a schedule result, not yet the completion
claim for #660: final integration still needs the independent method/spin and
multi-size acceptance gates, AOT/no-force-call-NVCC closure, and exact-head
production qualification.

## Why not async first?

#667 proposed a pinned ring / phase-async execution layer because the old
microbatch path synchronized every bounded page. The sweep shows that most of the
same latency can first be removed by using the already-supported larger bounded
page, with substantially less lifecycle and failure-isolation complexity. Async
submission remains a follow-up if larger-system scaling or residual phase
synchronization still warrants it.

Agent: ChatGPT
Model: GPT-5.6 Sol
