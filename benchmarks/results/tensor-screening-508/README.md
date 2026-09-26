# TensorIR representative-screening qualification

This is a bounded contract qualification for issue #508, not evidence that a
molecular method or overall tuning is faster. The two cases use real CUDA 12.9
`sm_120` execution on an allocated RTX 5090. Both retain the ordinary baseline:
none of the shortlisted candidates passes all existing performance gates.

`qualification.json` retains every screening and final A/B scalar timing, input
identities, device identity, numerical errors, compiled resource reports, source
file SHA-256 values, and complete local-evidence checksums. Full planner payloads
and per-sample diagnostics remain in the ignored local paths; the compact record
does not imply those files are downloadable from GitHub.

Each case screens three candidates on one fixture, then obtains fresh paired
measurements on all three fixtures for one finalist. This records 30 A/B pairs
rather than the 45 that unfiltered qualification would request at the same
repeat count. That is a work-count comparison, **not a wall-time speedup**; the
recorded whole-tuning times include compilation and have no matched control.

Reproduce inside a finite GPU allocation, with the documented CUDA compiler and
cuBLAS headers/library paths configured:

```sh
PYTHONPATH=python VIBEQC_TENSOR_CUDA_TEST=1 VIBEQC_TENSOR_ARCH=sm_120 \
  VIBEQC_NVCC="$NVCC" python -m pytest -q \
  tests/python/test_tensor_cuda_execution.py -k staged_tuning
```

The tests compare selected execution with independent NumPy formulas and do not
require a noisy timing outcome to become a speedup. Broader #508 acceptance,
including calibrated costs and winners on multiple representative workloads,
remains open.

Agent: ChatGPT
Model: GPT-6 Astra Pro
