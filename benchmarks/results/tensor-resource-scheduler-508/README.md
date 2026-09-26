# TensorIR resource-aware scheduler qualification (#508)

Date: 2026-09-20

This directory is evidence for the compiler-generic TensorIR CUDA scheduler.
It is not a molecular-method benchmark and makes no HF/DFT/CC speed claim.
Measurements ran on an allocated NVIDIA GeForce RTX 5090 with CUDA 12.9,
targeting `sm_120`. Endpoint timings include validation, staging, H2D/D2H,
packing, cuBLAS/generated kernels, allocation and error checks.

## Distinct qualified winners

### Layout-sensitive contraction

The first workload is a 1024x1024 contraction whose transposed GEMM result is
internal and then consumed by an elementwise add. Producer-layout planning can
therefore replace the packed contraction with direct GEMM while preserving the
named output ABI.

The selected schedule used `layouts=True`, 256 threads and two generic elements
per thread. Across two independent value fixtures, 10 interleaved A/B pairs per
fixture gave median endpoint speedups of 2.9215x and 2.9097x; bootstrap lower
bounds were 2.7421x and 2.7559x. Both shared noise gates passed.

### Packing-sensitive contraction
The second workload is a named transposed 2048x256 by 256x2048 contraction, so
the output layout keeps packing/scatter unavoidable. This run used the ordinary
default structured search: 256 generated candidates, 167 pruned before compile,
12 admitted by the static compile shortlist, 12 representative screens, three
full finalists and three accepted schedules.

The selected effective schedule used 512x512x128 panels. Its 10-pair complete
endpoint qualification measured a 2.3144x median speedup with a 2.2454x
bootstrap lower bound; the shared noise gate passed. `views=True` and
`fuse=True` appear in the requested schedule but are inert for this graph; the
material execution difference is the panel schedule.

These two winners exercise different execution mechanisms: producer-layout
selection/direct GEMM versus larger-panel packed execution. Promotion profiles
remain workload- and execution-environment-guarded.

## Negative evidence

Additional elementwise fusion, broadcast/fusion and CC-like residual campaigns
retained the deterministic baseline because one or more complete endpoint/noise
gates did not pass. Those outcomes are summarized in `qualification.json`;
no threshold was weakened to manufacture a winner.

`qualification.json` retains complete selected-candidate timing samples,
endpoint profiles, PTXAS resource facts, compile/resource calibration, promotion
profiles, the compact candidate ledger and negative-gate summaries.

## Validation

Host validation:
`PYTHONPATH=.:python python -m pytest tests/python/test_tensor_cuda_search.py tests/python/test_tensor_cuda_tune.py -q`

Real-device execution of the new work-per-thread, reduction-unroll and staging
axes:
`VIBEQC_TENSOR_CUDA_TEST=1 VIBEQC_NVCC=/path/to/cuda-12.9/bin/nvcc PYTHONPATH=.:python python -m pytest tests/python/test_tensor_cuda_execution.py::test_resource_aware_schedule_dimensions_execute_on_cuda -q`

Agent: ChatGPT
Model: GPT-5.6 Sol
