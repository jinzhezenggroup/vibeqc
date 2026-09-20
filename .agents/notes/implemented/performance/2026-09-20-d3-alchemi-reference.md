# #492 NVIDIA ALCHEMI D3 performance reference

Agent: ChatGPT
Model: GPT-5.6 Sol

## Decision

Use NVIDIA ALCHEMI Toolkit-Ops as the primary external GPU performance reference
for the D3(BJ) scheduling work under #492, alongside simple-dftd3 as the independent
correctness oracle and xTBloom as migration provenance.

The benchmark is intentionally optional and does not add ALCHEMI, PyTorch, Warp,
or downloaded D3 parameter files to VibeQC production dependencies. It compares
identical synthetic molecular workloads and damping/cutoff contracts, retains raw
samples, and reports ALCHEMI neighbor-list, D3-only, and complete pipeline latency.

## Interpretation boundary

ALCHEMI 0.4.x publishes FP32 D3 energy/force/CN outputs and stores its reference
parameters as FP32. VibeQC's production D3 path is FP64. Therefore ALCHEMI is a
scheduling/throughput reference rather than an equal-precision numerical baseline.
Every result records this distinction and the cross-implementation numerical delta.

The current VibeQC CUDA baseline remains one serial worker per molecule. A future
pair-parallel/tiled implementation should use the benchmark to identify the crossover
between launch overhead, pair work, and batch parallelism, without changing D3
scientific semantics merely to match a timing number.

## Local smoke evidence

On node3, the benchmark helper tests passed and an 8-atom x 2 CPU smoke run executed
both VibeQC and ALCHEMI Toolkit-Ops 0.4.1. With identical PBE-D3(BJ) damping and a
15-Angstrom hard cutoff, the cross-implementation maximum differences were about
6.8e-10 Eh in energy and 2.2e-9 Eh/bohr in gradient. These CPU timings are not
retained as performance evidence; they only qualify the harness and sign/unit mapping.

The NVIDIA side was also executed through Slurm on the RTX 5090 using
ALCHEMI Toolkit-Ops 0.4.1, Warp 1.16.0 and PyTorch 2.9.1+cu130. The 8-atom x 2
CUDA smoke completed successfully with separately synchronized neighbor-list,
D3-only and combined-pipeline measurements. Those two-sample smoke timings are
API qualification only, not retained performance/promotion evidence.
