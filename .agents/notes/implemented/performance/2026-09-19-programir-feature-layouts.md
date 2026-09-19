# Decision: propagate one DFT feature layout across ProgramIR into native XC

Status: implemented and locally qualified
Date: 2026-09-19
Issue: #460
Related: #509, #203

## Problem

The first #460 slice proved serial cross-provider lifetime planning, but the dense
CPU XC call remained opaque. Inspection of the next real boundary found no extra
pack between NativeAO and the feature contractions: AO jets are already emitted as
C-contiguous [jet, point, AO] and consumed directly by the existing BLAS path.

The materialization opportunity is one stage later. The DFT feature producer
already computes rho/gradient/sigma, while generated native scalar XC previously
repacked those values through a feature matrix copy, validation copy, active-point
gather, variable np.stack, and result scatter before Vxc assembly.

## Decision

Promote the existing TensorIR DenseLayout implementation to a backend-neutral
common contract and keep tensor.layout as a compatibility export. ProgramIR buffers
may bind one exact DenseLayout plus item size; layout is validated, serialized and
included in program identity. This does not import the TensorIR planner into DFT.

Add an internal polarized DensityFeatureBlock that owns one C-contiguous FP64
[feature, point] scalar buffer plus the GGA Cartesian gradient buffer when needed.
The public density_features API and numerical conventions remain unchanged.
NativeContractionProgram.scalar_values_packed validates that exact owner with
copy=False and passes the required row prefix directly to the generated point
function. Potential assembly then reuses the existing coefficient and AO
contraction implementation. The complete prepared path is tested with the legacy
scalar_values repacking entry point replaced by a failure.

No aliases, input donation, async leases, CUDA scheduling, general fusion,
unpolarized promotion, response/geometry changes, or new scientific equations are
part of this slice.

## Qualification

Local node3 validation on the candidate worktree:
- 222 ProgramIR/XC/Tensor-layout tests passed; 35 CUDA opt-in cases were skipped
  in the ordinary CPU shell.
- Compiler structure audit: 202 modules, zero dependency errors.
- Ruff check/format and git diff --check passed.
- Independent stored LDA/PBE energy and potential fixtures remain unchanged.
- Weak-reference lifetime checks still show no previous tile boundary payload at
  the next AO collocation.
- The native scalar input is verified to share memory with the DFT-owned feature
  buffer.

Matched complete fixed-density PBE E/V timing used the same
`build/libvibeqc.so`, one OpenBLAS/OMP thread, tile capacity 7, and the existing
`benchmarks/programir_xc_lifetimes.py` runner. Six rounds alternated baseline and
candidate process order; each process used 15 warmed complete endpoint repeats.

| fixture | baseline median | candidate median | paired ratio median |
| --- | ---: | ---: | ---: |
| H2, 2 AOs | 5.477135 ms | 4.803290 ms | 1.1379x |
| f-spherical, 16 AOs | 5.748916 ms | 5.054330 ms | 1.1377x |

Per-round baseline/candidate ratios were:
- H2: 1.1343, 1.1356, 1.1380, 1.1474, 1.1379, 1.1422.
- f-spherical: 1.1387, 1.1309, 1.1390, 1.1358, 1.1366, 1.1834.

These timings include the complete prepared fixed-density E/V call, including
feature contractions, generated scalar XC and Vxc assembly. They are not full SCF,
force, large-system, process-RSS, CUDA or universal DFT speedup claims.

A Slurm RTX 5090 run was also attempted for the 35 TensorIR CUDA layout tests.
The installed toolkit is CUDA 12.4 only; NVCC rejects `sm_120` before compiling
generated code (`Value 'sm_120' is not defined for option 'gpu-architecture'`).
The driver reports RTX 5090 / 580.95.05. This is an environment/toolkit blocker,
not a passed CUDA qualification and not evidence of a candidate code failure.

Agent: ChatGPT
Model: GPT-5.6 Sol
