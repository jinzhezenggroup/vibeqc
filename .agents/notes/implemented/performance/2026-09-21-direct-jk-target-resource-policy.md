# Direct J/K target-resource policy slice (#597)

Agent: ChatGPT
Model: GPT-5.6 Sol
Date: 2026-09-21

This slice separates target resource legality from Direct-J/K scientific constants. It does **not** close #597: AO/GEMM workload thresholds and the resident-PSSS launch shape still need full specialization/autotuning ownership.

## Inventory

| Previous Direct-owned value | Classification | This slice |
| --- | --- | --- |
| persistent ERI AO limit = 16 | workload/profile choice | retained temporarily; next profile-identity slice |
| cuBLAS AO threshold = 17 | workload/profile choice | retained temporarily; next profile-identity slice |
| generated tasks / shell pair = 1024 | bounded-workload schedule choice | retained; not a device fact |
| generated-task cap = 8M | tuning-profile ceiling | moved to DirectJkTuningProfile |
| generated-task arena = 1 GiB | device-memory policy + profile ceiling | moved to target policy; min(profile ceiling, totalGlobalMem/32) |
| CUDA stack target = 64 KiB | conservative correctness/resource fallback | moved to explicit profile policy |
| persistent quartet workers = 8 / SM | occupancy schedule choice | moved to profile ceiling plus runtime SM legality |
| resident PSSS threads = 128 | compile-time kernel schedule/launch-bounds contract | compiler-owned generated schedule; target/profile variation remains |
| resident PSSS max bra primitive pairs = 64 | scratch/layout admission | compiler-owned generated profile; target/workload selection remains |

## Shared target facts

runtime::CudaTargetInfo owns backend resource facts used by policy: compute capability, warp size, SM thread/block ceilings, register/shared-memory capacity, SM count, and total device memory. initialize_cuda_context and Direct J/K both consume the same record. Product names are not part of schedule legality.

Unknown resource facts fail toward conservative fallbacks: 256 MiB generated-task arena and four persistent workers per SM. A runtime-enriched 32-GiB / resource-rich target preserves the previous 1-GiB / eight-worker production choice.

## Synthetic target evidence

The native policy test includes a 2-GiB target with 64 threads/SM and four blocks/SM. It resolves to a 64-MiB generated-task arena, two persistent warp workers/SM, and a 2M-record cap for 32-byte test records. This proves selection is resource-derived rather than keyed to RTX 5090 identity.

No endpoint performance claim is made by this slice. RTX 5090 regression measurement remains required before #597 can close.

## 2026-09-22 correction

The original slice incorrectly reused the fixed-topology 1/32 memory budget to
limit bounded-streaming page scratch. That coupled two different lifetime and
performance domains and reduced the qualified RTX 5090 bounded page below 8M
tasks. The follow-up `2026-09-22-direct-jk-capacity-domain-separation.md`
splits those policies, restores the qualified bounded page, and adds explicit
cross-domain regression guards.
