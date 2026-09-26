# Decision: opt-in dense producer/consumer layouts inside TensorIR

Status: implemented, opt-in
Date: 2026-09-19
Issue: #509
Agent: ChatGPT
Model: GPT-6 Astra Pro

## Problem

The CUDA planner treated every materialized tensor as logical C-order. A producer
could therefore write an inconvenient batch/index order immediately before the
consumer gathered it into GEMM panels. Even a GEMM with otherwise direct operands
required packed scatter solely because its logical output order differed from
its natural grouped matrix order. Local transpose recognition could not solve
this producer/consumer decision.

## Decision

Introduce backend-neutral `DenseLayout` descriptors and an opt-in
`TensorSchedule(layouts=True)` pass. The descriptor stores axis ordinals rather
than scientific labels, so physical representation remains separate from the
logical TensorSpec and all mathematical serialization/AD rules.

A bounded trial jointly assigns the natural GEMM result order and either matrix
orientation for each operand. It is scored against the entire region, including
conflicting consumers, the producing contraction and non-C generic accesses.
Repeated references to the same producer must agree on one physical map. Only
strict cost improvements are accepted; the pass is bounded to 256 trials and
four sweeps for each tile configuration. This is neither a globally optimal
layout solver nor a certificate that no other schedule could fit a rejected
budget.

The score includes semantic panel conversion bytes, with repeated A packing for
N tiles and B packing for M tiles. An additional one-tensor-read/write equivalent
per non-C generic access conservatively penalizes strided access. That penalty is
not measured DRAM traffic or retained storage. Actual performance selection still
uses the existing complete-endpoint, numerical, compiled-resource and noise gates.

Generic producer writes are physically contiguous, with the logical coordinates
mapped before scalar evaluation. Reads and packed scatters perform the inverse
mapping. Direct GEMM recognition inspects physical label order. The native
runtime allocation/execution template and numerical scalar formulas are unchanged.

## Invariants and rejected alternatives

- Do not rewrite or transpose scientific TensorIR equations to encode a backend
  storage decision. The same primitive/AD graph is lowered in both layouts.
- Do not permute public inputs, constants or named outputs. Their pinned C-order
  spans are required by ordinary transfers, resident upload/download and borrowed
  tensor contracts. Caller noncontiguous/negative/overlapping views still stage
  through the existing checked C-order buffers.
- Do not directly address virtual operands at offset -1. Virtual views and fused
  expressions keep generated logical accessors and packed GEMM fallback. Affine
  alias borrowing requires a separate ownership/lifetime implementation.
- Do not add persistent packed copies for conflicting consumers. This first slice
  changes only equally sized dense buffers; arena slots and lifetimes are unchanged.
- Do not promote a static byte-count reduction to a default. Layout candidates are
  explicit and the old layout remains the default/fallback. End-to-end wins are
  domain-specific, not a claim about complete molecular CCSD/DFT calculations.
- Rerun layout selection after tile shrinkage. Otherwise costs could describe the
  requested tiles while allocation and repacking describe another actual schedule.
- Do not reuse an artifact merely because the scientific hash matches. Plan schema
  2 serializes layouts and decisions; `layout_identity` is also available as a #459
  guard fact. It complements rather than replaces full-plan/compiler/scientific
  identity. Native artifact checks still reject a different plan.

## Evidence and reproduction

CPU tests cover dense permutation bijections against independent NumPy layouts,
transpose equivalence, invalid/empty/singleton descriptors, joint operand choices,
conflicting/shared consumers, grouped GEMM labels, all four transpose flags,
unchanged public spans, virtual fallback, identity guards, trial limits, disjoint
live allocations and budget-driven tile changes. CUDA tests cover ordinary and
resident execution, caller strides, generic/direct/packed producer paths,
independent analytic JVP/VJP equations, error recovery and tiny packing panels.

```bash
PYTHONPATH=python python -m pytest -q tests/python/test_tensor*.py
PYTHONPATH=python python tools/check_compiler_structure.py
# In a finite GPU allocation, with a compiler matching the allocated target:
VIBEQC_TENSOR_CUDA_TEST=1 VIBEQC_NVCC=/path/to/nvcc VIBEQC_TENSOR_ARCH=sm_120 \
  PYTHONPATH=python python -m pytest -q tests/python/test_tensor_layout_cuda.py \
  tests/python/test_tensor_cuda_execution.py
PYTHONPATH=python python tools/tensor_layout_benchmark.py --nvcc /path/to/nvcc \
  --shape 33 7 65 31 --repeats 10 --output .artifacts/layout-small
PYTHONPATH=python python tools/tensor_layout_benchmark.py --nvcc /path/to/nvcc \
  --shape 65 9 97 63 --repeats 10 --output .artifacts/layout-large
```

The runner compares the same `out[bij] = sum_k x[ibk]^2 * y[bkj]` equation under
baseline and producer-layout plans. Two fixtures use C-order data and scaled
Fortran-order data, respectively. Paired warm samples include input validation,
staging, H2D/D2H, packing, all kernels/library calls, result allocation and error
checks. Startup and separately synchronized section profiles are not mixed into
warm performance gates. Both resource/numerical gates and both existing timing
noise/bootstrap gates must pass. All raw samples remain in local evidence; no
Release or external evidence host is required.

## Remaining scope / revisit conditions

Keep #509 open for affine virtual-view propagation, alternative shared pack-once
or cheap-recompute choices, richer budget-aware selection and cross-subsystem
ProgramIR propagation through #460. Qualify FP32 integration against #485 rather
than claiming coverage from the current FP64 master. Revisit the stride penalty
when larger shape/layout domains provide measured evidence for a better model.

## Qualification snapshot

Measured on an allocated RTX 5090 (`sm_120`), driver 580.95.05, with the
CUDA 12.9 compiler. The baseline is master `5a7fdeb2689553c0a304dad3338ba184d850ef60`
plus this source change in a dirty worktree. Exact compiler source/artifact and
binary hashes below identify what ran; this is not evidence for unmeasured builds.
No cross-device performance claim or global default promotion is made.

- Tensor CPU regression: **225 passed, 72 GPU opt-in skips**.
- Additional compiler/ownership checks: **42 passed**; compiler structure checks
  inspect **189 modules with zero dependency errors**. Ruff and whitespace checks pass.
- Allocated CUDA regression: **65 passed, 1 skipped**. All **35 new layout CUDA
  tests pass**. The existing provider rollback test skips because the measured
  provider allocation fits already allocated pages on this stack, not because
  CUDA was unavailable.
- Both sizes and both fixtures pass the existing numerical/resource, shared
  timing-noise, >2% speedup and bootstrap gates (10 paired warm repeats each).

| Shape (i,b,k,j) | Caller fixture | Baseline median ms | Layout median ms | Speedup | Bootstrap lower 95% |
| --- | --- | ---: | ---: | ---: | ---: |
| 33 x 7 x 65 x 31 | C-order | 0.363522 | 0.216244 | 1.6811x | 1.6580x |
| 33 x 7 x 65 x 31 | scaled Fortran-order | 0.373270 | 0.225812 | 1.6530x | 1.6369x |
| 65 x 9 x 97 x 63 | C-order | 0.640902 | 0.412955 | 1.5520x | 1.5252x |
| 65 x 9 x 97 x 63 | scaled Fortran-order | 0.699304 | 0.460061 | 1.5200x | 1.4742x |

Packing panels fall from **41,472 to 0 bytes** for the smaller case, and
**132,096 to 0 bytes** for the larger case. Arena capacities remain exactly
410,880 and 1,643,264 bytes respectively. Complete planned peaks fall from
105,735,624 to 105,694,152 bytes and from 107,957,176 to 107,825,080 bytes.
These are numeric-buffer capacities, not total process/device memory. Semantic
conversion reads+writes fall from 580,496 and 2,377,584 bytes to zero respectively.
Maximum absolute error is 3.33e-16 and 5.56e-16 respectively, against the existing
1e-11 absolute / 1e-10 relative gate.

The existing evidence-retention check fails identically on the base HEAD and this
change: `benchmarks/results/` contains 105,281,931 bytes, above its 100,663,296-byte
aggregate limit. This change modifies no files there and does not relax that guard,
delete unrelated evidence or publish external archives.

Compact raw paired measurements and exact identities are retained here; the runner
also writes the full local ledger, resource records and separate section profiles.
All times below are seconds in their actual interleaved execution order.

Compiler tensor source identity: `eaf2e1291fd5930fd75a344e9a3141ddbc0cdcfca5d04a7cde633847f6d61a34`

Runner SHA-256: `9e298cd3b830da50a6ef24019a329eeb0c2e96d1b23402763e6914a0283e136a`

```json
[{"shape_ibkj":[33,7,65,31],"seed":509,"baseline_plan":"8644e2f3878338209825e83ffb085d9a5c88d1a206bc8a72e72a5b4bfbb00f12","candidate_plan":"6762f52451991e9f5b2d96cb7e348b6f2767aaab8bc794dfa61c040467c0b10c","baseline_artifact":"8ce95eb8c6fb1eb2512306157286665d0833a3156808be1bedb60dae05c32c56","candidate_artifact":"73615bd84094c777a744e0a1a321f7e2d36ecc117f8036af7175ebaede06c011","baseline_binary_sha256":"007668d48f7bbb8e3c04d41a5214a8d1564f28aab0b2873443461007585dd034","candidate_binary_sha256":"499bf668d4a8483287f0ef530ab9124a95887c6474ae8a105c95f351c904b24e","max_absolute_error":3.3306690738754696e-16,"fixtures":[{"layout":"C-order","inputs_hash":"60c15d5f211e0c71af87e5f42f4eccef5a393b24cc135c097e41db629af2a18b","paired_seconds":[["baseline",0.0003724130801856518],["candidate",0.00021986104547977448],["candidate",0.0002104528248310089],["baseline",0.00036754412576556206],["baseline",0.00036255503073334694],["candidate",0.00021746614947915077],["candidate",0.00021645519882440567],["baseline",0.00036448799073696136],["baseline",0.00036828499287366867],["candidate",0.00022481102496385574],["candidate",0.0002123173326253891],["baseline",0.00038073910400271416],["baseline",0.0003604302182793617],["candidate",0.00020754802972078323],["candidate",0.00021603284403681755],["baseline",0.0003608008846640587],["baseline",0.0003601289354264736],["candidate",0.00023268628865480423],["candidate",0.00021372921764850616],["baseline",0.00036203302443027496]]},{"layout":"scaled Fortran-order","inputs_hash":"0c6391a09ce5bbb947922a15a4513a44c8bb9b8952d630d2898efe1f793fe9b9","paired_seconds":[["baseline",0.00038121966645121574],["candidate",0.00023016100749373436],["candidate",0.0002322150394320488],["baseline",0.00037331506609916687],["baseline",0.000378195196390152],["candidate",0.00022806599736213684],["candidate",0.0002259630709886551],["baseline",0.0003874208778142929],["baseline",0.00037322426214814186],["candidate",0.00021959980949759483],["candidate",0.00022794632241129875],["baseline",0.00037110084667801857],["baseline",0.00037043914198875427],["candidate",0.00021996069699525833],["candidate",0.00022566178813576698],["baseline",0.0003783837892115116],["baseline",0.0003728242591023445],["candidate",0.000220451969653368],["candidate",0.0002206927165389061],["baseline",0.0003726240247488022]]}]},{"shape_ibkj":[65,9,97,63],"seed":509,"baseline_plan":"fc0db438d6d90f232dc7b2c914d795db3999e87080161b4123c77f3c8c4228f0","candidate_plan":"8edf717a3fb0bc027dceb948c53f8ec972e113066c8ade6c2496e990cf388a25","baseline_artifact":"78dab653e330671d361febbe0c1987d4ff2eeeccb9212c36566175c048059df2","candidate_artifact":"c5d9c2366dd1c73b6fa4f06718c723373fb22ac9fe7aaee53212fcba2029c9e8","baseline_binary_sha256":"322ec1dec0fa067ab1d565b146cd8fb76c97b11d524c024c029eb40cd8c16db3","candidate_binary_sha256":"7998b93e371921e14129d2287079c353f889a9bb07409cddc14443add85cc18e","max_absolute_error":5.551115123125783e-16,"fixtures":[{"layout":"C-order","inputs_hash":"8e7eca63b2a51537c49867c34f701a577e2ed0998247aeb0ec2eebd9eb1c851a","paired_seconds":[["baseline",0.0006580399349331856],["candidate",0.00042042508721351624],["candidate",0.00041931308805942535],["baseline",0.0006491532549262047],["baseline",0.0006556650623679161],["candidate",0.00041155796498060226],["candidate",0.0004143528640270233],["baseline",0.0006479308940470219],["baseline",0.0006584711372852325],["candidate",0.0004258551634848118],["candidate",0.00041780993342399597],["baseline",0.0006324811838567257],["baseline",0.0006289943121373653],["candidate",0.0003914590924978256],["candidate",0.0003936430439352989],["baseline",0.0006312187761068344],["baseline",0.0006297659128904343],["candidate",0.00039612920954823494],["candidate",0.00039500603452324867],["baseline",0.0006338739767670631]]},{"layout":"scaled Fortran-order","inputs_hash":"6ea853bf15e83d8ddf17e53419f1fce843dbd046c20f986bbf208b85c49183d8","paired_seconds":[["baseline",0.0007026148959994316],["candidate",0.00048039015382528305],["candidate",0.000458328053355217],["baseline",0.0006997999735176563],["baseline",0.0006988081149756908],["candidate",0.0004763319157063961],["candidate",0.0004723849706351757],["baseline",0.0007147281430661678],["baseline",0.0006983070634305477],["candidate",0.0004611029289662838],["candidate",0.0004540998488664627],["baseline",0.0006962637417018414],["baseline",0.0007117129862308502],["candidate",0.00045901909470558167],["candidate",0.00045303720980882645],["baseline",0.0006973152048885822],["baseline",0.0007108505815267563],["candidate",0.0004802593030035496],["candidate",0.0004529869183897972],["baseline",0.0006967340596020222]]}]}]
```
