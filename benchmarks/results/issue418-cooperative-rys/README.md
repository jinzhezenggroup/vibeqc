# Cooperative DF Rys qualification (#418)

Status: mathematical, GPU derivative, sanitizer and class-workload qualification
passed. The combined library and complete 96/192/384/768 endpoint checks are
pending. These records do not establish an endpoint speedup or promote a mapping.
The accepted #415 low-angular 2.6% improvement remains the automatic baseline.

The candidate shares compiler-generated one-dimensional Gaussian moments across
Cartesian components for 13 canonical classes: 101/102/110/111/112, 201/202,
210/211/212 and 220/221/222. Roots are prepared once per primitive triple;
root/axis owners populate bounded subgroup storage. Packet scheduling, FP64,
public weights, normalization, screening and auxiliary-center translation remain
unchanged. The five previously promoted classes keep their delivered lowering.

## Independent numerical and resource gates

An 85-digit incomplete-gamma/Hankel audit rejects the existing three/four-root
weight interpolants: 158/969 and 188/969 arguments exceed 5e-14. Node errors
already satisfy the gate. The strict option reuses the common evaluator and
branch/layout infrastructure with degree-17 coefficients generated at 90 digits;
all 969 arguments then pass for each order. Direct/value consumers retain their
default tables. The coefficient generator reproduces the checked-in bytes.
Slurm 9891 verifies the emitted CUDA root arithmetic for all four root orders.
Its root header is byte-identical to the candidate header.

All 180 host derivative checks pass against libcint and high-precision center
differentiation. Two initially failed new high-angular T=1e300 checks are kept:
the recovered C derivative underflows while A/B remain normal and cancel. The
final recovered-center check bounds its error by the independently measured A/B
errors plus FP64 addition rounding. Original low-angular, ordinary force and
endpoint gates are unchanged. Both test runs are retained.

All 26 candidate/control objects pass the declared limits: at most 255 registers,
48 KiB shared storage, 128 bytes stack, zero spills and 2 MiB per object. Rys
candidates use 126–130 registers, 0/48/64 bytes stack, and at most 21,536 shared
bytes per block. Slurm 9894 validates Cartesian/spherical representations,
full/symmetric/packed weights, clipped auxiliary panels, asymmetric/coincident
centers, varied contractions and same-atom mapping. Both object groups pass
memcheck, racecheck, initcheck and synccheck without errors or race warnings.

## Class diagnostic (not the complete force endpoint)

Slurm 9895 measures five CUDA-event samples per signature from the retained
384/768 workload distributions, after compilation and sanitizer work finishes.
Every shell, primitive and active-component count matches the retained workload.
All 252 records pass the numerical gate; the maximum derivative error is
5.075e-17. Deterministic dense weights exercise real shell/primitive/panel
frequencies; they are not the frozen physical density used by the endpoint gate.

| AO | Selected 13 classes, polynomial (ms) | Cooperative Rys (ms) |
| --- | ---: | ---: |
| 384 | 157.557 | 60.726 |
| 768 | 1184.631 | 467.398 |

These are sums of per-signature median kernel times, excluding the other five
classes, SCF, response production and final endpoint work. Every class improves
on both distributions, including the old 101/110 negative controls. Their old
component-local results remain in `../issue394-batch-rys/`.

The combined campaign must still demonstrate >=0.20 s clean 768-AO complete
warm energy+force improvement, or >=15% complete shell-derivative improvement
with a clean endpoint win, using at least five interleaved pairs and identical
frozen starting density/operator work. It also requires the 384 control and
96/192 correctness checks. No class timing substitutes for that decision.

## Retention and reproduction

`manifest.json` records byte/hash-verified restoration of all retained records.
Large root audits and raw compiler records use deterministic gzip; binaries,
routine build output and temporary generated headers remain local. Header hashes,
source patch, compiler commands/resources, numerical failures, setup/recording
failures, sanitizer conclusions and every timing sample are retained. The first
ordering-sensitive root audit is preserved alongside the corrected paired-node
audit, so ordering differences are not misreported as physical errors.

`reproduction/` contains the actual local drivers; paths identify the Python,
CUDA 12.9.1 and frozen CPU-oracle library used on this machine. Run every GPU
driver through finite `srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1
--time=...`, preserving scheduler visibility. `identity/endpoint-protocol.json`
declares the next complete-endpoint experiment; its results are still pending.
