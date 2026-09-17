# Cooperative DF Rys qualification (#418)

Status: mathematical, GPU derivative, sanitizer, class-workload, combined warm
endpoint, standalone-baseline equivalence, integrated holdout and final production
qualification all passed. The five-pair campaign compares against the accepted
#415 low-angular baseline; final production promotes the cooperative mapping
within the existing automatic admission boundary.

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

## Complete warm endpoint campaign

Slurm 9896 passes all 48 native tests, with the GPU MP2 tests explicitly enabled.
Slurm 9897 then measures five interleaved auto/candidate pairs at each size and
property selection. Every pair starts from the same frozen checkpoint density.
No builds or profilers overlap this clean campaign. Separate component/journal
passes follow the clean samples; their times never enter these medians.

| AO | Baseline energy+force (s) | Candidate energy+force (s) | Improvement | Baseline energy (s) | Candidate energy (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.108953 | 0.109003 | -0.046% | 0.009817 | 0.009833 |
| 192 | 0.164843 | 0.130135 | 21.06% | 0.039649 | 0.039216 |
| 384 | 0.694017 | 0.622889 | 10.25% | 0.269893 | 0.269855 |
| 768 | 2.557756 | 2.000895 | 21.77% | 1.014737 | 1.015030 |

The 768-AO complete endpoint saves **0.556861 s** and passes the predeclared
>=0.20-s gate. The complete shell derivative component also falls from
1192.641 to 661.702 ms (**44.52%**), passing the alternative >=15% component gate
with a clean endpoint win. At 384 it falls from 172.039 to 89.284 ms. The
inclusive force stage falls from 1.631963 to 1.101979 s at 768. These diagnostic
times each describe one separate intrusive replay, not repeated clean medians.

All 80 clean samples and 16 diagnostics pass the unchanged energy/force gates
(1e-9 Eh / 1e-8 Eh/bohr). Maximum observed errors are 1.501e-11 / 1.620e-10.
Iterations, captured-SCF replay counts, explicit final Fock/eigensolve/density
work, final residuals, metric/rank, shell/primitive/component/public-weight work,
response contractions and transfers agree. Consecutive solves have distinct
epochs; each occupied factor is checked against its own final determinant.
Charged host/device memory and sampled device residency agree between arms.

The frozen campaign library is 209,746,608 bytes versus 206,443,312 for the
standalone baseline: +3,303,296 bytes (1.60%). Its native identity is
`de2f49cc07be2ab6f273504dd4ac2718274013873c91262c9fe18d585c039d7b` and its
library SHA-256 is `deefe6b2b95919628387a37c796fbb48a8b6c9609418cab946744e3c89d43d3f`.
The additional compiler/native/generated inventory hash in the build record is
a different, explicitly labeled hash domain.

Automatic admission remains restricted to the existing sm_120, 384/768-AO
equal-orbital/auxiliary domains. The 192-AO improvement above uses the explicit
candidate selector; this change does not expand automatic selection to 192 AO,
unequal auxiliary spaces or other architectures. The 96/192 automatic controls
retain their existing behavior, while the candidate holdouts verify broader
mathematical correctness.

Slurm 9898 verifies the independently frozen old library against the embedded
automatic baseline at 384/768: energy differences are zero, maximum force
differences are 9.60e-14 / 6.76e-14, and charged memory, device residency, semantic
work, graph replays and final validation agree. Its single replays establish
equivalence and are not pooled into the clean timing campaign.

Slurm 9899 passes four integrated RHF/UHF Cartesian/spherical shell route tests
and 12 response replay/discarded-metric tests, including batches, changed geometry
and bounded response storage. Two auxiliary-only-atom tests initially skip
because their suite has a separate opt-in; both skips are retained. Slurm 9900
enables that tier and passes both RHF/UHF tests, including unequal auxiliary
functions, an auxiliary-only center and finite-difference force checks.

Slurm 9901 passes all 48 native tests again on the final production library and
checks one final automatic replay plus a separate diagnostic at all four sizes.
The 384/768 automatic routes match the campaign candidate; 96/192 match the old
automatic route. Energies are identical and all force differences are below
1e-13. Selected classes, class resources, semantic work, final residuals, metric
and charged memory agree. These single qualification replays do not replace the
five-pair timing campaign.

The final library is 209,740,176 bytes, SHA-256
`ea39fae62486240a98021726d970ec80c965e8f4a687056e59636a7179faf5be`, with native
source identity `06fa1f9d7108f86edfaa4ba9285df58c20a1d191b3c20785a2500cad1798730d`.
Only the generated selection header and build identity change from the campaign;
all generated mathematics is byte-identical. The embedded comparison baseline
is removed from the qualified production manifest. All 21 policy host tests pass.

These results establish the stated warm frozen-density improvement. They do not
establish cold/changed-geometry or stock GPU4PySCF performance, and keep #206 open.

## Retention and reproduction

`manifest.json` records byte/hash-verified restoration of all retained records.
The final bundle contains 245 records, including the initial failures and skips.
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
declares the complete-endpoint experiment. `endpoints/endpoints-v1/analysis.json`
(compressed where indicated by the manifest) audits every sample and work gate.
