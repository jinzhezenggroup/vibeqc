# D4 production runtime ownership and generated derivative composition

Issue #493 promotes the already-qualified D4/EEQ mathematics from #551 and
#577 without making the qualification implementation the production owner.

## Ownership cut

The EEQ2019 evaluator is the qualified implicit charge/response custom
primitive. The fixed-charge D4 evaluator is the qualified energy and partial
derivative primitive. vibeqc_compiler.method.d4_derivative owns the complete
production derivative composition and CMake generates generated_d4_derivative.hpp.
Production calls that generated entry point and never the older complete D4
composition. The older complete evaluator remains only a qualification oracle.

D4Plan owns runtime state, resource admission, replay identity, errors and
transactional publication. The generated derivative identity, EEQ table digest,
EEQ parameter digest, native provider identity and scheduler identity are
exposed through the C ABI. Python requires native/compiler identities to agree
before execution and hashes them into its cache/replay identity.

## Scheduling and resources

CPU reuses one worst-system workspace. CUDA separates the two scientific
primitives instead of owning a second fixed-charge implementation. EEQ uses at
most 32 bounded worker blocks; each block atomically claims another ragged
member and reuses one worst-system solve workspace plus one dq/dR scratch.
Fixed-charge pair/ATM work is then delegated to the block-cooperative scheduler
owned by #735, whose workspace is linear in total atoms. Dense EEQ response
storage is therefore bounded by worker slots times the largest molecule rather
than every fleet member. maximum_bytes admission reduces the EEQ worker count
down to one before rejecting a plan. Prepared coordinates are uploaded once;
unchanged replay reuses them and changed geometry uploads the packed coordinate
buffer. The final chain-rule composition is compiler-generated.

Diagnostics retain plan/execution host bytes, device/table/workspace bytes,
worker/workspace slots, execution count, changed/unchanged replays, coordinate
H2D bytes, launch count, profile and ATM capability.

## Public capability boundary

PBE-D4(BJ-EEQ-ATM) is the canonical MethodIR composition. The native public
method pbe-d4-rks runs PBE RKS plus the production D4 correction and advertises
energy and batch execution only. The D4 correction itself has an analytic
gradient, but the complete PBE-D4 method does not advertise forces until the
PBE stationary-gradient work owned by #163 is production-ready.

Standard DFT-D4 uses EEQ2019 with ga=3,gc=2. r2SCAN-3c retains its explicit
ga=2,gc=1 override. GFN2 reference semantics remain qualification compatibility
data and are never a generic DFT-D4 default. No xTBloom or external dftd4
runtime dependency is introduced.

## Closure gates

test_d4_production.cpp compares production generated execution to independent
DFT-D4 4.2.0 fixtures and finite differences. test_d4_ragged.cpp exercises
bounded workspace, a 4100-member fleet, peer-local failure and replay counters.
test_dft_api.cpp checks the named endpoint and confirms force capability fails
closed.

Real CUDA promotion uses benchmarks/run_issue493_d4.slurm: Release sm_120 in a
finite Slurm RTX 5090 allocation, qualification/production/native API tests,
Compute Sanitizer and small/medium/ragged cold/replay/changed performance.
Raw evidence belongs under .artifacts; only reviewed compact evidence should be
promoted to benchmarks/results.
