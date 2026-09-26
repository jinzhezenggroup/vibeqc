# Decision: qualify complete CUDA WB97M-V stationary composition

Status: implemented (numerically accepted; scaling performance unqualified)
Date: 2026-09-26

## Problem

Master `872b52f5de002787d68a3d5c724fd1224572ce36` has the complete CUDA
SCF energy and separate stationary primitives. Those do not prove complete
public forces or README-style end-to-end performance. The public Python
consumer must assemble the exact same RSH/VV10 model under a live SCF token.

## Decision

Use a retained native integral source for hcore, overlap/Pulay, J, short-range
K and long-range K. Carry nuclear coordinate dual seeds through the shared
bounded range-moment recurrence, using `dM_n(T)/dT = -M_(n+1)(T)` at fixed
exponents and omega. Keep existing full-range value arithmetic unchanged.
Several CUDA blocks reduce each coordinate to avoid one-block-per-coordinate
underutilization; this changes reduction order and therefore needs FP64 gates.

Reuse compiler-owned semilocal geometry pullbacks for VV10 total-rho/sigma,
explicit pair-coordinate and partition-weight adjoints. Pack both VV10 domains
with the SCF molecular cutoff. The same AO/Becke generated operations own both
local and nonlocal motion. A twelve-source TensorIR reduction owns the sum;
the public result negates it to obtain forces.

Retain geometry-bound owners across warm replay; rebuild when their identity
changes. Preserve token checks and transactional output publication. Explicit
numeric bounds include the native derivative source and transient one-electron
bridge. Report the repeated final-state export separately from initial snapshot
work. CUDA module/driver stacks and compilation are outside numeric capacity.

## Rejected alternatives

- Promoting energy or isolated SR/LR/VV10 derivative tests as full force proof.
- Hiding CPU/PySCF derivatives or finite differences behind a CUDA result.
- Extending the diagnostic Python AO-quartet loop to README-scale systems.
- Calling finite memory a performance qualification: coordinate/quartet visits
  and VV10 pair counts are explicit, and large points can time out.
- Reporting GPU4PySCF force timings without moving-grid response or substituting
  its ordinary grid for the native quadrature when checking strict errors.

## Evidence and performance boundary

Host lowering tests and compiler structure check pass. Four complete public
CUDA acceptance tests pass on a Slurm-allocated RTX 5090. Tests cover RKS/UKS,
spherical d functions, reconverged finite differences, native
SR/LR fixed-density derivatives, geometry rebuild and failed-neighbor recovery.
The paired benchmark enforces the HF README's energy/force gates and retains
timeouts. Default-grid reference-only water timings and all attempt outcomes
are retained in `.artifacts/readme-wb97mv-20260926` during development.

The complete CUDA RKS H2, UKS H3, and spherical water/def2-SVP cases pass
independent GPU4PySCF energy/force and reconverged finite-difference gates.
Geometry rebuild, per-item failure isolation and stale-token recovery also pass.
The first public attempt exposed a CPU-only guard in the existing WB97M-V native model
proof; the proof now accepts the same token-bound complete model on CUDA.
Three CPU live-state/force regressions and 62 compiler regressions pass.

The VV10 resident provider already retains every point input/output and local
scale, but previously serialized logical row tiles into separate one/two-block
kernels. Flatten the logical tile and block indices into one CUDA launch, with
a finite launch-grid fallback. Each row retains its exact ordered j loop and
the ordered final energy reduction. Numeric allocation and logical tile/work
counts do not change. Independent CPU/CUDA value, feature and geometry checks
pass (eight tests). The completed default-grid water
endpoint passes the independent energy/force gates; its warm median is
19.216 s versus GPU4PySCF 2.310 s. These results do not qualify README-scale
performance. Performance analysis and optimization are deferred; retain the
attempts as acceptance evidence without adding a headline README chart.

The generated nuclear-only module uses raw primitive kind zero. The generic
stationary integral helper adds spherical component tags for multi-term d AOs;
those tags must not be used for a nuclear primitive with no AO operands. The
water test protects this boundary, which H2/H3 alone cannot exercise.

## Revisit when

Source-driven shell derivative schedules eliminate repeated coordinate scans;
native retained AO/grid leases avoid feature downloads and duplicate collocation;
or a shared complete C-native stationary consumer replaces Python composition.
The native public C registry stays energy-only in this implementation.
