# Partial second integral derivatives

Second-order integral requests use an explicit `SecondDerivative` consumer
and IntegralIR schema version 3. Existing first-force requests, their schema
versions, result layout and registry categories remain unchanged. A represented
order-two request is separate from an executable backend capability.

The output choices are a raw Hessian block, a fixed-weight Hessian
`sum(W * d2I/dR2)`, and a fixed-weight Hessian-vector product. The derivative
holds external W, the direction, primitive exponents, contraction coefficients,
nuclear charges and omega fixed. Electronic response and full molecular
Hessian assembly are outside this primitive interface. Range, DF and ECP
second derivatives are not executable in the initial lowering.

`build_one_electron_second_ir` declares S/T/V shell pairs;
`build_eri_second_ir` declares four-center Coulomb requests. Public angular
momenta run through f. `build_second_derivative_kernel` selects one Cartesian
AO component for a raw output, or one to 64 components for a weighted output.
It contracts weights before differentiation. Explicit `output_indices` select
coordinate outputs before generation and reject an impossible output budget.
Unrequested recurrence states and derivative outputs can be pruned by scalar
lowering. These DAGs are generated once; there is no runtime coordinate AD tape.

## Coordinate layout and invariants

Dense coordinates follow requested mathematical-center order, with xyz inside
each center. Hessian axes are `(center_row, xyz_row, center_column, xyz_column)`;
raw outputs append the shell AO axes. HVP outputs have `(center, xyz)` axes.
Svec traverses the upper coordinate triangle by row and multiplies every
off-diagonal entry by sqrt(2). Its ordinary packed dot product therefore
equals the full Frobenius inner product. Decoding divides by those same
factors. An asymmetric matrix cannot silently become a packed symmetric one.

Let R contain independent-center identity rows and the dependent-center
negative-sum row. The full Hessian is `(R kron I3) H_ind (R kron I3).T`.
Both derivative indices require recovery. An HVP first projects its direction
with `R.T`, differentiates the resulting directional scalar, then recovers
the output with R. It does not construct a dense Hessian in the DAG.
Partial center requests use translation only when every required center is
present. The chosen dependent center can be any declared operator center.

For multiple mathematical centers on the same atom, expand the atom direction
to all corresponding center slots before HVP evaluation, then sum those slots
when publishing the atom response. This applies the chain rule `A.T H A`.
Coincident coordinates alone never identify two mathematical centers.

`SecondAtomMap` expands directions and scatters HVPs using sorted distinct
physical atom labels; it also provides the small diagnostic two-index Hessian
scatter. Its mapping is explicit and cannot be inferred from coordinates.

## Bounded native execution

`compile_second_derivative` accepts an explicit `CppCompilerAdapter` or
`CudaCompilerAdapter` and uses the shared hash-verified native compiler cache.
It compiles one to twelve coordinate outputs and at most 64 selected Cartesian
components. `second_coordinate_tiles` partitions a dense/svec Hessian or HVP
into explicit coordinate selections, with six outputs per tile by default.
Geometry may be recomputed between tiles to bound scalar liveness. Native
raw output requires one selected AO component per compiled helper.

`PreparedSecondDerivative` retains the shared weighted-integral runtime owner
with a coordinate-output policy. It consumes a single-pass iterable of
`SecondPrimitive` records, each containing mathematical-center geometry,
packed fixed weights, a fixed normalization scale and an optional direction.
Its tagged record has the established 208-byte primitive prefix, one double
per selected component and twelve direction doubles. Stride and tag checks
prevent the new suffix from being interpreted as a first-gradient record.
Second-order ERI geometry owns fifteen Boys moments. The legacy fourteen-moment
API still rejects order fourteen without modifying its output buffer.

The shared resource planner accounts for record staging, native publication
storage, compact chunk results, detached results and the CUDA arena. A retained
owner rejects impossible capacities before allocating or loading a device.
Partial and empty chunks are supported. Invalid inputs, nonfinite arithmetic
and late stream failures do not publish partial results; the owner can be
replayed afterward. Call stacks, compiler/Python metadata, caller-owned inputs
and CUDA context overhead are named scope exclusions. Kernel register, stack
and spill reports remain visible as compiler-resource diagnostics.

`prepare_second_shell_stream` freezes contracted shell inputs using the same
public-weight pullback and angular normalization as first derivatives. Input
primitive coefficients already include the native radial normalization.
Weighted requests accept full shell `WeightTile` inputs, including padded
strides. Real-spherical inputs require an explicit public `ShellSignature` and
Cartesian-to-public projection matrices. Every nonzero pulled-back component
must belong to the compiled subset; incomplete coverage fails explicitly.
A weighted Hessian with a unit public cotangent evaluates a raw public
component, including spherical components. The adapter's numeric peak is
composed with the prepared owner's budget before execution.

`query_integral_capability` exposes the separate `cpu_second_derivatives` and
`cuda_second_derivatives` backends. Eligibility requires explicit coordinate
tiling where the full output exceeds twelve coordinates. It does not change
the existing direct-HF/first-gradient reports. Execution diagnostics separately
record source lowering, compilation, successful execution, independent
numerical-gate status and the unavailable method endpoint. A successful call
does not itself run or claim an independent numerical gate.

## Scientific validation

The optional `tools/vibeqc_validation/second_derivatives.py` uses Libcint's
analytic second-derivative blocks and independent moment quadrature for small
fixtures. Its imports are outside the installed compiler/runtime dependency
path. The tests cover asymmetric p/d and f cases, independent ERI diagonal,
same-pair and cross-pair blocks, translation on both indices, same-center
limits, packed normalization, arbitrary signed weights, raw/HVP agreement,
output tiling and optimize-before/after-differentiation schedules. Contracted
Cartesian/spherical HVPs use independent first-gradient finite differences.

Second derivatives use a separate scale-aware gate: with
`s = max(1, max(abs(H)))`, analytic raw blocks have absolute tolerance
`5e-11` and relative tolerance `2e-11`; translation and symmetry residuals
must not exceed `5e-12*s`. Central differences of first gradients at
`1e-3`, `3e-4` and `1e-4` Bohr must reach `2e-7*s`, and the final error must
be below 3% of the first error or the `5e-11*s` roundoff floor. Agreement
between two generated paths supplements these independent checks.

Native/compiler, numerical-evidence and method-endpoint acceptance are separate
stages. Passing these scalar gates alone does not promote a production
schedule or advertise a molecular Hessian.

`tools/validate_second_derivatives.py` reproduces compact raw/weighted/HVP
evidence for nine asymmetric/coincident S/T/V and ERI cases, including a
minimum coordinate/component f/f/f/f tile. It records three signed primitive
contributions, independent analytic blocks, three-step Libcint first-gradient
differences, cold/warm timing samples and actual compiler/resource identities.
It refuses to publish dirty source or an incomplete selected case inventory.
The timing scope is the prepared primitive provider, with external-weight
preparation and independent reference construction outside the samples.

`tools/validate_second_ownership.py --backend cpu --output <new-directory>`
builds ASan/UBSan lifecycle executables. The CUDA mode uses compute-sanitizer
memcheck, including leak checks; execute it under a finite Slurm allocation.
It exercises attraction HVPs, four-center HVPs and the f/f/f/f raw moment bound,
with invalid/empty chunks and successful replay after failure.
