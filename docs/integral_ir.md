# Integral intent and bounded shell blocks

`IntegralIR` can describe overlap, kinetic, nuclear attraction, Coulomb metric,
three-center Coulomb, and four-center ERI values and first derivatives.
`RawBlock` and `WeightedDerivative` consumers do not require an RHF/UHF density.
This is a semantic and host reference interface; it does **not** add production
kernels for these consumers or for the new operator families.

| Operator | Basis slots | External positions | Permutable basis slots |
| --- | --- | --- | --- |
| overlap(A,B) | orbital A, B | none | A/B |
| kinetic(A,B) | orbital A, B | none | A/B |
| attraction(A,B;C) | orbital A, B | nucleus C with charge Z | A/B |
| metric(P,Q) | auxiliary P, Q | none | P/Q |
| three-center(A,B;P) | orbital A, B; auxiliary P | none | A/B |
| ERI(A,B,C,D) | orbital A, B, C, D | none | within each pair and pair exchange |

Permutations are explicit declarations, expressed as complete basis-slot
permutations. They do not silently reorder tensors or imply an eightfold weight
prefactor. The three-center auxiliary slot cannot be exchanged with an orbital
slot. An attraction nucleus is a `NuclearCenter`, never a zero-angular-momentum
Gaussian shell.

## Centers and derivatives

`BasisShell(slot, center, angular, role, convention)` describes a basis slot.
`ShellSignature.center_bindings` maps **every** mathematical center, including
an attraction nucleus, to a physical atom. Basis slots and center IDs are
distinct concepts. Center IDs may be nonconsecutive; bindings follow the order
in `OperatorSpec.centers`. Multiple bindings may point to the same atom.

Two shells on the same atom retain distinct position variables. Differentiate
these variables first, recover any dependent derivative, and only then apply
the chain rule:

```text
dI/dR_atom = sum(dI/dR_center for centers bound to atom)
```

All six operators admit simultaneous translation of their complete center
inventory. Attraction requires `dA + dB + dC = 0`. An invariant over only A and
B is rejected. At most one center is recovered from this relation, preventing
circular or duplicate recovery. A partial derivative selection that omits any
invariant center performs no recovery. Data providers supply exactly the
independent centers; supplying the recovered center again is an error.

`DerivativeSpec(order=2, ...)` can preserve legacy higher-order intent, but it
does not describe an executable Hessian. CUDA scheduling, block execution, and
translation recovery explicitly reject orders other than one. The current raw
layout is a value/first-derivative contract; no higher-order layout is defined.

## Tensor and numerical conventions

Coordinates are in Bohr and operators use atomic units. Overlap is
dimensionless, kinetic is `-1/2 nabla^2`, attraction is `-Z/|r-R_C|`, and all
Coulomb operators use `1/r12`. AO coefficients/exponents and normalization are
supplied by the caller's existing basis context; this compile-time signature
does not serialize a molecule. Normalized Cartesian components use VibeQC's
CCA order (descending x, then y exponents), and real spherical components use
the existing PySCF/libcint convention. Mixed conventions in one signature are
rejected. Declaring a uniform spherical signature does not enable CUDA
spherical-block lowering.

Tensor axes follow basis-slot order, named `shell_0`, `shell_1`, etc. Four-center
blocks use chemists' `(AB|CD)` order, three-center blocks `(AB|P)`, and metrics
`(P|Q)`. There is no spin index or implicit RHF density factor. Derivative raw
blocks prepend `(center, xyz)`, with xyz ordered x/y/z and centers in requested
order. Input reference center buffers use dense `(xyz, shell_0, ...)` order.

`TensorLayout` declares FP64 shape and strides **in elements**. Dense axis
permutations and padding are accepted; overlapping, broadcast, negative, and
interleaved strides are rejected. Shapes, strides, products, and total byte
counts are checked against signed 64-bit indexing before allocation.

## Bounded requests and external weights

The consumer layout describes one shell tile. Its basis extents may be smaller
than the full shell dimensions. `BlockRequest` adds an ID, `ShellTile` offsets
and extents, a consumer index, and complete atom bindings. Optional
`shell_indices` identify concrete shells in the caller's orbital/auxiliary
basis spaces; omission is useful for synthetic reference requests. A weight
callback receives this complete request and its source descriptor. The source
key resolves through caller-owned state; serialized IR never contains a Python
callback or a global provider registry.

`assemble_raw_block(request, data)` packs supplied values or independent
derivatives. `contract_weighted_derivative(request, derivatives, provider)`
contracts a `WeightTile` supplied by `provider(descriptor, request)`. Both are
host reference helpers, not integral evaluators. The contraction is exactly

```text
output_sign * weight.sign * weight.prefactor * sum(W_tile * dI_tile)
```

Weights may be arbitrary, nonfactorizable tensors. Their axes and physical
strides must match the descriptor. Padding does not contribute to the sum.
Neither HF exchange factors nor permutation multiplicities are inferred.
Forces conventionally set `output_sign=-1`; the output name alone never changes
the sign. Nuclear derivative outputs have `(center, xyz)` rows in requested
order. Atomic outputs have `(atom, xyz)` rows for sorted distinct physical atom
indices. These are additive contributions from this tile and the selected
parameters; the caller accumulates tiles into its molecular result.

The explicit memory budget covers FP64 input, reconstructed center workspace,
weight-buffer span, and output-buffer span. It excludes Python object overhead
and provider-owned caches. Requests exceeding the budget fail before calling a
provider or allocating output. No helper allocates molecular `N^4` or
`3N x N^4` arrays. `BlockResponse` carries the request ID, status, tile,
layout, derivative-center order, physical row maps, and data. An unsupported
response has a reason and no data; it cannot masquerade as a successful zero.

## Backend and compatibility boundaries

`query_integral_capability()` queries eligibility at the CUDA semantic input
boundary. Its success does not establish emission for a particular schedule,
compilation, numerical correctness, endpoint validation, or production
selection. Use the existing [shell capability report](codegen_capabilities.json)
and architecture manifest for those later stages. The new operators and
raw/weighted consumers currently report unavailable CUDA lowering. Explicit
shell signatures also require a future runtime task adapter before lowering.

Existing `build_integral_ir(ShellClassSpec, consumers=("fock", "force"))` calls
retain their behavior. `integral.signature` exposes a compatibility adapter;
`ShellSignature.from_shell_class()` / `to_shell_class()` preserve catalog names
and angular orders without renumbering the 55 canonical classes. Legacy atom
bindings remain unresolved until populated from `task.atom[center]`; a quartet
slot must never be guessed to be its physical atom index.

The new serialization schema is `vibeqc.integral_ir`, version 1. Decoding rejects
unknown versions, fields, and scalar layouts. `integral_cache_key()` uses the
existing SHA-256 content-addressing approach over the versioned payload,
including external charges, bindings, layouts, weight descriptors, and signs.
This is an intent hash, not evidence of a compiled kernel. The existing NVRTC
cache, production manifest schema 2, and CUDA generator ABI 1 remain unchanged:
no generated symbol, runtime layout, registry, or source changes are necessary.
Existing profile loaders still reject incompatible generator ABIs.

The [compatibility fixture](../tests/reference_data/integral_ir_legacy_artifacts.json)
records baseline revision `2414c57`, the manifest hash, catalog ordering, and
SHA-256 hashes of all ten generated artifacts for `sm_120` and portable `sm_90`.
The test regenerates these bundles using the existing four-shard mechanism and
requires byte identity. There is no generated-source diff. Future intentional
changes to the manifest or generator must update this fixture with reviewed
source differences and the appropriate ABI/version decision.

## Reproducible examples

[integral_ir_examples.json](integral_ir_examples.json) contains a two-center
overlap value, a three-center derivative, and a four-center external-weight
request. Every example includes its unavailable CUDA status. Regenerate with:

```bash
python tools/generate_integral_ir_examples.py --output docs/integral_ir_examples.json
python -m pytest tests/python/test_integral_contracts.py -q
```

These contracts provide independent synthetic acceptance tests while the shared
numerical evidence protocol in #138 is developed. They make no numerical or
performance claim for new operator kernels.
