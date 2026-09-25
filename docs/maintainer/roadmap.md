# Implementation roadmap

VibeQC's long-term goal is broad quantum-chemistry coverage behind one
accelerator-native interface. The generated
[public method table](../public_methods.md) and
[methods guide](../user/methods.md) are authoritative for capabilities that are
actually exposed today. This page records development directions only; it does
not promote a method, backend, derivative, or performance claim.

The roadmap intentionally avoids issue-by-issue history, one-off benchmark
numbers, and kernel optimization diaries. Reproducible measurements belong
under `benchmarks/results/`, operational qualification rules belong in the
Maintainer Guide, and durable engineering rationale belongs in
`.agents/notes/`.

## Production foundation

The shared foundation is already in place:

- registry-driven C/C++/Python method discovery and prepared execution;
- CPU and CUDA execution with explicit unsupported-case failures;
- contracted Gaussian basis handling, direct and density-fitted integral
  infrastructure, scalar Gaussian ECP support, and generated kernel paths;
- RHF/UHF energies and analytic forces, prepared batches, replay, diagnostics,
  and per-item failure isolation;
- compiler-owned IR/code-generation layers plus target-aware runtime/resource
  planning; and
- independent numerical validation and reproducible performance-evidence
  workflows.

These components are infrastructure, not blanket evidence that every method is
qualified on every backend, basis, or property.

## DFT and derivative coverage

Continue from the public LDA/PBE, meta-GGA, global-hybrid, and composite DFT
identities toward complete method-owned execution:

- broaden functional coverage without duplicating XC formulas across backends;
- complete analytic nuclear gradients across the supported DFT families;
- extend the same derivative ownership to Hessians or efficient Hessian-vector
  products;
- keep grid, exact-exchange, range-separated, dispersion, nonlocal-correlation,
  ECP, and response terms explicit in the method contract; and
- qualify CPU/CUDA behavior with matched scientific definitions rather than
  backend-specific approximations.

See the developer documentation for
[XC integration](../developer/xc_integration.md),
[stationary problems](../developer/stationary_problem.md), and
[Hessians](../developer/hessian.md).

## Correlated methods

Build correlated methods on the shared physical-reference, integral-provider,
response, and tensor/compiler boundaries:

- finish the remaining MP2 variants and density-fitted force coverage;
- qualify RCCSD and RCCSD(T) production endpoints across their declared
  CPU/CUDA domains;
- extend analytic gradients and response with bounded memory/resource plans;
  and
- add broader open-shell, frozen-core, ECP, and higher-order variants only
  after their mathematical and failure contracts are explicit.

Current implementation details live in
[MP2](../developer/mp2.md),
[RCCSD](../developer/rccsd.md), and
[RCCSD(T)](../developer/rccsd_t.md).

## Compiler and backend convergence

Reduce the number of independently maintained execution paths:

- represent reusable scientific work in shared IR rather than method-specific
  handwritten kernels;
- generate CPU/CUDA kernels and derivative products from the same declared
  semantics where practical;
- make schedule, precision, residency, and target specialization explicit
  compiler/runtime decisions;
- retain auditable reference implementations without silently routing
  production requests through them; and
- keep complete-endpoint performance as the acceptance metric rather than
  isolated microkernel speed.

See
[compiler architecture](../developer/compiler_architecture.md),
[Program IR](../developer/program_ir.md), and
[performance engineering](performance_engineering.md).

## Broader chemistry

After the existing families have complete scientific and backend contracts,
extend the same interfaces to additional Hartree-Fock variants,
multireference and excited-state methods, periodic systems, embedding,
relativistic Hamiltonians, and other reserved families. Reserved names or IR
slots are not implementation claims.

## Promotion gates

Every roadmap item remains unimplemented or partially supported until the
relevant endpoint satisfies all of the following:

1. mathematical conventions and public behavior are documented;
2. energies and requested derivatives agree with independent references over a
   representative domain;
3. CPU/GPU/backend boundaries and unsupported combinations fail explicitly;
4. memory, lifetime, replay, and batch semantics are bounded and tested;
5. performance claims use complete endpoint timings with reproducible evidence;
   and
6. generated/public capability metadata is updated only after those gates pass.

Detailed acceptance procedures are maintained in
[validation](validation.md),
[performance engineering](performance_engineering.md), and
[evidence retention](evidence_retention.md).
