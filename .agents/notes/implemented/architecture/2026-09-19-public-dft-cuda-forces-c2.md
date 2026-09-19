# Public semilocal CUDA force endpoint — #163 C2

Refs #163 C2 and stacked B3 PR #553.

## Capability

The public Python `Calculator` and `PreparedBatch` expose `energy+forces` for CUDA LDA/PBE RKS/UKS by composing the existing native resident KS owner with the compiler-owned `StationaryGradientPlan` and the qualified C1 CUDA seven-source consumer. Public force publication applies `F=-dE/dR` exactly once. No second named-functional gradient equation is introduced.

The native C method registry intentionally remains energy-only. C2 depends on the Python compiler/runtime layer to materialize generated CUDA artifacts, so native C callers and CPU callers must not inherit a capability they cannot execute.

## Lifetime and failure semantics

Each force is generated from an opaque, current native KS snapshot. Changed geometry rebuilds the native item before a fresh snapshot is acquired. Successful neighboring batch items are differentiated independently; a gradient failure changes only that item's status and publishes no force. Snapshot leases are closed after consumption and stale owners cannot renew a force result.

## Resource boundary

The shared KS resource request advertises energy+force observables on CUDA and includes conservative 512 MiB device / 256 MiB host serialized generated-force staging caps. The C1 consumer independently enforces the same caps plus its grid, primitive-record, and pair-visit work bounds. The pre-existing resident KS owner, CUDA context/modules, allocator metadata, and caller outputs retain their documented separate scopes.

## Qualified scientific domain

Initial public promotion is the already-qualified C1 domain: real FP64, all-electron conventional Coulomb LDA/PBE RKS/UKS, native snapshot v3, stable atom-centered grid branch, and s/p AO bases within the bounded diagnostic atom/AO/work limits. No CPU scientific fallback is allowed. ECP, DF, hybrid/meta-GGA/RSH, higher angular momentum, and higher derivatives remain separately gated.

Agent: ChatGPT
Model: GPT-5.6 Sol
