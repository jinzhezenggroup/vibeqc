# MethodIR execution type checking

Date: 2026-09-19

## Decision

Add a small static type-checking boundary between MethodIR and backend lowering
instead of embedding backend policy in MethodIR or adding a second tensor algebra.

The mathematical MethodIR identity remains unchanged. A separately hashed
TypedMethodIR records the selected backend, dtype, derivative order and inferred
logical feature types.

## First slice

The checker validates:

- float32/float64 execution dtype;
- polarized/unpolarized spin support;
- derivative order 0/1/2 against an explicit backend capability;
- required MethodIR ingredients and operators;
- logical per-point feature shapes:
  - unpolarized rho/sigma/tau -> one component;
  - polarized rho/tau -> two components;
  - polarized sigma -> three components;
- feature dtype/spin agreement, with implicit precision casts rejected.

StationaryGradientPlan now calls the checker before generating TensorIR. Its
current capability is deliberately narrow: FP64, semilocal rho/sigma, first
derivatives.

## Boundaries

BackendCapability is a finite lowering declaration, not proof that a public
runtime endpoint exists or is numerically qualified. Primitive-specific
scientific derivative rules remain independently checked by their owners.

Feature shapes are logical component shapes at one point; runtime batching and
memory layout remain owned by lower layers.

## Rejected alternatives

- Adding dtype/backend fields directly to MethodIR would fragment the method's
  scientific identity when only the execution schedule changed.
- Inferring backend support from primitive representation would incorrectly
  promote representability into executable capability.
- Reusing TensorSpec directly at MethodIR level would introduce orbital/index
  layout concerns into a layer that only needs logical XC feature contracts.
- Allowing implicit mixed-dtype features would make precision changes invisible;
  future mixed precision should use explicit cast/schedule semantics.
