# Decision: keep alternate TensorIR reference namespaces bounded and fail closed

Status: implemented
Date: 2026-09-22

## Problem

Issue #633 needs the same TensorIR mathematical program to act as a validation oracle through an alternate array namespace without turning Python per-node execution into a production backend. Reusing the NumPy interpreter by converting alternate arrays through `numpy.asarray` would silently move device data to the host and erase backend/device evidence. Replacing the established NumPy path wholesale would also risk changing its numerical-order contracts.

## Decision

`tensor.execute(..., namespace=...)` keeps the existing NumPy interpreter unchanged for the default and explicit NumPy cases. A separate bounded namespace interpreter handles portable immutable Array API-style operations and publishes detached outputs on the namespace device. It never imports CuPy, JAX, PyTorch, or another optional backend.

The alternate path fails closed when a TensorIR primitive lacks a declared portable lowering. In particular, indexed scatter/segment operations, range-safe `scaled_bilinear`, and mixed-accumulation execution remain unsupported there. `einsum` is accepted only when the requested namespace exposes a NumPy-like `einsum` extension. These limitations are validation-boundary capabilities, not changes to TensorIR legality or identity.

## Rejected alternatives

- Converting every alternate feed/result through NumPy was rejected because it can hide host transfers and makes device-preservation claims false.
- Rewriting the established NumPy oracle around the generic namespace adapter was rejected because its exact mixed-accumulation and range-safe arithmetic behavior is already an independent numerical contract.
- Adding a mandatory CuPy/JAX/PyTorch dependency was rejected because #633 explicitly keeps external frameworks optional and outside scientific identity.

## Invariants

- The default NumPy reference backend remains `numpy-cpu-interpreter` and keeps its existing numerical behavior.
- Selecting an alternate namespace does not alter `Program.logical_hash` or TensorIR legality.
- Alternate execution does not silently route unsupported primitives through NumPy.
- Inputs with a declared device may not change device while entering the requested namespace; generated constants/scratch and detached outputs stay on the inferred namespace device.
- This interpreter is for validation only; native CPU/CUDA production execution does not dispatch Python per TensorIR node.

## Evidence

`tests/python/test_tensor_namespace_execution.py` compares a captured Array-frontend program against the NumPy oracle, checks identity and detached outputs, verifies explicit fail-closed `einsum` behavior, and (when installed) executes the same program with `array_api_strict` while checking dtype and device preservation.

## Consequences

The B2 experiment can grow one portable primitive at a time without weakening the reference oracle. Namespaces with richer extensions can cover more TensorIR programs, while unsupported semantics remain explicit rather than producing an accidental host fallback.

## Revisit when

Revisit the split only if a standard namespace can express the current NumPy numerical-order contracts (including mixed accumulation and range-safe bilinear arithmetic) and indexed update semantics without host fallback or framework-specific mutation APIs.

## References

- #633
- `python/vibeqc_compiler/tensor/interpreter.py`
- `python/vibeqc_compiler/tensor/namespace_interpreter.py`
- `tests/python/test_tensor_namespace_execution.py`
