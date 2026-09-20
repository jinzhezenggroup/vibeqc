# Decision: COSX explicit-point derivative is a separate partial derivative primitive

Status: implemented
Date: 2026-09-20

## Problem

Complete molecular COSX forces require several responses at once: AO/basis-center
motion, quadrature-point motion, quadrature-weight response, any fitting or
symmetrization response, and the orbital/Pulay terms required by the stationary
energy expression. Implementing one contribution behind the molecular force API
would incorrectly present a partial derivative as a complete force.

The explicit quadrature-point contribution is nevertheless independently useful
and can be validated without coupling to the KS or MethodIR force path.

## Decision

Expose a narrowly named explicit-point derivative primitive. It differentiates
the discrete COSX exchange energy with respect to each supplied quadrature point
coordinate while holding the AO density, quadrature weights, Gaussian centers,
and basis data fixed.

The CPU oracle combines order-one AO spatial jets with analytic ESP probe-center
derivatives from the existing Jet Coulomb recurrence, then contracts each
point-axis response directly into dE/dR_point.

The CUDA implementation lives in a separate translation unit,
`src/dft/cuda_cosx_derivative.cu`. It uses order-one AO grid jets and the
generated one-electron attraction derivative DAG. The ordinary energy-only
`cuda_cosx.cu` remains unchanged and does not include the large derivative
header.

For the signed unit-charge attraction V used by generated CUDA kernels, positive
ESP is -V. Translational invariance gives

`d(-V)/dC = dV/dA + dV/dB`

for a moving ESP probe C with Gaussian centers A and B fixed. This relation
supplies the CUDA ESP probe derivative without introducing another recurrence.

## Rejected alternatives

- Wiring the partial derivative into molecular force dispatch was rejected
  because it would omit basis-center, weight, orbital/Pulay, and other required
  terms.
- Including the generated derivative header in the existing energy-only COSX
  translation unit was rejected because it materially increases compilation
  work for users that only build/use COSX energies.
- Materializing a global point-by-axis-by-AO-squared COSX derivative tensor was
  rejected in favor of bounded point tiles and early scalar contraction.

## Invariants

- The primitive is an explicit-point partial derivative, never a complete
  nuclear gradient.
- Density, quadrature weights, Gaussian centers, and basis data are fixed.
- Point-gradient storage is point-major xyz.
- CUDA work remains bounded by the requested point tile.
- Energy-only COSX compilation and runtime behavior remain unchanged.
- Full COSX force dispatch stays unsupported until every required response term
  is assembled and validated.

## Evidence

- CPU ESP probe derivatives are checked against central finite differences.
- CPU discrete COSX point derivatives are checked against central finite
  differences of the same fixed-weight point model.
- CUDA derivatives are compared against the independent CPU analytic oracle
  across multiple tile sizes and spherical s/d/f AO expansion.

## Consequences

This lands a reusable, testable force building block without claiming full
analytic COSX forces. The CUDA derivative translation unit has a non-trivial
compile cost because it consumes the generated one-electron derivative DAG, but
that cost is isolated from the existing energy-only translation unit.

## Revisit when

Revisit this boundary when basis-center AO response, quadrature-weight response,
and the stationary/orbital terms are ready to be assembled into a complete COSX
molecular gradient.

## References

- Issue #246
- `src/dft/cosx_reference.cpp`
- `src/dft/cuda_cosx_derivative.cu`
- `tests/native/test_cosx_reference.cpp`
- `tests/native/test_cosx_cuda.cu`
