# Decision: Bound CUDA ECP force extension through d

Status: proposed
Date: 2026-09-22

## Problem

The public CUDA semilocal ECP force owner stopped at s/p even though the
compiler and native ECP provider already validated d orbitals. The stationary
diagnostic also compiled a Cartesian request product eagerly and discarded
multi-component spherical AO records.

## Decision

Admit s/p/d ECP records in the bounded CUDA diagnostic. Expand each packed AO
record into its finite Cartesian component terms, canonicalize first-derivative
requests through the shared schedule, and carry a destination atom plus axis for
each of the four derivative centers in the CUDA reduction map. Keep all-electron
CUDA forces at s/p and retain the existing atom/AO/primitive/byte/work caps.

## Invariants

The shared first-derivative graphs remain the only mathematical lowering. Signed
ordered weights, center/axis permutations, contraction coefficients and source
IDs must be preserved. Admission counts expanded primitive/component records
before compilation and allocation. A failed owner remains transactional and a
fresh snapshot is required for recovery.

## Revisit when

Higher-angular ECP force records or broader all-electron CUDA forces have an
independent source-size, resource, numerical and complete-endpoint
qualification.
