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


## Review repairs and remaining boundary

Canonical derivative bindings map both centers and coordinate axes. The first
CUDA record implementation rotated only the output scatter, unlike the shared
CPU component owner. Pack input coordinates with both binding permutations;
keep the existing inverse scatter. Four nontrivial axis cases failed the host
record check before repair, while an identity-axis control passed.

The 26-double record and 12-int64 map retain 38 eight-byte entries per slot.
The initial 34-entry host charge undercounted the actual arrays; the device
allocation already used the correct map width. Charge all retained entries.
A Cartesian d shell has six components, whereas a real-spherical d shell has
five. Keep both independent oracle/finite-difference matrices, but do not apply
the s/p-only cross-representation equivalence assertion to d.

These local fixes do not establish production acceptance. Mainline has since
moved to the native task/topology interface and packaged all-electron AOT;
porting this proposal must preserve those accepted owners rather than restore
the retired host-record ABI. The requested GPU allocation for a narrow record
probe was unavailable, and no new full CUDA endpoint result is claimed here.

Agent: ChatGPT (Even-PR Review R9 hjhmhw3o)
Model: GPT-6 Astra Pro
