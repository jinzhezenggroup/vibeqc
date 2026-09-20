# Decision: fixed-state GFN2 electronic TensorIR

Status: implemented, compiler slice; no SCC execution promotion
Date: 2026-09-20

## Equations and ownership

The pinned xTBloom/tblite convention is ket-AO-origin multipoles. For each
canonical upper AO pair, the forward dipole/quadrupole integral multiplies the
column atom potential and its reverse integral multiplies the row atom
potential, with the recorded minus-one-half coefficients. Scalar shell shifts
use minus one half of overlap times the two shell potentials. H0 remains an
explicit prepared input. Canonical pair corrections are reflected consistently
without silently symmetrizing the caller's H0 data.

Flattened matrix blocks have explicit per-system offsets; atom, shell and AO
maps are immutable and cannot cross systems. Orbitals of one shell must share
one atomic owner. The matrix/component index kinds describe actual semantic
axes and do not replace the existing history, orbital or pair spaces.

Restricted and unrestricted graphs have different identities. Unrestricted
channels use charge plus/minus magnetization potentials. TensorIR owns the
S/D/Q reverse programs; topology and fixed potentials are not differentiated.
The result is fixed-state Hamiltonian algebra, not an SCC solve, population
analysis, relaxed total-energy derivative or complete nuclear force.

## Evidence and limits

`test_gfn2_electronic_ir.py` retains the independent ragged two-system values
from xTBloom `tests/cuda_hamiltonian_test.cu` at revision
`2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`, including directed nonsymmetric
inputs. It checks spin-channel semantics, reference/generated adjoints and
CUDA lowering. `test_gfn2_electronic_contracts.py` independently compares every
S/D/Q output-channel cotangent against two-step recomputed-primal differences
and rejects a within-system split-atom shell. These derivative gates do not
replace the pinned external primal oracle.

No real-device throughput or full GFN2 runtime promotion follows from source
lowering. Matrix topology and tensor intermediates remain bounded by explicit
sizes but can require quadratic storage/work; static ragged maps are not a
claim of scalable dynamic SCC execution. The #504 geometry compiler remains a
separate owner and keeps its own reference and stale-topology checks.

## Rejected alternatives and revisit criteria

Do not copy a second hand-derived gradient or infer multipole orientation from
symmetric test data. Do not conflate shell and atom ownership merely because
both indices lie in the same system. Mutable SCC history, occupations, mixing,
DIIS and eigensolvers stay outside the compiler graph. Wider spin layouts,
dynamic maps, population/energy bookkeeping and production #560 integration
require new independent equation, resource, lifetime and endpoint evidence.

References: #505, #624; method/gfn2_electronic.py; the two tests named above.

Agent: ChatGPT
Model: GPT-6 Astra Pro
