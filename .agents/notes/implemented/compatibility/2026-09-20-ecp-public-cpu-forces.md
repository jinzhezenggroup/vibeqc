# Decision: bounded Python CPU ECP force publication

Status: implemented
Date: 2026-09-20

The CPU work gate (#638) completes a prerequisite of #171 but cannot itself
advertise public forces. Public singlepoint and prepared replay now share the
existing compiled stationary consumer with a numeric host-capacity inventory,
SCF resource reservation and transactional snapshot cleanup. CPU capabilities
are promoted only for s/p ECP basis records, including their all-electron
fragments; the native C registry and standalone all-electron CPU capability
remain unchanged.

The production CPU ECP provider is explicitly the independent checked native
two-grid implementation, consistent with the CPU energy owner. Generated
TensorIR owns density weights and nine-source reduction. PySCF is only a test
oracle. Treating the provider as generated code, silently using a reference
selector, or promoting an endpoint based on diagnostic numerical agreement
alone would conceal remaining compiler-retirement and resource obligations.

The extra host reservation is 256 MiB. Metadata admission before snapshot
export includes native/immutable snapshot copies, AO ownership, ECP grid and
derivative buffers, XC tiles, primitive staging and bounded TensorIR/grid
arenas. CPU ECP quadrature capacity is independent of generated CUDA policy.
This is numeric storage capacity, excluding compiler/Python/runtime/allocator
overhead; no whole-process peak or speed claim is made. Full ordered integral
work is retained and bounded by #638. Subsequent larger-domain or performance
promotion must supply independent endpoint evidence.

Validation gates: four LDA/PBE RKS/UKS methods, Cartesian and real-spherical
NaH, independent PySCF analytic gradients (1e-7 Eh/bohr), two-step reconverged
energy finite differences (2e-7 Eh/bohr), translation, serialized input,
equivalent representation, cold/warm/rebuilt mixed batches, exact resource
reservation, work/byte rejection, snapshot closure and successful recovery.
Tests: test_ecp_public_cpu.py, test_stationary_cpu_work.py and existing
test_ecp_stationary_cpu.py. This supersedes the public-endpoint deferral in
2026-09-20-cpu-stationary-work-admission.md without changing that historical
work-only contract. Higher angular domains and generated projector retirement
remain separate #171 acceptance work.
