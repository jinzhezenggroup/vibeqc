# Decision: qualify B3LYP through the shared private interior CPU SCF path

Status: implemented
Date: 2026-09-21

## Problem and decision

A registered method name or audited fixed-density expression does not establish
production-grid, prepared-state or force capability. B3LYP now obtains its
semilocal scalar and first derivatives from the existing canonical MethodIR
(including VWN-RPA rather than silently substituting VWN5), and obtains the exact
exchange fraction from that same definition. Native RKS/UKS reuse the existing
CPU J/K providers and physical-stationarity SCF state machine. No second SCF,
exchange provider or handwritten B3LYP expression is introduced.

The B3-family point-layout and tile integration code is shared with CAM-B3LYP.
CAM-specific attenuation admission remains in its own evaluator. Both preserve
the strict interior-v1 density, spin-gradient and reduced-gradient bounds; this
slice does not invent vacuum, empty-spin, asymptotic or production-tail behavior.
The private point bridge reserves selector 3 for B3LYP, retaining selectors 0/1/2
and their compatibility wrappers. GGA publishes zero kinetic coefficients.

## Rejected alternatives and invariants

Do not infer a public B3LYP method from private SCF convergence. Public registration,
prepared snapshots and complete analytic forces remain separate qualification
steps. Force requests still reject in the shared driver. A mismatched full-range
exchange strategy must reject before SCF; restricted occupation-two and same-spin
unrestricted exchange conventions remain the existing Fock contracts.
Do not relax the domain to make a production grid pass or copy CAM attenuation
into the global hybrid. Scientific identities remain generated from MethodIR.

## Evidence and revisit conditions

Native tests pin independent Libxc scalar/feature values, verify the density
variation of the integrated potential, and exercise RKS/UKS agreement, warm replay,
wrong-exchange rejection and force rejection. Python tests cover the private
Cartesian point adapter and independently displaced collocation geometry.
These are not a replacement for independently converged production-grid molecular
endpoints. Revisit public admission only after the tail policy, live snapshot
ownership, complete stationary force assembly and backend-specific acceptance
are separately implemented and tested. No GPU or performance improvement is
claimed by this CPU/private compatibility decision.

References: #165; #749; `src/dft/xc.cpp`; `src/dft/rks.cpp`;
`src/dft/uks.cpp`; `tools/generate_xc_cpu.py`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
