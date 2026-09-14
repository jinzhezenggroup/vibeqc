# Retained B with bounded contraction panels (#308)

The 768-AO source retains its complete transformed B tensor under both a 4-GiB
and an 8-GiB DF value allowance. Setup generates each raw integral exactly once;
the three subsequent J, dense-K and occupied-K calls generate no raw integrals.
Every D/J/K component passes the independent PySCF comparison at the unchanged
1e-9 absolute threshold. This removes the previous requirement for three
full-tensor K buffers just to retain B.

| DF value allowance | Actual Q | Setup (s) | Repeated J (s) | Repeated dense K (s) | Repeated occupied K (s) | Sampled process GPU peak (MiB) |
| --- | ---: | ---: | --- | --- | --- | ---: |
| 4 GiB | 23 | 103.7683 | 0.005726, 0.005702 | 0.832756, 0.832725 | 0.213481, 0.213512 | 4,458 |
| 8 GiB | 128 | 103.9982 | 0.005681, 0.005672 | 0.829374, 0.829451 | 0.210302, 0.210217 | 5,874 |

These are instrumented fixed-density diagnostics with progress/component
tracing and memory sampling. Setup and the first calls are retained separately
in `summary.json`. They do not establish clean endpoint performance, cold SCF,
full analytic forces or GPU4PySCF parity.

The extended native probe selects storage/tiles using the **actual 9,963,210
source-device bytes** before creating the value plan. The 4-GiB and 8-GiB
source-inclusive diagnostic peaks are respectively 4,250,432,681 and
5,736,789,161 bytes. A DF value allowance is not a whole-process or whole-HF
budget: process peaks include CUDA/provider allocations, and these probes do
not allocate the complete force/SCF live set. The existing #203 qualification
range and guards remain unchanged.

Both layouts materialize 3,623,878,656 raw bytes and retain the same-size B.
The narrower plan uses 34 setup panels, each feeding all transformed Q.
Maximum independent J error is 4.164e-12; dense/occupied K errors are below
9.88e-13, and their mutual difference is below 1.78e-14. No response term,
metric direction or scientific tolerance changed.

Slurm job 9552 passed 78 Python GPU tests (two mocked-CUDA tests skipped) and
the native DF/capture-recovery suites. Coverage includes Q=1/Q=5 independent
96-AO comparisons; tiny Cartesian/spherical s/p/d auxiliary cases forcing
raw-P splitting; batch 1/4; truncated metric response; and energy/force budget
transitions with both one-electron providers. All 129 host resource, policy,
ownership and structure tests passed. Jobs 9553/9554 ran memcheck through the
Python-spawned and direct native retained probes, with zero reported errors;
the direct outputs also pass the complete independent J/K comparison.

The large diagnostic job is 9553. All GPU work and hardware queries used finite
`srun` allocations on `main` with `gpu:5090:1`, preserving assigned visibility.
The measured library SHA-256 is
`7adf03dbcf5a81a18097918db7e7c40903a44487a12611ace0e81380b0118d1a`;
native source identity is
`22698c1caa743d027f76502e28ae5d860e97d503bdbb8da4a65cbc71348018e7`.
The frozen library remains in `.artifacts/issue308-frozen-retained-initial/`.

`evidence.zip` retains the exact uncommitted reconstruction patch against
`13c1547f2e3425b30ada4e2291a220784cd6fbaa`, runner/probe source, build/input/library
identities, full journals, memory samples, test/sanitizer logs and first 24 rows
of all four output matrices. Every full transient array was independently
checked and hashed; the complete arrays remain local. The independent 768-AO
checkpoint is retained in `../issue308-large-diagnostics/reference.zip`.
Every archive member was restored and compared byte for byte; exact hashes are
listed in `summary.json` and the evidence-policy exception.
