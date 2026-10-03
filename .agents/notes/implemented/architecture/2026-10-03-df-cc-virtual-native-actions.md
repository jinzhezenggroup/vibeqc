# Decision: derive native DF virtual response from the audited CC inventory

Status: implemented
Date: 2026-10-03

## Problem

The qualified conventional force endpoint stops at 56 AOs. At hundreds of AOs,
retaining full virtual integrals and differentiating dense four-index parameters
is a structural obstacle. For the independent 230-AO ethane fixture, `vvvv`
alone is 19,083,546,248 bytes; for 264-AO benzene it is 27,894,275,208 bytes.
Those counts exclude every amplitude, response, solver and device-staging owner.

The existing correlation-only DF validation implementation omits `ovvv/vvvv`,
but its virtual corrections are hand-transcribed NumPy operations. They are
not the source of the native CC code or its generated amplitude/factor
derivatives. Simply enabling DF on the conventional native solver would still
request the dense virtual blocks and would not produce correct DF forces.

## Decision

Select the virtual-integral contributions from the existing fully expanded
RCCSD inventory, substitute `g[pqrs] = sum_Q B[Q,pq] B[Q,rs]`, and represent one
auxiliary slice at a time. Enforce Hamiltonian linearity at generation time;
multiple integral operands would require independent auxiliary sums.

Use the existing binary-contraction planner with an optional intermediate-axis
limit. Its subset search skips a forbidden intermediate but continues searching
other trees; if no lower-degree allowed tree exists, it retains the original
direct contraction. Existing unconstrained consumers keep their schedules.
For the DF virtual program, at most two virtual axes are allowed. This is a
representation rule, not a claim about the total live allocation peak.

Use existing dense-symmetry AD for amplitude JVP/VJP and factor VJP, then share
the conventional runtime-shape CPU/CUDA emitter. The native interface exposes
scratch queries, equation identities and operation counts. Its outputs borrow
the caller's arena. A future execution owner must accumulate every Q slice,
admit all retained inputs/outputs/staging, and bind the complete physical state.

## Evidence

Small independent dense four-index residuals and the older separately written
NumPy factor contractions agree with the generated primal. Auxiliary rotations
leave the summed result invariant. Amplitude dot products, projected B_vv/T2
cotangents, and three central-difference steps of the dense functional validate
the reverse actions. Shape checks include occupied-rich cases and the dimensions
of the targeted hundreds-AO systems.

Native CPU and real Slurm CUDA actions agree with the independently executed
TensorIR for four runtime shapes and all four actions. Exact CPU scratch arenas
retain canaries; one-element-short arenas reject before numerical work. Full,
unfiltered CUDA memcheck covers each action with zero errors. All five existing
conventional CC/triples/Fock generated files remain byte-identical.

The [retained qualification](../../../../benchmarks/results/df-cc-native-actions-20261003/publication.json)
contains all 32 native action comparisons without rounding, with maximum
absolute error 8.673617379884035e-19 under the fixed 3e-12 absolute / 1e-12
relative gate. It also retains exact generated-source/binary identities,
unfiltered sanitizer output, source reconstruction, and the complete original
large-reference records. Slurm job 12117 used the local RTX 5090; the independent
230/264-AO references used n2 CPU jobs 2152/2151 and both passed residual replay.

The 230-AO and 264-AO independent oracle uses PySCF 2.14.0, conventional RHF,
explicit symmetric metric whitening at relative cutoff 1e-10, and the same B
factors injected into PySCF's DF CCSD/(T) owner. Physical CC replay residuals
must be <=1e-9. This establishes reference targets, not a production endpoint.
The script preserves ideal molecular symmetry instead of perturbing fixtures
to avoid same-space degeneracies.

## Limits and next work

Each virtual action has at most fifth-degree work for one Q and fourth-degree
storage. The complete Q sum still adds auxiliary work: this is not a claim of
lowering the formal CCSD N^6 or standard-(T) N^7 scaling. Whole-call timing must
include auxiliary loops and all other phases. Invariant-hoisting, bounded reuse,
and the triples contraction schedule remain necessary performance work.

There is no new native Calculator DF registration, complete large-system
energy/force qualification, or permission to lift the existing force limit in
this change. Next consumers must provide the native factor/source owner, full
CCSD/(T) solve, corrected Lambda, reference-orbital response and DF metric/source
pullbacks, followed by independent complete-endpoint validation. Production must
not borrow the validation-only PySCF state or this test harness.

## References

- `python/generativeqc_compiler/cc/df_equations.py`
- `tools/generate_df_ccsd_native.py`
- `tests/python/test_df_cc_virtual_equations.py`
- `tests/python/test_df_cc_native_codegen.py`
- `benchmarks/df_ccsdt_large_oracle.py`
