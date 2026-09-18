# Decision: remove the external-engine dependency from the RHF Hessian chain

Status: implemented
Date: 2026-09-19

## Problem

PR #449 routed generated second-integral skeletons and shared response through
an active PySCF `System`, repeated PySCF SCF, and `Hessian.make_h1` / libcint first
derivatives. Numerical agreement was useful integration evidence but did not
show that VibeQC itself could produce all Hessian inputs. The prior
[response-boundary note](2026-09-19-hessian-analytic-response-boundary.md)
deliberately kept that external first-order provider. This decision supersedes
that provider choice, not the validated RHS/gauge or raw-symmetry conventions.

## Decision

Use one live native `NativeSource` and its existing `export_rhf` snapshot. A
`NativeRHFState` owns the immutable reference and borrows the source lifetime.
No helper reruns SCF, supplies placeholder convergence residuals, or replaces
native density/Fock/overlap data with an external engine's values.

Reuse the compiler's S/T/V first-derivative DAG and weighted full-Coulomb ERI
primitive emitter. The new raw first-component adapter compiles selected unit
cotangent subsets; a generic native runtime streams contracted primitives and
publishes only finite completed outputs. Runtime loops live in `src/integrals`;
mathematical emission and artifact identity stay in the compiler. The template
is included as an installed compiler asset.

At the method level, contract each first ERI derivative immediately into the
frozen Fock with the same reference density. Retain both AO-index motions and
the independent nuclear-attraction operator center. The existing RHS contract
then includes metric-density response and calls #179 with streamed native J/K.
Reuse #178 for second derivatives; fold density weights per shell quartet.

## Rejected alternatives

- Replacing PySCF first derivatives with the native dense `integral_derivatives`
  oracle would remove the import but leave a global integral Jacobian on the
  calculation side. It is retained only as an independent small-system test.
- Creating another SCF/CPHF/GMRES implementation would duplicate existing owners.
- Treating the scalar final HF force as a complete perturbation operator would
  lose the open AO indices needed for the response RHS.
- Relabeling the PySCF-backed module without completing the native input chain
  would not address the requested implementation repair.
- Promoting this bounded CPU integration to a GPU, production-size, or HVP
  capability would exceed the evidence and the implemented execution domain.

## Invariants

Native calculation runs with PySCF absent; oracle imports remain optional and
explicit. Reference/source identity and lifetime are checked before execution.
Radial normalization is shared with native basis conventions; angular factors
are applied once. The first derivative uses fixed density, with density response
owned by CPHF. The occupied metric term and independent ordered response blocks
remain mandatory. Invalid records leave native output unchanged; failed chunks
do not poison a later valid call.

## Evidence

The tests in `test_hessian_analytic.py` and `test_first_derivatives_native.py`
exercise these invariants, optional external comparisons, native-force
directional differences, and source/resource failures. The absence-of-oracle
regression runs in a fresh process and blocks both imports and dense oracle
calls, so a test session that already imported PySCF cannot hide the dependency.

Local CPU qualification for this repair: 68 passed / one explicit slow skip in
the combined Hessian/reference/response/assembly/weight suite; the expanded
first-component suite separately passed all ten cases. Adjacent first/second
integral-input and response tests passed 95 cases with eleven unavailable
backend cases skipped. Native CTest passed 31/31. A Python environment without
PySCF passed the three no-oracle/state-domain/lifetime checks.

At the seven-AO water geometry `(0,0,0), (0,1.43,1.1), (0,-1.43,1.1)` in Bohr,
using identical native basis records on the external comparison side, the
maximum Hessian error was 2.10e-12 Eh/Bohr^2, raw asymmetry 6.22e-15, and the
per-axis translation defects about 1.42e-12. These are correctness measurements,
not speed claims. The opt-in complete 12-AO d-shell Hessian was not run; its
primitive and reduced-response checks are not a substitute for that slow gate.

## Consequences and remaining scope

The existing native CPU SCF export is a small dense solver plus CPU NumPy
canonicalization. The integration is limited to 12 Cartesian AOs/four atoms and
retains coordinate H1/S1, response vectors and a dense molecular Hessian. Native
first/second kernel compilation and Python shell orchestration are not a
performance claim. #180 still owns production-size bounded execution, molecular
HVP, and separately qualified CUDA/DF/ECP/DFT extensions.

## Revisit when

A qualified compiler-owned directional consumer can replace full H1/S1 storage,
or native device response and complete resource accounting support promotion.
The mathematical source and RHS contracts must remain the same.

Agent: ChatGPT
Model: GPT-6 Astra Pro
