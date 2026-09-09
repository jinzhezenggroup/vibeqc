# Numerical accuracy evidence

`vibeqc.accuracy` separates the resolved scientific model, observable-error
requirements and evidence. It is an explicit comparison API. It does not yet
select precision, screening or iteration tolerances inside `Calculator`.

`Calculator.resolved_model(atoms, charge=..., multiplicity=...)` hashes the
actual ordered geometry, basis/representation, electron populations and
Hamiltonian. It reuses the existing basis mathematical identity and canonical
profile hashing. Conventional and density-fitted HF have different identities;
the latter includes actual auxiliary data and the metric threshold. Backend,
schedule, screening and convergence settings are excluded, so experiments can
compare numerical choices against a fixed model. Distinct metric thresholds
remain distinct approximations and cannot be relabeled as identical models.

```python
from vibeqc import (
    Calculator, ObservableTarget, TargetAccuracy,
    compare_observables,
)

atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
target = TargetAccuracy((
    ObservableTarget("energy", "absolute", "Eh", absolute=1e-6),
    ObservableTarget("forces", "max_abs", "Eh/bohr", absolute=1e-7),
))
calc = Calculator(energy_tolerance=1e-6, target_accuracy=target)
model = calc.resolved_model(atoms)
result = calc.singlepoint(atoms)
assert result.accuracy.status == "unverified"

# A fully relaxed audit runs the requested target again with tighter numerics.
strict = Calculator(energy_tolerance=1e-13, density_tolerance=1e-11,
                    screening_tolerance=1e-14)
reference = strict.singlepoint(atoms)
report = compare_observables(
    model, calc.resolved_model(atoms), strict.resolved_model(atoms), target,
    {"energy": result.energy, "forces": result.forces},
    {"energy": reference.energy, "forces": reference.forces},
    scope="relaxed_target",
    provenance=(("reference", "independent-strict-native-solve"),),
    converged=result.converged and reference.converged,
)
print(report.to_dict())
```

All requirements must hold independently. The allowance for each norm is
`absolute + relative * reference_norm`; a zero reference norm supplies zero
relative allowance. Force RMS means RMS over all `3*Natom` Cartesian
components. Force maximum means maximum absolute Cartesian component. No unit
conversion or broadcasting of mismatched force arrays is implicit.

Evidence records the target, evaluated and reference model IDs, source, units,
norm, audit scope, optional conditioning/calibration metadata and provenance.
Assessment rejects mismatched identities. Fixed-density audits remain useful
operator diagnostics but cannot satisfy a relaxed-target request. Per-source
screening/arithmetic/residual estimates are retained independently; their sum
has no assumed observable-error interpretation.

The reported states are `unverified`, `unconverged`, `observed_met`,
`observed_unmet`, `estimated_below_target` and `estimated_above_target`.
Observed agreement describes a measured difference, including the strict
reference's own unresolved uncertainty. It establishes neither a proven error
bound nor electronic stability/intended-root identity. Calibrated predictions
remain estimates; a recorded held-out error overrides an optimistic estimate.
There is no supported `certified` result, and merely labeling evidence a proven
bound is rejected.

`Calculator(target_accuracy=...)` attaches the request to each result without
changing native tolerances or arithmetic. Prepared batches use each item's
actual execution geometry; a failed item has no accuracy assessment. Changing
convergence controls or the target invalidates prepared settings. Native C/C++
descriptor layouts and default numerical results are unchanged.

## Strict HF audits and experiments

`tools.vibeqc_numerics.audit` provides a native RHF/UHF final-state probe and an
independent fixed-density operator audit. The latter contracts unscreened raw
integrals in NumPy FP64 to compute the energy, physical Fock commutator,
electron traces and metric idempotency. A fully relaxed audit performs a
separate strict native solve and compares its energy and forces against pinned
PySCF references. These dense diagnostic tools reject systems larger than
12 orbital or 24 auxiliary AOs; they are not production runtime estimators.

`python -m tools.validate_accuracy --output /tmp/accuracy` records independent
and coupled convergence sweeps. CUDA additionally sweeps screening. Saved
densities and forces have checksum-linked manifests and can be replayed with
`python -m tools.vibeqc_numerics.replay /tmp/accuracy/report.json` without SCF.
Failures remain in the report. Metric-rank changes use distinct model identities.
An additional CUDA driver compares experimental mixed-Fock requests at matched
SCF controls; its requested threshold does not establish actual FP32 work.

## Limited empirical calibration

`vibeqc.accuracy_estimator.EmpiricalHFEstimator` relates measured physical
residuals to observed energy and force errors using a conservative training
envelope. Water and methane are the training families. Hydrogen, ammonia and
neutral HF are held out as entire families, including all numerical settings.
The versioned domain includes CPU FP64 conventional Cartesian RHF with verified
STO-3G data, size and conditioning limits. UHF, different basis families, CUDA,
missing/small gaps and poorly conditioned overlap matrices are rejected.

To consume the recorded calibration after producing a matching probe and audit:

```python
import json
from pathlib import Path
from vibeqc import AccuracyAssessment
from vibeqc.accuracy_estimator import EmpiricalHFEstimator
from tools.vibeqc_numerics.audit import error_features

estimator = EmpiricalHFEstimator.from_dict(json.loads(
    Path("benchmarks/results/accuracy-173/cpu/estimator.json").read_text()
))
features = error_features(source, probe, physical_audit)
evidence = estimator.predict(
    probe.model, features,
    energy_reference_norm=abs(reference.energy),
    force_reference_norm=abs(reference.forces).max(),
)
assessment = AccuracyAssessment(probe.model, target, evidence, probe.converged)
```

A prediction remains empirical even when below the target. A quantitative
residual-to-observable bound would require control of the inverse SCF Jacobian
or an observable-specific adjoint, together with nonlinear remainder estimates.
The orbital gap and AO condition number do not provide that control or prove
electronic stability. Successive energy/density changes are convergence controls,
not substitutes for physical residuals or independently requested force errors.

The [measured report](../benchmarks/results/accuracy-173/README.md) includes raw
errors, holdout misses and excess conservatism, audit overhead, model/rank changes,
failures and reproduction commands. This issue supplies evidence and calibration;
the precision selection/refinement controller remains owned by #174.
