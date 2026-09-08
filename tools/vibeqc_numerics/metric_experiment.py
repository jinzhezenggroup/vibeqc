"""Measure DF metric approximation changes without relabeling the target model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from vibeqc import Calculator, compare_observables

from tools.validate_accuracy import TARGET
from tools.vibeqc_numerics.audit import ProbeControls, StrictHFAudit, probe_hf
from tools.vibeqc_posthf.sources import NativeSource


def run():
    """Hold geometry/basis fixed while independently sweeping metric truncation."""
    atoms = [("He", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    rows = []
    for method, charge, multiplicity in (("rhf", 1, 1), ("uhf", 0, 2)):
        target_model = Calculator(
            method=method,
            density_fitting="cpu",
            auxiliary_basis="sto-3g",
            density_fitting_relative_threshold=1e-10,
        ).resolved_model(atoms, charge=charge, multiplicity=multiplicity)
        with NativeSource(atoms, auxiliary_basis="sto-3g", charge=charge) as source:
            reference = probe_hf(source, target_model)
            if not reference.converged:
                raise RuntimeError("metric experiment strict reference failed")
            for threshold in (1e-12, 1e-10, 1e-4, 0.05, 0.2):
                evaluated_model = replace(
                    target_model, metric_relative_threshold=threshold
                )
                row = {
                    "method": method,
                    "controls": asdict(ProbeControls()),
                    "target_model": target_model.to_dict(),
                    "evaluated_model": evaluated_model.to_dict(),
                }
                try:
                    probe = probe_hf(source, evaluated_model)
                    row.update(probe.scalars())
                    if not probe.converged:
                        row["status"] = "unconverged"
                    else:
                        audit = StrictHFAudit(source, evaluated_model).evaluate(probe)
                        row.update(
                            status="observed",
                            audit=audit,
                            energy_difference_from_target=abs(
                                probe.energy - reference.energy
                            ),
                            force_max_difference_from_target=float(
                                np.max(np.abs(probe.forces - reference.forces))
                            ),
                        )
                        try:
                            comparison = compare_observables(
                                target_model,
                                evaluated_model,
                                target_model,
                                TARGET,
                                {"energy": probe.energy, "forces": probe.forces},
                                {
                                    "energy": reference.energy,
                                    "forces": reference.forces,
                                },
                                scope="relaxed_target",
                                provenance=(("reference", target_model.identity),),
                                converged=True,
                            )
                            row["numerical_accuracy_status"] = comparison.status
                        except ValueError as error:
                            row["numerical_accuracy_status"] = "changed_model"
                            row["reason"] = str(error)
                except (ValueError, RuntimeError) as error:
                    row.update(status="failed", reason=str(error))
                rows.append(row)
    return {
        "schema_version": 1,
        "source": "DF metric truncation",
        "backend": "cpu",
        "interpretation": "Changed-model observable differences are not numerical-error certification for the fixed target.",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run()
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            [
                {
                    "method": r["method"],
                    "threshold": r["evaluated_model"]["metric_relative_threshold"],
                    "rank": r.get("audit", {}).get("metric_rank"),
                    "status": r.get("numerical_accuracy_status", r["status"]),
                }
                for r in result["rows"]
            ]
        )
    )


if __name__ == "__main__":
    main()
