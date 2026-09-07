"""Project the existing shell capability catalog into distinct evidence stages."""

import argparse
import json
from pathlib import Path

from .fixtures import ROOT
from .schema import file_hash, outcome


def capability_table(catalog: Path = ROOT / "docs/codegen_capabilities.json") -> dict:
    """Reuse all 55 classes, including #135's 34 f-containing classes.

    A committed emitter catalog records structural support, not a new compile
    or numerical run. Manifest selection is historical state and can coexist
    with absent acceptance evidence; FPPS explicitly remains provisional.
    """
    payload = json.loads(catalog.read_text())
    rows = []
    for entry in payload["shell_classes"]:
        selected = entry["production"]["status"] == "manifest_selected"
        rows.append(
            {
                "shell_class": entry["shell_class"],
                "angular": entry["angular"],
                "contains_f": 3 in entry["angular"],
                "recurrences": entry["recurrences"],
                "stages": {
                    "representation": outcome(
                        "pass",
                        scope="four-center ERI, first derivative, direct HF consumers",
                    ),
                    "source": outcome(
                        "pass" if entry["generic_fused"]["supported"] else "not-run",
                        None
                        if entry["generic_fused"]["supported"]
                        else "not represented in emitter catalog",
                        provenance="committed capability catalog",
                        schedules=entry["generic_fused"]["schedules"],
                    ),
                    "compilation": outcome(
                        "not-run",
                        "attach #135/autotune toolchain-specific compilation evidence",
                    ),
                    "numerical": outcome(
                        "not-run",
                        "attach independent class-isolated CUDA evidence; host self-checks do not establish this stage",
                    ),
                    "endpoint": outcome(
                        "not-run",
                        "attach molecular evidence with actual loaded angular momenta",
                    ),
                    "production": {
                        "selected": selected,
                        "manifest": entry["production"],
                        "acceptance_status": "provisional"
                        if entry["shell_class"] == "fpps" and selected
                        else "requires-attached-evidence",
                    },
                },
            }
        )
    return {
        "schema": "vibeqc.validation.capabilities",
        "schema_version": 1,
        "catalog_sha256": file_hash(catalog),
        "architecture": payload["architecture"],
        "shell_classes": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="six-stage JSON capability table"
    )
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(capability_table(), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
