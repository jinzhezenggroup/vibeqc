"""Check scientific compiler dependencies and report maintainability metrics."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vibeqc_compiler.common.structure import audit_structure


def main():
    """Exit unsuccessfully on an ownership violation; optionally print inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_structure()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for error in report["errors"]:
            print(error, file=sys.stderr)
        print(
            f"Checked {len(report['modules'])} compiler modules; {len(report['errors'])} dependency errors"
        )
    return bool(report["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
