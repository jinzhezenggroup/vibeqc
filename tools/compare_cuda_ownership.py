"""Separate physical CUDA edits from unchanged lines whose ownership changed.

Both reports must describe the supplied source trees exactly. The ordinary
ownership report remains the semantic authority; this comparison explains its
net deltas without presenting reclassification as physical code removal.
"""

import argparse
import collections
import difflib
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.report_cuda_ownership import ROLES, code_lines, validate_baseline


def lines_and_roles(root, row):
    """Bind the counting lexer and semantic regions to exact source bytes."""
    if row is None:
        return [], [], []
    source = (root / row["path"]).read_text()
    if hashlib.sha256(source.encode()).hexdigest() != row["sha256"]:
        raise ValueError(f"source/report identity mismatch: {row['path']}")
    lines = source.splitlines()
    roles = [row["role"]] * len(lines)
    for region in row["regions"]:
        begin, end = region["first_line"] - 1, region["last_line"]
        roles[begin:end] = [region["role"]] * (end - begin)
    return lines, roles, code_lines(source)


def compare(old_root, old, new_root, new):
    """Reconcile physical additions/removals and role shifts with report totals.

    SequenceMatcher's popularity heuristic is disabled so repeated braces and
    loop bodies do not disappear from the alignment of large translation units.
    Equal text can still enter or leave a multiline comment; those changes count
    as physical code additions/removals, not semantic reclassification.
    """
    for report in (old, new):
        validate_baseline(report)
    old_rows, new_rows = (
        {row["path"]: row for row in report["files"]} for report in (old, new)
    )
    added, removed, reclassified = (collections.Counter() for _ in range(3))
    for path in sorted(old_rows.keys() | new_rows.keys()):
        a, ar, active_a = lines_and_roles(old_root, old_rows.get(path))
        b, br, active_b = lines_and_roles(new_root, new_rows.get(path))
        for tag, i, j, k, end in difflib.SequenceMatcher(
            None, a, b, autojunk=False
        ).get_opcodes():
            if tag == "equal":
                for ia, ib in zip(range(i, j), range(k, end), strict=True):
                    if active_a[ia] and active_b[ib]:
                        if ar[ia] != br[ib]:
                            reclassified[ar[ia] + " -> " + br[ib]] += 1
                    elif active_a[ia]:
                        removed[ar[ia]] += 1
                    elif active_b[ib]:
                        added[br[ib]] += 1
            else:
                removed.update(ar[n] for n in range(i, j) if active_a[n])
                added.update(br[n] for n in range(k, end) if active_b[n])
    delta = {}
    for role in ROLES:
        shift = sum(
            count
            for transition, count in reclassified.items()
            if transition.endswith(" -> " + role)
        ) - sum(
            count
            for transition, count in reclassified.items()
            if transition.startswith(role + " -> ")
        )
        delta[role] = (
            new["maintained_code_lines"][role] - old["maintained_code_lines"][role]
        )
        if delta[role] != added[role] - removed[role] + shift:
            raise ValueError(f"physical edits do not reconcile with {role} totals")
    return {
        "definition": "Nonblank/noncomment physical code lines, SequenceMatcher(autojunk=False); unchanged lines changing role are separate from physical edits.",
        "added": dict(added),
        "removed": dict(removed),
        "unchanged_lines_reclassified": dict(reclassified),
        "net_role_delta": delta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline_root",
        "baseline_report",
        "candidate_root",
        "candidate_report",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.baseline_root,
        json.loads(args.baseline_report.read_text()),
        args.candidate_root,
        json.loads(args.candidate_report.read_text()),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
