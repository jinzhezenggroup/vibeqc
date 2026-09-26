"""Count executed generated loop bodies from the observed primitive histogram.

Usage: python generated_work.py COMPONENT_TRACE_JSONL OUTPUT_JSON
This is symbolic work before compiler optimization, not SASS or DRAM traffic.
"""

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
records = [json.loads(line) for line in source.read_text().splitlines()]
counters = next(
    row["counters"] for row in records if row["operation"] == "force_response"
)
classes = {}
for key, count in counters.items():
    match = re.fullmatch(r"shell_([0-3])([0-3])([0-3])_p(\d+)_(\d+)_(\d+)", key)
    if match:
        a, b, c, pa, pb, pc = map(int, match.groups())
        classes[a, b, c] = classes.get((a, b, c), 0) + count * pa * pb * pc


def powers(degree):
    return [
        (i, j, degree - i - j) for i in range(degree + 1) for j in range(degree - i + 1)
    ]


rows = []
for (a, b, c), primitives in sorted(classes.items()):
    terms = 0
    for x, y, z in itertools.product(powers(a), powers(b), powers(c)):
        degrees = [sum(p) for p in zip(x, y, z)]
        terms += sum(
            (degrees[i] + 2) * (degrees[(i + 1) % 3] + 1) * (degrees[(i + 2) % 3] + 1)
            for i in range(3)
        )
    rows.append(
        {
            "angular": [a, b, c],
            "primitive_products": primitives,
            "cartesian_components": len(powers(a)) * len(powers(b)) * len(powers(c)),
            "other_axis_product_evaluations_per_primitive_before": 2 * terms,
            "other_axis_product_evaluations_per_primitive_after": terms,
            "axis_cache_doubles_before": (a + 2)
            * (b + 2)
            * (c + 1)
            * (a + b + c + 4)
            // 2,
            "axis_cache_doubles_after": (a + 2)
            * (b + 1)
            * (c + 1)
            * (a + b + c + 3)
            // 2,
            "moment_polynomials_per_primitive_before": 3
            * ((a + 2) * (b + 2) - 1)
            * (c + 1),
            "moment_polynomials_per_primitive_after": 3 * (a + 2) * (b + 1) * (c + 1),
        }
    )
assert sum(classes.values()) == counters["shell_primitive_products"]
result = {
    "scope": "Exact generated loop-body counts before optimizer; not measured FLOPs or global transactions. Geometry/Boys preparation remains once per primitive product.",
    "counter_trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "classes": rows,
    "primitive_products": sum(classes.values()),
    "weighted_other_axis_products_before": sum(
        row["primitive_products"]
        * row["other_axis_product_evaluations_per_primitive_before"]
        for row in rows
    ),
    "weighted_other_axis_products_after": sum(
        row["primitive_products"]
        * row["other_axis_product_evaluations_per_primitive_after"]
        for row in rows
    ),
}
output.write_text(json.dumps(result, indent=2) + "\n")
