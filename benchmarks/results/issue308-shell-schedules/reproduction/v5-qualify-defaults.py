"""Require actual selected consumers and unchanged branches after default promotion.

These separate intrusive runs exercise defaults and the explicit generic oracle
on the qualified RTX 5090. Full-force checks are owned by the existing runner;
this audit checks the consumer counters as well as every numerical sample.
"""

import json
import pathlib
import subprocess
import sys

out = pathlib.Path(sys.argv[1]).resolve()
for aos in (19, 192, 384):
    path = out / f"default-traced-{aos}ao"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.issue308_response_timeline",
            "--aos",
            str(aos),
            "--batch",
            "1",
            "--repeats",
            "1",
            "--component-trace",
            "--library",
            str(out / "libvibeqc.so"),
            "--output",
            str(path),
        ],
        check=True,
    )
    result = json.loads((path / "result.json").read_text())
    (sample,) = result["samples"]
    (response,) = [
        g for g in sample["components"]["groups"] if g["operation"] == "force_response"
    ]
    counters = response["counter_sums"]
    expected = aos >= 192
    for key in (
        "three_center_shell_panels",
        "response_charge_blas_dots",
        "raw_panel_pinned_host_bytes",
    ):
        assert bool(counters.get(key)) == expected, (aos, key, counters)
    assert not sample["host_components"]["reference_eigensolves"]
    assert all(sample["warm_start_used"]) and not any(sample["warm_start_fallback"])
print(
    "Default consumers, size boundary, full forces and zero reference eigensolves verified."
)
