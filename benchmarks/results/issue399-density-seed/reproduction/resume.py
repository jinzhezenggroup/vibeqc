"""Resume missing bounded #399 runs, retaining interrupted evidence separately."""

import json
import os
import shutil
import sys
import time
from pathlib import Path

from benchmarks.df_policy_endpoint import main

root = Path(".artifacts/issue399/v1")


def run_missing(output, expected, args):
    try:
        data = json.loads(output.read_text())
        assert len(data["samples"]) == expected
        if expected == 10 and not output.stem.endswith("energy"):
            assert len(data["diagnostics"]) == 2
        assert all(
            s["iterations"] == [3]
            and s["maximum_energy_error"] <= 1e-9
            and (s["maximum_force_error"] is None or s["maximum_force_error"] <= 1e-8)
            for s in data["samples"]
        )
        return
    except (OSError, ValueError, KeyError, AssertionError):
        if output.exists():
            partial = root / ("interrupted-" + str(time.time_ns()))
            partial.mkdir()
            for artifact in root.glob(output.stem + ".*"):
                shutil.move(artifact, partial / artifact.name)
    print("Starting", output, flush=True)
    sys.argv = ["benchmarks.df_policy_endpoint", *args, "--output", str(output)]
    main()


for aos in (768, 384):
    os.environ.update(VIBEQC_DF_SEED_EXCHANGE="dense", VIBEQC_DF_FINAL_EXCHANGE="dense")
    case = "water-hexadecamer-2s4" if aos == 384 else "water-32mer-4s4"
    common = [
        "--aos",
        str(aos),
        "--expected-iterations",
        "3",
        "--skip-cold",
        "--warm-checkpoint-in",
        f"/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final/ablations/{aos}-VIBEQC_DF_DIIS_DOTS.checkpoint",
        "--reference",
        f"benchmarks/results/issue377-379-df/gpu4pyscf/{case}-def2-svp-spherical.json",
    ]
    for kind, control, policies in [
        ("seed", "VIBEQC_DF_SEED_EXCHANGE", ["dense", "factor"]),
        ("final", "VIBEQC_DF_FINAL_EXCHANGE", ["dense", "occupied"]),
    ]:
        if kind == "final":
            os.environ["VIBEQC_DF_SEED_EXCHANGE"] = "factor"
        for energy in (False, True):
            name = f"{aos}-{kind}" + ("-energy" if energy else "") + ".json"
            args = common + [
                "--control",
                control,
                "--policies",
                *policies,
                "--repeats",
                "5",
            ]
            args += ["--energy-only"] if energy else ["--components-after"]
            run_missing(root / name, 10, args)
    os.environ.update(VIBEQC_DF_SEED_VERIFY="1", VIBEQC_DF_FINAL_EXCHANGE="dense")
    run_missing(
        root / f"{aos}-verification.json",
        1,
        common
        + [
            "--control",
            "VIBEQC_DF_SEED_EXCHANGE",
            "--policies",
            "factor",
            "--repeats",
            "1",
            "--trace",
            "--journal",
        ],
    )
    os.environ.pop("VIBEQC_DF_SEED_VERIFY")
