"""Matched baseline/default qualification of the final automatic selectors."""

import os
import sys

from benchmarks.df_policy_endpoint import main

os.environ["VIBEQC_DF_EXCHANGE"] = "auto"
for energy in (False, True):
    sys.argv = [
        "df_policy_endpoint",
        "--aos",
        "768",
        "--control",
        "VIBEQC_DF_SEED_EXCHANGE",
        "--policies",
        "dense",
        "auto",
        "--policy-controls",
        '{"dense":{"VIBEQC_DF_FINAL_EXCHANGE":"dense"},"auto":{"VIBEQC_DF_FINAL_EXCHANGE":"auto"}}',
        "--repeats",
        "5",
        "--expected-iterations",
        "3",
        "--skip-cold",
        "--warm-checkpoint-in",
        "/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final/ablations/768-VIBEQC_DF_DIIS_DOTS.checkpoint",
        "--reference",
        "benchmarks/results/issue377-379-df/gpu4pyscf/water-32mer-4s4-def2-svp-spherical.json",
        "--output",
        ".artifacts/issue399/v2/768-default" + ("-energy" if energy else "") + ".json",
    ]
    sys.argv += ["--energy-only"] if energy else ["--components-after"]
    main()
