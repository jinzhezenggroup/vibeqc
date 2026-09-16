"""Separate Nsight windows for seed-only/final attribution and device transfers."""

import os
import sys

from benchmarks.df_policy_endpoint import main

os.environ["VIBEQC_DF_EXCHANGE"] = "occupied"
for aos in (768, 384):
    policies = ["dense", "factor", "auto"] if aos == 768 else ["dense", "factor"]
    mapping = (
        (
            '{"dense":{"VIBEQC_DF_FINAL_EXCHANGE":"dense"},'
            '"factor":{"VIBEQC_DF_FINAL_EXCHANGE":"dense"},'
            '"auto":{"VIBEQC_DF_FINAL_EXCHANGE":"auto"}}'
        )
        if aos == 768
        else (
            '{"dense":{"VIBEQC_DF_FINAL_EXCHANGE":"dense"},'
            '"factor":{"VIBEQC_DF_FINAL_EXCHANGE":"occupied"}}'
        )
    )
    case = "water-32mer-4s4" if aos == 768 else "water-hexadecamer-2s4"
    sys.argv = [
        "df_policy_endpoint",
        "--aos",
        str(aos),
        "--control",
        "VIBEQC_DF_SEED_EXCHANGE",
        "--policies",
        *policies,
        "--policy-controls",
        mapping,
        "--repeats",
        "1",
        "--expected-iterations",
        "3",
        "--skip-cold",
        "--cuda-profile",
        "--host-trace",
        "--journal",
        "--warm-checkpoint-in",
        f"/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final/ablations/{aos}-VIBEQC_DF_DIIS_DOTS.checkpoint",
        "--reference",
        f"benchmarks/results/issue377-379-df/gpu4pyscf/{case}-def2-svp-spherical.json",
        "--output",
        f".artifacts/issue399/profiles/{aos}.json",
    ]
    main()
