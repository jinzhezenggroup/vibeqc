"""Clean-kernel Nsight windows: counters and component tracing stay disabled."""

import os
import sys

from benchmarks.df_policy_endpoint import main

os.environ.update(
    VIBEQC_DF_EXCHANGE="auto",
    VIBEQC_DF_SEED_EXCHANGE="auto",
    VIBEQC_DF_FINAL_EXCHANGE="auto",
)
for aos in (768, 384):
    case = "water-32mer-4s4" if aos == 768 else "water-hexadecamer-2s4"
    sys.argv = [
        "df_policy_endpoint",
        "--aos",
        str(aos),
        "--control",
        "VIBEQC_DF_SHELL_MATH_000",
        "--policies",
        "polynomial",
        "rys",
        "--repeats",
        "1",
        "--expected-iterations",
        "3",
        "--skip-cold",
        "--cuda-profile",
        "--host-trace",
        "--warm-checkpoint-in",
        f"/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final/ablations/{aos}-VIBEQC_DF_DIIS_DOTS.checkpoint",
        "--reference",
        f"benchmarks/results/issue377-379-df/gpu4pyscf/{case}-def2-svp-spherical.json",
        "--output",
        f".artifacts/issue394-000/profiles/{aos}.json",
    ]
    main()
