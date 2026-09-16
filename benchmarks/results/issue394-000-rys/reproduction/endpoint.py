"""Qualify only 000 Rys, charging the complete endpoint after #399."""

import os
import sys

from benchmarks.df_policy_endpoint import main

os.environ.update(
    VIBEQC_DF_EXCHANGE="auto",
    VIBEQC_DF_SEED_EXCHANGE="auto",
    VIBEQC_DF_FINAL_EXCHANGE="auto",
)


def run(aos, diagnostic):
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
        "1" if diagnostic else "5",
        "--expected-iterations",
        "3",
        "--skip-cold",
        "--warm-checkpoint-in",
        f"/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final/ablations/{aos}-VIBEQC_DF_DIIS_DOTS.checkpoint",
        "--reference",
        f"benchmarks/results/issue377-379-df/gpu4pyscf/{case}-def2-svp-spherical.json",
        "--output",
        f".artifacts/issue394-000/v1/{aos}-"
        + ("work" if diagnostic else "endpoint")
        + ".json",
    ]
    sys.argv += ["--trace"] if diagnostic else ["--components-after"]
    main()


for aos in (768, 384):
    run(aos, False)
# Detailed source-operation atomics never contaminate the clean or basic-resource pass.
os.environ.update(VIBEQC_DF_SHELL_WORK="1", VIBEQC_DF_SHELL_COUNTERS="1")
for aos in (768, 384):
    run(aos, True)
