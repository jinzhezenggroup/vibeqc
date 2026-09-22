"""Reproduce the README's nested water clusters with the audited HF comparator.

All comparator options, numerical gates, fixed-density replay and binary
provenance are retained. DF callers must supply an explicit auxiliary basis.
"""

from dataclasses import replace

from benchmarks import compare_gpu4pyscf_batch as comparator
from benchmarks._cases import benchmark_cases


def scaling_cases() -> dict:
    """Use identical nested geometry prefixes for direct and fitted HF."""
    base = benchmark_cases()["water-32mer-4s4-def2-svp-spherical"]
    return {
        f"water-{atoms}": replace(
            base,
            description=f"nested {atoms}-atom water cluster, def2-SVP spherical",
            atoms=base.atoms[:atoms],
            expected_ao_count=atoms * 8,
        )
        for atoms in (3, 6, 12, 24, 48, 96)
    }


if __name__ == "__main__":
    comparator.benchmark_cases = scaling_cases
    comparator.main()
