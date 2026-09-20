"""CPU work admission metadata, independently enumerated without a native library."""

import re
import typing
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc._stationary_cpu import _admit_work
from vibeqc_compiler.common.paths import asset_path


def inputs(*, ecp: bool = True) -> tuple[typing.Any, typing.Any, tuple[int, ...]]:
    # Unequal contraction sizes catch AO-count and unique-primitive substitutes.
    counts = (1, 2, 3)
    rows = np.zeros((3, 16))
    rows[:, 2], rows[:, 3] = counts, 1
    basis = SimpleNamespace(
        natom=3,
        nao=3,
        nprimitive=sum(counts),
        shells=(),
        packed=np.concatenate((np.zeros(9 + 2 * sum(counts)), rows.ravel())),
    )
    state = SimpleNamespace(
        grid=SimpleNamespace(points=range(11)),
        _source=SimpleNamespace(
            hamiltonian="scalar-semilocal-ecp" if ecp else "all-electron",
            ecp_cores=(10, 0, 10),
            ecp_terms=(1, 2),
        ),
    )
    return state, basis, counts


def admit(
    state: typing.Any, basis: typing.Any, execution: str = "native", **kwargs: int
) -> dict[str, int]:
    limits = {
        "max_primitive_records": 2_000_000,
        "max_grid_points": 1_000_000,
        "max_grid_pair_visits": 100_000_000,
        "max_ecp_pair_samples": 100_000_000,
    }
    limits.update(kwargs)
    return _admit_work(state, basis, execution, 4, **limits)


@pytest.mark.parametrize("execution", ["native", "reference"])
def test_work_counts_follow_contractions_and_both_provider_grids(
    execution: str,
) -> None:
    state, basis, counts = inputs()
    work = admit(state, basis, execution)
    # Explicitly enumerate primitive tuples for every public AO pair/quartet.
    records = 3  # distinct nuclear pairs
    for rank, repeats in ((2, 5), (4, 1)):
        for aos in product(range(3), repeat=rank):
            records += repeats * sum(
                1 for _ in product(*(range(counts[a]) for a in aos))
            )
    assert work["primitive_record_bound"] == records
    assert work["grid_pair_work_bound"] == (75 if execution == "native" else 378)
    # Read the independent CPU provider's actual checked calls, so a future
    # policy change cannot silently undercount its two-grid work.
    source = asset_path("src/integrals/ecp.cpp").read_text()
    grids = re.findall(r"ecp_integrals\(system, (\d+), (\d+), derivatives\)", source)
    assert len(grids) == 2
    expected = 0
    for radial, polar in grids:
        for _center in (0, 2):
            for a in range(3):
                for _b in range(a + 1):
                    expected += int(radial) * int(polar) * (2 * int(polar))
    assert work["ecp_quadrature_pair_samples"] == expected
    assert admit(state, basis, execution, max_ecp_pair_samples=expected) == work | {
        "ecp_pair_sample_budget": expected
    }
    with pytest.raises(ValueError, match="ECP.*work budget"):
        admit(state, basis, execution, max_ecp_pair_samples=expected - 1)


def test_all_electron_has_no_ecp_work_or_dense_provider_limit() -> None:
    state, basis, _ = inputs(ecp=False)
    basis.nao = 17
    assert (
        admit(state, basis, max_ecp_pair_samples=1)["ecp_quadrature_pair_samples"] == 0
    )


@pytest.mark.parametrize(
    "field,value",
    [("nao", 17), ("natom", 9), ("nprimitive", 129), ("ecp_terms", (0,) * 129)],
)
def test_ecp_dense_provider_domain_is_explicit(field: str, value: typing.Any) -> None:
    state, basis, _ = inputs()
    # Preserve the packed AO rows when changing header dimensions.
    rows = basis.packed[-48:]
    setattr(state._source if field == "ecp_terms" else basis, field, value)
    basis.packed = np.concatenate(
        (np.zeros(3 * basis.natom + 2 * basis.nprimitive), rows)
    )
    with pytest.raises(ValueError, match="dense-export domain"):
        admit(state, basis)
