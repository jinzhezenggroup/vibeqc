"""CPU work admission metadata, independently enumerated without a native library."""

import typing
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc._cpu_force_resources import CPU_FORCE_HOST_CAP, cpu_force_inventory
from vibeqc._stationary_cpu import _admit_work


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
    # Independent literal prescription. Native tests separately compare both
    # generated production grids with the retained pre-migration CPU oracle.
    grids = ((160, 32), (224, 44))
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


def test_hybrid_admission_counts_exact_exchange_eri_derivative_pass() -> None:
    state, basis, counts = inputs(ecp=False)
    semilocal = admit(state, basis)["primitive_record_bound"]
    state._source.method_ir = SimpleNamespace(full_range_exact_exchange=0.25)
    hybrid = admit(state, basis)["primitive_record_bound"]
    extra = sum(counts) ** 4
    assert hybrid == semilocal + extra
    with pytest.raises(ValueError, match="primitive work budget"):
        admit(state, basis, max_primitive_records=hybrid - 1)


def test_all_electron_has_no_ecp_work_or_dense_provider_limit() -> None:
    state, basis, _ = inputs(ecp=False)
    basis.nao = 17
    assert (
        admit(state, basis, max_ecp_pair_samples=1)["ecp_quadrature_pair_samples"] == 0
    )


def test_multicomponent_work_counts_and_exact_budget() -> None:
    state, basis, counts = inputs()
    rows = basis.packed[-48:].reshape(3, 16)
    components = (1, 2, 3)
    rows[:, 3] = components
    records = 3
    for rank, repeats in ((2, 5), (4, 1)):
        for aos in product(range(3), repeat=rank):
            for _terms in product(*(range(components[a]) for a in aos)):
                records += repeats * sum(
                    1 for _ in product(*(range(counts[a]) for a in aos))
                )
    assert (
        admit(state, basis, max_primitive_records=records)["primitive_record_bound"]
        == records
    )
    with pytest.raises(ValueError, match="primitive work budget"):
        admit(state, basis, max_primitive_records=records - 1)


@pytest.mark.parametrize("components", [0, 4])
def test_unsupported_component_count_rejected(components: int) -> None:
    state, basis, _ = inputs()
    basis.packed[-48:].reshape(3, 16)[0, 3] = components
    with pytest.raises(NotImplementedError, match="Cartesian components"):
        admit(state, basis)


def test_f_shell_rejected_before_work() -> None:
    state, basis, _ = inputs()
    basis.shells = (SimpleNamespace(angular_momentum=3),)
    with pytest.raises(NotImplementedError, match="s/p/d"):
        admit(state, basis)


@pytest.mark.parametrize("invalid", ["tile", "shell", "components"])
def test_component_executor_rejects_invalid_metadata_before_compilation(
    invalid: str, monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    from vibeqc import _stationary_cpu_components as module

    _, basis, _ = inputs()
    tile = 1
    if invalid == "tile":
        tile = 0
    elif invalid == "shell":
        basis.shells = (SimpleNamespace(angular_momentum=3),)
    else:
        basis.packed[-48:].reshape(3, 16)[0, 3] = 4

    def forbidden(*args: typing.Any) -> None:
        pytest.fail("invalid component metadata reached source generation")

    monkeypatch.setattr(module, "derivative_sources", forbidden)
    with pytest.raises((ValueError, NotImplementedError)):
        module.ComponentPrimitiveExecutor(basis, tmp_path, tile, None)


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


def test_cpu_numeric_inventory_covers_provider_arrays_and_export_copies() -> None:
    _, basis, _ = inputs()
    inventory = cpu_force_inventory(basis, grid_points=11, ecp_terms=2)
    # Independently enumerate CPU's peak refined-grid storage, retaining both
    # coarse and fine derivatives plus sphere vector growth, AO and projections.
    n, a, q = basis.nao, basis.natom, 2 * 44**2
    provider = 2 * (2 * n**2 + 2 * 3 * a * n**2) * 8
    provider += 2 * q * (3 + 1 + 16) * 8 + n * q * 4 * 8 + n * 16 * 4 * 8
    assert inventory["ecp_provider"] >= provider
    assert inventory["ecp_export_and_contraction"] >= 4 * 6 * a * n**2 * 8
    assert sum(inventory.values()) < CPU_FORCE_HOST_CAP
    huge = cpu_force_inventory(basis, grid_points=1_000_000, ecp_terms=2)
    assert sum(huge.values()) > CPU_FORCE_HOST_CAP
    with pytest.raises(ValueError, match="grid point work"):
        cpu_force_inventory(basis, grid_points=1_000_001, ecp_terms=2)


@pytest.mark.parametrize(
    "field,limit", [("nao", 16), ("natom", 8), ("nprimitive", 128), ("ecp_terms", 128)]
)
def test_cpu_force_inventory_rejects_each_oversized_dimension(
    field: str, limit: int
) -> None:
    _, basis, _ = inputs()
    options = {"grid_points": 11, "ecp_terms": 2}
    if field == "ecp_terms":
        options[field] = limit
    else:
        setattr(basis, field, limit)
    assert sum(cpu_force_inventory(basis, **options).values()) < CPU_FORCE_HOST_CAP
    if field == "ecp_terms":
        options[field] += 1
    else:
        setattr(basis, field, limit + 1)
    with pytest.raises(ValueError, match="dense-export domain"):
        cpu_force_inventory(basis, **options)
