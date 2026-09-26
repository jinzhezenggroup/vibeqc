"""Pinned larger SCF states and independent AO/feature/XC workload gates."""

import typing

# Reuse the established native program fixture.
# ruff: noqa: F811
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_xc_contractions_native import native_factory  # noqa: F401
from vibeqc.autotune import dft_density_candidates
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.xc.prepared import PreparedXCContractions

from tools.density_workload_matrix import (
    batch_resources,
    load_workloads,
    original_source,
)

DIRECTORY = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/density-candidates/inputs"
)
NAMES = (
    "water_svp",
    "water_tzvp",
    "co2_tzvp",
    "water4_svp",
    "water8_svp",
    "oh_diffuse",
    "oh_minus_diffuse",
)


@pytest.fixture(scope="module")
def workloads() -> typing.Any:
    result = {
        name: (meta, data, grid) for name, meta, data, grid in load_workloads(DIRECTORY)
    }
    assert tuple(result) == NAMES
    return result


@pytest.mark.parametrize("name", NAMES)
def test_independent_current_state_and_all_sampled_features(
    workloads: typing.Any, name: typing.Any
) -> None:
    meta, data, grid = workloads[name]
    assert meta["reference"]["converged"]
    assert meta["reference"]["gradient_norm"] < 1e-6
    with NativeAO(**basis_arguments(meta)) as basis:
        source = original_source(basis, meta, data)
        assert source.source_kind == "orbitals"
        assert all(np.all(f == 1) for f in source.occupations)
        jets = basis.evaluate(grid.points[data["sample_ids"]], 1)
        np.testing.assert_allclose(jets, data["sample_jets"], atol=1e-11, rtol=1e-10)
        for route in ("density_matrix", "orbitals"):
            values = source.features(
                jets, stamp=source.stamp, route=route, ingredients=("rho", "gradient")
            )
            np.testing.assert_allclose(
                values["rho"], data["sample_rho_gradient"][:, 0], atol=1e-11, rtol=1e-10
            )
            np.testing.assert_allclose(
                values["gradient"],
                data["sample_rho_gradient"][:, 1:].transpose(0, 2, 1),
                atol=1e-11,
                rtol=1e-10,
            )


def test_producer_runtime_is_not_an_executable_workload_identity(
    workloads: typing.Any, native_factory: typing.Any
) -> None:
    meta, data, grid = workloads["water_svp"]
    changed = deepcopy(meta)
    changed["reference"]["seconds"] += 100
    # A reference snapshot audits every provenance byte. Mathematical inputs
    # and actual candidate signatures intentionally have a separate identity.
    assert canonical_hash(changed) != canonical_hash(meta)
    assert (
        changed["inputs_hash"] == meta["inputs_hash"] == canonical_hash(meta["inputs"])
    )
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        PreparedXCContractions(
            native_factory("LDA_XC_PW", "potential"), basis, grid
        ) as endpoint,
    ):
        sources = [original_source(basis, item, data) for item in (meta, changed)]
        registrations = [
            dft_density_candidates(endpoint, s, stamp=s.stamp) for s in sources
        ]
        assert [c.workload.identity for c in registrations[0]] == [
            c.workload.identity for c in registrations[1]
        ]
        assert [c.identity for c in registrations[0]] == [
            c.identity for c in registrations[1]
        ]


@pytest.mark.parametrize("name", ["water_svp", "oh_diffuse"])
@pytest.mark.parametrize("functional", ["LDA_XC_PW", "PBE"])
def test_full_independent_ev_and_replica_resource_composition(
    workloads: typing.Any,
    native_factory: typing.Any,
    name: typing.Any,
    functional: typing.Any,
) -> None:
    meta, data, grid = workloads[name]
    spin = "unpolarized" if meta["layout"] == "total" else "polarized"
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        PreparedXCContractions(
            native_factory(functional, "potential", spin), basis, grid, tile_points=256
        ) as endpoint,
    ):
        sources = tuple(
            original_source(basis, meta, data, generation=i) for i in range(4)
        )
        plan = batch_resources(endpoint, sources)
        source_bytes = sum(
            s.density.nbytes
            + sum(c.nbytes for c in s.coefficients)
            + sum(f.nbytes for f in s.occupations)
            for s in sources
        )
        assert (
            plan.peak_bytes["host"]
            >= endpoint.resource_plan.peak_bytes["host"] + source_bytes
        )
        candidate = dft_density_candidates(
            endpoint, sources[0], stamp=sources[0].stamp
        )[0]
        value, _ = candidate.execute(stamp=sources[0].stamp)
        potential = (
            value["potential"].mean(axis=0)
            if spin == "unpolarized"
            else value["potential"]
        )
        np.testing.assert_allclose(
            value["energy"], data[functional + "_energy"][0], atol=1e-11, rtol=1e-10
        )
        np.testing.assert_allclose(
            potential, data[functional + "_potential"], atol=1e-11, rtol=1e-10
        )
