"""Actual dense native CPU XC tile ownership, not a modeled allocator speedup."""

import shutil
import typing
import weakref
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.program import ProgramIR
from vibeqc_compiler.common.program_storage import (
    CallDonationBinding,
    ProgramStoragePlan,
)
from vibeqc_compiler.common.resources import MAX_BYTES
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.features import density_feature_block, density_features
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.coefficients import coefficient_program
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture
from vibeqc_compiler.xc.native import NativeContractionProgram
from vibeqc_compiler.xc.prepared import PreparedXCContractions
from vibeqc_compiler.xc.program_ir import fixed_density_tile_program


def describe(
    name: typing.Any = "PBE", spin: typing.Any = "polarized", **changes: typing.Any
) -> typing.Any:
    options = {
        "nao": 12,
        "tile_points": 7,
        "basis_bytes": 100,
        "grid_bytes": 1000,
        "basis_identity": "basis-v1",
        "native_identity": "native-v1",
        "packed_features": False,
    }
    options.update(changes)
    return fixed_density_tile_program(
        ContractionProgram(functional(name, spin=spin)).contract, **options
    )


@pytest.mark.parametrize("name,jets", [("LDA_XC_PW", 1), ("PBE", 4)])
@pytest.mark.parametrize("spin,nspin", [("unpolarized", 1), ("polarized", 2)])
def test_fixed_density_description_is_bound_to_real_contract(
    name: typing.Any, jets: typing.Any, spin: typing.Any, nspin: typing.Any
) -> None:
    p = describe(name, spin)
    assert ProgramIR.from_payload(p.to_payload()) == p
    sizes = {buffer.name: buffer.bytes for buffer in p.buffers}
    assert sizes["quadrature"] == 1000
    assert sizes["jets"] == 8 * jets * 7 * 12
    assert sizes["contribution"] == 8 * nspin * 12 * 12 + 24
    assert p.release_after("collocation") == ()  # Vxc still needs AO values/jets!
    assert p.release_after("xc") == ("jets",)
    assert p.calls[0].provider.startswith("dft.")
    assert p.calls[1].provider.startswith("xc.")
    assert len(p.calls) == 2  # Do not pretend internal XC stages are separate kernels.


@pytest.mark.parametrize(
    "changes",
    [
        {"nao": 0},
        {"nao": True},
        {"tile_points": -1},
        {"tile_points": MAX_BYTES},
        {"basis_bytes": -1},
        {"grid_bytes": -1},
        {"basis_identity": ""},
        {"native_identity": None},
    ],
)
def test_invalid_tile_boundaries(changes: typing.Any) -> None:
    with pytest.raises(ValueError):
        describe(**changes)


def test_provider_and_shape_changes_invalidate_graph_identity() -> None:
    for changes in (
        {"nao": 13},
        {"tile_points": 8},
        {"basis_identity": "moved"},
        {"native_identity": "other"},
        {"packed_features": True},
    ):
        assert describe(**changes).identity != describe().identity


@pytest.mark.parametrize("name,gradient", [("LDA_XC_PW", False), ("PBE", True)])
def test_packed_feature_program_records_real_cross_subsystem_layouts(
    name: typing.Any, gradient: typing.Any
) -> None:
    p = describe(name, "polarized", packed_features=True)
    assert tuple(call.name for call in p.calls) == (
        "collocation",
        "features",
        "scalar_xc",
        "vxc",
    )
    assert p.calls[1].provider == "dft.density_feature_block"
    assert p.calls[2].provider.startswith("xc.")
    layouts = {buffer.name: buffer.layout for buffer in p.buffers}
    assert layouts["jets"].shape == (
        1 if name == "LDA_XC_PW" else 4,
        7,
        12,
    )
    assert layouts["feature_scalar"].shape == (7, 7)
    assert layouts["xc_rows"].shape == (8, 7)
    assert layouts["coefficients"].shape == ((2 if name == "LDA_XC_PW" else 8), 7)
    assert ("feature_gradient" in layouts) is gradient
    releases = p.release_after("vxc")
    assert releases[0] == "jets"
    assert "feature_scalar" in releases and "xc_rows" in releases
    assert ("feature_gradient" in releases) is gradient
    assert ProgramIR.from_payload(p.to_payload()) == p


def test_pbe_program_storage_donates_scalar_owner_to_coefficients() -> None:
    p = describe("PBE", "polarized", packed_features=True)
    baseline = ProgramStoragePlan(p).storage_analysis()
    donated = ProgramStoragePlan(
        p,
        donations=(CallDonationBinding("vxc", "xc_rows", "coefficients"),),
    ).storage_analysis()
    sizes = {buffer.name: buffer.bytes for buffer in p.buffers}

    assert donated.donations == ((4, "xc_rows", "coefficients"),)
    assert donated.slot_for("coefficients") == donated.slot_for("xc_rows")
    assert (
        baseline.peak_by_space["pageable"] - donated.peak_by_space["pageable"]
        == sizes["xc_rows"]
    )


def test_packed_feature_program_rejects_unpolarized_and_non_boolean_selection() -> None:
    with pytest.raises(ValueError, match="polarized"):
        describe("PBE", "unpolarized", packed_features=True)
    with pytest.raises(ValueError, match="bool"):
        describe("PBE", "polarized", packed_features=1)


def test_non_potential_contracts_remain_outside_phase_a() -> None:
    with pytest.raises(TypeError):
        fixed_density_tile_program(
            None,
            nao=1,
            tile_points=1,
            basis_bytes=1,
            grid_bytes=1,
            basis_identity="b",
            native_identity="n",
        )
    with pytest.raises(ValueError, match="potential"):
        fixed_density_tile_program(
            ContractionProgram(functional("PBE"), "response").contract,
            nao=1,
            tile_points=1,
            basis_bytes=1,
            grid_bytes=1,
            basis_identity="b",
            native_identity="n",
        )


@pytest.fixture(scope="module")
def native_factory(tmp_path_factory: typing.Any) -> typing.Any:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native XC requires c++")
    cache = tmp_path_factory.mktemp("programir-native")
    programs = {}

    def make(
        name: typing.Any,
        spin: typing.Any = "polarized",
        observable: typing.Any = "potential",
    ) -> typing.Any:
        key = name, spin, observable
        if key not in programs:
            programs[key] = NativeContractionProgram(
                functional(name, spin=spin),
                observable,
                compiler=CppCompilerAdapter(Path(compiler)),
                cache=cache,
            )
        return programs[key]

    return make


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize("case", ["h2", "f_spherical"])
def test_native_cpu_releases_previous_tile_before_next_ao_allocation(
    native_factory: typing.Any,
    monkeypatch: typing.Any,
    name: typing.Any,
    case: typing.Any,
) -> None:
    meta, data, grid = load_integration_fixture(case)
    program = native_factory(name)
    refs, observed = [], []
    with NativeAO(**basis_arguments(meta)) as basis:
        old_ao, old_xc = basis.evaluate, program.potential_from_rows

        def collocate(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            # Weak references do not themselves extend an allocation lifetime.
            observed.append(tuple(label for label, ref in refs if ref() is not None))
            jets = old_ao(*args, **kwargs)
            refs.append(("jets", weakref.ref(jets)))
            return jets

        def contract(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            result = old_xc(*args, **kwargs)
            for key in ("potential", "electrons"):
                refs.append((key, weakref.ref(result[key])))
            return result

        monkeypatch.setattr(basis, "evaluate", collocate)
        monkeypatch.setattr(program, "potential_from_rows", contract)
        monkeypatch.setattr(
            program,
            "scalar_values",
            lambda *_a, **_kw: pytest.fail("packed path called legacy scalar repack"),
        )
        with PreparedXCContractions(program, basis, grid, tile_points=7) as prepared:
            description = prepared.tile_program
            actual = prepared.execute(data["density_spin"])
            assert len(observed) > 1
            assert observed == [()] * len(observed)
            assert all(ref() is None for _, ref in refs)
            assert prepared.statistics["tile_program_identity"] == description.identity
            releases = prepared.statistics["tile_boundary_releases"]
            assert releases[0] == "jets"
            assert "feature_scalar" in releases and "xc_rows" in releases
            if name == "PBE":
                assert "feature_gradient" in releases
            assert prepared.statistics["tile_layouts"]["feature_scalar"]["shape"] == (
                7,
                7,
            )
            assert prepared.statistics["scalar_calls"] == len(observed)
            np.testing.assert_allclose(
                actual["energy"], data[f"{name}_spin_energy"][0], rtol=1e-10, atol=1e-11
            )
            np.testing.assert_allclose(
                actual["potential"],
                data[f"{name}_spin_potential"],
                rtol=1e-10,
                atol=1e-11,
            )
            # Repeated calls and changed D still recompute the full endpoint.
            repeated = prepared.execute(data["density_spin"])
            np.testing.assert_array_equal(repeated["potential"], actual["potential"])
            changed = prepared.execute(0.9 * data["density_spin"])
            assert changed["energy"] != actual["energy"]
            with pytest.raises(AttributeError):
                prepared.tile_program = None
            assert prepared.tile_program is description


def test_packed_scalar_rows_feed_coefficients_without_gradient_repack(
    native_factory: typing.Any, monkeypatch: typing.Any
) -> None:
    program = native_factory("PBE")
    npoint = 11
    scalar = np.zeros((7, npoint))
    scalar[0] = 0.7
    scalar[1] = 0.4
    scalar[2] = 0.02
    scalar[3] = 0.005
    scalar[4] = 0.03

    rows = program.scalar_values_packed(scalar, donate_coefficients=True)
    assert program.metadata["packed_layouts"]["scalar"] == {
        "variables": ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"),
        "outputs": 8,
        "root_rows": (0, 1, 2, 3, 4, 5),
    }
    assert rows.feature_gradient.shape == (7, npoint)
    assert rows.feature_gradient.flags.c_contiguous
    assert np.count_nonzero(rows.feature_gradient[5:]) == 0
    for index in range(5):
        assert np.shares_memory(rows[(index,)], rows.feature_gradient[index])

    v = program._gradient(rows, npoint)
    assert np.shares_memory(v, rows.feature_gradient)
    gradient = np.arange(2 * npoint * 3, dtype=np.float64).reshape(2, npoint, 3)
    gradient *= 1e-3
    expected = coefficient_program("polarized", "gga").evaluate(gradient, np.asarray(v))
    monkeypatch.setattr(
        program.coefficients.function,
        "evaluate",
        lambda *_args, **_kwargs: pytest.fail("packed path used generic stacked ABI"),
    )
    owner = rows.coefficient_output_owner
    assert owner is not None
    owner_pointer = owner.ctypes.data
    actual = program.coefficients.evaluate(gradient, v, output_owner=owner)
    assert actual.owner is owner
    assert actual.owner.ctypes.data == owner_pointer
    assert not actual.owner.flags.writeable
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
        assert np.shares_memory(actual[key], actual.owner)
        assert not actual[key].flags.writeable


def test_pbe_packed_donation_matches_independent_complete_tile(
    native_factory: typing.Any,
) -> None:
    rng = np.random.default_rng(831)
    npoint, nao = 13, 6
    jets = rng.normal(size=(4, npoint, nao)) * 0.25
    density = []
    for _ in range(2):
        factor = rng.normal(size=(nao, 3)) * 0.15
        density.append(factor @ factor.T)
    density = np.asarray(density)
    weights = rng.uniform(0.01, 0.2, size=npoint)

    reference = ContractionProgram(
        functional("PBE", spin="polarized"), "potential"
    ).evaluate(jets, density, weights)
    program = native_factory("PBE")
    block = density_feature_block(
        jets, density, ingredients=("rho", "gradient", "sigma")
    )
    rows = program.scalar_values_packed(block.scalar, donate_coefficients=True)
    owner = rows.coefficient_output_owner
    assert owner is not None
    owner_pointer = owner.ctypes.data

    actual = program.potential_from_rows(jets, block.features(), weights, rows)

    assert owner.ctypes.data == owner_pointer
    assert not owner.flags.writeable
    np.testing.assert_array_equal(actual["energy"], reference["energy"])
    np.testing.assert_array_equal(actual["electrons"], reference["electrons"])
    np.testing.assert_array_equal(actual["potential"], reference["potential"])


def test_feature_block_matches_public_features_and_native_scalar_consumes_owner(
    native_factory: typing.Any, monkeypatch: typing.Any
) -> None:
    meta, data, grid = load_integration_fixture("h2")
    program = native_factory("PBE")
    with NativeAO(**basis_arguments(meta)) as basis:
        jets = basis.evaluate(grid.points[:7], order=1)
        block = density_feature_block(
            jets,
            data["density_spin"],
            ingredients=("rho", "gradient", "sigma"),
        )
        public = density_features(
            jets,
            data["density_spin"],
            ingredients=("rho", "gradient", "sigma"),
        )
        features = block.features()
        for key in public:
            np.testing.assert_array_equal(features[key], public[key])
        assert np.shares_memory(features["rho"], block.scalar)
        assert np.shares_memory(features["sigma"], block.scalar)
        assert not block.scalar.flags.writeable

        seen = []
        assert program._packed_scalar is not None
        original = program._packed_scalar.evaluate_matrix

        def evaluate_matrix(values: typing.Any) -> typing.Any:
            seen.append(values)
            return original(values)

        monkeypatch.setattr(program._packed_scalar, "evaluate_matrix", evaluate_matrix)
        direct = program.scalar_values_packed(block.scalar)
        baseline = program.scalar_values(features)
        for key in baseline:
            np.testing.assert_array_equal(direct[key], baseline[key])
        assert seen
        assert np.shares_memory(seen[0], block.scalar)


def test_other_native_observables_do_not_advertise_programir(
    native_factory: typing.Any,
) -> None:
    meta, data, grid = load_integration_fixture("h2")
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        PreparedXCContractions(
            native_factory("PBE", observable="energy"), basis, grid, tile_points=7
        ) as prepared,
    ):
        assert prepared.tile_program is None
        prepared.execute(data["density_spin"])
        assert "tile_program_identity" not in prepared.statistics
