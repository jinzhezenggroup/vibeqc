"""Native compact XC parity, bounded tile execution and fixed-mask contracts."""

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.dft.spatial import SpatialPolicy
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture as fixture
from vibeqc_compiler.xc.native import NativeContractionProgram
from vibeqc_compiler.xc.prepared import PreparedXCContractions


@pytest.fixture(scope="module")
def native_factory(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native XC requires a C++ compiler")
    cache = tmp_path_factory.mktemp("xc-native")
    programs = {}

    def build(name, observable, spin="polarized"):
        key = name, observable, spin
        if key not in programs:
            programs[key] = NativeContractionProgram(
                functional(name, spin=spin),
                observable,
                compiler=CppCompilerAdapter(Path(compiler)),
                cache=cache,
            )
        return programs[key]

    return build


def compare(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        if name == "geometry":
            for field in ("centers", "points", "weights"):
                np.testing.assert_allclose(
                    getattr(actual[name], field),
                    getattr(expected[name], field),
                    atol=1e-11,
                    rtol=1e-10,
                )
        else:
            np.testing.assert_allclose(
                actual[name], expected[name], atol=1e-11, rtol=1e-10
            )


@pytest.mark.parametrize("case", ["h2", "f_spherical"])
@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize("observable", ["energy", "potential", "response", "geometry"])
def test_native_complete_endpoint_tiles_and_two_budgets(
    native_factory, case, name, observable
):
    meta, data, grid = fixture(case)
    native = native_factory(name, observable)
    diagnostic = ContractionProgram(functional(name), observable)
    density = data["density_spin"]
    options = {"delta_density": 0.03 * density} if observable == "response" else {}
    with NativeAO(**basis_arguments(meta)) as basis:
        whole_options = dict(options)
        if observable == "geometry":
            ao_atoms = np.repeat(
                [s.atom_index for s in basis.shells],
                [
                    2 * s.angular_momentum + 1
                    if basis.representation == "real_spherical"
                    else (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2
                    for s in basis.shells
                ],
            )
            whole_options.update(ao_atoms=ao_atoms, natom=basis.natom)
        expected = diagnostic.evaluate(
            basis.evaluate(grid.points, diagnostic.contract.ao_order),
            density,
            grid.weights,
            **whole_options,
        )
        for tile, budget in ((7, 2 << 20), (19, 4 << 20)):
            with PreparedXCContractions(
                native,
                basis,
                grid,
                tile_points=tile,
                resource_budget=ResourceBudget(host_bytes=budget),
            ) as prepared:
                actual = prepared.execute(density, **options)
                compare(actual, expected)
                assert prepared.resource_plan.peak_bytes["host"] <= budget
                assert (
                    prepared.statistics["tiles"]
                    == (len(grid.weights) + tile - 1) // tile
                )
                if observable == "potential":
                    np.testing.assert_allclose(
                        actual["energy"],
                        data[f"{name}_spin_energy"][0],
                        atol=1e-11,
                        rtol=1e-10,
                    )
                    np.testing.assert_allclose(
                        actual["potential"],
                        data[f"{name}_spin_potential"],
                        atol=1e-11,
                        rtol=1e-10,
                    )
                assert native.artifact.metadata["compile_seconds"] > 0
            with pytest.raises(RuntimeError, match="closed"):
                prepared.execute(density, **options)


def test_native_budget_preflight_precedes_collocation(native_factory, monkeypatch):
    meta, _, grid = fixture("h2")
    with NativeAO(**basis_arguments(meta)) as basis:
        monkeypatch.setattr(
            basis,
            "evaluate",
            lambda *a, **k: pytest.fail("allocated jets before budget preflight"),
        )
        with pytest.raises(MemoryError, match="no supported plan fits"):
            PreparedXCContractions(
                native_factory("PBE", "potential"),
                basis,
                grid,
                resource_budget=ResourceBudget(host_bytes=1),
            )


def test_native_spatial_mask_matches_independent_zeroed_collocation(native_factory):
    meta, data, grid = fixture("h2")
    program = native_factory("PBE", "potential")
    density = data["density_spin"]
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        PreparedSpatialGrid(
            basis,
            grid,
            policy=SpatialPolicy(
                region_points=5, screening="absolute_ao_jet", cutoff=1e-4
            ),
            tile_points=3,
        ) as spatial,
        PreparedXCContractions(
            program,
            basis,
            grid,
            spatial=spatial,
            resource_budget=ResourceBudget(host_bytes=4 << 20),
        ) as prepared,
    ):
        result = prepared.execute(density)
        expected = {
            "energy": 0.0,
            "potential": np.zeros_like(density),
            "electrons": np.zeros(2),
        }
        oracle = ContractionProgram(functional("PBE"))
        for task in spatial.tasks.tasks:
            jets = basis.evaluate(grid.points[task.point_ids], 1).copy()
            omitted = np.ones(basis.nao, dtype=bool)
            omitted[task.ao_ids] = False
            jets[:, :, omitted] = 0
            tile = oracle.evaluate(jets, density, grid.weights[task.point_ids])
            for key in expected:
                expected[key] += tile[key]
        compare(result, expected)
        spatial.reconfigure(basis, replace(grid, points=grid.points + 0.01))
        with pytest.raises(ValueError, match="stale"):
            prepared.execute(density)


@pytest.mark.parametrize("observable", ["energy", "potential", "response", "geometry"])
def test_empty_native_spatial_masks_have_zero_native_calls(
    native_factory, monkeypatch, observable
):
    from vibeqc_compiler.dft.ao import jet_indices

    meta, data, grid = fixture("h2")
    grid = replace(grid, points=grid.points + 100)
    native = native_factory("PBE", observable)
    monkeypatch.setattr(
        native._scalar,
        "evaluate",
        lambda *a: pytest.fail("empty mask reached scalar code"),
    )
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        PreparedSpatialGrid(
            basis,
            grid,
            policy=SpatialPolicy(
                screening="absolute_ao_jet", cutoff=1e-4, derivatives=jet_indices(2)
            ),
            tile_points=7,
        ) as spatial,
        PreparedXCContractions(
            native,
            basis,
            grid,
            spatial=spatial,
            resource_budget=ResourceBudget(host_bytes=4 << 20),
        ) as prepared,
    ):
        assert all(len(t.ao_ids) == 0 for t in spatial.tasks.tasks)
        options = (
            {"delta_density": data["density_spin"]} if observable == "response" else {}
        )
        result = prepared.execute(data["density_spin"], **options)
        assert result["energy"] == 0
        assert prepared.statistics["tiles"] > 0
        assert all(
            prepared.statistics[k] == 0
            for k in (
                "scalar_calls",
                "point_coefficient_calls",
                "ao_pullback_calls",
                "matrix_assembly_products",
            )
        )
        for value in result.values():
            if hasattr(value, "centers"):
                for field in ("centers", "points", "weights"):
                    assert np.count_nonzero(getattr(value, field)) == 0
            else:
                assert np.count_nonzero(value) == 0


def test_native_full_spin_solver_adapter_and_source_staleness(native_factory):
    from tools.vibeqc_response.xc import FixedDensityXCDerivativeKernel

    meta, data, grid = fixture("h2")
    spec = functional("PBE")
    d = data["density_spin"]
    direction = d * np.array([0.03, -0.02])[:, None, None]
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        PreparedXCContractions(
            native_factory("PBE", "response"), basis, grid, tile_points=7
        ) as prepared,
    ):
        native = FixedDensityXCDerivativeKernel(spec, basis, grid, d, prepared=prepared)
        diagnostic = FixedDensityXCDerivativeKernel(
            spec, basis, grid, d, tile_points=11
        )
        actual = native.apply_spin(direction)
        np.testing.assert_allclose(
            actual, diagnostic.apply_spin(direction), atol=1e-11, rtol=1e-10
        )
        np.testing.assert_allclose(
            native.apply(direction), actual.mean(axis=0), atol=1e-12
        )
        with pytest.raises(ValueError, match="mismatch"):
            FixedDensityXCDerivativeKernel(
                functional("LDA_XC_PW"), basis, grid, d, prepared=prepared
            )
        # An alternate native metadata identity cannot reuse an existing owner.
        original = prepared.program.metadata
        try:
            prepared.program.metadata = {**original, "source_sha256": "0" * 64}
            with pytest.raises(ValueError, match="stale"):
                prepared.execute(d, delta_density=direction)
        finally:
            prepared.program.metadata = original


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize("observable", ["energy", "potential", "response", "geometry"])
def test_unpolarized_native_endpoints(native_factory, name, observable):
    meta, data, grid = fixture("f_cartesian")
    density = data["density_total"]
    program = native_factory(name, observable, "unpolarized")
    diagnostic = ContractionProgram(functional(name, spin="unpolarized"), observable)
    options = {"delta_density": 0.03 * density} if observable == "response" else {}
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        PreparedXCContractions(program, basis, grid, tile_points=11) as prepared,
    ):
        expected_options = dict(options)
        if observable == "geometry":
            expected_options.update(
                ao_atoms=np.repeat(
                    [s.atom_index for s in basis.shells],
                    [
                        (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2
                        for s in basis.shells
                    ],
                ),
                natom=basis.natom,
            )
        compare(
            prepared.execute(density, **options),
            diagnostic.evaluate(
                basis.evaluate(grid.points, diagnostic.contract.ao_order),
                density,
                grid.weights,
                **expected_options,
            ),
        )


def test_native_energy_rejects_quadrature_overflow(native_factory):
    jets, density, weights = np.ones((1, 1, 1)), np.array([[1e9]]), np.array([1e298])
    for consumer in (
        native_factory("LDA_XC_PW", "energy"),
        ContractionProgram(functional("LDA_XC_PW"), "energy"),
    ):
        with pytest.raises(ArithmeticError, match="integrated XC energy"):
            consumer.evaluate(jets, density, weights)


@pytest.mark.parametrize("observable", ["energy", "potential", "response", "geometry"])
def test_nonempty_strict_spatial_subsets_preserve_all_observables(
    native_factory, observable
):
    from vibeqc_compiler.dft import ExplicitGrid
    from vibeqc_compiler.dft.ao import jet_indices

    from tools.benchmark_xc_contractions import diagnostic

    meta, data, _ = fixture("f_cartesian")
    args = basis_arguments(meta)
    args["atoms"] = [(2, (0.0, 0.0, 0.0)), (1, (12.0, 0.0, 0.0))]
    rng = np.random.default_rng(2367)
    points = np.concatenate(
        [
            np.asarray(center) + 0.2 * rng.normal(size=(8, 3))
            for _, center in args["atoms"]
        ]
    )
    grid = ExplicitGrid(
        points, np.full(16, 0.03), (0,) * 8 + (1,) * 8, {"case": "strict-local"}
    )
    density, direction = data["density_spin"], data["density_spin"] * 0.03
    with (
        NativeAO(**args) as basis,
        PreparedSpatialGrid(
            basis,
            grid,
            policy=SpatialPolicy(
                screening="absolute_ao_jet",
                cutoff=1e-4,
                region_points=4,
                derivatives=jet_indices(2),
            ),
            tile_points=3,
        ) as spatial,
        PreparedXCContractions(
            native_factory("PBE", observable), basis, grid, spatial=spatial
        ) as prepared,
    ):
        assert all(0 < len(t.ao_ids) < basis.nao for t in spatial.tasks.tasks)
        options = {"delta_density": direction} if observable == "response" else {}
        expected = diagnostic(
            ContractionProgram(functional("PBE"), observable),
            basis,
            grid,
            density,
            3,
            spatial,
            direction,
        )
        compare(prepared.execute(density, **options), expected)


def test_concurrent_native_source_publication_and_matching_cache_hits(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    from vibeqc_compiler.common.provenance import canonical_hash

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native XC requires a C++ compiler")
    original_write = Path.write_text

    def no_live_source_rewrite(path, *args, **kwargs):
        # Direct writes expose a truncated compiler input to concurrent readers.
        # Logs and unrelated files still use their normal writers.
        if path.name == "xc.cpp":
            pytest.fail("rewrote the live native compiler input")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", no_live_source_rewrite)

    def build(_):
        return NativeContractionProgram(
            functional("PBE"),
            "potential",
            compiler=CppCompilerAdapter(Path(compiler)),
            cache=tmp_path,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        programs = list(pool.map(build, range(4)))
    assert len({p.artifact.metadata["key"] for p in programs}) == 1
    for program in programs:
        assert (
            program.artifact.metadata["identity"]["source"]
            == program.metadata["source_sha256"]
        )
    source = tmp_path / "source" / canonical_hash(programs[0].metadata) / "xc.cpp"
    before = source.stat()
    build(None)
    after = source.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
    assert list(source.parent.glob(".xc-*")) == []
    original_write(source, "truncated source")
    with pytest.raises(ValueError, match="native XC source identity mismatch"):
        build(None)
