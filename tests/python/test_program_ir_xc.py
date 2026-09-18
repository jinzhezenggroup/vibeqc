"""Actual dense native CPU XC tile ownership, not a modeled allocator speedup."""

import shutil
import weakref
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.program import ProgramIR
from vibeqc_compiler.common.resources import MAX_BYTES
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture
from vibeqc_compiler.xc.native import NativeContractionProgram
from vibeqc_compiler.xc.prepared import PreparedXCContractions
from vibeqc_compiler.xc.program_ir import fixed_density_tile_program


def describe(name="PBE", spin="polarized", **changes):
    options = {
        "nao": 12,
        "tile_points": 7,
        "basis_bytes": 100,
        "grid_bytes": 1000,
        "basis_identity": "basis-v1",
        "native_identity": "native-v1",
    }
    options.update(changes)
    return fixed_density_tile_program(
        ContractionProgram(functional(name, spin=spin)).contract, **options
    )


@pytest.mark.parametrize("name,jets", [("LDA_XC_PW", 1), ("PBE", 4)])
@pytest.mark.parametrize("spin,nspin", [("unpolarized", 1), ("polarized", 2)])
def test_fixed_density_description_is_bound_to_real_contract(name, jets, spin, nspin):
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
def test_invalid_tile_boundaries(changes):
    with pytest.raises(ValueError):
        describe(**changes)


def test_provider_and_shape_changes_invalidate_graph_identity():
    for changes in (
        {"nao": 13},
        {"tile_points": 8},
        {"basis_identity": "moved"},
        {"native_identity": "other"},
    ):
        assert describe(**changes).identity != describe().identity


def test_non_potential_contracts_remain_outside_phase_a():
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
def native_factory(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native XC requires c++")
    cache = tmp_path_factory.mktemp("programir-native")
    programs = {}

    def make(name, spin="polarized", observable="potential"):
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
    native_factory, monkeypatch, name, case
):
    meta, data, grid = load_integration_fixture(case)
    program = native_factory(name)
    refs, observed = [], []
    with NativeAO(**basis_arguments(meta)) as basis:
        old_ao, old_xc = basis.evaluate, program.evaluate

        def collocate(*args, **kwargs):
            # Weak references do not themselves extend an allocation lifetime.
            observed.append(tuple(label for label, ref in refs if ref() is not None))
            jets = old_ao(*args, **kwargs)
            refs.append(("jets", weakref.ref(jets)))
            return jets

        def contract(*args, **kwargs):
            result = old_xc(*args, **kwargs)
            for key in ("potential", "electrons"):
                refs.append((key, weakref.ref(result[key])))
            return result

        monkeypatch.setattr(basis, "evaluate", collocate)
        monkeypatch.setattr(program, "evaluate", contract)
        with PreparedXCContractions(program, basis, grid, tile_points=7) as prepared:
            description = prepared.tile_program
            actual = prepared.execute(data["density_spin"])
            assert len(observed) > 1
            assert observed == [()] * len(observed)
            assert all(ref() is None for _, ref in refs)
            assert prepared.statistics["tile_program_identity"] == description.identity
            assert prepared.statistics["tile_boundary_releases"] == ("jets",)
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


def test_other_native_observables_do_not_advertise_programir(native_factory):
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
