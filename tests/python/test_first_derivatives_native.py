"""Generated first-component CPU execution: independent values and failure gates."""

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Primitive, Shell
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.integral.first_derivatives_execute import (
    FirstDerivativeEvaluator,
    FirstDerivativeShellEvaluator,
    compile_first_derivative,
    compile_first_derivative_shell,
    first_derivative_component_tiles,
    first_derivative_shell_identity,
)
from vibeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
)
from vibeqc_compiler.integral.weight_pullback import normalized_radial_primitives
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir

from tools.vibeqc_posthf.sources import NativeSource


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    executable = shutil.which("c++")
    if executable is None:
        pytest.skip("CPU C++ compiler required")
    return compile_first_derivative(
        build_weighted_eri_ir((0, 0, 0, 0)),
        CppCompilerAdapter(Path(executable)),
        tmp_path_factory.mktemp("first-native"),
        component_indices=(0,),
    )


@pytest.fixture(scope="module")
def full_shell_artifact(tmp_path_factory):
    executable = shutil.which("c++")
    if executable is None:
        pytest.skip("CPU C++ compiler required")
    return compile_first_derivative_shell(
        build_weighted_eri_ir((2, 1, 0, 0)),
        CppCompilerAdapter(Path(executable)),
        tmp_path_factory.mktemp("first-full-shell"),
        tile_size=5,
    )


def test_raw_eri_component_matches_independent_native_source(artifact):
    coordinates = (
        (0.13, -0.31, 0.24),
        (-0.43, 0.27, 0.51),
        (0.68, -0.14, -0.22),
        (-0.21, 0.48, -0.63),
    )
    exponents = (0.6, 0.8, 1.1, 0.9)
    shells = tuple(Shell(i, 0, (Primitive(a, 1.0),)) for i, a in enumerate(exponents))
    primitives = tuple(normalized_radial_primitives(0, ((a, 1.0),)) for a in exponents)
    actual = FirstDerivativeEvaluator(artifact).contract(primitives, coordinates)[0]
    with NativeSource([(1, x) for x in coordinates], basis=shells) as source:
        value = source._read("four_center_eri", (0, 1, 2, 3), (1, 1, 1, 1)).item()
        derivatives = source.integral_derivatives()["eri"][:, 0, 1, 2, 3]
    np.testing.assert_allclose(actual[0], value, atol=3e-13)
    np.testing.assert_allclose(actual[1:], derivatives, atol=2e-12)
    np.testing.assert_allclose(actual[1:].reshape(4, 3).sum(axis=0), 0, atol=2e-13)


def test_full_shell_eri_tiles_match_independent_native_oracle(full_shell_artifact):
    from vibeqc_compiler.integral.weight_pullback import normalized_cartesian_components

    assert [len(tile.component_indices) for tile in full_shell_artifact.tiles] == [
        5,
        5,
        5,
        3,
    ]
    executor = FirstDerivativeShellEvaluator(
        full_shell_artifact, record_capacity=2, budget_bytes=8 << 20
    )
    cases = (
        (
            np.array(
                [
                    [0.13, -0.31, 0.24],
                    [-0.43, 0.27, 0.51],
                    [0.68, -0.14, -0.22],
                    [-0.21, 0.48, -0.63],
                ]
            ),
            (
                ((0.60, 1.0), (0.22, 0.35)),
                ((0.80, 1.0),),
                ((1.10, 1.0),),
                ((0.90, 1.0),),
            ),
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.31, -0.27, 0.19],
                    [0.72, -0.11, 0.43],
                    [0.7200001, -0.1100002, 0.4300003],
                ]
            ),
            (
                ((8.0, 1.0),),
                ((0.08, 1.0),),
                ((1.7, 1.0),),
                ((0.25, 1.0),),
            ),
        ),
    )
    angular = full_shell_artifact.integral.signature.angular
    factors = np.array(
        [
            scale
            for _, scale in normalized_cartesian_components(
                angular, np.ones(full_shell_artifact.integral.signature.component_shape)
            )
        ]
    )
    for coordinates, primitive_specs in cases:
        primitives = tuple(
            normalized_radial_primitives(l, specs)
            for l, specs in zip(angular, primitive_specs, strict=True)
        )
        actual = executor.contract(primitives, coordinates)
        actual *= factors[:, None]
        shells = tuple(
            Shell(
                atom,
                l,
                tuple(
                    Primitive(exponent, coefficient) for exponent, coefficient in specs
                ),
            )
            for atom, (l, specs) in enumerate(
                zip(angular, primitive_specs, strict=True)
            )
        )
        with NativeSource([(1, xyz) for xyz in coordinates], basis=shells) as source:
            sizes = tuple(source.shell_sizes)
            offsets = tuple(int(x) for x in np.cumsum((0, *sizes[:-1]), dtype=np.int64))
            value = source._read("four_center_eri", offsets, sizes).reshape(-1)
            slices = tuple(
                slice(offset, offset + size)
                for offset, size in zip(offsets, sizes, strict=True)
            )
            derivative = (
                source.integral_derivatives()["eri"][(slice(None), *slices)]
                .reshape(12, -1)
                .T
            )
        np.testing.assert_allclose(actual[:, 0], value, atol=3e-11, rtol=2e-10)
        np.testing.assert_allclose(actual[:, 1:], derivative, atol=5e-11, rtol=2e-10)
        np.testing.assert_allclose(
            actual[:, 1:].reshape(-1, 4, 3).sum(axis=1),
            0.0,
            atol=3e-11,
        )


def test_full_shell_metadata_and_budget_are_bounded(full_shell_artifact):
    ir = full_shell_artifact.integral
    assert first_derivative_component_tiles(ir, tile_size=5) == tuple(
        tile.component_indices for tile in full_shell_artifact.tiles
    )
    assert full_shell_artifact.program_identity == first_derivative_shell_identity(
        ir, tile_size=5
    )
    ffff_tiles = first_derivative_component_tiles(build_weighted_eri_ir((3, 3, 3, 3)))
    assert len(ffff_tiles) == 157 and len(ffff_tiles[-1]) == 16
    assert tuple(index for tile in ffff_tiles for index in tile) == tuple(range(10_000))
    for invalid in (0, 65, True):
        with pytest.raises(ValueError, match="tile size"):
            first_derivative_component_tiles(ir, tile_size=invalid)
    with pytest.raises(ValueError, match="budget"):
        FirstDerivativeShellEvaluator(full_shell_artifact, budget_bytes=1)
    with pytest.raises(ValueError, match="tile identity"):
        FirstDerivativeShellEvaluator(
            replace(full_shell_artifact, tiles=full_shell_artifact.tiles[::-1]),
            budget_bytes=8 << 20,
        )
    with pytest.raises(ValueError, match="tile identity"):
        FirstDerivativeShellEvaluator(
            replace(full_shell_artifact, program_identity="unrelated"),
            budget_bytes=8 << 20,
        )


def test_first_component_rejects_bad_metadata_and_budget(artifact):
    with pytest.raises(TypeError, match="compiled first"):
        FirstDerivativeEvaluator(object())
    with pytest.raises(ValueError, match="budget"):
        FirstDerivativeEvaluator(artifact, budget_bytes=1)
    with pytest.raises(ValueError):
        FirstDerivativeEvaluator(replace(artifact, component_indices=(1,)))
    with pytest.raises(ValueError, match="identity"):
        FirstDerivativeEvaluator(replace(artifact, program_identity="unrelated"))


def test_failed_late_chunk_does_not_poison_next_call(artifact):
    executor = FirstDerivativeEvaluator(artifact, record_capacity=1)
    good = (((0.7, 1.0),),) * 4
    centers = np.arange(12).reshape(4, 3) * 0.07
    expected = executor.contract(good, centers)
    with pytest.raises(ValueError, match="must be real"):
        executor.contract((((0.7, 1.0j),), *good[1:]), centers)
    malformed = (((0.7, 1.0), (-1.0, 1.0)), *good[1:])
    with pytest.raises(ValueError):
        executor.contract(malformed, centers)
    np.testing.assert_array_equal(executor.contract(good, centers), expected)
    with pytest.raises(ValueError, match="centers"):
        executor.contract(good, np.full((4, 3), np.nan))
    # Check transactional native publication independently of Python admission.
    output = np.full(executor.shape, 173.0)
    records = np.array([[0.7] * 4 + list(centers.ravel()) + [1.0]])
    pointer = records.ctypes.data
    assert (
        executor.run(pointer, 1, executor.stride - 1, output.ctypes.data, output.size)
        == 1
    )
    np.testing.assert_array_equal(output, 173.0)
    records[0, 0] = -1.0
    assert (
        executor.run(pointer, 1, executor.stride, output.ctypes.data, output.size) == 1
    )
    np.testing.assert_array_equal(output, 173.0)
    assert executor.run(None, 0, executor.stride, output.ctypes.data, output.size) == 0
    np.testing.assert_array_equal(output, 0.0)


def test_one_electron_component_matches_closed_form_overlap(tmp_path):
    ir = build_one_electron_derivative_ir("overlap", (0, 0))
    art = compile_first_derivative(
        ir,
        CppCompilerAdapter(Path(shutil.which("c++") or "c++")),
        tmp_path,
        component_indices=(0,),
    )
    a, b = 0.7, 1.2
    centers = np.array([[0.2, -0.3, 0.1], [-0.1, 0.7, -0.8]])
    delta = centers[0] - centers[1]
    mu = a * b / (a + b)
    value = (np.pi / (a + b)) ** 1.5 * np.exp(-mu * np.dot(delta, delta))
    actual = FirstDerivativeEvaluator(art).contract(
        (((a, 1.0),), ((b, 1.0),)), centers
    )[0]
    expected = np.r_[value, -2 * mu * delta * value, 2 * mu * delta * value]
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("angular", (1, 2, 3))
def test_higher_components_and_coincident_slots_match_native_oracle(tmp_path, angular):
    from vibeqc_compiler.integral.weight_pullback import normalized_cartesian_components

    ir = build_weighted_eri_ir((angular, 0, 0, 0))
    indices = tuple(range(ir.signature.component_count))
    art = compile_first_derivative(
        ir,
        CppCompilerAdapter(Path(shutil.which("c++") or "c++")),
        tmp_path,
        component_indices=indices,
    )
    coords = np.array([[0.13, -0.31, 0.24], [-0.43, 0.27, 0.51]])
    shells = (
        Shell(0, angular, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(0.8, 1.0),)),
    )
    radial = normalized_radial_primitives(angular, ((0.6, 1.0),))
    scalar = normalized_radial_primitives(0, ((0.8, 1.0),))
    actual = FirstDerivativeEvaluator(art).contract(
        (radial, scalar, scalar, scalar), coords[[0, 1, 1, 1]]
    )
    factors = np.array(
        [
            scale
            for _, scale in normalized_cartesian_components(
                ir.signature.angular, np.ones(ir.signature.component_shape)
            )
        ]
    )
    actual *= factors[:, None]
    with NativeSource([(1, c) for c in coords], basis=shells) as source:
        nfirst = source.shell_sizes[0]
        value = source._read(
            "four_center_eri", (0, nfirst, nfirst, nfirst), (nfirst, 1, 1, 1)
        ).reshape(-1)
        derivative = source.integral_derivatives()["eri"][
            :, :nfirst, nfirst, nfirst, nfirst
        ]
    atom_derivative = np.concatenate(
        (actual[:, 1:4], actual[:, 4:].reshape(-1, 3, 3).sum(axis=1)), axis=1
    )
    np.testing.assert_allclose(actual[:, 0], value, atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(atom_derivative.T, derivative, atol=2e-11, rtol=1e-10)


@pytest.mark.parametrize("angular", (2, 3))
def test_generated_stv_includes_both_basis_motions_and_operator_nucleus(
    tmp_path, angular
):
    from vibeqc_compiler.integral.weight_pullback import normalized_cartesian_components

    coords = np.array([[0.13, -0.31, 0.24], [-0.43, 0.27, 0.51]])
    shells = (
        Shell(0, angular, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(0.8, 1.0),)),
    )
    prims = (
        normalized_radial_primitives(angular, ((0.6, 1.0),)),
        normalized_radial_primitives(0, ((0.8, 1.0),)),
    )
    compiler = CppCompilerAdapter(Path(shutil.which("c++") or "c++"))
    count = (angular + 1) * (angular + 2) // 2
    scales = np.array(
        [
            weight
            for _, weight in normalized_cartesian_components(
                (angular, 0), np.ones((count, 1))
            )
        ]
    )
    sums = {key: np.zeros((count, 7)) for key in ("overlap", "hcore")}
    specs = [
        ("overlap", "overlap", None),
        ("kinetic", "hcore", None),
        ("nuclear_attraction", "hcore", 0),
        ("nuclear_attraction", "hcore", 1),
    ]
    for family, destination, nucleus in specs:
        ir = build_one_electron_derivative_ir(family, (angular, 0))
        artifact = compile_first_derivative(
            ir, compiler, tmp_path, component_indices=tuple(range(count))
        )
        atoms = (0, 1) if nucleus is None else (0, 1, nucleus)
        values = FirstDerivativeEvaluator(artifact).contract(prims, coords[list(atoms)])
        values *= scales[:, None]
        sums[destination][:, 0] += values[:, 0]
        for center, atom in enumerate(atoms):
            sums[destination][:, 1 + 3 * atom : 1 + 3 * (atom + 1)] += values[
                :, 1 + 3 * center : 1 + 3 * (center + 1)
            ]
    with NativeSource([(1, c) for c in coords], basis=shells) as source:
        overlap, hcore = source.one_electron()
        derivatives = source.integral_derivatives()
    for key, value in (("overlap", overlap), ("hcore", hcore)):
        np.testing.assert_allclose(
            sums[key][:, 0], value[:count, count], atol=2e-12, rtol=1e-10
        )
        np.testing.assert_allclose(
            sums[key][:, 1:].T,
            derivatives[key][:, :count, count],
            atol=2e-11,
            rtol=1e-10,
        )


def test_installed_asset_layout_contains_first_runtime(tmp_path, monkeypatch):
    import tomllib
    from vibeqc_compiler.common import paths

    repository = Path(__file__).resolve().parents[2]
    package = tmp_path / "installed" / "vibeqc_compiler"
    headers = (
        "src/integrals/first_component_runtime.hpp",
        "src/integrals/eri_geometry.hpp",
        "src/integrals/range_moments.hpp",
    )
    manifest = tomllib.loads((repository / "pyproject.toml").read_text())
    included = manifest["tool"]["scikit-build"]["wheel"]["force-include"]
    for header in headers:
        destination = package / "assets" / header
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / header, destination)
        assert included[header] == "vibeqc_compiler/assets/" + header
    monkeypatch.setattr(paths, "PACKAGE", package)
    artifact = compile_first_derivative(
        build_weighted_eri_ir((0, 0, 0, 0)),
        CppCompilerAdapter(Path(shutil.which("c++") or "c++")),
        tmp_path / "cache",
        component_indices=(0,),
    )
    value = FirstDerivativeEvaluator(artifact).contract(
        (((0.7, 1.0),),) * 4, np.arange(12).reshape(4, 3) * 0.1
    )
    assert np.isfinite(value).all() and value[0, 0] > 0
