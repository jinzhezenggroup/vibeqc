"""Complete diagnostic gradients: independent analytic and reconverged oracles.

PySCF is used only here. Its own libcint, libxc, SCF and analytic Becke response
evaluate the same input basis and atomic quadrature, without calling any of the
generated derivative graphs under test. Public DFT force capabilities stay off.
"""

import ctypes as ct
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions, method_capabilities
from vibeqc._dft_gradient import StationaryDerivativeContract, StationaryKsState
from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic
from vibeqc_compiler.dft import NativeAO

ATOMS = [("O", (0.1, -0.1, 0.0)), ("H", (0.1, 0.2, 1.7)), ("H", (1.6, -0.2, -0.5))]
GRID = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)


def calculator(method, **kwargs):
    return Calculator(
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        **kwargs,
    )


def independent_gradient(basis, state, method):
    """PySCF full-response RKS with native atomic quadrature, independent algebra."""
    from pyscf import dft, gto, lib
    from pyscf.data.elements import ELEMENTS
    from pyscf.grad import rks

    lib.num_threads(1)
    labels = [f"{ELEMENTS[a.atomic_number]}{i}" for i, a in enumerate(basis.atoms)]
    shells = {label: [] for label in labels}
    for shell in basis.shells:
        shells[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )
    mol = gto.M(
        atom=[(label, a.position) for label, a in zip(labels, basis.atoms)],
        basis=shells,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "PBE" if method == "pbe-rks" else "LDA_X,LDA_C_PW"
    mf.grids.coords = np.array(state.grid.points)
    mf.grids.weights = np.array(state.grid.weights)
    mf.grids.radii_adjust = None
    owners = np.asarray(state.grid.owners)
    tab = {
        mol.atom_symbol(a): (
            np.array(state.grid.points[owners == a] - mol.atom_coord(a)),
            np.array(state._source.atomic_weights[owners == a]),
        )
        for a in range(mol.natm)
    }
    mf.grids.gen_atomic_grids = lambda *args, **kwargs: tab
    mf.small_rho_cutoff = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 150
    mf.kernel()
    assert mf.converged
    grad = mf.nuc_grad_method()
    grad.grid_response = True
    total = grad.kernel()
    d, w = mf.make_rdm1(), grad.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    h, s, j = grad.hcore_generator(mol), grad.get_ovlp(mol), grad.get_j(mol, d)
    grid_and_weight, vxc = rks.get_vxc_full_response(
        mf._numint, mol, mf.grids, mf.xc, d
    )
    components = {
        name: np.zeros_like(total)
        for name in ("one_electron", "coulomb", "overlap_pulay", "xc_ao", "xc_weight")
    }
    for a, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        components["one_electron"][a] = np.einsum("xij,ij->x", h(a), d)
        components["coulomb"][a] = 2 * np.einsum("xij,ij->x", j[:, p0:p1], d[p0:p1])
        components["overlap_pulay"][a] = -2 * np.einsum(
            "xij,ij->x", s[:, p0:p1], w[p0:p1]
        )
        components["xc_ao"][a] = 2 * np.einsum("xij,ij->x", vxc[:, p0:p1], d[p0:p1])
    for points, _, dw in rks.grids_response_cc(mf.grids):
        ao = mf._numint.eval_ao(mol, points, deriv=1)
        kind = "GGA" if method == "pbe-rks" else "LDA"
        rho = mf._numint.eval_rho(mol, ao if kind == "GGA" else ao[0], d, xctype=kind)
        exc = mf._numint.eval_xc(mf.xc, rho, deriv=0)[0]
        density = rho[0] if kind == "GGA" else rho
        components["xc_weight"] += np.einsum("p,p,axp->ax", exc, density, dw)
    components["xc_grid"] = grid_and_weight - components["xc_weight"]
    components["nuclear"] = grad.grad_nuc()
    np.testing.assert_allclose(sum(components.values()), total, atol=2e-12, rtol=0)
    return mf.e_tot, total, components


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks"])
def test_complete_asymmetric_water_analytic_and_reconverged_fd(method, record_property):
    pytest.importorskip("pyscf", reason="independent analytic reference requires PySCF")
    calc = calculator(method)
    with calc.prepare_batch([ATOMS]) as batch, NativeAO(ATOMS) as basis:
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        result = complete_rks_gradient_diagnostic(
            state, basis, cache=".cache/b22-tests"
        )
        reference_energy, reference, components = independent_gradient(
            basis, state, method
        )
        assert energy == pytest.approx(reference_energy, abs=2e-9)
        for name, value in result.components.items():
            np.testing.assert_allclose(
                value, components[name], atol=1e-7, rtol=0, err_msg=name
            )
        np.testing.assert_allclose(result.gradient, reference, atol=1e-7, rtol=0)
        record_property(
            "analytic_max_error", float(np.max(np.abs(result.gradient - reference)))
        )
        record_property(
            "source_max_error",
            float(
                max(
                    np.max(np.abs(value - components[name]))
                    for name, value in result.components.items()
                )
            ),
        )
        assert result.work["ordered_quartets"] == basis.nao**4
        assert result.work["grid_directional_points"] == 9 * len(state.grid.points)
        # All seven signed sources are nontrivial here. An omitted/reversed
        # grid, Pulay or nuclear term cannot pass by molecular symmetry.
        for name in ("xc_grid", "xc_weight", "overlap_pulay", "nuclear"):
            assert np.max(np.abs(result.components[name])) > 1e-4
            assert (
                np.max(
                    np.abs(result.gradient - 2 * result.components[name] - reference)
                )
                > 1e-4
            )
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=2e-10, rtol=0)

        coordinates = np.array([a[1] for a in ATOMS])
        estimates = []
        for step in (1e-3, 3e-4, 1e-4):
            fd = np.empty_like(coordinates)
            for a, axis in product_coordinates():
                direction = np.zeros_like(coordinates)
                direction[a, axis] = step
                energies = [
                    calc.singlepoint(
                        [
                            (label, pos)
                            for (label, _), pos in zip(
                                ATOMS, coordinates + sign * direction
                            )
                        ]
                    ).energy
                    for sign in (1, -1)
                ]
                fd[a, axis] = (energies[0] - energies[1]) / (2 * step)
            estimates.append(fd)
        # A multistep stable region and the full componentwise force gate;
        # every displaced density, grid, Fock and SCF solve is recomputed.
        np.testing.assert_allclose(estimates[-1], estimates[-2], atol=1e-6, rtol=0)
        np.testing.assert_allclose(result.gradient, estimates[-1], atol=1e-6, rtol=0)
        richardson = (9 * estimates[-1] - estimates[-2]) / 8
        np.testing.assert_allclose(result.gradient, richardson, atol=1e-7, rtol=0)
        record_property(
            "finite_difference_max_error",
            float(np.max(np.abs(result.gradient - estimates[-1]))),
        )
        record_property(
            "richardson_max_error", float(np.max(np.abs(result.gradient - richardson)))
        )

        shift, order = np.array([0.25, -0.37, 0.18]), [2, 0, 1]
        moved = [(ATOMS[i][0], coordinates[i] + shift) for i in order]
        with calc.prepare_batch([moved]) as other, NativeAO(moved) as moved_basis:
            other.execute(strict=True)
            moved_state = StationaryKsState.from_native(other, moved_basis)
            moved_result = complete_rks_gradient_diagnostic(
                moved_state,
                moved_basis,
                cache=".cache/b22-tests",
                tile_points=137,
                integral_terms=17,
                primitive_tile=29,
            )
            np.testing.assert_allclose(
                moved_result.gradient, result.gradient[order], atol=1e-8, rtol=0
            )
        # Warm replay must generate a new lease and reproduce the same gradient.
        replay = batch.execute(strict=True).items[0]
        assert replay.warm_start_used
        with pytest.raises(ValueError, match="stale"):
            complete_rks_gradient_diagnostic(state, basis, cache=".cache/b22-tests")
        current = StationaryKsState.from_native(batch, basis)
        warm = complete_rks_gradient_diagnostic(
            current, basis, cache=".cache/b22-tests"
        )
        np.testing.assert_allclose(warm.gradient, result.gradient, atol=1e-9, rtol=0)


def product_coordinates():
    return ((a, axis) for a in range(3) for axis in range(3))


def test_failure_isolation_native_malformed_geometry_and_detached_state():
    calc = calculator("pbe-rks")
    with calc.prepare_batch([ATOMS, ATOMS]) as batch, NativeAO(ATOMS) as basis:
        batch.execute(strict=True)
        old = StationaryKsState.from_native(batch, basis)
        contract = StationaryDerivativeContract(old.identity)
        with pytest.raises(ValueError, match="current native.*snapshot"):
            contract.validate(replace(old, _source=None))
        with pytest.raises(ValueError):
            contract.validate(
                replace(old, weighted_density=old.weighted_density * 1.01)
            )
        result = batch.execute(coordinates=[[0.0], None], strict=False)
        assert not result.items[0].succeeded and result.items[1].succeeded
        with pytest.raises(ValueError, match="stale"):
            contract.validate(old)
        live = StationaryKsState.from_native(batch, basis, index=1)
        # Bypass Python shape checks: malformed native invocation revokes all
        # prior results before descriptor validation, including the good neighbor.
        lib = batch._library
        from vibeqc import _native

        bad = _native.BatchInputDescriptor()
        status = lib.vibeqc_batch_execute(batch._batch, ct.byref(bad), 1, None, 0)
        assert status != 0
        with pytest.raises(ValueError, match="stale"):
            StationaryDerivativeContract(live.identity).validate(live)
        changed = np.array([a[1] for a in ATOMS])
        changed[1, 0] += 0.02
        batch.execute(coordinates=[changed, None], strict=True)
        with pytest.raises(ValueError, match="basis/overlap source"):
            StationaryKsState.from_native(batch, basis)
        moved = [(a[0], p) for a, p in zip(ATOMS, changed)]
        with NativeAO(moved) as moved_basis:
            new = StationaryKsState.from_native(batch, moved_basis)
            assert new.identity.owner != old.identity.owner
        for method in ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"):
            assert method_capabilities(method).supported_properties == frozenset(
                {"energy"}
            )
        with pytest.raises(ValueError, match="does not support properties"):
            calc.singlepoint(ATOMS, properties=("energy", "forces"))


@pytest.mark.parametrize("fail_publication", [False, True])
def test_compiler_source_publication_is_atomic(tmp_path, monkeypatch, fail_publication):
    from vibeqc import _stationary_cpu as module

    path = tmp_path / "source.cpp"
    path.write_text("old complete source")
    replace_file = module.os.replace

    def check_then_publish(temporary, destination):
        assert path.read_text() == "old complete source"
        assert temporary.read_text() == "new complete source"
        if fail_publication:
            raise OSError("injected publication failure")
        replace_file(temporary, destination)

    monkeypatch.setattr(module.os, "replace", check_then_publish)
    if fail_publication:
        with pytest.raises(OSError, match="injected publication"):
            module._publish_source(path, "new complete source")
        assert path.read_text() == "old complete source"
    else:
        module._publish_source(path, "new complete source")
        assert path.read_text() == "new complete source"
    assert sorted(tmp_path.iterdir()) == [path]


def test_cpu_diagnostic_bounds_and_late_provider_failure(tmp_path, monkeypatch):
    from pathlib import Path

    from vibeqc import _stationary_cpu as module
    from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter

    atoms = [("H", (0.1, 0.2, -0.6)), ("H", (0.2, -0.1, 0.8))]
    calc = calculator("pbe-rks")
    compiler = CppCompilerAdapter(Path("c++"))
    with calc.prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
        batch.execute(strict=True)
        state = StationaryKsState.from_native(batch, basis)
        for name, value in (
            ("tile_points", 0),
            ("tile_points", True),
            ("integral_terms", 129),
            ("primitive_tile", 4097),
        ):
            with pytest.raises(ValueError, match=name):
                complete_rks_gradient_diagnostic(
                    state, basis, cache=tmp_path, **{name: value}
                )
        with pytest.raises(TypeError, match="compiler adapter"):
            complete_rks_gradient_diagnostic(
                state, basis, cache=tmp_path, compiler=object()
            )
        original = module._PrimitiveExecutor._run
        calls = 0

        def fail_late(self, kind, count):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise ArithmeticError("injected late primitive failure")
            return original(self, kind, count)

        with monkeypatch.context() as patch:
            patch.setattr(module._PrimitiveExecutor, "_run", fail_late)
            with pytest.raises(ArithmeticError, match="late primitive"):
                complete_rks_gradient_diagnostic(
                    state, basis, cache=tmp_path, compiler=compiler
                )
        assert calls == 3
        assert StationaryDerivativeContract(state.identity).validate(state) is state
        good = complete_rks_gradient_diagnostic(
            state, basis, cache=tmp_path, compiler=compiler
        )
        assert np.isfinite(good.gradient).all()
        with pytest.raises(ValueError):
            good.gradient.flags.writeable = True
    with (
        NativeAO(ATOMS, basis="def2-svp") as d_basis,
        pytest.raises(NotImplementedError, match="s/p"),
    ):
        module._PrimitiveExecutor(d_basis, tmp_path, 128, compiler)


def test_fresh_process_gradient_has_no_external_oracle_dependency(tmp_path):
    import subprocess
    import sys

    code = r"""
import importlib.abc
import sys
import numpy as np
class BlockOracle(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyscf', 'gpu4pyscf', 'cupy'}:
            raise AssertionError('unexpected external oracle: ' + fullname)
sys.meta_path.insert(0, BlockOracle())
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc_compiler.dft import NativeAO
from vibeqc._dft_gradient import StationaryKsState
from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic
atoms = [('H', (.1, .2, -.6)), ('H', (.2, -.1, .8))]
calc = Calculator(method='pbe-rks', device='cpu',
    ks_options=KsOptions(grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)),
    energy_tolerance=1e-12, density_tolerance=1e-10)
with calc.prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
    batch.execute(strict=True)
    state = StationaryKsState.from_native(batch, basis)
    result = complete_rks_gradient_diagnostic(state, basis, cache=sys.argv[1])
    assert np.isfinite(result.gradient).all()
    np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=1e-10)
assert not any(name.split('.')[0] in {'pyscf', 'gpu4pyscf', 'cupy'} for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
