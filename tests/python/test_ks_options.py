"""Explicit KS composition/grid identity, native snapshots and budget shapes."""

import os
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from vibeqc import (
    Atom,
    Calculator,
    GridSpec,
    KsOptions,
    ResourceBudget,
    estimate_ks_resources,
)
from vibeqc.ks import resolve_ks_options
from vibeqc_compiler.dft.grid import MolecularGrid
from vibeqc_compiler.method import MethodSpec, SemilocalXCPrimitive, resolve_method
from vibeqc_compiler.xc.spec import functional

H2 = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
CUSTOM = GridSpec(
    radial_points=32,
    angular_polar=10,
    angular_azimuth=20,
    element_radii=((1, 1.3),),
    partition_iterations=2,
)


@pytest.fixture(params=("cpu", "cuda"))
def device(request):
    if request.param == "cuda" and os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    return request.param


def test_functional_composition_resolves_only_required_ingredients():
    lda = resolve_ks_options("lda-rks")
    pbe = resolve_ks_options("pbe-uks")
    assert lda.ao_order == 0 and lda.functional.ingredients == ("rho",)
    assert pbe.ao_order == 1 and pbe.functional.ingredients == ("rho", "sigma")
    assert lda.functional.spin == "unpolarized" and pbe.functional.spin == "polarized"
    assert lda.method_ir.identifier == "LDA_XC_PW"
    assert pbe.method_ir.identifier == "PBE"
    assert isinstance(pbe.method_ir.primitives[0], SemilocalXCPrimitive)
    assert pbe.functional.identity == functional("PBE", spin="polarized").identity
    assert (
        lda.functional.identity == functional("LDA_XC_PW", spin="unpolarized").identity
    )
    assert pbe.to_payload()["method_ir_identity"] == pbe.method_ir.identity
    assert "tau" not in pbe.to_payload()["required_ingredients"]
    assert pbe.to_payload()["scalar_derivative_order"] == 1
    assert pbe.to_payload()["scf_domain"].endswith("pbe-spin-c2-1e-18")


def test_method_ir_capability_gate_rejects_non_semilocal_graph(monkeypatch):
    import vibeqc.ks as ks_module

    hybrid = resolve_method("PBE0", spin="unpolarized")
    monkeypatch.setattr(ks_module, "resolve_method", lambda *args, **kwargs: hybrid)
    with pytest.raises(NotImplementedError, match="exactly one supported semilocal"):
        ks_module.resolve_ks_options("pbe-rks")


@pytest.mark.parametrize(
    "method, identifier, spin",
    (
        ("lda-rks", "LDA_XC_PW", "unpolarized"),
        ("lda-uks", "LDA_XC_PW", "polarized"),
        ("pbe-rks", "PBE", "unpolarized"),
        ("pbe-uks", "PBE", "polarized"),
    ),
)
def test_method_ir_projection_preserves_catalog_identity(method, identifier, spin):
    options = resolve_ks_options(method)
    expected = functional(identifier, spin=spin)
    assert options.functional.identity == expected.identity
    assert options.method_ir.spin == spin
    assert options.method_ir.primitives[0].semantic_payload() == (
        SemilocalXCPrimitive(expected).semantic_payload()
    )
    assert options.to_payload()["method_ir"] == options.method_ir.to_payload()


@pytest.mark.parametrize(
    "graph",
    (
        resolve_method(
            MethodSpec(
                "PBE",
                (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
            )
        ),
        resolve_method(MethodSpec("PBE", (("GGA_X_PBE", Fraction(1)),))),
        resolve_method(
            MethodSpec(
                "PBE",
                (
                    ("GGA_X_PBE", Fraction(1)),
                    ("GGA_C_PBE", Fraction(1)),
                    ("LDA_X", Fraction(1)),
                ),
            )
        ),
        resolve_method("LDA_XC_PW"),
        resolve_method("PBE", spin="polarized"),
    ),
    ids=("coefficient", "missing-component", "extra-component", "family", "spin"),
)
@pytest.mark.parametrize("consumer", ("options", "calculator", "resources"))
def test_method_ir_mismatch_fails_before_native_load(monkeypatch, graph, consumer):
    import vibeqc.ks as ks_module
    from vibeqc import _native

    def forbidden(*args, **kwargs):
        pytest.fail("inconsistent MethodIR reached native loading")

    # The common resolver is authoritative for composition, but native selectors
    # still execute fixed LDA/PBE mathematics. Any drift must fail at the boundary.
    monkeypatch.setattr(ks_module, "resolve_method", lambda *args, **kwargs: graph)
    monkeypatch.setattr(_native, "load_library", forbidden)
    with pytest.raises(RuntimeError, match="disagrees with native KS XC catalog"):
        if consumer == "options":
            resolve_ks_options("pbe-rks")
        elif consumer == "calculator":
            Calculator(method="pbe-rks")
        else:
            estimate_ks_resources([H2], method="pbe-rks")


def test_method_ir_projection_treats_identifiers_as_descriptive(monkeypatch):
    import vibeqc.ks as ks_module

    graph = replace(resolve_method("PBE"), identifier="descriptive-pbe-alias")
    monkeypatch.setattr(ks_module, "resolve_method", lambda *args, **kwargs: graph)
    options = resolve_ks_options("pbe-rks")
    assert options.method_ir is graph
    assert options.functional.identity == functional("PBE", spin="unpolarized").identity


def test_unsupported_compositions_and_policy_fail_before_native_load(monkeypatch):
    from vibeqc import _native

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported KS model reached native loading")

    monkeypatch.setattr(_native, "load_library", forbidden)
    pbe = functional("PBE", spin="unpolarized")
    for spec in (
        functional("LDA_XC_PW", spin="unpolarized"),
        replace(pbe, spin="polarized"),
        replace(pbe, exact_exchange=Fraction(1, 4)),
        replace(
            pbe, components=(("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1)))
        ),
    ):
        with pytest.raises(NotImplementedError, match="composition/spin"):
            Calculator(method="pbe-rks", ks_options=KsOptions(functional=spec))
    with pytest.raises(NotImplementedError, match="domain"):
        KsOptions(scf_domain="unversioned-clipping")
    with pytest.raises(ValueError, match="RKS/UKS"):
        Calculator(method="rhf", ks_options=KsOptions())


def test_custom_model_changes_plan_identity_without_materializing_grid(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run materialized a scientific array")

    monkeypatch.setattr(MolecularGrid, "__init__", forbidden)
    monkeypatch.setattr(np, "empty", forbidden)
    default = estimate_ks_resources([H2])
    custom = estimate_ks_resources(
        [H2], ks_options=KsOptions(grid=CUSTOM, tile_points=31)
    )
    assert default.identity != custom.identity
    assert custom.resident_bytes["host"] < default.resident_bytes["host"]
    changed_radius = estimate_ks_resources(
        [H2],
        ks_options=KsOptions(
            grid=replace(CUSTOM, element_radii=((1, 1.7),)), tile_points=31
        ),
    )
    assert custom.identity != changed_radius.identity
    assert custom.peak_bytes == changed_radius.peak_bytes


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"))
def test_custom_native_grid_matches_independent_scf_and_budget(method, device):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    pyscf.lib.num_threads(1)
    uks = method.endswith("uks")
    charge, multiplicity = (1, 2) if uks else (0, 1)
    options = KsOptions(grid=CUSTOM, tile_points=31)
    calculator = Calculator(
        method=method,
        device=device,
        ks_options=options,
        resource_budget=ResourceBudget(),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    atoms = tuple(Atom.from_value(a) for a in H2)
    basis = {"H0": [], "H1": []}
    for shell in calculator._shells_for_atoms(atoms):
        basis[f"H{shell.atom_index}"].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )
    mol = gto.M(
        atom=[(f"H{i}", atom.position) for i, atom in enumerate(atoms)],
        basis=basis,
        charge=charge,
        spin=multiplicity - 1,
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    grid = MolecularGrid(
        atoms, spec=CUSTOM, charge=charge, multiplicity=multiplicity
    ).explicit()
    native = calculator.singlepoint(atoms, charge=charge, multiplicity=multiplicity)
    assert native.converged and native.physical_residual_rms < 1e-9
    assert native.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    assert native.ks_diagnostic.grid_points == len(grid.weights)
    assert native.ks_diagnostic.tile_points == options.tile_points
    assert native.ks_diagnostic.ao_order == calculator.ks_options.ao_order
    for guess in ("minao", "1e"):
        reference = dft.UKS(mol) if uks else dft.RKS(mol)
        reference.xc = "PBE" if method.startswith("pbe") else "LDA_X,LDA_C_PW"
        reference.grids.coords = np.array(grid.points)
        reference.grids.weights = np.array(grid.weights)
        reference.small_rho_cutoff = 0
        reference.conv_tol, reference.conv_tol_grad = 1e-13, 1e-9
        reference.kernel(dm0=reference.get_init_guess(key=guess))
        assert reference.converged
        assert abs(native.energy - reference.e_tot) < 1e-8
    default = Calculator(method=method, device=device).singlepoint(
        atoms, charge=charge, multiplicity=multiplicity
    )
    assert abs(default.energy - native.energy) > 1e-8
    # Tile shape changes scheduling and capacity, while this fixed-grid energy
    # remains the same discrete model up to FP64 reduction order.
    other = Calculator(
        method=method, device=device, ks_options=replace(options, tile_points=128)
    )
    assert (
        abs(
            other.singlepoint(atoms, charge=charge, multiplicity=multiplicity).energy
            - native.energy
        )
        < 1e-9
    )
    with calculator.prepare_batch(
        [atoms, atoms], charges=[charge] * 2, multiplicities=[multiplicity] * 2
    ) as batch:
        batch.execute(strict=True)
        assert batch.execute(strict=True).items[0].warm_start_used
        if device == "cuda":
            ledger = batch.resource_diagnostics["preparation"]["device_ledger"]
            assert ledger["live_bytes"] == batch.resource_plan.resident_bytes["device"]
        # Replacing the model must fail before any old density/DIIS can run.
        calculator._ks_options = resolve_ks_options(
            method, replace(options, grid=GridSpec())
        )
        with pytest.raises(RuntimeError, match="model identity changed"):
            batch.execute(strict=True)


def test_older_native_library_cannot_silently_ignore_custom_options(monkeypatch):
    from vibeqc import _native

    library = _native.load_library(device="cpu")
    monkeypatch.setattr(library, "vibeqc_ks_options_version", None)
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: library)
    with pytest.raises(NotImplementedError, match="model options"):
        Calculator(method="pbe-rks", ks_options=KsOptions(grid=CUSTOM))
    assert Calculator(method="pbe-rks").singlepoint(H2).converged
