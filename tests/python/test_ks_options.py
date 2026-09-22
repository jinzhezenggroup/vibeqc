"""Explicit KS composition/grid identity, native snapshots and budget shapes."""

import json
import os
import typing
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc import (
    Atom,
    Calculator,
    GridPolicy,
    GridSpec,
    KsOptions,
    ResourceBudget,
    estimate_ks_resources,
)
from vibeqc.ks import native_ks_options, resolve_ks_options
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft.grid import (
    GRID_POLICY_RADII_SOURCE,
    GRID_POLICY_UPSTREAM_REVISION,
    MolecularGrid,
    grid_policy_provenance,
)
from vibeqc_compiler.method import (
    MethodSpec,
    SemilocalXCPrimitive,
    original_nonlocal_correlation,
    resolve_method,
)
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
def device(request: typing.Any) -> typing.Any:
    if request.param == "cuda" and os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    return request.param


def test_functional_composition_resolves_only_required_ingredients() -> None:
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


def test_production_grid_policy_is_resolved_element_aware_and_versioned() -> None:
    lda = resolve_ks_options("lda-rks")
    pbe = resolve_ks_options("pbe-rks")
    tight = resolve_ks_options("pbe-rks", KsOptions(grid_accuracy="tight"))
    derivative = GridPolicy().resolve("pbe-rks", derivative_order=1)
    derivative_profile = GridPolicy().profile("pbe-rks", derivative_order=1)

    assert lda.grid.version == pbe.grid.version == tight.grid.version == 2
    assert (
        lda.grid.radial_points,
        lda.grid.angular_polar,
        lda.grid.angular_azimuth,
    ) == (
        54,
        16,
        32,
    )
    assert (
        pbe.grid.radial_points,
        pbe.grid.angular_polar,
        pbe.grid.angular_azimuth,
    ) == (
        54,
        16,
        32,
    )
    assert tight.grid == derivative
    assert derivative_profile.pruning == "none"
    assert derivative_profile.screening == "none"
    assert derivative_profile.topology == "atom-radial-polar-azimuth"
    assert (
        tight.grid.radial_points,
        tight.grid.angular_polar,
        tight.grid.angular_azimuth,
    ) == (
        72,
        24,
        48,
    )
    radii = dict(pbe.grid.element_radii)
    assert len(radii) == 86
    assert radii[1] != 1.0
    assert radii[26] > 0.0  # representative transition metal, Fe
    assert radii[54] > 0.0  # representative heavier element, Xe
    assert lda.identity != pbe.identity
    assert pbe.identity != tight.identity
    assert pbe.to_payload()["grid"]["version"] == 2
    assert pbe.to_payload()["grid_provenance"] == GridPolicy().provenance
    assert pbe.to_payload()["grid_provenance"] == grid_policy_provenance(pbe.grid)
    assert GridSpec(**pbe.to_payload()["grid"]) == pbe.grid
    changed_provenance = json.loads(json.dumps(pbe.to_payload()))
    changed_provenance["grid_provenance"]["upstream_revision"] = "different"
    assert canonical_hash(changed_provenance) != pbe.identity

    native = native_ks_options(pbe)
    assert native.grid_version == 2
    assert native.element_radius_count == 119
    assert native.element_radii[26] == pytest.approx(radii[26], rel=0, abs=0)
    assert native.element_radii[87] == 0.0

    custom_points = replace(pbe.grid, radial_points=pbe.grid.radial_points + 1)
    custom_radii = replace(
        pbe.grid,
        element_radii=tuple(
            (z, radius * 1.01 if z == 1 else radius)
            for z, radius in pbe.grid.element_radii
        ),
    )
    for custom in (custom_points, custom_radii):
        provenance = grid_policy_provenance(custom)
        assert provenance == {"policy_version": 2, "contract": "explicit-grid-v2"}
        resolved_custom = resolve_ks_options("pbe-rks", KsOptions(grid=custom))
        assert resolved_custom.to_payload()["grid_provenance"] == provenance
        assert resolved_custom.identity != pbe.identity


def test_production_grid_radii_match_pinned_provenance_and_unknowns_fail_closed() -> (
    None
):
    root = Path(__file__).resolve().parents[2]
    model = json.loads(
        (
            root / "upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1.json"
        ).read_text()
    )
    source = [item["covalent_radius_bohr"] for item in model["elements"]]
    policy = GridPolicy()
    spec = policy.resolve("lda-rks")
    assert GRID_POLICY_RADII_SOURCE.endswith(
        "92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111"
    )
    assert GRID_POLICY_UPSTREAM_REVISION == "2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3"
    assert policy.provenance["radii_source"] == GRID_POLICY_RADII_SOURCE
    assert [r for _, r in spec.element_radii] == source

    # Historical v1 remains an exact one-Bohr reference fallback.
    legacy = MolecularGrid([Atom(87, (0.0, 0.0, 0.0))], spec=GridSpec())
    assert legacy.resolved_radii == (1.0,)
    # Production v2 never silently turns an unsourced element into one Bohr.
    with pytest.raises(ValueError, match="no sourced radius.*87"):
        MolecularGrid([Atom(87, (0.0, 0.0, 0.0))], spec=spec)


def test_grid_policy_capability_boundaries_fail_closed() -> None:
    policy = GridPolicy()
    for method in (
        "r2scan-rks",
        "r2scan-uks",
        "scan-rks",
        "scan-uks",
        "vv10-rks",
        "vv10-uks",
        "pbe0-rks",
        "pbe0-uks",
    ):
        with pytest.raises(NotImplementedError, match="qualified only"):
            policy.resolve(method)
    with pytest.raises(NotImplementedError, match="orders 0 and 1"):
        policy.resolve("pbe-rks", derivative_order=2)
    with pytest.raises(ValueError, match="accuracy"):
        GridPolicy("turbo").resolve("pbe-rks")


@pytest.mark.parametrize(
    ("replacement", "rejection"),
    (
        ("PBE0", "disagrees with native KS selector"),
        ("PBE-D4(BJ-EEQ-ATM)", "one semilocal XC primitive"),
    ),
)
def test_named_pbe_selector_cannot_silently_change_to_hybrid(
    monkeypatch: typing.Any,
    replacement: str,
    rejection: str,
) -> None:
    import vibeqc.ks as ks_module

    hybrid = resolve_method(replacement, spin="unpolarized")
    monkeypatch.setattr(ks_module, "resolve_method", lambda *args, **kwargs: hybrid)
    with pytest.raises(RuntimeError, match=rejection):
        ks_module.resolve_ks_options("pbe-rks")


@pytest.mark.parametrize(
    "method,spin,coefficients",
    (
        ("pbe0-rks", "unpolarized", (0.75, 1.0, -0.125)),
        ("pbe0-uks", "polarized", (0.75, 1.0, -0.25)),
    ),
)
def test_pbe0_named_selector_resolves_common_methodir_composition(
    method: typing.Any, spin: typing.Any, coefficients: typing.Any
) -> None:
    options = resolve_ks_options(method, KsOptions(grid=CUSTOM))
    assert options.method_ir.identifier == "PBE0"
    assert options.method_ir.spin == spin
    assert options.coefficients == coefficients
    assert options.requires_composition_v2
    assert len(options.method_ir.primitives) == 2


@pytest.mark.parametrize(
    "method, identifier, spin",
    (
        ("lda-rks", "LDA_XC_PW", "unpolarized"),
        ("lda-uks", "LDA_XC_PW", "polarized"),
        ("pbe-rks", "PBE", "unpolarized"),
        ("pbe-uks", "PBE", "polarized"),
    ),
)
def test_method_ir_projection_preserves_catalog_identity(
    method: typing.Any, identifier: typing.Any, spin: typing.Any
) -> None:
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
def test_method_ir_mismatch_fails_before_native_load(
    monkeypatch: typing.Any, graph: typing.Any, consumer: typing.Any
) -> None:
    import vibeqc.ks as ks_module
    from vibeqc import _native

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
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


def test_method_ir_projection_treats_identifiers_as_descriptive(
    monkeypatch: typing.Any,
) -> None:
    import vibeqc.ks as ks_module

    graph = replace(resolve_method("PBE"), identifier="descriptive-pbe-alias")
    monkeypatch.setattr(ks_module, "resolve_method", lambda *args, **kwargs: graph)
    options = resolve_ks_options("pbe-rks")
    assert options.method_ir is graph
    assert options.functional.identity == functional("PBE", spin="unpolarized").identity


def test_unsupported_compositions_and_policy_fail_before_native_load(
    monkeypatch: typing.Any,
) -> None:
    from vibeqc import _native

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
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


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks"))
def test_unqualified_hybrid_default_grid_fails_closed(method: str) -> None:
    with pytest.raises(NotImplementedError, match="explicit GridSpec"):
        resolve_ks_options(method)


def test_ks_options_v2_suffix_preserves_v1_prefix_and_pbe0_coefficients() -> None:
    from vibeqc import _native

    pure = resolve_ks_options("pbe-rks")
    hybrid = resolve_ks_options("pbe0-rks", KsOptions(grid=CUSTOM))
    old = native_ks_options(pure, version=1)
    new = native_ks_options(hybrid, version=2)
    assert old.struct_size == _native.KsOptionsDescriptor.composition_version.offset
    assert new.struct_size > old.struct_size
    assert new.composition_version == 1
    assert (
        new.semilocal_exchange_scale,
        new.semilocal_correlation_scale,
        new.fock_exchange_coefficient,
    ) == (0.75, 1.0, -0.125)
    with pytest.raises(NotImplementedError, match="cannot serialize composition"):
        native_ks_options(hybrid, version=1)


def test_ks_options_v3_schedule_suffix_preserves_older_prefixes() -> None:
    from vibeqc import _native

    fused = resolve_ks_options("pbe-rks")
    unfused = resolve_ks_options("pbe-rks", KsOptions(xc_schedule="host_unfused"))
    v1 = native_ks_options(fused, version=1)
    v2 = native_ks_options(fused, version=2)
    v3 = native_ks_options(unfused, version=3)

    assert v1.struct_size == _native.KsOptionsDescriptor.composition_version.offset
    assert v2.struct_size == _native.KsOptionsDescriptor.xc_execution_schedule.offset
    assert v3.struct_size > v2.struct_size
    assert v3.xc_execution_schedule == _native.XC_EXECUTION_HOST_UNFUSED
    assert fused.identity != unfused.identity
    assert fused.to_payload()["xc_schedule"] == "device_fused"
    assert unfused.to_payload()["xc_schedule"] == "host_unfused"
    for version in (1, 2):
        with pytest.raises(NotImplementedError, match="cannot serialize execution"):
            native_ks_options(unfused, version=version)
    with pytest.raises(ValueError, match="XC schedule"):
        KsOptions(xc_schedule="unknown")


def test_ks_options_v4_preserves_v1_v2_v3_prefixes_and_methodir_plan() -> None:
    import ctypes

    from vibeqc import _native

    pure = resolve_ks_options("pbe-rks")
    hybrid = resolve_ks_options("pbe0-rks", KsOptions(grid=CUSTOM))
    old = native_ks_options(pure, version=1)
    legacy_composition = native_ks_options(hybrid, version=2)
    new = native_ks_options(hybrid, version=4)
    assert old.struct_size == _native.KsOptionsDescriptor.composition_version.offset
    assert (
        legacy_composition.struct_size
        == _native.KsOptionsDescriptor.xc_execution_schedule.offset
    )
    legacy_schedule = native_ks_options(hybrid, version=3)
    assert (
        legacy_schedule.struct_size
        == _native.KsOptionsDescriptor.execution_plan_version.offset
    )
    assert legacy_schedule.execution_plan_version == 0

    class LegacyKsOptionsV3(ctypes.Structure):
        _fields_ = _native.KsOptionsDescriptor._fields_[
            : next(
                i
                for i, (name, _) in enumerate(_native.KsOptionsDescriptor._fields_)
                if name == "reserved_v3_padding"
            )
        ]

    assert ctypes.sizeof(LegacyKsOptionsV3) == legacy_schedule.struct_size
    for name, _ in LegacyKsOptionsV3._fields_:
        assert (
            getattr(LegacyKsOptionsV3, name).offset
            == getattr(_native.KsOptionsDescriptor, name).offset
        )
    assert (
        new.struct_size
        == _native.KsOptionsDescriptor.nonlocal_correlation_version.offset
    )
    current = native_ks_options(hybrid, version=5)
    assert current.struct_size == ctypes.sizeof(_native.KsOptionsDescriptor)
    assert current.nonlocal_correlation_version == 0
    assert new.composition_version == 1
    assert (
        new.semilocal_exchange_scale,
        new.semilocal_correlation_scale,
        new.fock_exchange_coefficient,
    ) == (0.75, 1.0, -0.125)
    assert (
        new.execution_plan_version,
        new.spin_channels,
        new.semilocal_family,
    ) == (1, 1, 1)


@pytest.mark.parametrize(
    "method, options, expected",
    (
        ("lda-rks", None, (1, 0)),
        ("pbe-uks", None, (2, 1)),
        ("pbe-d4-rks", None, (1, 1)),
        ("pbe0-uks", KsOptions(grid=CUSTOM), (2, 1)),
        ("r2scan-rks", None, (1, 2)),
    ),
)
def test_native_v4_execution_selector_is_derived_from_methodir(
    method: str, options: KsOptions | None, expected: tuple[int, int]
) -> None:
    resolved = resolve_ks_options(method, options)
    native = native_ks_options(resolved, version=4)
    assert native.execution_plan_version == 1
    assert (native.spin_channels, native.semilocal_family) == expected


def test_custom_model_changes_plan_identity_without_materializing_grid(
    monkeypatch: typing.Any,
) -> None:
    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
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
def test_custom_native_grid_matches_independent_scf_and_budget(
    method: typing.Any, device: typing.Any
) -> None:
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


def test_older_native_library_cannot_claim_pbe0_without_composition_v2(
    monkeypatch: typing.Any,
) -> None:
    from vibeqc import _native

    library = _native.load_library(device="cpu")

    class VersionOne:
        argtypes = None
        restype = None

        def __call__(self) -> typing.Any:
            return 1

    monkeypatch.setattr(library, "vibeqc_ks_options_version", VersionOne())
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: library)
    with pytest.raises(NotImplementedError, match="composition options v2"):
        Calculator(method="pbe0-rks", ks_options=KsOptions(grid=CUSTOM))


def test_older_native_library_cannot_silently_ignore_custom_options(
    monkeypatch: typing.Any,
) -> None:
    from vibeqc import _native

    library = _native.load_library(device="cpu")
    monkeypatch.setattr(library, "vibeqc_ks_options_version", None)
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: library)
    with pytest.raises(NotImplementedError, match="model options"):
        Calculator(method="pbe-rks", ks_options=KsOptions(grid=CUSTOM))
    assert Calculator(method="pbe-rks").singlepoint(H2).converged


@pytest.mark.parametrize(
    "method",
    (
        "lda-rks",
        "lda-uks",
        "pbe-rks",
        "pbe-uks",
        "pbe0-rks",
        "pbe0-uks",
        "r2scan-rks",
        "r2scan-uks",
    ),
)
def test_resolved_ks_options_preserve_catalog_identity(method: str) -> None:
    first = resolve_ks_options(method, KsOptions(grid=CUSTOM, tile_points=31))
    second = resolve_ks_options(method, first)
    assert second == first
    assert second.identity == first.identity
    assert second.method_ir is first.method_ir
    assert second.functional is first.functional


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks"))
def test_named_hybrid_resource_planning_preserves_explicit_grid(method: str) -> None:
    options = KsOptions(grid=CUSTOM, tile_points=31)
    resolved = resolve_ks_options(method, options)
    assert resolved.requires_composition_v2
    assert (
        estimate_ks_resources([H2], method=method, ks_options=options).identity
        == estimate_ks_resources([H2], method=method, ks_options=resolved).identity
    )


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_budgeted_custom_hybrid_preserves_resolved_methodir(spin: str) -> None:
    method = "pbe-rks" if spin == "unpolarized" else "pbe-uks"
    graph = resolve_method(
        MethodSpec(
            "PBE50-budgeted",
            (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 2),
        ),
        spin=spin,
    )
    options = KsOptions(composition=graph, grid=CUSTOM, tile_points=31)
    resolved = resolve_ks_options(method, options)
    again = resolve_ks_options(method, resolved)
    assert again.identity == resolved.identity
    assert again.method_ir is graph
    assert again.coefficients == (0.5, 1.0, -0.25 if spin == "unpolarized" else -0.5)
    assert (
        estimate_ks_resources([H2], method=method, ks_options=options).identity
        == estimate_ks_resources([H2], method=method, ks_options=again).identity
    )
    charge, multiplicity = (0, 1) if spin == "unpolarized" else (1, 2)
    ordinary = Calculator(method=method, ks_options=options).singlepoint(
        H2, charge=charge, multiplicity=multiplicity
    )
    budgeted = Calculator(
        method=method, ks_options=options, resource_budget=ResourceBudget()
    ).singlepoint(H2, charge=charge, multiplicity=multiplicity)
    assert ordinary.converged and budgeted.converged
    assert budgeted.energy == pytest.approx(ordinary.energy, abs=2e-12)


def test_native_v5_serializes_nonlocal_primitive_without_named_method_branch() -> None:
    graph = resolve_method(
        MethodSpec(
            "PBE+VV10-v5",
            (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
            nonlocal_correlation=original_nonlocal_correlation("vv10"),
        ),
        spin="unpolarized",
    )
    options = resolve_ks_options(
        "pbe-rks",
        KsOptions(
            composition=graph,
            grid=GridSpec(radial_points=3, angular_polar=2, angular_azimuth=4),
            tile_points=16,
            nonlocal_memory_budget_bytes=1 << 20,
        ),
    )
    assert options.requires_nonlocal_v5
    with pytest.raises(NotImplementedError, match="nonlocal correlation v5"):
        native_ks_options(options, version=4)
    native = native_ks_options(options, version=5)
    assert native.nonlocal_correlation_version == 1
    assert native.nonlocal_variant == 1
    assert native.nonlocal_b == pytest.approx(5.9)
    assert native.nonlocal_c == pytest.approx(0.0093)
    assert native.nonlocal_coefficient == pytest.approx(1.0)
    assert native.nonlocal_maximum_bytes == 1 << 20
