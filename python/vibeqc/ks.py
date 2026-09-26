"""Immutable native semilocal KS model and execution controls.

The compiler's FunctionalSpec supplies composition and ingredient requirements.
Native SCF has its own audited tail/spin domain, distinct from the compiler's
interior-only reference contract. Unsupported compositions fail before prepare.
"""

import math
import typing
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft.grid import (
    GridPolicy,
    GridSpec,
    checked_int,
    grid_policy_provenance,
)
from vibeqc_compiler.dft.nonlocal_policy import (
    MOLECULAR_VV10_DENSITY_POLICY,
    MOLECULAR_VV10_DENSITY_THRESHOLD,
)
from vibeqc_compiler.method import (
    D4Spec,
    DispersionCorrectionPrimitive,
    MethodIR,
    SemilocalXCPrimitive,
    compile_ks_execution_plan,
    resolve_method,
)
from vibeqc_compiler.xc._generated_split_hybrids import SPLIT_HYBRIDS
from vibeqc_compiler.xc.spec import CATALOG, FunctionalSpec, functional

from ._generated_methods import METHOD_METADATA

SCF_DOMAIN = "semilocal-scaled-v1/pbe-spin-c2-1e-18"
B3LYP_SCF_DOMAIN = "b3lyp-vwn-rpa-tail-v1/density-vacuum-1e-18"
WB97MV_SCF_DOMAIN = "libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16"
SPLIT_HYBRID_SCF_DOMAIN = "libxc-7.0/split-global-hybrid-v1"
_NATIVE_SCF_DOMAINS = frozenset(
    (SCF_DOMAIN, B3LYP_SCF_DOMAIN, WB97MV_SCF_DOMAIN, SPLIT_HYBRID_SCF_DOMAIN)
)

# Manifest aliases share exactly the canonical method/spin binding.
_NATIVE_KS_METHODS = {
    selector: (metadata["compiler_method"], metadata["spin"])
    for name, metadata in METHOD_METADATA.items()
    if metadata["provider"] == "dft"
    for selector in (name, *metadata["aliases"])
}


@dataclass(frozen=True)
class KsOptions:
    """A snapshotted native KS composition, quadrature, and bounded XC tile.

    RKS requires unpolarized MethodIR semantics; UKS requires polarized. Native
    execution accepts audited LDA/PBE/r2SCAN plus canonical B3LYP. CPU global
    hybrids consume MethodIR-owned full-range exact exchange through common J/K.
    CUDA hybrids and unsupported primitive families fail closed. A different
    model requires a new prepared owner.
    """

    functional: FunctionalSpec | None = None
    composition: MethodIR | None = None
    grid: GridSpec | None = None
    grid_accuracy: str = "standard"
    tile_points: int = 256
    xc_schedule: str = "device_fused"
    scf_domain: str = SCF_DOMAIN
    nonlocal_memory_budget_bytes: int = 256 << 20
    _method_ir: MethodIR | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.functional is not None and not isinstance(
            self.functional, FunctionalSpec
        ):
            raise TypeError("KS functional must be a FunctionalSpec")
        if self.composition is not None and not isinstance(self.composition, MethodIR):
            raise TypeError("KS composition must be a resolved MethodIR")
        if self.functional is not None and self.composition is not None:
            raise ValueError("provide KS functional or composition, not both")
        if self.grid is not None and not isinstance(self.grid, GridSpec):
            raise TypeError("KS grid must be a GridSpec or None")
        if self.grid_accuracy not in ("standard", "tight"):
            raise ValueError("KS grid_accuracy must be 'standard' or 'tight'")
        checked_int(self.tile_points, "KS XC tile points")
        if self.xc_schedule not in ("device_fused", "host_unfused"):
            raise ValueError("KS XC schedule must be 'device_fused' or 'host_unfused'")
        if self.scf_domain not in _NATIVE_SCF_DOMAINS:
            raise NotImplementedError("unsupported native KS tail/spin domain policy")
        if (
            type(self.nonlocal_memory_budget_bytes) is not int
            or not 0 < self.nonlocal_memory_budget_bytes < 2**64
        ):
            raise ValueError(
                "nonlocal_memory_budget_bytes must be a positive uint64 integer"
            )

    @property
    def method_ir(self) -> typing.Any:
        """Resolved method graph consumed by this native KS option set."""
        if self._method_ir is None:
            raise ValueError("resolve KS options against a method first")
        return self._method_ir

    @property
    def execution_plan(self) -> typing.Any:
        """Method-name-free scientific contribution plan for this KS calculation."""
        return compile_ks_execution_plan(self.method_ir)

    @property
    def coefficients(self) -> typing.Any:
        """Resolved (semilocal X, semilocal C, raw Fock K) coefficients."""
        if _is_pbe_d4_composition(self.method_ir):
            return (1.0, 1.0, 0.0)
        return ks_coefficients(self.method_ir)

    @property
    def has_nondefault_composition(self) -> bool:
        return self.coefficients != (1.0, 1.0, 0.0)

    @property
    def uses_host_xc_schedule(self) -> bool:
        return self.xc_schedule != "device_fused"

    @property
    def has_nonlocal_correlation(self) -> bool:
        return self.execution_plan.nonlocal_correlation is not None

    @property
    def has_range_exchange(self) -> bool:
        return any(
            term.operator in ("short-range", "long-range")
            for term in self.execution_plan.exchange
        )

    @property
    def ao_order(self) -> typing.Any:
        """SCF needs the potential; GGA/meta-GGA compositions need first AO jets."""
        if self.functional is None:
            raise ValueError("resolve KS options against a method first")
        return int("sigma" in self.functional.ingredients)

    def to_payload(self) -> typing.Any:
        """Keep composition provenance and the effective SCF domain explicit."""
        if self.functional is None:
            raise ValueError("resolve KS options against a method first")
        if self.grid is None:
            raise ValueError("resolve KS grid policy against a method first")
        payload = {
            "functional": self.functional.to_payload(),
            "scf_domain": self.scf_domain,
            "grid": asdict(self.grid),
            "grid_provenance": grid_policy_provenance(self.grid),
            "tile_points": self.tile_points,
            "xc_schedule": self.xc_schedule,
            "required_ao_order": self.ao_order,
            "required_ingredients": self.functional.ingredients,
            "scalar_derivative_order": 1,
            "observable": "scf-energy",
        }
        if self._method_ir is not None:
            payload["method_ir"] = self._method_ir.to_payload()
            payload["method_ir_identity"] = self._method_ir.identity
            if self.execution_plan.nonlocal_correlation is not None:
                payload["nonlocal_memory_budget_bytes"] = (
                    self.nonlocal_memory_budget_bytes
                )
        if self.scf_domain == WB97MV_SCF_DOMAIN:
            payload["nonlocal_density_policy"] = {
                "version": MOLECULAR_VV10_DENSITY_POLICY,
                "threshold": str(MOLECULAR_VV10_DENSITY_THRESHOLD),
                "active_comparison": ">=",
            }
        return payload

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True)
class ProfiledKsSelection:
    """Batch-local KS options plus exact-profile qualification provenance."""

    options: KsOptions | None
    exact_profile_match: bool = False


def _native_execution_plan(method_ir: typing.Any) -> typing.Any:
    """Project the common KS plan onto primitive lowerers available in native v2."""
    plan = compile_ks_execution_plan(method_ir)
    if plan.post_scf:
        raise NotImplementedError(
            "native KS electronic projection requires one semilocal XC primitive "
            "and supported exchange only; geometry-only post-SCF corrections "
            "require a separate qualified composition owner"
        )
    full_range = tuple(term for term in plan.exchange if term.operator == "full-range")
    range_terms = tuple(
        term for term in plan.exchange if term.operator in ("short-range", "long-range")
    )
    unsupported = tuple(
        term
        for term in plan.exchange
        if term.operator not in ("full-range", "short-range", "long-range")
    )
    if unsupported:
        raise NotImplementedError(
            "native KS execution plan requires unavailable lowerers: "
            + ", ".join(f"{term.operator}-exchange" for term in unsupported)
        )
    if full_range and range_terms:
        raise NotImplementedError(
            "native KS execution cannot mix full-range and range-separated exchange"
        )
    if len(full_range) > 1:
        raise NotImplementedError(
            "native KS accepts at most one full-range exchange contribution"
        )
    if range_terms:
        operators = {term.operator for term in range_terms}
        omegas = {term.omega for term in range_terms}
        if len(range_terms) != 2 or operators != {"short-range", "long-range"}:
            raise NotImplementedError(
                "native KS range separation requires one short- and one long-range exchange contribution"
            )
        if len(omegas) != 1:
            raise NotImplementedError(
                "native KS range-separated exchange requires one shared omega"
            )
    return plan


def _is_pbe_d4_composition(method_ir: typing.Any) -> bool:
    if not isinstance(method_ir, MethodIR) or len(method_ir.primitives) != 2:
        return False
    semilocal, correction = method_ir.primitives
    pbe = functional("PBE", spin="unpolarized")
    return (
        type(semilocal) is SemilocalXCPrimitive
        and type(correction) is DispersionCorrectionPrimitive
        and isinstance(correction.specification, D4Spec)
        and correction.specification.charge_model == "eeq2019"
        and correction.specification.reference_model == "eeq"
        and semilocal.functional.spin == "unpolarized"
        and semilocal.functional.ingredients == pbe.ingredients
        and SemilocalXCPrimitive(semilocal.functional).semantic_payload()
        == SemilocalXCPrimitive(pbe).semantic_payload()
    )


def _native_pbe_d4_semilocal(method_ir: typing.Any) -> typing.Any:
    if not _is_pbe_d4_composition(method_ir):
        raise NotImplementedError(
            "public PBE-D4 requires one PBE semilocal primitive plus one D4(BJ)-EEQ correction"
        )
    return typing.cast("SemilocalXCPrimitive", method_ir.primitives[0]).functional


def _native_semilocal(method_ir: typing.Any) -> typing.Any:
    return _native_execution_plan(method_ir).semilocal.functional


def _split_hybrid_record(method_ir: typing.Any) -> typing.Any:
    """Return the generated split-hybrid record for one exact canonical MethodIR."""

    plan = _native_execution_plan(method_ir)
    components = dict(plan.semilocal.functional.components)
    for record in SPLIT_HYBRIDS.values():
        expected = {
            name: Fraction(coefficient) for name, coefficient in record["components"]
        }
        if components != expected:
            continue
        if len(plan.exchange) != 1:
            continue
        exchange = plan.exchange[0]
        if (
            exchange.operator == "full-range"
            and exchange.coefficient == Fraction(record["exact_exchange"])
            and exchange.omega == 0
        ):
            return record
    return None


def cuda_global_hybrid_force_eligible(method_ir: MethodIR) -> bool:
    """Check derivative source coverage for an already admitted CUDA KS graph.

    Native preparation still enforces the SCF point-domain and composition
    contract. This adds no name-based method admission: only one semilocal
    source plus positive full-range exchange has a complete CUDA pullback.
    Range separation, dispersion and nonlocal correlation need other providers.
    """
    plan = compile_ks_execution_plan(method_ir)
    return (
        len(method_ir.primitives) == 2
        and plan.semilocal is not None
        and plan.semilocal.functional.ingredients
        in (("rho",), ("rho", "sigma"), ("rho", "sigma", "tau"))
        and len(plan.exchange) == 1
        and plan.exchange[0].operator == "full-range"
        and plan.exchange[0].coefficient > 0
        and plan.exchange[0].omega == 0
        and plan.nonlocal_correlation is None
    )


def _native_semilocal_family(method_ir: typing.Any) -> int:
    """Return the primitive-family selector consumed by native KS execution."""
    # The named PBE-D4 ABI retains its separately qualified native correction
    # owner. Do not route that explicit composition through electronic-only
    # admission, or generalize its exception to arbitrary post-SCF corrections.
    if _is_pbe_d4_composition(method_ir):
        return 1
    plan = _native_execution_plan(method_ir)
    split = _split_hybrid_record(method_ir)
    if split is not None:
        return int(split["functional_code"])
    components = dict(plan.semilocal.functional.components)
    if components == {"LDA_X": Fraction(1), "LDA_C_PW": Fraction(1)}:
        return 0
    if set(components) <= {"GGA_X_PBE", "GGA_C_PBE"}:
        return 1
    if components == {
        "MGGA_X_R2SCAN": Fraction(1),
        "MGGA_C_R2SCAN": Fraction(1),
    }:
        return 2
    if components == dict(
        _native_semilocal(
            resolve_method("B3LYP", spin=plan.semilocal.functional.spin)
        ).components
    ):
        return 3
    if components == {"MGGA_X_WB97M_V": Fraction(1), "MGGA_C_WB97M_V": Fraction(1)}:
        canonical = compile_ks_execution_plan(
            resolve_method("WB97M-V", spin=method_ir.spin)
        )
        # The current native evaluator is generated for this exact composition.
        # Names are not selectors, and changed omega/NLC terms must not alias it.
        if (
            plan.semilocal.semantic_payload() != canonical.semilocal.semantic_payload()
            or plan.exchange != canonical.exchange
            or plan.nonlocal_correlation != canonical.nonlocal_correlation
        ):
            raise NotImplementedError(
                "native B97M lowerer requires canonical WB97M-V composition"
            )
        return 4
    raise NotImplementedError("native KS semilocal family has no qualified lowerer")


def ks_coefficients(method_ir: typing.Any) -> typing.Any:
    """Lower one supported MethodIR graph to explicit native X/C/K coefficients."""
    if not isinstance(method_ir, MethodIR):
        raise TypeError("KS coefficients require a resolved MethodIR")
    plan = _native_execution_plan(method_ir)
    spec = plan.semilocal.functional
    components = dict(spec.components)
    split = _split_hybrid_record(method_ir)
    if split is not None:
        exchange_scale = correlation_scale = Fraction(1)
    elif set(components) <= {"GGA_X_PBE", "GGA_C_PBE"}:
        exchange_scale = components.get("GGA_X_PBE", Fraction(0))
        correlation_scale = components.get("GGA_C_PBE", Fraction(0))
    elif (
        components
        in (
            {"LDA_X": Fraction(1), "LDA_C_PW": Fraction(1)},
            {"MGGA_X_R2SCAN": Fraction(1), "MGGA_C_R2SCAN": Fraction(1)},
        )
        and len(method_ir.primitives) == 1
    ):
        exchange_scale = correlation_scale = Fraction(1)
    elif components == {
        "LDA_X": Fraction(2, 25),
        "GGA_X_B88": Fraction(18, 25),
        "LDA_C_VWN_RPA": Fraction(19, 100),
        "GGA_C_LYP": Fraction(81, 100),
    }:
        # The generated B3LYP semilocal primitive already owns its internal
        # component coefficients; native X/C scales stay unity.
        exchange_scale = correlation_scale = Fraction(1)
    elif components == {
        "MGGA_X_WB97M_V": Fraction(1),
        "MGGA_C_WB97M_V": Fraction(1),
    }:
        exchange_scale = correlation_scale = Fraction(1)
    else:
        raise NotImplementedError("unsupported native KS semilocal composition")
    if not plan.exchange:
        fock_exchange = Fraction(0)
    elif len(plan.exchange) == 1:
        fock_exchange = plan.exchange[0].fock_coefficient
    else:
        fock_exchange = next(
            term.fock_coefficient
            for term in plan.exchange
            if term.operator == "short-range"
        )
    values = tuple(
        float(value) for value in (exchange_scale, correlation_scale, fock_exchange)
    )
    if (
        not all(math.isfinite(value) for value in values)
        or exchange_scale < 0
        or correlation_scale < 0
    ):
        raise NotImplementedError("native KS composition coefficients are invalid")
    return values


def ks_range_exchange_parameters(method_ir: typing.Any) -> typing.Any:
    """Return MethodIR-owned (short, long, omega) RSH parameters, if present."""
    if not isinstance(method_ir, MethodIR):
        raise TypeError("KS range exchange requires a resolved MethodIR")
    plan = compile_ks_execution_plan(method_ir)
    range_terms = tuple(
        term for term in plan.exchange if term.operator in ("short-range", "long-range")
    )
    if not range_terms:
        return None
    operators = {term.operator for term in range_terms}
    omegas = {term.omega for term in range_terms}
    if len(range_terms) != 2 or operators != {"short-range", "long-range"}:
        raise NotImplementedError(
            "native KS range separation requires one short- and one long-range exchange contribution"
        )
    if len(omegas) != 1:
        raise NotImplementedError(
            "native KS range-separated exchange requires one shared omega"
        )
    short = next(term for term in range_terms if term.operator == "short-range")
    long = next(term for term in range_terms if term.operator == "long-range")
    return float(short.coefficient), float(long.coefficient), float(short.omega)


def resolve_ks_method(method: typing.Any) -> typing.Any:
    """Resolve a public native KS selector through its generated MethodIR binding."""
    if method not in _NATIVE_KS_METHODS:
        raise ValueError("KS options require a supported native RKS/UKS method")
    identifier, spin = _NATIVE_KS_METHODS[method]
    method_ir = resolve_method(identifier, spin=spin)

    # MethodIR is authoritative for scientific composition.  The public
    # manifest owns only the stable name -> compiler-method/spin binding.
    # Catalog-backed primitives retain their qualified native projection and
    # declaration identity. Discover them from the existing catalog, not a
    # second hand-maintained public-name or coefficient table.
    if identifier in CATALOG:
        # Reject unavailable post-SCF operators before the named-selector
        # consistency check, preserving the established capability exception.
        semilocal = _native_semilocal(method_ir)
        if len(method_ir.primitives) != 1:
            raise RuntimeError("MethodIR composition disagrees with native KS selector")
        runtime_functional = functional(identifier, spin=spin)
        if SemilocalXCPrimitive(semilocal).semantic_payload() != (
            SemilocalXCPrimitive(runtime_functional).semantic_payload()
        ):
            raise RuntimeError(
                "MethodIR semilocal node disagrees with native KS XC catalog"
            )
        return method_ir, runtime_functional

    if _is_pbe_d4_composition(method_ir):
        return method_ir, functional("PBE", spin=spin)

    semilocal = _native_semilocal(method_ir)
    ks_coefficients(method_ir)
    return method_ir, semilocal


def _scf_domain_for_ir(method_ir: typing.Any) -> str:
    """Select the native work domain from the resolved, possibly renamed IR."""
    code = _native_semilocal_family(method_ir)
    if code >= 0x10000:
        return SPLIT_HYBRID_SCF_DOMAIN
    return {3: B3LYP_SCF_DOMAIN, 4: WB97MV_SCF_DOMAIN}.get(code, SCF_DOMAIN)


def scf_domain_for_method(method: typing.Any) -> str:
    """Return the exact native point-domain identity for one public KS method."""
    method_ir, _ = resolve_ks_method(method)
    return _scf_domain_for_ir(method_ir)


def native_xc_functional_code(method: typing.Any) -> int:
    """Return the lowerer code selected from the method's resolved semilocal IR."""
    method_ir, _ = resolve_ks_method(method)
    return _native_semilocal_family(method_ir)


def resolve_ks_options(method: typing.Any, options: typing.Any = None) -> typing.Any:
    """Validate a MethodIR-resolved model before resource/native allocation."""
    named_ir, expected = resolve_ks_method(method)
    options = KsOptions() if options is None else options
    if not isinstance(options, KsOptions):
        raise TypeError("ks_options must be KsOptions")

    method_ir = named_ir
    # Calculator passes its resolved options to resource planning. Keep that
    # graph authoritative: rebinding the descriptive selector loses custom K.
    composition = options.composition or options._method_ir
    if composition is not None:
        method_ir = composition
        if _is_pbe_d4_composition(named_ir):
            if method_ir.identity != named_ir.identity:
                raise NotImplementedError(
                    "public PBE-D4 requires the pinned named MethodIR without parameter overrides"
                )
            selected = _native_pbe_d4_semilocal(method_ir)
        else:
            selected = _native_semilocal(method_ir)
        # The native selector chooses only the ingredient/spin family; all
        # scientific coefficients remain explicit in the supplied MethodIR.
        if (
            method_ir.spin != named_ir.spin
            or selected.spin != expected.spin
            or selected.ingredients != expected.ingredients
        ):
            raise NotImplementedError(
                "KS composition/spin disagrees with native family selector"
            )
        if not _is_pbe_d4_composition(named_ir):
            ks_coefficients(method_ir)
        # Preserve the independent catalog projection's declaration order and
        # identity when options have already been resolved.
        resolved = selected if options.functional is None else options.functional
        if SemilocalXCPrimitive(resolved).semantic_payload() != (
            SemilocalXCPrimitive(selected).semantic_payload()
        ):
            raise ValueError("resolved KS functional disagrees with its MethodIR")
    else:
        resolved = expected if options.functional is None else options.functional
        if (
            resolved.spin != expected.spin
            or resolved.version != expected.version
            or sorted(resolved.components) != sorted(expected.components)
            or resolved.exact_exchange != expected.exact_exchange
            or resolved.range_omega != expected.range_omega
            or resolved.long_range_exchange != expected.long_range_exchange
        ):
            raise NotImplementedError(
                "KS FunctionalSpec does not match the method's supported composition/spin"
            )

    grid = options.grid
    if grid is None:
        if _native_semilocal_family(named_ir) == 2:
            # The v2 policy has no qualified meta-GGA profile. Preserve the
            # existing explicit v1 default rather than assigning a GGA grid.
            if options.grid_accuracy != "standard":
                raise NotImplementedError(
                    "r2SCAN grid accuracy profiles require an explicit GridSpec"
                )
            grid = GridSpec()
        elif _is_pbe_d4_composition(named_ir):
            grid = GridPolicy(options.grid_accuracy).resolve(
                "pbe-rks", derivative_order=0
            )
        elif ks_coefficients(method_ir)[2] != 0.0:
            raise NotImplementedError(
                "global-hybrid grid policy requires an explicit GridSpec"
            )
        else:
            grid = GridPolicy(options.grid_accuracy).resolve(method, derivative_order=0)
    domain = _scf_domain_for_ir(method_ir)
    if options.scf_domain not in (SCF_DOMAIN, domain):
        raise NotImplementedError(
            "KS tail/spin domain does not match the selected method"
        )
    result = replace(
        options, functional=resolved, composition=None, grid=grid, scf_domain=domain
    )
    object.__setattr__(result, "_method_ir", method_ir)
    return result


def profiled_ks_selection(
    options: KsOptions | None,
    diagnostics: dict[str, typing.Any],
    systems: typing.Any,
    *,
    charges: typing.Any,
    multiplicities: typing.Any,
) -> ProfiledKsSelection:
    """Resolve one exact local DFT09 winner and retain qualification provenance."""

    if (
        options is None
        or options.xc_schedule != "device_fused"
        or diagnostics.get("source") != "local"
    ):
        return ProfiledKsSelection(options)
    target = diagnostics.get("target")
    if not isinstance(target, dict):
        return ProfiledKsSelection(options)
    device = target.get("device")
    source_identity = target.get("source_identity")
    if (
        not isinstance(device, dict)
        or type(device.get("major")) is not int
        or type(device.get("minor")) is not int
        or not isinstance(source_identity, str)
        or not source_identity
    ):
        return ProfiledKsSelection(options)

    from vibeqc_compiler.dft.xc_schedule import (
        grid_xc_schedule,
        molecular_grid_xc_workload,
    )

    from .profiles import select_dft_schedule

    selected = []
    architecture = f"sm_{device['major']}{device['minor']}"
    for atoms, charge, multiplicity in zip(
        systems, charges, multiplicities, strict=True
    ):
        workload = molecular_grid_xc_workload(
            architecture=architecture,
            functional=options.functional,
            atoms=atoms,
            grid_spec=options.grid,
            charge=charge,
            multiplicity=multiplicity,
            source_identity=source_identity,
            screening_identity=None,
            observable="potential",
            density_route="density_matrix",
        )
        payload = select_dft_schedule(diagnostics, workload.to_payload())
        if payload is None:
            return ProfiledKsSelection(options)
        schedule = grid_xc_schedule(payload)
        if schedule.point_tile is None:
            raise ValueError("local DFT schedule winner must resolve its point tile")
        selected.append(schedule)
    if not selected or any(item != selected[0] for item in selected[1:]):
        return ProfiledKsSelection(options)

    winner = selected[0]
    result = replace(
        options,
        xc_schedule=winner.name,
        tile_points=winner.point_tile,
    )
    object.__setattr__(result, "_method_ir", options._method_ir)
    return ProfiledKsSelection(result, exact_profile_match=True)


def profiled_ks_options(
    options: KsOptions | None,
    diagnostics: dict[str, typing.Any],
    systems: typing.Any,
    *,
    charges: typing.Any,
    multiplicities: typing.Any,
) -> KsOptions | None:
    """Apply one exact local DFT09 winner to a batch, otherwise preserve the portable plan."""

    return profiled_ks_selection(
        options,
        diagnostics,
        systems,
        charges=charges,
        multiplicities=multiplicities,
    ).options


def native_ks_options(options: typing.Any) -> typing.Any:
    """Lower one compiler KS execution plan into the current semantic C ABI.

    The ABI carries MethodIR primitives directly.  No named-method/family code
    and no historical suffix version is selected here.
    """
    import ctypes

    from . import _native

    grid = options.grid
    if grid is None:
        raise ValueError("native KS options require a resolved GridSpec")
    radii = None
    if grid.element_radii:
        fill = 0.0 if grid.version >= 2 else 1.0
        radii = (ctypes.c_double * 119)(*[fill] * 119)
        for z, radius in grid.element_radii:
            radii[z] = radius

    if _is_pbe_d4_composition(options.method_ir):
        semilocal = typing.cast("SemilocalXCPrimitive", options.method_ir.primitives[0])
        exchange = ()
        nonlocal_primitive = None
    else:
        plan = _native_execution_plan(options.method_ir)
        semilocal = plan.semilocal
        exchange = plan.exchange
        nonlocal_primitive = plan.nonlocal_correlation

    component_ids = tuple(
        name.encode("ascii") for name, _ in semilocal.functional.components
    )
    component_type = _native.KsSemilocalComponentDescriptor * len(component_ids)
    components = component_type(
        *(
            _native.KsSemilocalComponentDescriptor(component_id, float(coefficient))
            for component_id, (_, coefficient) in zip(
                component_ids, semilocal.functional.components, strict=True
            )
        )
    )

    operators = {
        "full-range": _native.KS_EXCHANGE_FULL_RANGE,
        "short-range": _native.KS_EXCHANGE_SHORT_RANGE,
        "long-range": _native.KS_EXCHANGE_LONG_RANGE,
    }
    exchange_type = _native.KsExchangeTermDescriptor * len(exchange)
    exchange_terms = exchange_type(
        *(
            _native.KsExchangeTermDescriptor(
                operators[term.operator],
                float(term.coefficient),
                float(term.omega),
                float(term.fock_coefficient),
            )
            for term in exchange
        )
    )

    variants = {"vv10": _native.NONLOCAL_VV10, "rvv10": _native.NONLOCAL_RVV10}
    if nonlocal_primitive is not None:
        try:
            nonlocal_variant = variants[nonlocal_primitive.spec.variant]
        except KeyError as error:
            raise NotImplementedError(
                f"unsupported native nonlocal variant {nonlocal_primitive.spec.variant!r}"
            ) from error
        nonlocal_b = float(nonlocal_primitive.spec.b)
        nonlocal_c = float(nonlocal_primitive.spec.c)
        nonlocal_coefficient = float(nonlocal_primitive.coefficient)
    else:
        nonlocal_variant = 0
        nonlocal_b = nonlocal_c = nonlocal_coefficient = 0.0

    domain = options.scf_domain.encode("ascii")
    descriptor = _native.KsOptionsDescriptor(
        ctypes.sizeof(_native.KsOptionsDescriptor),
        _native.ABI_VERSION,
        domain,
        grid.version,
        grid.radial_points,
        grid.angular_polar,
        grid.angular_azimuth,
        grid.partition_iterations,
        grid.coincident_tolerance,
        options.tile_points,
        radii,
        119 if radii is not None else 0,
        _native.XC_EXECUTION_DEVICE_FUSED
        if options.xc_schedule == "device_fused"
        else _native.XC_EXECUTION_HOST_UNFUSED,
        1 if options.method_ir.spin == "unpolarized" else 2,
        components,
        len(components),
        float(semilocal.functional.range_omega),
        exchange_terms if exchange_terms else None,
        len(exchange_terms),
        1 if nonlocal_primitive is not None else 0,
        nonlocal_variant,
        nonlocal_b,
        nonlocal_c,
        nonlocal_coefficient,
        options.nonlocal_memory_budget_bytes if nonlocal_primitive is not None else 0,
    )
    # ctypes pointer fields do not own their pointees.  Retain every borrowed
    # object for exactly as long as the short-lived descriptor is alive.
    descriptor._domain_owner = domain
    descriptor._component_ids_owner = component_ids
    descriptor._components_owner = components
    descriptor._exchange_owner = exchange_terms
    descriptor._radii_owner = radii
    return descriptor
