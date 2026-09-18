"""Immutable native semilocal KS model and execution controls.

The compiler's FunctionalSpec supplies composition and ingredient requirements.
Native SCF has its own audited tail/spin domain, distinct from the compiler's
interior-only reference contract. Unsupported compositions fail before prepare.
"""

from dataclasses import asdict, dataclass, field, replace

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft.grid import GridSpec, checked_int
from vibeqc_compiler.method import MethodIR, SemilocalXCPrimitive, resolve_method
from vibeqc_compiler.xc.spec import FunctionalSpec, functional

SCF_DOMAIN = "semilocal-scaled-v1/pbe-spin-c2-1e-18"

_NATIVE_KS_METHODS = {
    "lda-rks": ("LDA_XC_PW", "unpolarized"),
    "pbe-rks": ("PBE", "unpolarized"),
    "lda-uks": ("LDA_XC_PW", "polarized"),
    "pbe-uks": ("PBE", "polarized"),
}


@dataclass(frozen=True)
class KsOptions:
    """A snapshotted LDA/PBE composition, quadrature, and bounded XC tile.

    An absent functional resolves from the method name. RKS requires an
    unpolarized FunctionalSpec; UKS requires polarized. The only supported
    compositions have unit LDA_X/LDA_C_PW or GGA_X_PBE/GGA_C_PBE coefficients,
    with no exact exchange. A different model requires a new prepared owner.
    """

    functional: FunctionalSpec | None = None
    grid: GridSpec = field(default_factory=GridSpec)
    tile_points: int = 256
    scf_domain: str = SCF_DOMAIN
    _method_ir: MethodIR | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.functional is not None and not isinstance(
            self.functional, FunctionalSpec
        ):
            raise TypeError("KS functional must be a FunctionalSpec")
        if not isinstance(self.grid, GridSpec):
            raise TypeError("KS grid must be a GridSpec")
        checked_int(self.tile_points, "KS XC tile points")
        if self.scf_domain != SCF_DOMAIN:
            raise NotImplementedError("unsupported native KS tail/spin domain policy")

    @property
    def method_ir(self):
        """Resolved method graph consumed by this native KS option set."""
        if self._method_ir is None:
            raise ValueError("resolve KS options against a method first")
        return self._method_ir

    @property
    def ao_order(self):
        """SCF needs the potential; only GGA composition needs first AO jets."""
        if self.functional is None:
            raise ValueError("resolve KS options against a method first")
        return int("sigma" in self.functional.ingredients)

    def to_payload(self):
        """Keep composition provenance and the effective SCF domain explicit."""
        if self.functional is None:
            raise ValueError("resolve KS options against a method first")
        payload = {
            "functional": self.functional.to_payload(),
            "scf_domain": self.scf_domain,
            "grid": asdict(self.grid),
            "tile_points": self.tile_points,
            "required_ao_order": self.ao_order,
            "required_ingredients": self.functional.ingredients,
            "scalar_derivative_order": 1,
            "observable": "scf-energy",
        }
        if self._method_ir is not None:
            payload["method_ir"] = self._method_ir.to_payload()
            payload["method_ir_identity"] = self._method_ir.identity
        return payload

    @property
    def identity(self):
        return canonical_hash(self.to_payload())


def resolve_ks_method(method):
    """Resolve one native KS name through MethodIR and project its semilocal node."""
    if method not in _NATIVE_KS_METHODS:
        raise ValueError("KS options require a native LDA/PBE RKS/UKS method")
    identifier, spin = _NATIVE_KS_METHODS[method]
    method_ir = resolve_method(identifier, spin=spin)
    if len(method_ir.primitives) != 1 or not isinstance(
        method_ir.primitives[0], SemilocalXCPrimitive
    ):
        raise NotImplementedError(
            "native KS requires exactly one supported semilocal XC primitive"
        )
    # MethodIR canonicalizes component order for semantic/cache identity, while
    # the established native KS FunctionalSpec identity retains audited declaration
    # order. Compare both compositions in canonical form before returning the
    # catalog representation; overwriting the node's components would hide changed
    # coefficients or missing terms. Bind the catalog to the requested native
    # selector, whose fixed kernels cannot execute a different family or spin.
    runtime_functional = functional(identifier, spin=spin)
    if (
        method_ir.primitives[0].semantic_payload()
        != SemilocalXCPrimitive(runtime_functional).semantic_payload()
    ):
        raise RuntimeError(
            "MethodIR semilocal node disagrees with native KS XC catalog"
        )
    return method_ir, runtime_functional


def resolve_ks_options(method, options=None):
    """Validate a MethodIR-resolved model before resource/native allocation."""
    method_ir, expected = resolve_ks_method(method)
    options = KsOptions() if options is None else options
    if not isinstance(options, KsOptions):
        raise TypeError("ks_options must be KsOptions")
    resolved = expected if options.functional is None else options.functional
    # Identifiers are descriptive; only audited component/parameter identity
    # determines supported mathematics. Zero or modified terms are not ignored.
    if (
        resolved.spin != expected.spin
        or resolved.version != expected.version
        or sorted(resolved.components) != sorted(expected.components)
        or resolved.exact_exchange
        or resolved.range_omega
        or resolved.long_range_exchange
    ):
        raise NotImplementedError(
            "KS FunctionalSpec does not match the method's supported composition/spin"
        )
    result = replace(options, functional=resolved)
    object.__setattr__(result, "_method_ir", method_ir)
    return result


def native_ks_options(options):
    """Pack a short-lived C descriptor; ctypes retains its radius-array owner."""
    import ctypes

    from . import _native

    grid = options.grid
    radii = None
    if grid.element_radii:
        radii = (ctypes.c_double * 119)(*[1.0] * 119)
        for z, radius in grid.element_radii:
            radii[z] = radius
    return _native.KsOptionsDescriptor(
        ctypes.sizeof(_native.KsOptionsDescriptor),
        _native.ABI_VERSION,
        1,
        grid.version,
        grid.radial_points,
        grid.angular_polar,
        grid.angular_azimuth,
        grid.partition_iterations,
        grid.coincident_tolerance,
        options.tile_points,
        radii,
        119 if radii is not None else 0,
    )
