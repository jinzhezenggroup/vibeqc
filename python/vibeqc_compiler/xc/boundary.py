"""Deterministic physical boundary probes for semilocal XC qualification.

The probes are compiler inputs, not reference values.  Independent Libxc oracle
generation lives in tools/generate_libxc_boundary_reference.py so the VibeQC
Graph never participates in producing its own expected answers.
"""

from __future__ import annotations

from dataclasses import dataclass

BOUNDARY_SEMANTICS = "semilocal-boundary-v1"

_POLARIZED_FEATURES = (
    "rho_a",
    "rho_b",
    "sigma_aa",
    "sigma_ab",
    "sigma_bb",
    "lapl_a",
    "lapl_b",
    "tau_a",
    "tau_b",
)
_UNPOLARIZED_FEATURES = ("rho", "sigma", "lapl", "tau")


@dataclass(frozen=True)
class BoundaryProbe:
    """One named physical endpoint/tail probe in an explicit feature layout."""

    label: str
    spin: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.values, strict=True))


def bulk_feature_names(family: str, spin: str) -> tuple[str, ...]:
    """Return the physical feature order used by the bulk Libxc adapter."""
    if family not in ("lda", "gga", "mgga"):
        raise ValueError(f"unsupported semilocal family {family!r}")
    if spin not in ("polarized", "unpolarized"):
        raise ValueError(f"unsupported spin layout {spin!r}")
    if spin == "polarized":
        names = list(_POLARIZED_FEATURES[:2])
        if family != "lda":
            names.extend(_POLARIZED_FEATURES[2:5])
        if family == "mgga":
            names.extend(_POLARIZED_FEATURES[5:])
    else:
        names = ["rho"]
        if family != "lda":
            names.append("sigma")
        if family == "mgga":
            names.extend(("lapl", "tau"))
    return tuple(names)


def _polarized_states() -> tuple[tuple[str, dict[str, float]], ...]:
    rho = 0.073
    tiny = rho * 1.0e-14
    return (
        (
            "zero-minority",
            {
                "rho_a": rho,
                "rho_b": 0.0,
                "sigma_aa": 4.0 * rho * rho,
                "sigma_ab": 0.0,
                "sigma_bb": 0.0,
                "lapl_a": 0.0,
                "lapl_b": 0.0,
                "tau_a": 0.7 * rho,
                "tau_b": 0.0,
            },
        ),
        (
            "near-zero-minority",
            {
                "rho_a": rho,
                "rho_b": tiny,
                "sigma_aa": 4.0 * rho * rho,
                "sigma_ab": 0.0,
                "sigma_bb": 4.0 * tiny * tiny,
                "lapl_a": 0.0,
                "lapl_b": 0.0,
                "tau_a": 0.7 * rho,
                "tau_b": 0.7 * tiny,
            },
        ),
        (
            "zero-majority",
            {
                "rho_a": 0.0,
                "rho_b": rho,
                "sigma_aa": 0.0,
                "sigma_ab": 0.0,
                "sigma_bb": 4.0 * rho * rho,
                "lapl_a": 0.0,
                "lapl_b": 0.0,
                "tau_a": 0.0,
                "tau_b": 0.7 * rho,
            },
        ),
        (
            "zero-gradient",
            {
                "rho_a": 0.61,
                "rho_b": 0.19,
                "sigma_aa": 0.0,
                "sigma_ab": 0.0,
                "sigma_bb": 0.0,
                "lapl_a": 0.0,
                "lapl_b": 0.0,
                "tau_a": 0.33,
                "tau_b": 0.12,
            },
        ),
        (
            "polarized-tail",
            {
                "rho_a": 7.3e-20,
                "rho_b": 2.1e-20,
                "sigma_aa": 4.0 * (7.3e-20) ** 2,
                "sigma_ab": 0.0,
                "sigma_bb": 4.0 * (2.1e-20) ** 2,
                "lapl_a": 0.0,
                "lapl_b": 0.0,
                "tau_a": 0.7 * 7.3e-20,
                "tau_b": 0.7 * 2.1e-20,
            },
        ),
    )


def _unpolarized_states() -> tuple[tuple[str, dict[str, float]], ...]:
    return (
        (
            "zero-gradient",
            {"rho": 0.8, "sigma": 0.0, "lapl": 0.0, "tau": 0.46},
        ),
        (
            "unpolarized-tail",
            {"rho": 1.0e-20, "sigma": 4.0e-40, "lapl": 0.0, "tau": 7.0e-21},
        ),
    )


def semilocal_boundary_probes(
    feature_names: tuple[str, ...], *, spin: str
) -> tuple[BoundaryProbe, ...]:
    """Project the standard boundary suite into an existing feature layout."""
    if spin == "polarized":
        allowed = set(_POLARIZED_FEATURES)
        states = _polarized_states()
    elif spin == "unpolarized":
        allowed = set(_UNPOLARIZED_FEATURES)
        states = _unpolarized_states()
    else:
        raise ValueError(f"unsupported spin layout {spin!r}")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("boundary qualification requires unique feature names")
    unknown = set(feature_names) - allowed
    if unknown:
        raise ValueError(f"unsupported boundary features: {sorted(unknown)!r}")
    probes = []
    for label, state in states:
        missing = [name for name in feature_names if name not in state]
        if missing:
            raise ValueError(f"boundary state {label!r} lacks features {missing!r}")
        probes.append(
            BoundaryProbe(
                label=label,
                spin=spin,
                feature_names=feature_names,
                values=tuple(float(state[name]) for name in feature_names),
            )
        )
    return tuple(probes)
