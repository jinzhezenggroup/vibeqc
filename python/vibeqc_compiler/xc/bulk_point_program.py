"""Bind generic bulk Libxc AOT products to the native semilocal point ABI.

This is executable plumbing for #1119/#1121.  It deliberately does not create
production-domain, molecular-SCF, or public-method evidence.  The binding keeps
the imported capability identity, the projected point-expression identity, and
the exact emitted artifact identity separate so later qualification can verify
all three without granting capability from source generation alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter

from .bulk_aot import BACKENDS, SourceVariant

if TYPE_CHECKING:
    from .bulk_runtime import BulkRuntimeProgram

POINT_PROGRAM_BINDING_SCHEMA = "vibeqc.libxc-bulk-point-program-binding/v1"

_POINT_LAYOUTS = {
    ("rho_a", "rho_b"): 1,
    ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"): 7,
    (
        "rho_a",
        "rho_b",
        "sigma_aa",
        "sigma_ab",
        "sigma_bb",
        "tau_a",
        "tau_b",
    ): 15,
}


def _emit_runtime_source(program: BulkRuntimeProgram, backend: str) -> str:
    """Emit the projected runtime Graph through the existing scalar lowerers."""
    variables = {
        name: f"features[{index}]" for index, name in enumerate(program.spec.features)
    }
    if backend == "cuda":
        from vibeqc_compiler.integral.cuda import CudaEmitter

        emitter = CudaEmitter(program.graph, variables)
        prefix = 'extern "C" __device__ '
    else:
        emitter = ScalarCEmitter(program.graph, variables)
        prefix = ""
    emitter.emit(program.roots)
    return "\n".join(
        [
            "/* Generated from pinned MPL-2.0 Libxc sources by VibeQC. */",
            "#include <math.h>",
            f"{prefix}void bulk_xc_point(const double *features, double *outputs) {{",
            *["  " + line for line in emitter.lines],
            *[
                f"  outputs[{index}] = {emitter.reference(root)};"
                for index, root in enumerate(program.roots)
            ],
            "}",
            "",
        ]
    )


def inspect_runtime_program(
    program: BulkRuntimeProgram, backend: str = "cpu"
) -> SourceVariant:
    """Project one BulkRuntimeProgram into the existing AOT artifact contract.

    Unlike :func:`bulk_aot.inspect_program`, this consumes the runtime-projected
    rho/sigma/tau ABI.  Tau-only meta-GGAs therefore do not regain dead
    Laplacian inputs when they cross the AOT boundary.
    """
    if backend not in BACKENDS:
        raise ValueError("backend must be cpu or cuda")
    if not program.roots or not program.spec.features:
        raise ValueError("bulk runtime program must have roots and features")
    energy_ssa = program.graph.analyze_ssa((program.roots[0],))
    full_ssa = program.graph.analyze_ssa(program.roots)
    return SourceVariant(
        name=program.spec.identifier,
        family=program.spec.family,
        spin=program.spec.spin,
        import_identity=program.spec.source_identity,
        domain=program.spec.domain,
        features=program.spec.features,
        derivative_order=program.order,
        backend=backend,
        source=_emit_runtime_source(program, backend),
        energy_nodes=energy_ssa.reachable_node_count,
        ssa=full_ssa.to_payload(),
    )


@dataclass(frozen=True)
class SemilocalPointBinding:
    """One CPU AOT artifact bound to the native SemilocalPointProgram ABI."""

    variant: SourceVariant
    capability_identity: str
    point_expression_identity: str
    domain_version: int

    def __post_init__(self) -> None:
        if self.variant.backend != "cpu":
            raise ValueError("native semilocal point binding currently requires CPU")
        if self.variant.spin != "polarized":
            raise ValueError(
                "native semilocal point binding requires the polarized two-spin ABI"
            )
        if self.variant.derivative_order != 1:
            raise ValueError("native semilocal point binding requires E/vxc output")
        if self.variant.features not in _POINT_LAYOUTS:
            raise ValueError(
                "native semilocal point binding requires compact rho/sigma/tau features"
            )
        if not self.capability_identity or not self.point_expression_identity:
            raise ValueError("point binding identities must be nonempty")
        if type(self.domain_version) is not int or self.domain_version <= 0:
            raise ValueError("domain_version must be a positive integer")

    @property
    def ingredient_mask(self) -> int:
        return _POINT_LAYOUTS[self.variant.features]

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": POINT_PROGRAM_BINDING_SCHEMA,
            "name": self.variant.name,
            "capability_identity": self.capability_identity,
            "point_expression_identity": self.point_expression_identity,
            "artifact_emission_identity": self.variant.emission_identity,
            "artifact_source_sha256": self.variant.source_sha256,
            "import_identity": self.variant.import_identity,
            "domain": self.variant.domain,
            "domain_version": self.domain_version,
            "features": list(self.variant.features),
            "ingredient_mask": self.ingredient_mask,
        }

    def emit_source(self) -> str:
        """Return one self-contained C++ point-program translation unit.

        The imported scalar artifact remains the mathematical owner.  This
        wrapper only maps Cartesian spin gradients to sigma coordinates and the
        raw tau derivative to the native vtau/2 weak-form coefficient.
        """
        feature_count = len(self.variant.features)
        outputs = feature_count + 1
        if self.ingredient_mask == 1:
            feature_lines = ["  double features[2]{rho[0], rho[1]};"]
        else:
            feature_lines = [
                "  const double sigma_aa = gradient[0][0] * gradient[0][0] +",
                "      gradient[0][1] * gradient[0][1] + gradient[0][2] * gradient[0][2];",
                "  const double sigma_ab = gradient[0][0] * gradient[1][0] +",
                "      gradient[0][1] * gradient[1][1] + gradient[0][2] * gradient[1][2];",
                "  const double sigma_bb = gradient[1][0] * gradient[1][0] +",
                "      gradient[1][1] * gradient[1][1] + gradient[1][2] * gradient[1][2];",
            ]
            tail = ", tau[0], tau[1]" if self.ingredient_mask == 15 else ""
            feature_lines.append(
                "  double features["
                + str(feature_count)
                + "]{rho[0], rho[1], sigma_aa, sigma_ab, sigma_bb"
                + tail
                + "};"
            )

        mapping = [
            f"  double outputs[{outputs}]{{}};",
            "  ::bulk_xc_point(features, outputs);",
            "  SemilocalPointValue out{};",
            "  out.energy = outputs[0];",
            "  out.rho[0] = outputs[1];",
            "  out.rho[1] = outputs[2];",
        ]
        if self.ingredient_mask != 1:
            mapping.extend(
                [
                    "  for (unsigned axis = 0; axis < 3; ++axis) {",
                    "    out.gradient[0][axis] =",
                    "        2.0 * outputs[3] * gradient[0][axis] + outputs[4] * gradient[1][axis];",
                    "    out.gradient[1][axis] =",
                    "        outputs[4] * gradient[0][axis] + 2.0 * outputs[5] * gradient[1][axis];",
                    "  }",
                ]
            )
        if self.ingredient_mask == 15:
            mapping.extend(
                [
                    "  out.kinetic[0] = 0.5 * outputs[6];",
                    "  out.kinetic[1] = 0.5 * outputs[7];",
                ]
            )
        mapping.append("  return out;")

        q = json.dumps
        return (
            self.variant.source
            + '\n#include "dft/xc.hpp"\n'
            + "\nnamespace vibeqc::dft::bulk_generated {\n"
            + f"inline constexpr const char* kBindingIdentity = {q(self.identity)};\n"
            + f"inline constexpr const char* kCapabilityIdentity = {q(self.capability_identity)};\n"
            + f"inline constexpr const char* kArtifactEmissionIdentity = {q(self.variant.emission_identity)};\n"
            + f"inline constexpr const char* kPointExpressionIdentity = {q(self.point_expression_identity)};\n"
            + "inline SemilocalPointValue evaluate_point(const double rho[2],\n"
            + "                                         const double (&gradient)[2][3],\n"
            + "                                         const double tau[2]) {\n"
            + "\n".join(feature_lines + mapping)
            + "\n}\n"
            + "inline constexpr SemilocalPointProgram kPointProgram{\n"
            + f"    {q(self.variant.name)}, kPointExpressionIdentity, {self.ingredient_mask}U, "
            + f"{self.domain_version}U, evaluate_point}};\n"
            + "}  // namespace vibeqc::dft::bulk_generated\n"
        )


def bind_runtime_semilocal_point_program(
    program: BulkRuntimeProgram, *, domain_version: int
) -> SemilocalPointBinding:
    """Bind one complete polarized E/vxc runtime Graph to the CPU point ABI.

    This function is intentionally not an admission producer.  Callers still
    need independent compiled-CPU and production-domain evidence before the
    resulting executable can enter the molecular-SCF qualification runner.
    """
    expected = ((), *((index,) for index in range(len(program.spec.features))))
    if program.outputs != expected:
        raise ValueError(
            "native semilocal point binding requires the complete E/vxc output set"
        )
    variant = inspect_runtime_program(program, "cpu")
    return SemilocalPointBinding(
        variant=variant,
        capability_identity=program.spec.capability_identity,
        point_expression_identity=program.expression_hash,
        domain_version=domain_version,
    )


__all__ = [
    "POINT_PROGRAM_BINDING_SCHEMA",
    "SemilocalPointBinding",
    "bind_runtime_semilocal_point_program",
    "inspect_runtime_program",
]
