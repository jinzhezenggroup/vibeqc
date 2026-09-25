"""Bounded, offline compilation census for bulk Libxc programs (#1123).

This module measures compiler products, not numerical or production admission.
Exact emitted-source/ABI matches share a build task; every registration retains
its independent imported identity. No persistent binary cache is introduced.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibeqc_compiler.common.compiler_process import run_compiler
from vibeqc_compiler.common.cuda_resources import parse_resources
from vibeqc_compiler.common.provenance import canonical_hash, file_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from .libxc_bulk import BulkProgram

SCHEMA = "vibeqc.libxc-aot-census/v1"
POINT_ABI = "bulk-xc-point/features-to-E-vxc-packed-fxc/f64/v1"
BACKENDS = ("cpu", "cuda")


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class CudaResourceLimits:
    """Explicit PTXAS limits required before a CUDA probe is package-eligible."""

    maximum_registers: int
    maximum_stack_bytes: int
    maximum_local_bytes: int
    maximum_shared_bytes: int
    maximum_spill_bytes: int

    def __post_init__(self) -> None:
        _positive_int(self.maximum_registers, "maximum_registers")
        for name in (
            "maximum_stack_bytes",
            "maximum_local_bytes",
            "maximum_shared_bytes",
            "maximum_spill_bytes",
        ):
            _nonnegative_int(getattr(self, name), name)

    def to_payload(self) -> dict[str, int]:
        return {
            "maximum_registers": self.maximum_registers,
            "maximum_stack_bytes": self.maximum_stack_bytes,
            "maximum_local_bytes": self.maximum_local_bytes,
            "maximum_shared_bytes": self.maximum_shared_bytes,
            "maximum_spill_bytes": self.maximum_spill_bytes,
        }


@dataclass(frozen=True)
class PackageBudget:
    """Hard bounds on unique build tasks, not on scientific support claims."""

    max_artifacts: int = 64
    max_source_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        _positive_int(self.max_artifacts, "max_artifacts")
        _positive_int(self.max_source_bytes, "max_source_bytes")


@dataclass(frozen=True)
class SourceVariant:
    """One scientific registration mapped to an exact executable source."""

    name: str
    family: str
    spin: str
    import_identity: str
    domain: str
    features: tuple[str, ...]
    derivative_order: int
    backend: str
    source: str
    energy_nodes: int
    ssa: dict[str, Any]

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError("backend must be cpu or cuda")
        if type(self.derivative_order) is not int or self.derivative_order not in (
            0,
            1,
            2,
        ):
            raise ValueError("derivative_order must be 0, 1 or 2")
        if not self.source or not self.features or not self.domain:
            raise ValueError("source, features and domain must be nonempty")

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    @property
    def emission_identity(self) -> str:
        """Conservative exact-source equivalence, not algebraic equivalence.

        Import provenance stays in the per-registration record. An importer or
        source-owner change that produces identical code may reuse a build task,
        but it never reuses numerical/admission evidence through this identity.
        """
        return canonical_hash(
            {
                "schema": SCHEMA,
                "abi": POINT_ABI,
                "domain": self.domain,
                "spin": self.spin,
                "features": self.features,
                "derivative_order": self.derivative_order,
                "backend": self.backend,
                "source_sha256": self.source_sha256,
            }
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "spin": self.spin,
            "import_identity": self.import_identity,
            "domain": self.domain,
            "features": list(self.features),
            "derivative_order": self.derivative_order,
            "backend": self.backend,
            "emission_identity": self.emission_identity,
            "source_sha256": self.source_sha256,
            "source_bytes": len(self.source.encode("utf-8")),
            "energy_nodes": self.energy_nodes,
            "ssa": self.ssa,
        }


def inspect_program(
    program: BulkProgram, derivative_order: int, backend: str, *, domain: str
) -> SourceVariant:
    """Use the existing Graph analysis and unchanged scalar C/CUDA emitters."""
    if backend not in BACKENDS:
        raise ValueError("backend must be cpu or cuda")
    roots = program.roots(derivative_order)
    return SourceVariant(
        name=program.name,
        family=program.family,
        spin=program.spin,
        import_identity=program.identity,
        domain=domain,
        features=program.features,
        derivative_order=derivative_order,
        backend=backend,
        source=program.emit_source(derivative_order, cuda=backend == "cuda"),
        energy_nodes=program.graph.analyze_ssa((program.energy,)).reachable_node_count,
        ssa=program.graph.analyze_ssa(roots).to_payload(),
    )


def plan_package(
    variants: Iterable[SourceVariant], budget: PackageBudget | None = None
) -> dict[str, Any]:
    """Select exact-source groups in a stable order within explicit bounds.

    This is an offline *build plan*, not an installable/production package.
    Unselected groups retain a precise budget blocker. The budget is applied
    after deduplication so aliases never consume multiple slots or byte quotas.
    """
    if budget is None:
        budget = PackageBudget()
    groups: dict[str, list[dict[str, Any]]] = {}
    registrations: set[tuple[str, str, int, str]] = set()
    for variant in variants:
        registration = (
            variant.name,
            variant.spin,
            variant.derivative_order,
            variant.backend,
        )
        if registration in registrations:
            raise ValueError(f"duplicate census variant: {registration}")
        registrations.add(registration)
        groups.setdefault(variant.emission_identity, []).append(variant.to_payload())
    used_bytes = selected = 0
    unique_source_bytes = per_registration_source_bytes = 0
    artifacts: list[dict[str, Any]] = []
    for identity, members in sorted(groups.items()):
        members.sort(key=lambda item: (item["name"], item["import_identity"]))
        representative = members[0]
        size = representative["source_bytes"]
        reason = None
        if selected >= budget.max_artifacts:
            reason = "artifact-count-budget"
        elif used_bytes + size > budget.max_source_bytes:
            reason = "source-byte-budget"
        if reason is None:
            selected += 1
            used_bytes += size
        unique_source_bytes += size
        per_registration_source_bytes += size * len(members)
        artifacts.append(
            {
                "emission_identity": identity,
                "backend": representative["backend"],
                "source_sha256": representative["source_sha256"],
                "source_bytes": size,
                "selected": reason is None,
                "blocker": reason,
                "registrations": members,
            }
        )
    return {
        "schema": SCHEMA,
        "kind": "offline-build-plan",
        "capability_claims": [],
        "budget": {
            "max_artifacts": budget.max_artifacts,
            "max_source_bytes": budget.max_source_bytes,
        },
        "summary": {
            "registration_variants": len(registrations),
            "unique_artifacts": len(artifacts),
            "reused_build_tasks": len(registrations) - len(artifacts),
            "selected_artifacts": selected,
            "selected_source_bytes": used_bytes,
            "unique_source_bytes": unique_source_bytes,
            "per_registration_source_bytes": per_registration_source_bytes,
        },
        "artifacts": artifacts,
    }


def cuda_probe_source(source: str) -> str:
    """Retain the device function in an offline resource-measurement kernel.

    A device function alone can be eliminated from an object. This entry point
    makes its outputs observable; its resources are *probe*, not KS resources.
    """
    return (
        source
        + """
extern "C" __global__ void bulk_xc_census_probe(
    const double *features, double *outputs) {
  if (blockIdx.x == 0 && threadIdx.x == 0) bulk_xc_point(features, outputs);
}
"""
    )


def ptxas_resources(log: str) -> dict[str, int | None]:
    """Extract conservative maxima from canonical per-function PTXAS records."""
    records = parse_resources(log)
    declared = re.findall(r"Function properties for (\S+)", log)
    # The shared parser returns complete records only. Never let a truncated
    # or unsupported function record disappear from the artifact-level gate.
    if not records or sorted(record.function for record in records) != sorted(declared):
        return {
            "registers": None,
            "stack_bytes": None,
            "spill_store_bytes": None,
            "spill_load_bytes": None,
            "shared_bytes": None,
            "local_bytes": None,
        }
    local_values = [record.local_bytes for record in records]
    return {
        "registers": max(record.registers for record in records),
        "stack_bytes": max(record.stack_bytes for record in records),
        "spill_store_bytes": max(record.spill_store_bytes for record in records),
        "spill_load_bytes": max(record.spill_load_bytes for record in records),
        # PTXAS omits the smem token for a zero-shared-memory function; the
        # canonical parser intentionally normalizes that omission to zero.
        "shared_bytes": max(record.shared_bytes for record in records),
        # Keep local memory fail-closed when any function omits the lmem field.
        "local_bytes": (
            max(value for value in local_values if value is not None)
            if all(value is not None for value in local_values)
            else None
        ),
    }


def gate_cuda_resources(
    evidence: dict[str, Any], limits: CudaResourceLimits
) -> dict[str, Any]:
    """Fail closed unless every required PTXAS resource is observed and bounded.

    This is a compiler-artifact gate only. Passing it does not imply numerical,
    GPU-runtime, SCF, force, or public-method qualification.
    """
    if not isinstance(limits, CudaResourceLimits):
        raise TypeError("limits must be CudaResourceLimits")
    observed = evidence.get("resources")
    if evidence.get("status") != "compiled":
        return {
            "status": "unavailable",
            "package_eligible": False,
            "reasons": [f"compile-status:{evidence.get('status', 'missing')}"],
            "limits": limits.to_payload(),
            "observed": None,
        }
    if not isinstance(observed, dict):
        return {
            "status": "unavailable",
            "package_eligible": False,
            "reasons": ["ptxas-resources-missing"],
            "limits": limits.to_payload(),
            "observed": None,
        }
    fields = (
        "registers",
        "stack_bytes",
        "spill_store_bytes",
        "spill_load_bytes",
        "shared_bytes",
        "local_bytes",
    )
    missing = [field for field in fields if type(observed.get(field)) is not int]
    if missing:
        return {
            "status": "unavailable",
            "package_eligible": False,
            "reasons": ["unknown-resource:" + field for field in missing],
            "limits": limits.to_payload(),
            "observed": {field: observed.get(field) for field in fields},
        }
    values = {field: int(observed[field]) for field in fields}
    if any(value < 0 for value in values.values()):
        raise ValueError("PTXAS resource observations must be nonnegative")
    reasons = []
    if values["registers"] > limits.maximum_registers:
        reasons.append("register-limit")
    if values["stack_bytes"] > limits.maximum_stack_bytes:
        reasons.append("stack-limit")
    if values["local_bytes"] > limits.maximum_local_bytes:
        reasons.append("local-memory-limit")
    if values["shared_bytes"] > limits.maximum_shared_bytes:
        reasons.append("shared-memory-limit")
    if (
        values["spill_store_bytes"] + values["spill_load_bytes"]
        > limits.maximum_spill_bytes
    ):
        reasons.append("spill-limit")
    return {
        "status": "passed" if not reasons else "rejected",
        "package_eligible": not reasons,
        "reasons": reasons,
        "limits": limits.to_payload(),
        "observed": values,
    }


def compile_probe(
    variant: SourceVariant,
    *,
    compiler: str | None = None,
    timeout: float = 60,
    cuda_arch: str = "sm_80",
) -> dict[str, Any]:
    """Explicit offline compilation; never called by runtime or import hooks.

    No persistent cache is read or written: the caller deduplicates tasks inside
    one invocation. A recipe hash is diagnostic, not a reusable binary-cache key
    (system headers, libdevice and downstream toolchain files are not hashed).
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if not re.fullmatch(r"sm_[0-9]{2,3}[af]?", cuda_arch):
        raise ValueError("cuda_arch must be a concrete sm_XX target")
    executable = shutil.which(
        compiler or ("nvcc" if variant.backend == "cuda" else "cc")
    )
    if executable is None:
        return {"status": "unavailable", "reason": "compiler-not-found"}
    path = Path(executable).resolve()
    try:
        version = run_compiler(
            [str(path), "--version"],
            timeout,
            label="XC census compiler identification",
        )
        if version.timed_out or version.returncode != 0:
            return {
                "status": "timed-out" if version.timed_out else "failed",
                "reason": "compiler-identification",
                "diagnostic": version.stdout + version.stderr,
            }
        compiler_identity = {
            "executable_sha256": file_hash(path),
            "version": (version.stdout + version.stderr).strip(),
        }
    except OSError as error:
        return {
            "status": "failed",
            "reason": "compiler-identification",
            "diagnostic": str(error),
        }
    cuda = variant.backend == "cuda"
    source = cuda_probe_source(variant.source) if cuda else variant.source
    flags = (
        ["-O1", "-arch=" + cuda_arch, "-Xptxas=-v"]
        if cuda
        else ["-std=c99", "-O1", "-fPIC"]
    )
    recipe = {
        "emission_identity": variant.emission_identity,
        "translation_unit_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "compiler": compiler_identity,
        "flags": flags,
    }
    evidence: dict[str, Any] = {
        "recipe": recipe,
        "recipe_sha256": canonical_hash(recipe),
        "translation_unit_bytes": len(source.encode("utf-8")),
        "runtime_validation": "not-run",
    }
    with tempfile.TemporaryDirectory(prefix="vibeqc-xc-census-") as directory:
        root = Path(directory)
        input_path = root / ("point.cu" if cuda else "point.c")
        output_path = root / "point.o"
        input_path.write_text(source, encoding="utf-8")
        try:
            result = run_compiler(
                [str(path), *flags, "-c", str(input_path), "-o", str(output_path)],
                timeout,
                label="XC census probe",
            )
            evidence["compile_seconds"] = result.duration_seconds
            log = (result.stdout + result.stderr).replace(str(root), "<build>")
            evidence.update(returncode=result.returncode, diagnostic=log)
            if result.timed_out:
                return {**evidence, "status": "timed-out"}
            if result.returncode != 0 or not output_path.is_file():
                return {**evidence, "status": "failed", "reason": "object-compilation"}
            evidence.update(
                status="compiled",
                object_bytes=output_path.stat().st_size,
                object_sha256=file_hash(output_path),
                resources=ptxas_resources(log) if cuda else None,
            )
        except OSError as error:
            evidence.update(
                status="failed", reason="compiler-io", diagnostic=str(error)
            )
    return evidence


def measure_plan(
    plan: dict[str, Any],
    load_variant: Callable[[dict[str, Any]], SourceVariant],
    *,
    compilers: dict[str, str | None] | None = None,
    timeout: float = 60,
    cuda_arch: str = "sm_80",
    cuda_resource_limits: CudaResourceLimits | None = None,
) -> dict[str, Any]:
    """Rebuild and compile only selected groups, one source at a time.

    A streaming census retains metadata, not every emitted source. Regeneration
    is measured separately and checked against both provenance and executable
    identity before compilation. Aliases share the single representative task.
    """
    measurements = {}
    for artifact in plan["artifacts"]:
        identity = artifact["emission_identity"]
        if not artifact["selected"]:
            measurements[identity] = {
                "status": "not-selected",
                "reason": artifact["blocker"],
            }
            continue
        record = artifact["registrations"][0]
        start = time.perf_counter()
        variant = load_variant(record)
        reemit_seconds = time.perf_counter() - start
        if (
            variant.emission_identity != identity
            or variant.import_identity != record["import_identity"]
        ):
            raise ValueError("source or import identity changed after the census")
        measurement = {
            **compile_probe(
                variant,
                compiler=(compilers or {}).get(variant.backend),
                timeout=timeout,
                cuda_arch=cuda_arch,
            ),
            "reemit_seconds": reemit_seconds,
        }
        if variant.backend == "cuda":
            measurement["resource_gate"] = (
                gate_cuda_resources(measurement, cuda_resource_limits)
                if cuda_resource_limits is not None
                else {
                    "status": "not-run",
                    "package_eligible": False,
                    "reasons": ["resource-policy-not-supplied"],
                }
            )
        measurements[identity] = measurement
    return measurements


def census_catalog(
    *,
    names: Sequence[str] | None = None,
    spins: Sequence[str] = ("polarized", "unpolarized"),
    derivative_orders: Sequence[int] = (1,),
    backends: Sequence[str] = BACKENDS,
    work_limit: int = 1_000_000,
    budget: PackageBudget | None = None,
    source_root: Path | None = None,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Stream a selected catalog through existing, work-bounded bulk imports.

    The default requests E/vxc only, never the Cartesian product of all
    derivative orders. Unsupported registrations and budget-exhausted imports
    are retained as explicit observations; they do not become support claims.
    """
    from vibeqc_compiler.common.compiler_work import (
        CompilerWorkLimit,
        compiler_work_budget,
    )
    from vibeqc_compiler.common.paths import asset_path

    from . import libxc_bulk

    if budget is None:
        budget = PackageBudget()
    from .libxc_maple import MapleImportError

    _positive_int(work_limit, "work_limit")
    for values, allowed, label in (
        (spins, ("polarized", "unpolarized"), "spins"),
        (backends, BACKENDS, "backends"),
        (derivative_orders, (0, 1, 2), "derivative_orders"),
    ):
        if (
            not values
            or len(set(values)) != len(values)
            or any(value not in allowed for value in values)
        ):
            raise ValueError(f"invalid or duplicate {label}")
    if any(type(order) is not int for order in derivative_orders):
        raise ValueError("derivative_orders must contain integers")
    catalog = libxc_bulk.read_catalog(catalog_path or libxc_bulk.CATALOG_PATH)
    records = {record["name"]: record for record in catalog["registrations"]}
    requested = sorted(records if names is None else {name.upper() for name in names})
    if not requested or any(name not in records for name in requested):
        raise ValueError("empty selection or unknown Libxc registration")
    root = (
        asset_path(libxc_bulk.SOURCE_ASSET)
        if source_root is None
        else Path(source_root)
    )
    observations: list[dict[str, Any]] = []
    blocked = [
        {"name": name, "reason": records[name].get("reason", "graph-not-imported")}
        for name in requested
        if records[name]["graph_status"] != "imported"
    ]

    def variants() -> Iterable[SourceVariant]:
        for name in requested:
            if records[name]["graph_status"] != "imported":
                continue
            for spin in sorted(spins):
                for order in sorted(derivative_orders):
                    # Separate budgets keep backend results independent of the
                    # requested backend ordering or another emitter's work.
                    for backend in sorted(backends):
                        observation: dict[str, Any] = {
                            "name": name,
                            "spin": spin,
                            "derivative_order": order,
                            "backend": backend,
                        }
                        start = time.perf_counter()
                        variant: SourceVariant | None = None
                        try:
                            with compiler_work_budget(work_limit) as counter:
                                assert counter is not None
                                program = libxc_bulk.build_record(
                                    records[name],
                                    catalog["source_files"],
                                    root,
                                    spin=spin,
                                )
                                variant = inspect_program(
                                    program,
                                    order,
                                    backend,
                                    domain=libxc_bulk.BULK_SEMANTICS,
                                )
                            observation.update(
                                status="emitted", symbolic_work=counter.used
                            )
                        except CompilerWorkLimit as error:
                            observation.update(
                                status="work-budget-exhausted", reason=str(error)
                            )
                        except (
                            MapleImportError,
                            ValueError,
                            ArithmeticError,
                            RecursionError,
                        ) as error:
                            observation.update(
                                status="generation-failed",
                                reason=f"{type(error).__name__}: {error}",
                            )
                        observation["generation_seconds"] = time.perf_counter() - start
                        observations.append(observation)
                        if variant is not None:
                            yield variant

    plan = plan_package(variants(), budget)
    plan.update(
        catalog_sha256=canonical_hash(catalog),
        request={
            "names": requested,
            "spins": sorted(spins),
            "derivative_orders": sorted(derivative_orders),
            "backends": sorted(backends),
            "work_limit": work_limit,
        },
        blocked_imports=blocked,
        observations=observations,
    )
    return plan
