"""Build split Libxc global-hybrid semilocal programs from pinned sources.

This module is build/test tooling, not public method admission.  It composes the
separately owned hybrid-exchange and correlation Maple programs while keeping
full-range exact exchange outside the semilocal Graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.xc import libxc_bulk
from vibeqc_compiler.xc.libxc_maple import MapleImportError

from tools.libxc_bulk_metadata import extract_registrations
from tools.libxc_method_metadata import extract_method_registrations


@dataclass(frozen=True)
class LibxcWorkPolicy:
    """Pinned work_gga/work_mgga boundary policy for one component owner."""

    density_threshold: float
    sigma_threshold: float
    tau_threshold: float
    needs_tau: bool
    enforce_fhc: bool


@dataclass(frozen=True)
class SplitGlobalHybridProgram:
    identifier: str
    exact_exchange: Fraction
    exchange_registration: str
    correlation_registration: str
    exchange: libxc_bulk.BulkProgram
    correlation: libxc_bulk.BulkProgram
    exchange_policy: LibxcWorkPolicy
    correlation_policy: LibxcWorkPolicy


def _root() -> Path:
    return asset_path(libxc_bulk.SOURCE_ASSET)


def _catalog() -> dict[str, Any]:
    return libxc_bulk.read_catalog()


def _records(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["name"]: record for record in catalog["registrations"]}


def _method_inventory() -> dict[str, dict[str, Any]]:
    catalog = _catalog()
    records = _records(catalog)
    root = _root()
    owners = sorted(
        {
            record["owner"]
            for record in records.values()
            if record["name"].startswith(("HYB_GGA_X_", "HYB_MGGA_X_"))
        }
    )
    result: dict[str, dict[str, Any]] = {}
    for owner in owners:
        source = (root / owner).read_text(encoding="utf-8")
        for row in extract_method_registrations(
            source, PurePosixPath(owner).name
        ):
            if row.get("status") != "generated" or "split_exchange_component" not in row:
                continue
            exchange = row["split_exchange_component"]
            correlation = row["paired_correlation_component"]
            if exchange not in records or correlation not in records:
                continue
            exact = Fraction(row["exact_exchange"])
            if exact <= 0:
                continue
            identifier = row["identifier"]
            if identifier in result:
                raise MapleImportError(
                    f"duplicate split global-hybrid identifier: {identifier}"
                )
            result[identifier] = row
    return result


def available_split_global_hybrids() -> tuple[str, ...]:
    """Return source-derived global split hybrids with both component owners."""

    return tuple(sorted(_method_inventory()))


def _bound_component(
    name: str, *, allow_hybrid_exchange: bool
) -> dict[str, Any]:
    catalog = _catalog()
    records = _records(catalog)
    try:
        skeleton = records[name]
    except KeyError as error:
        raise MapleImportError(f"unknown split-hybrid component: {name}") from error
    root = _root()
    owner = root / skeleton["owner"]
    entry = root / skeleton["entry"]
    header = root / "src" / "util.h"
    rows = extract_registrations(
        owner.read_text(encoding="utf-8"),
        entry.read_text(encoding="utf-8"),
        header.read_text(encoding="utf-8"),
        allow_hybrid_exchange=allow_hybrid_exchange,
    )
    row = next((item for item in rows if item["name"] == name), None)
    if row is None or row.get("metadata_status") != "bound":
        reason = "registration was not extracted" if row is None else row.get("reason")
        raise MapleImportError(
            f"split-hybrid component {name} is not bindable: {reason}"
        )
    return {**row, "entry": skeleton["entry"], "owner": skeleton["owner"]}


@cache
def _validate_work_policy_sources() -> None:
    root = asset_path("upstream/libxc/7.0.0")
    required = {
        root / "functionals.c": (
            "func->sigma_threshold = pow(func->info->dens_threshold, 4.0/3.0);",
            "func->tau_threshold   = 1e-20;",
        ),
        root / "work_mgga_inc.c": (
            "if(dens < p->dens_threshold)",
            "my_rho[0] = m_max(p->dens_threshold, VAR(rho, ip, 0));",
            "my_sigma[0] = m_max(p->sigma_threshold * p->sigma_threshold, VAR(sigma, ip, 0));",
            "my_tau[0] = m_max(p->tau_threshold, VAR(tau, ip, 0));",
            "my_rho[1] = m_max(p->dens_threshold, VAR(rho, ip, 1));",
            "my_sigma[1] = (my_sigma[1] >= -s_ave ? my_sigma[1] : -s_ave);",
            "my_sigma[1] = (my_sigma[1] <= +s_ave ? my_sigma[1] : +s_ave);",
        ),
    }
    for path, snippets in required.items():
        text = path.read_text(encoding="utf-8")
        if any(snippet not in text for snippet in snippets):
            raise MapleImportError(
                f"Libxc split-hybrid work policy source changed: {path.name}"
            )


def _work_policy(record: dict[str, Any]) -> LibxcWorkPolicy:
    _validate_work_policy_sources()
    bindings = record.get("bindings")
    flags = record.get("flags")
    if not isinstance(bindings, dict) or not isinstance(flags, str):
        raise MapleImportError("split-hybrid component lacks Libxc work metadata")
    try:
        density = float(bindings["p_a_dens_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise MapleImportError("split-hybrid component lacks density threshold") from error
    if not density > 0.0:
        raise MapleImportError("split-hybrid production policy requires positive density threshold")
    sigma = density ** (4.0 / 3.0)
    needs_tau = "XC_FLAGS_NEEDS_TAU" in flags
    return LibxcWorkPolicy(
        density_threshold=density,
        sigma_threshold=sigma,
        tau_threshold=1.0e-20 if needs_tau else 0.0,
        needs_tau=needs_tau,
        enforce_fhc="XC_FLAGS_ENFORCE_FHC" in flags,
    )


def build_split_global_hybrid(
    identifier: str, *, spin: str = "polarized"
) -> SplitGlobalHybridProgram:
    """Build semilocal exchange/correlation programs for one split global hybrid."""

    inventory = _method_inventory()
    try:
        method = inventory[identifier]
    except KeyError as error:
        raise MapleImportError(
            f"unknown or unrepresentable split global hybrid: {identifier!r}"
        ) from error
    exchange_name = method["split_exchange_component"]
    correlation_name = method["paired_correlation_component"]
    exchange_record = _bound_component(
        exchange_name, allow_hybrid_exchange=True
    )
    correlation_record = _bound_component(
        correlation_name, allow_hybrid_exchange=False
    )
    exact = Fraction(method["exact_exchange"])
    source_exact = Fraction(exchange_record["exact_exchange_parameter"])
    if source_exact != exact:
        raise MapleImportError(
            f"split-hybrid exact exchange mismatch for {identifier}: "
            f"{source_exact} != {exact}"
        )
    catalog = _catalog()
    root = _root()
    exchange = libxc_bulk.build_record(
        exchange_record, catalog["source_files"], root, spin=spin
    )
    correlation = libxc_bulk.build_record(
        correlation_record, catalog["source_files"], root, spin=spin
    )
    if exchange.features != correlation.features:
        raise MapleImportError(
            f"split-hybrid feature mismatch for {identifier}: "
            f"{exchange.features!r} != {correlation.features!r}"
        )
    return SplitGlobalHybridProgram(
        identifier=identifier,
        exact_exchange=exact,
        exchange_registration=exchange_name,
        correlation_registration=correlation_name,
        exchange=exchange,
        correlation=correlation,
        exchange_policy=_work_policy(exchange_record),
        correlation_policy=_work_policy(correlation_record),
    )
