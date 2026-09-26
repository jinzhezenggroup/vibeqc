"""Qualify an imported Libxc semilocal functional through native molecular SCF.

The generated, evidence-bound CPU point program is linked against the existing
generic RKS/UKS consumer. Native cold, warm-replay, and changed-geometry
endpoints are compared with independent PySCF 2.14.0 / Libxc 7.0.0 SCF on the
exact native quadrature points and weights. The resulting receipt is the
molecular-SCF evidence consumed by automatic Libxc promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from vibeqc_compiler.common.evidence import canonical_hash
from vibeqc_compiler.method.bulk_ks import resolve_bulk_ks_candidate
from vibeqc_compiler.xc.bulk_point_program import bind_runtime_semilocal_point_program
from vibeqc_compiler.xc.bulk_runtime import (
    PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    build_bulk_runtime_program,
)
from vibeqc_compiler.xc.molecular_scf_evidence import (
    build_result,
    stage_evidence,
)

from tools.generate_validation_references import pyscf_molecule

CAMPAIGN_SCHEMA = "vibeqc.libxc-molecular-scf-campaign/v1"
FIXTURE_SCHEMA = "vibeqc.libxc-molecular-scf-fixture/v1"
REFERENCE_SCHEMA = "vibeqc.libxc-molecular-scf-reference/v1"
SPINS = ("polarized", "unpolarized")
PHASES = ("cold", "warm-replay", "changed-geometry")


def _stage_payload(value: Mapping[str, Any], expected: str) -> dict[str, Any]:
    candidate: Any = value.get("stage_evidence", value)
    if not isinstance(candidate, Mapping):
        raise TypeError(f"{expected} evidence must be a mapping")
    if candidate.get("stage") != expected:
        raise ValueError(f"expected {expected} stage evidence")
    return dict(candidate)


def _load_stage(path: Path, expected: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError(f"{expected} evidence file must contain an object")
    return _stage_payload(raw, expected)


def _runner_source(binding_source: str, binding_identity: str) -> str:
    suffix = f"""
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"

namespace vibeqc::dft::bulk_generated {{
const SemilocalPointProgram& qualification_program() {{ return kPointProgram; }}
}}

namespace {{
using namespace vibeqc;

core::System h2(double displacement = 0.0) {{
  core::System system;
  system.atoms = {{{{1, {{0.0, 0.0, 0.0}}}}, {{1, {{0.1, 0.2, 1.4 + displacement}}}}}};
  system.shells = {{
      {{0, 0, {{{{3.425250914, 0.1543289673}},
                {{0.6239137298, 0.5353281423}},
                {{0.168855404, 0.4446345422}}}}}},
      {{1, 0, {{{{3.425250914, 0.1543289673}},
                {{0.6239137298, 0.5353281423}},
                {{0.168855404, 0.4446345422}}}}}}}};
  std::string detail;
  if (molecule::validate_and_normalize(system, detail) != VIBEQC_STATUS_SUCCESS)
    throw std::runtime_error("invalid molecular-SCF qualification fixture: " + detail);
  return system;
}}

void print_grid(std::string_view label, const dft::MolecularGrid& grid) {{
  std::cout << "GRID " << label << ' ' << grid.point_count();
  std::cout << std::setprecision(17);
  for (double value : grid.points()) std::cout << ' ' << value;
  for (double value : grid.weights()) std::cout << ' ' << value;
  std::cout << '\\n';
}}

void print_row(std::string_view spin, std::string_view phase, const scf::ScfResult& result) {{
  std::cout << "ROW " << spin << ' ' << phase << ' ' << (result.converged ? 1 : 0)
            << ' ' << result.iterations << ' ' << std::setprecision(17) << result.energy
            << ' ' << result.physical_residual_rms
            << ' ' << (result.initial_density_used ? 1 : 0) << '\\n';
}}

void run_spin(std::string_view spin, bool unrestricted) {{
  const auto cold_system = h2();
  const auto moved_system = h2(0.08);
  const dft::AoBasis cold_basis(cold_system);
  const dft::AoBasis moved_basis(moved_system);
  const dft::GridSpec grid_spec{{1, 1, 2, 4, 3, 1e-12}};
  const dft::MolecularGrid cold_grid(cold_system, grid_spec);
  const dft::MolecularGrid moved_grid(moved_system, grid_spec);

  scf::FockBuildSpec spec;
  spec.spin = unrestricted ? scf::FockSpin::Unrestricted : scf::FockSpin::Restricted;
  spec.exchange.present = false;
  spec.derivative_order = 0;
  const auto strategy = scf::resolve_fock_build(spec, scf::FockBackend::Cpu);
  const scf::PreparedFockPlan cold_plan(cold_system, nullptr, strategy);
  const scf::PreparedFockPlan moved_plan(moved_system, nullptr, strategy);

  scf::ScfOptions options;
  options.compute_forces = false;
  options.max_iterations = 200;
  options.diis_history = 8;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;

  const auto& program = dft::bulk_generated::qualification_program();
  const auto cold =
      unrestricted ? scf::run_semilocal_uks(cold_plan, cold_basis, cold_grid, options, program)
                   : scf::run_semilocal_rks(cold_plan, cold_basis, cold_grid, options, program);
  const auto warm = unrestricted
                        ? scf::run_semilocal_uks(cold_plan, cold_basis, cold_grid, options, program,
                                                 &cold.density)
                        : scf::run_semilocal_rks(cold_plan, cold_basis, cold_grid, options, program,
                                                 &cold.density);
  const auto moved = unrestricted
                         ? scf::run_semilocal_uks(moved_plan, moved_basis, moved_grid, options,
                                                  program, &cold.density)
                         : scf::run_semilocal_rks(moved_plan, moved_basis, moved_grid, options,
                                                  program, &cold.density);
  print_row(spin, "cold", cold);
  print_row(spin, "warm-replay", warm);
  print_row(spin, "changed-geometry", moved);
}}

}}  // namespace

int main() {{
  try {{
    const auto cold_system = h2();
    const auto moved_system = h2(0.08);
    const dft::GridSpec grid_spec{{1, 1, 2, 4, 3, 1e-12}};
    const dft::MolecularGrid cold_grid(cold_system, grid_spec);
    const dft::MolecularGrid moved_grid(moved_system, grid_spec);
    const auto& program = dft::bulk_generated::qualification_program();
    std::cout << "META {binding_identity} " << program.expression_identity << ' '
              << program.ingredient_mask << ' ' << program.domain_version << '\\n';
    print_grid("cold", cold_grid);
    print_grid("changed-geometry", moved_grid);
    run_spin("polarized", true);
    run_spin("unpolarized", false);
    return 0;
  }} catch (const std::exception& error) {{
    std::cerr << error.what() << '\\n';
    return 1;
  }}
}}
"""
    return binding_source + "\n" + suffix


def _find_library(build_dir: Path) -> Path:
    direct = (
        build_dir / "libvibeqc.so",
        build_dir / "libvibeqc.dylib",
        build_dir / "vibeqc.dll",
    )
    for candidate in direct:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(
        path.resolve()
        for pattern in ("libvibeqc.so", "libvibeqc.dylib", "vibeqc.dll")
        for path in build_dir.rglob(pattern)
        if path.is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one built VibeQC library below {build_dir}, found {matches}"
        )
    return matches[0]


def _compiler(cxx: str | None) -> Path:
    requested = cxx or os.environ.get("CXX")
    found = shutil.which(requested) if requested else None
    if found is None:
        found = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if found is None:
        raise RuntimeError("C++ compiler unavailable for molecular-SCF qualification")
    return Path(found).resolve()


def _compile_and_run(
    source: str,
    *,
    build_dir: Path,
    cxx: str | None,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("molecular-SCF timeout must be positive and finite")
    library = _find_library(build_dir)
    compiler = _compiler(cxx)
    with tempfile.TemporaryDirectory(prefix="vibeqc-libxc-scf-") as directory:
        root = Path(directory)
        source_path = root / "molecular_scf.cpp"
        executable = root / "molecular_scf"
        source_path.write_text(source, encoding="utf-8")
        command = [
            str(compiler),
            "-std=c++20",
            "-O2",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            str(source_path),
            "-L",
            str(library.parent),
            "-lvibeqc",
            "-pthread",
            f"-Wl,-rpath,{library.parent}",
            "-o",
            str(executable),
        ]
        compiled = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if compiled.returncode != 0:
            raise RuntimeError(
                "molecular-SCF qualification compilation failed: "
                + (compiled.stdout + compiled.stderr).strip()
            )
        executed = subprocess.run(
            [str(executable)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if executed.returncode != 0:
            raise RuntimeError(
                "molecular-SCF qualification execution failed: "
                + (executed.stdout + executed.stderr).strip()
            )
    metadata = {
        "compiler": str(compiler),
        "library": str(library),
        "translation_unit_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "stdout_sha256": hashlib.sha256(executed.stdout.encode("utf-8")).hexdigest(),
    }
    return executed.stdout, metadata


def _parse_native(
    output: str,
    *,
    binding_identity: str,
    expression_identity: str,
    ingredient_mask: int,
    domain_version: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: dict[str, dict[str, Any]] = {}
    grids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metadata_seen = False
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "META":
            if len(fields) != 5:
                raise ValueError("native molecular-SCF metadata is malformed")
            if (
                fields[1] != binding_identity
                or fields[2] != expression_identity
                or int(fields[3]) != ingredient_mask
                or int(fields[4]) != domain_version
            ):
                raise ValueError("native molecular-SCF program identity mismatch")
            metadata_seen = True
        elif fields[0] == "GRID":
            if len(fields) < 4:
                raise ValueError("native molecular-SCF grid record is malformed")
            label = fields[1]
            count = int(fields[2])
            values = np.asarray([float(item) for item in fields[3:]], dtype=np.float64)
            if values.size != 4 * count:
                raise ValueError("native molecular-SCF grid record has wrong size")
            grids[label] = (values[: 3 * count].reshape(count, 3), values[3 * count :])
        elif fields[0] == "ROW":
            if len(fields) != 8:
                raise ValueError("native molecular-SCF row is malformed")
            spin, phase = fields[1], fields[2]
            if spin not in SPINS or phase not in PHASES:
                raise ValueError("native molecular-SCF row has unsupported spin/phase")
            key = f"{spin}:{phase}"
            if key in rows:
                raise ValueError("duplicate native molecular-SCF row")
            rows[key] = {
                "spin": spin,
                "phase": phase,
                "converged": fields[3] == "1",
                "iterations": int(fields[4]),
                "energy_hartree": float(fields[5]),
                "physical_residual_rms": float(fields[6]),
                "initial_density_used": fields[7] == "1",
            }
        else:
            raise ValueError(f"unexpected native molecular-SCF output record {fields[0]!r}")
    if not metadata_seen:
        raise ValueError("native molecular-SCF output omitted program metadata")
    expected_rows = {f"{spin}:{phase}" for spin in SPINS for phase in PHASES}
    if set(rows) != expected_rows:
        raise ValueError("native molecular-SCF output omitted required lifecycle rows")
    if set(grids) != {"cold", "changed-geometry"}:
        raise ValueError("native molecular-SCF output omitted exact quadrature grids")
    return rows, grids


def _inputs(displacement: float) -> dict[str, Any]:
    return {
        "atomic_numbers": [1, 1],
        "coordinates": [[0.0, 0.0, 0.0], [0.1, 0.2, 1.4 + displacement]],
        "shells": [
            {
                "atom_index": atom,
                "angular_momentum": 0,
                "primitives": [
                    [3.425250914, 0.1543289673],
                    [0.6239137298, 0.5353281423],
                    [0.168855404, 0.4446345422],
                ],
            }
            for atom in (0, 1)
        ],
        "basis_representation": "cartesian",
        "charge": 0,
        "multiplicity": 1,
    }


def _physical_residual_rms(
    fock: np.ndarray, density: np.ndarray, overlap: np.ndarray
) -> float:
    if fock.ndim == 2:
        blocks = ((fock, density),)
    else:
        blocks = tuple(zip(fock, density, strict=True))
    residuals = [
        fock_block @ density_block @ overlap - overlap @ density_block @ fock_block
        for fock_block, density_block in blocks
    ]
    joined = np.concatenate([item.ravel() for item in residuals])
    return float(np.sqrt(np.mean(joined * joined)))


def _reference(
    name: str,
    spin: str,
    inputs: dict[str, Any],
    points: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    import pyscf
    from pyscf import dft, lib

    if pyscf.__version__ != "2.14.0" or dft.libxc.__version__ != "7.0.0":
        raise RuntimeError(
            "molecular-SCF qualification requires PySCF 2.14.0 / Libxc 7.0.0"
        )
    lib.num_threads(1)
    mol, _, _ = pyscf_molecule(inputs)
    solver = dft.UKS(mol) if spin == "polarized" else dft.RKS(mol)
    solver.xc = name
    solver.grids.coords = np.asarray(points, dtype=np.float64)
    solver.grids.weights = np.asarray(weights, dtype=np.float64)
    solver.grids.non0tab = None
    solver.small_rho_cutoff = 0.0
    solver.conv_tol = 1.0e-12
    solver.conv_tol_grad = 1.0e-10
    solver.max_cycle = 200
    solver.diis_space = 8
    solver.direct_scf_tol = 1.0e-14
    energy = float(solver.kernel())
    density = np.asarray(solver.make_rdm1(), dtype=np.float64)
    fock = np.asarray(solver.get_fock(dm=density), dtype=np.float64)
    overlap = np.asarray(solver.get_ovlp(), dtype=np.float64)
    return {
        "energy_hartree": energy,
        "converged": bool(solver.converged),
        "physical_residual_rms": _physical_residual_rms(fock, density, overlap),
        "pyscf": pyscf.__version__,
        "libxc": dft.libxc.__version__,
    }


def _geometry_identity(inputs: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "schema": "vibeqc.libxc-molecular-scf-geometry/v1",
            "atomic_numbers": inputs["atomic_numbers"],
            "coordinates": inputs["coordinates"],
            "charge": inputs["charge"],
            "multiplicity": inputs["multiplicity"],
        }
    )


def _fixture_identity(
    name: str,
    spin: str,
    inputs: Mapping[str, Any],
    points: np.ndarray,
    weights: np.ndarray,
) -> str:
    return canonical_hash(
        {
            "schema": FIXTURE_SCHEMA,
            "functional": name,
            "spin": spin,
            "basis": inputs["shells"],
            "grid_points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
            "grid_weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        }
    )


def _reference_identity(
    name: str,
    spin: str,
    geometry_identity: str,
    fixture_identity: str,
    reference: Mapping[str, Any],
) -> str:
    return canonical_hash(
        {
            "schema": REFERENCE_SCHEMA,
            "functional": name,
            "spin": spin,
            "geometry_identity": geometry_identity,
            "fixture_identity": fixture_identity,
            "pyscf": reference["pyscf"],
            "libxc": reference["libxc"],
            "scf": {
                "conv_tol": "1e-12",
                "conv_tol_grad": "1e-10",
                "max_cycle": 200,
                "direct_scf_tol": "1e-14",
            },
        }
    )


def qualify_molecular_scf(
    name: str,
    *,
    compiled_cpu_evidence: Mapping[str, Any],
    production_domain_evidence: Mapping[str, Any],
    build_dir: Path,
    evidence: str,
    cxx: str | None = None,
    timeout: float = 120.0,
    energy_tolerance: float = 1.0e-8,
    residual_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Run exact dual-spin native/PySCF lifecycle qualification."""
    for value, label in (
        (energy_tolerance, "energy tolerance"),
        (residual_tolerance, "residual tolerance"),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and nonnegative")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("molecular-SCF qualification requires an evidence reference")

    prerequisites = {
        "compiled-cpu": _stage_payload(compiled_cpu_evidence, "compiled-cpu"),
        "production-domain": _stage_payload(
            production_domain_evidence, "production-domain"
        ),
    }
    resolutions = {
        spin: resolve_bulk_ks_candidate(
            name,
            spin=spin,
            backend="cpu",
            evidence=prerequisites,
        )
        for spin in SPINS
    }

    program = build_bulk_runtime_program(
        name,
        spin="polarized",
        order=1,
        domain=PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    )
    binding = bind_runtime_semilocal_point_program(program)
    for resolution in resolutions.values():
        if resolution.compiled_cpu_binding_identity != binding.identity:
            raise RuntimeError(
                "molecular-SCF point binding disagrees with compiled-CPU evidence"
            )

    source = _runner_source(binding.emit_source(), binding.identity)
    output, native_metadata = _compile_and_run(
        source,
        build_dir=build_dir,
        cxx=cxx,
        timeout=timeout,
    )
    native_rows, grids = _parse_native(
        output,
        binding_identity=binding.identity,
        expression_identity=binding.point_expression_identity,
        ingredient_mask=binding.ingredient_mask,
        domain_version=binding.domain_version,
    )

    cold_inputs = _inputs(0.0)
    moved_inputs = _inputs(0.08)
    geometry = {
        "cold": _geometry_identity(cold_inputs),
        "changed-geometry": _geometry_identity(moved_inputs),
    }
    references: dict[str, dict[str, Any]] = {}
    fixtures: dict[str, str] = {}
    for spin in SPINS:
        for phase, inputs in (
            ("cold", cold_inputs),
            ("changed-geometry", moved_inputs),
        ):
            points, weights = grids[phase]
            reference = _reference(name, spin, inputs, points, weights)
            key = f"{spin}:{phase}"
            references[key] = reference
            fixtures[key] = _fixture_identity(
                name, spin, inputs, points, weights
            )

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for spin in SPINS:
        for phase in PHASES:
            native = native_rows[f"{spin}:{phase}"]
            reference_phase = (
                "changed-geometry" if phase == "changed-geometry" else "cold"
            )
            reference = references[f"{spin}:{reference_phase}"]
            fixture_identity = fixtures[f"{spin}:{reference_phase}"]
            geometry_identity = geometry[reference_phase]
            reference_identity = _reference_identity(
                name,
                spin,
                geometry_identity,
                fixture_identity,
                reference,
            )
            energy_error = abs(
                native["energy_hartree"] - reference["energy_hartree"]
            )
            failures = []
            if not native["converged"]:
                failures.append("native SCF did not converge")
            if not reference["converged"]:
                failures.append("independent SCF did not converge")
            if energy_error > energy_tolerance:
                failures.append(
                    f"energy mismatch {energy_error:.6e} > {energy_tolerance:.6e}"
                )
            if native["physical_residual_rms"] > residual_tolerance:
                failures.append(
                    "native physical residual exceeds qualification tolerance"
                )
            if reference["physical_residual_rms"] > residual_tolerance:
                failures.append(
                    "independent physical residual exceeds qualification tolerance"
                )
            if phase != "cold" and not native["initial_density_used"]:
                failures.append("native lifecycle phase did not consume its warm density")
            status = "pass" if not failures else "fail"
            reason = None if not failures else "; ".join(failures)
            rows.append(
                {
                    "spin": spin,
                    "phase": phase,
                    "status": status,
                    "reason": reason,
                    "fixture_identity": fixture_identity,
                    "geometry_identity": geometry_identity,
                    "reference_identity": reference_identity,
                    "converged": bool(native["converged"]),
                    "iterations": int(native["iterations"]),
                    "energy_hartree": float(native["energy_hartree"]),
                    "reference_energy_hartree": float(reference["energy_hartree"]),
                    "absolute_energy_error_hartree": float(energy_error),
                    "energy_tolerance_hartree": float(energy_tolerance),
                    "physical_residual_rms": float(native["physical_residual_rms"]),
                    "residual_tolerance": float(residual_tolerance),
                }
            )
            details.append(
                {
                    "spin": spin,
                    "phase": phase,
                    "native": native,
                    "independent": reference,
                    "absolute_energy_error_hartree": energy_error,
                    "status": status,
                    "reason": reason,
                }
            )

    receipt = build_result(name, resolutions, rows, evidence=evidence)
    envelope = stage_evidence(name, receipt)
    return {
        "schema": CAMPAIGN_SCHEMA,
        "functional": name,
        "point_binding": binding.to_payload(),
        "native_execution": native_metadata,
        "gates": {
            "absolute_energy_error_hartree": energy_tolerance,
            "physical_residual_rms": residual_tolerance,
        },
        "details": details,
        "receipt": receipt,
        "stage_evidence": envelope,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--compiled-cpu", type=Path, required=True)
    parser.add_argument("--production-domain", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--cxx")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = qualify_molecular_scf(
        args.name,
        compiled_cpu_evidence=_load_stage(args.compiled_cpu, "compiled-cpu"),
        production_domain_evidence=_load_stage(
            args.production_domain, "production-domain"
        ),
        build_dir=args.build_dir,
        evidence=args.evidence,
        cxx=args.cxx,
        timeout=args.timeout,
        energy_tolerance=args.energy_tolerance,
        residual_tolerance=args.residual_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    status = payload["stage_evidence"]["status"]
    print(f"{args.name}: molecular-scf={status} -> {args.output}")
    return 1 if args.require_pass and status != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
