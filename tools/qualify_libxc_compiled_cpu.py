"""Compile and execute one exact automatic Libxc CPU point binding.

The scientific production-domain oracle is separate. This tool proves that the
content-addressed `SemilocalPointBinding` can be compiled as C++ and that the
resulting native point ABI executes the same first-order mathematical program.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.compiler_process import run_compiler
from vibeqc_compiler.common.provenance import atomic_json, canonical_hash, file_hash
from vibeqc_compiler.xc.bulk_point_program import (
    bind_runtime_semilocal_point_program,
)
from vibeqc_compiler.xc.bulk_runtime import (
    PRODUCTION_CANDIDATE_DOMAIN,
    build_bulk_runtime_program,
)
from vibeqc_compiler.xc.compiled_cpu_evidence import build_result, stage_evidence

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCHEMA = "vibeqc.libxc-compiled-cpu-smoke-input/v1"


def _smoke_input() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.asarray([0.7, 0.4], dtype=np.float64)
    gradient = np.asarray([[0.1, 0.2, 0.05], [0.05, -0.1, 0.15]], dtype=np.float64)
    tau = np.asarray([0.35, 0.21], dtype=np.float64)
    return rho, gradient, tau


def _feature_values(
    features: tuple[str, ...],
    rho: np.ndarray,
    gradient: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    sigma = {
        "sigma_aa": float(np.dot(gradient[0], gradient[0])),
        "sigma_ab": float(np.dot(gradient[0], gradient[1])),
        "sigma_bb": float(np.dot(gradient[1], gradient[1])),
    }
    values = {
        "rho_a": float(rho[0]),
        "rho_b": float(rho[1]),
        **sigma,
        "tau_a": float(tau[0]),
        "tau_b": float(tau[1]),
    }
    return np.asarray([values[name] for name in features], dtype=np.float64)


def _native_expected(
    raw: np.ndarray,
    ingredient_mask: int,
    gradient: np.ndarray,
) -> list[float]:
    result = [float(raw[0]), float(raw[1]), float(raw[2])]
    if ingredient_mask == 1:
        result.extend([0.0] * 6)
    else:
        alpha = 2.0 * raw[3] * gradient[0] + raw[4] * gradient[1]
        beta = raw[4] * gradient[0] + 2.0 * raw[5] * gradient[1]
        result.extend(float(value) for value in alpha)
        result.extend(float(value) for value in beta)
    if ingredient_mask == 15:
        result.extend((0.5 * float(raw[6]), 0.5 * float(raw[7])))
    else:
        result.extend((0.0, 0.0))
    return result


def _cpp_array(values: np.ndarray) -> str:
    return ", ".join(repr(float(value)) for value in values.reshape(-1))


def _translation_unit(
    binding_source: str,
    binding_identity: str,
    domain_version: int,
    rho: np.ndarray,
    gradient: np.ndarray,
    tau: np.ndarray,
) -> str:
    return (
        binding_source
        + """
#include <iomanip>
#include <iostream>

int main() {
"""
        + f'  std::cout << "{binding_identity}" << " {domain_version}\\n";\n'
        + f"  const double rho[2]{{{_cpp_array(rho)}}};\n"
        + "  const double gradient[2][3]{{"
        + _cpp_array(gradient[0])
        + "}, {"
        + _cpp_array(gradient[1])
        + "}};\n"
        + f"  const double tau[2]{{{_cpp_array(tau)}}};\n"
        + """  const auto value = vibeqc::dft::bulk_generated::evaluate_point(
      rho, gradient, tau);
  std::cout << std::setprecision(17)
            << value.energy << ' ' << value.rho[0] << ' ' << value.rho[1];
  for (const auto& spin : value.gradient)
    for (double item : spin) std::cout << ' ' << item;
  std::cout << ' ' << value.kinetic[0] << ' ' << value.kinetic[1] << '\\n';
}
"""
    )


def _compiler_identity(path: Path, timeout: float) -> tuple[dict, str | None]:
    try:
        result = run_compiler(
            [str(path), "--version"],
            timeout,
            label="Libxc compiled-CPU compiler identification",
        )
    except OSError as exc:
        return {}, f"compiler identification failed: {exc}"
    if result.timed_out or result.returncode != 0:
        return {}, "compiler identification failed: " + result.stdout + result.stderr
    return {
        "executable_sha256": file_hash(path),
        "version": (result.stdout + result.stderr).strip(),
    }, None


def qualify_compiled_cpu(
    name: str,
    *,
    evidence: str,
    cxx: str | None = None,
    timeout: float = 60.0,
    absolute_tolerance: float = 2.0e-12,
) -> dict:
    """Compile one production-candidate binding and return exact stage evidence."""
    if timeout <= 0.0 or not np.isfinite(timeout):
        raise ValueError("compiled-CPU timeout must be positive and finite")
    if absolute_tolerance < 0.0 or not np.isfinite(absolute_tolerance):
        raise ValueError("compiled-CPU tolerance must be nonnegative and finite")

    program = build_bulk_runtime_program(
        name,
        spin="polarized",
        order=1,
        domain=PRODUCTION_CANDIDATE_DOMAIN,
    )
    binding = bind_runtime_semilocal_point_program(program)
    rho, gradient, tau = _smoke_input()
    features = _feature_values(program.spec.features, rho, gradient, tau)
    raw = program.evaluate(features.reshape(-1, 1))[:, 0]
    expected = _native_expected(raw, binding.ingredient_mask, gradient)
    input_payload = {
        "schema": SMOKE_SCHEMA,
        "features": list(program.spec.features),
        "rho": rho.tolist(),
        "gradient": gradient.tolist(),
        "tau": tau.tolist(),
    }
    input_identity = canonical_hash(input_payload)

    compiler = (
        shutil.which(cxx)
        if cxx
        else (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"))
    )
    if compiler is None:
        outcome = {
            "status": "not-run",
            "reason": "c++ compiler unavailable",
            "compiler": None,
            "translation_unit_sha256": None,
            "executable_sha256": None,
            "smoke": None,
        }
        result = build_result(name, binding, outcome, evidence=evidence)
        return {"result": result, "stage_evidence": stage_evidence(name, result)}

    compiler_path = Path(compiler).resolve()
    compiler_identity, compiler_error = _compiler_identity(compiler_path, timeout)
    if compiler_error is not None:
        outcome = {
            "status": "fail",
            "reason": compiler_error,
            "compiler": None,
            "translation_unit_sha256": None,
            "executable_sha256": None,
            "smoke": None,
        }
        result = build_result(name, binding, outcome, evidence=evidence)
        return {"result": result, "stage_evidence": stage_evidence(name, result)}

    source = _translation_unit(
        binding.emit_source(),
        binding.identity,
        binding.domain_version,
        rho,
        gradient,
        tau,
    )
    translation_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()

    with tempfile.TemporaryDirectory(prefix="vibeqc-libxc-cpu-") as directory:
        root = Path(directory)
        source_path = root / "point.cpp"
        executable_path = root / "point"
        source_path.write_text(source, encoding="utf-8")
        try:
            compiled = run_compiler(
                [
                    str(compiler_path),
                    "-std=c++20",
                    "-O2",
                    "-I",
                    str(ROOT / "src"),
                    "-I",
                    str(ROOT / "include"),
                    str(source_path),
                    "-o",
                    str(executable_path),
                ],
                timeout,
                label="Libxc compiled-CPU point binding",
            )
        except OSError as exc:
            compiled = None
            compile_reason = f"compiler execution failed: {exc}"
        else:
            compile_reason = None

        if (
            compiled is None
            or compiled.timed_out
            or compiled.returncode != 0
            or not executable_path.is_file()
        ):
            if compile_reason is None:
                diagnostic = (
                    ""
                    if compiled is None
                    else (compiled.stdout + compiled.stderr).replace(
                        str(root), "<build>"
                    )
                )
                compile_reason = "point binding compilation failed"
                if diagnostic.strip():
                    compile_reason += ": " + diagnostic.strip()
            outcome = {
                "status": "fail",
                "reason": compile_reason,
                "compiler": compiler_identity,
                "translation_unit_sha256": translation_sha,
                "executable_sha256": None,
                "smoke": None,
            }
        else:
            executable_sha = file_hash(executable_path)
            try:
                executed = subprocess.run(
                    [str(executable_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                execution_reason = f"compiled point execution failed: {exc}"
                observed: list[float] = []
                metadata_ok = False
            else:
                lines = executed.stdout.splitlines()
                metadata = lines[0].split() if lines else []
                metadata_ok = (
                    executed.returncode == 0
                    and len(metadata) == 2
                    and metadata[0] == binding.identity
                    and metadata[1] == str(binding.domain_version)
                    and len(lines) >= 2
                )
                try:
                    observed = (
                        [float(value) for value in lines[1].split()]
                        if len(lines) >= 2
                        else []
                    )
                except ValueError:
                    observed = []
                execution_reason = None

            aligned = len(observed) == len(expected)
            max_error = (
                max(
                    abs(left - right)
                    for left, right in zip(observed, expected, strict=True)
                )
                if aligned
                else float("inf")
            )
            finite_observed = bool(np.isfinite(observed).all())
            smoke_pass = (
                execution_reason is None
                and metadata_ok
                and aligned
                and finite_observed
                and max_error <= absolute_tolerance
            )
            finite_error = max_error if np.isfinite(max_error) else 1.0e300
            smoke = {
                "status": "pass" if smoke_pass else "fail",
                "input_identity": input_identity,
                "expected": expected,
                "observed": observed if finite_observed else [],
                "absolute_tolerance": absolute_tolerance,
                "maximum_absolute_error": finite_error,
            }
            reason = None
            if not smoke_pass:
                reason = execution_reason or (
                    "compiled point smoke mismatch: "
                    f"metadata_ok={metadata_ok}, aligned={aligned}, "
                    f"max_abs={finite_error:.17g}"
                )
            outcome = {
                "status": "pass" if smoke_pass else "fail",
                "reason": reason,
                "compiler": compiler_identity,
                "translation_unit_sha256": translation_sha,
                "executable_sha256": executable_sha,
                "smoke": smoke,
            }

    result = build_result(name, binding, outcome, evidence=evidence)
    return {"result": result, "stage_evidence": stage_evidence(name, result)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--cxx")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-12)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = qualify_compiled_cpu(
        args.name,
        evidence=args.evidence,
        cxx=args.cxx,
        timeout=args.timeout,
        absolute_tolerance=args.absolute_tolerance,
    )
    atomic_json(args.output, payload)
    status = payload["stage_evidence"]["status"]
    print(f"{args.name}: compiled-cpu={status} -> {args.output}")
    return 1 if args.require_pass and status != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
