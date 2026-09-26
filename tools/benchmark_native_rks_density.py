"""Measure explicit D/C through the existing native RKS loop on larger bases.

The generated bridge contains input tables and calls run_lda_rks/run_pbe_rks;
it contains no SCF iteration or XC equations. Cold and common-density warm
starts use the same quadrature/model and convergence controls for both routes.
"""

import argparse
import csv
import os
import platform
import shutil
import subprocess
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from vibeqc.autotune import source_identity
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.evidence import (
    block_error,
    new_evidence,
    outcome,
    write_evidence,
)
from vibeqc_compiler.common.provenance import canonical_hash, file_hash

from tools.density_workload_matrix import load_workloads

BRIDGE = r"""
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include "dft/xc.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"
using namespace vibeqc;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point begin) {
  return std::chrono::duration<double>(Clock::now()-begin).count();
}
extern "C" int measure(const char* destination, unsigned repeats) {
 try {
  std::ofstream out(destination);
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "case,method,scenario,repeat,route,nao,nocc,npoint,iterations,energy,energy_error,density_error,residual,density_calls,orbital_calls,fallbacks,packed_elements,factor_peak,xc_peak,observed_peak,seconds,owner_setup_seconds\n";
  std::vector<std::pair<std::string,core::System>> inputs;
  @INPUTS@
  for (auto& [name, system] : inputs) {
    auto started = Clock::now();
    std::string detail;
    if (molecule::validate_and_normalize(system,detail) != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail);
    scf::FockBuildSpec spec;
    spec.derivative_order=0;
    spec.exchange.present=false;
    scf::PreparedFockPlan plan(system,nullptr,scf::resolve_fock_build(spec,scf::FockBackend::Cpu));
    dft::AoBasis basis(system);
    dft::MolecularGrid grid(system,{1,16,8,16,3,1e-12});
    const double setup=seconds(started);
    scf::ScfOptions options;
    options.compute_forces=false;
    options.max_iterations=150;
    for (auto run : {scf::run_lda_rks, scf::run_pbe_rks}) {
      options.xc_density_route=dft::XcDensityRoute::DensityMatrix;
      const auto baseline=run(plan,basis,grid,options,nullptr);
      if (!baseline.converged) throw std::runtime_error("native D reference did not converge: "+name);
      for (unsigned warm : {0U,1U}) {
        for (unsigned repeat=0; repeat<repeats+1; ++repeat) {
          for (unsigned position : {0U,1U}) {
            const unsigned route=(position+repeat)%2;
            options.xc_density_route=route ? dft::XcDensityRoute::OccupiedOrbitals : dft::XcDensityRoute::DensityMatrix;
            options.strict_initial_density=bool(warm);
            runtime::cpu_resource_observation={.active=true};
            started=Clock::now();
            const auto result=run(plan,basis,grid,options,warm ? &baseline.density : nullptr);
            const double elapsed=seconds(started);
            const auto peak=runtime::cpu_resource_observation.peak_bytes;
            runtime::cpu_resource_observation={};
            const auto& diagnostic=result.xc_density_diagnostic;
            double error=0;
            for (std::size_t i=0;i<baseline.density.size();++i)
              error=std::max(error,std::abs(result.density[i]-baseline.density[i]));
            if (!result.converged || !std::isfinite(result.energy) || !std::isfinite(diagnostic.physical_residual) ||
                !std::all_of(result.density.begin(),result.density.end(),[](double x){return std::isfinite(x);}) || error>1e-8 || std::abs(result.energy-baseline.energy)>1e-9 ||
                diagnostic.physical_residual>options.density_tolerance ||
                diagnostic.density_calls+diagnostic.orbital_calls!=result.fock_builds ||
                (route && (!result.xc_density_factor || !result.xc_density_factor->matches(
                  diagnostic.final_identity,scf::DensityFactorSpin::Restricted,result.density))) ||
                diagnostic.fallback_calls!=(route && warm ? 1U : 0U))
              throw std::runtime_error("native D/C endpoint gate failed: "+name);
            if (repeat==0) continue; // Every route/scenario receives one unmeasured warmup.
            out << name << ',' << (run==scf::run_lda_rks ? "LDA_XC_PW" : "PBE-tail-v2") << ','
                << (warm ? "common-warm-density" : "cold-core-guess") << ',' << repeat-1 << ','
                << (route ? "orbitals" : "density_matrix") << ',' << basis.nao << ',' << system.electron_count/2 << ','
                << grid.point_count() << ',' << result.iterations << ',' << result.energy << ','
                << std::abs(result.energy-baseline.energy) << ',' << error << ',' << diagnostic.physical_residual << ','
                << diagnostic.density_calls << ',' << diagnostic.orbital_calls << ',' << diagnostic.fallback_calls << ','
                << diagnostic.packed_coefficient_elements << ',' << diagnostic.factor_peak_bytes << ','
                << diagnostic.xc_peak_bytes << ',' << peak+runtime::vector_bytes(baseline.density) << ',' << elapsed << ',' << setup << '\n';
            out.flush();
          }
        }
      }
    }
  }
  return out ? 0 : 2;
 } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
"""


def bridge_source(workloads: typing.Any) -> typing.Any:
    """Lower only validated finite molecular input tables into C++ literals."""
    blocks = []
    for name, meta, _, _ in workloads:
        if name not in ("water_svp", "water_tzvp"):
            continue
        inputs = meta["inputs"]
        lines = [
            "{ core::System system;",
            f"system.charge={inputs['charge']};",
            f"system.multiplicity={inputs['multiplicity']};",
            "system.basis_representation=VIBEQC_BASIS_SPHERICAL;",
        ]
        for z, xyz in zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True):
            lines.append(
                f"system.atoms.push_back({{{z},{{{','.join(map(repr, xyz))}}}}});"
            )
        for shell in inputs["shells"]:
            primitives = ",".join(
                "{" + ",".join(map(repr, p)) + "}" for p in shell["primitives"]
            )
            lines.append(
                f"system.shells.push_back({{{shell['atom_index']},{shell['angular_momentum']},{{{primitives}}}}});"
            )
        lines.append(f'inputs.emplace_back("{name}",std::move(system)); }}')
        blocks.append("\n".join(lines))
    if len(blocks) != 2:
        raise ValueError("native endpoint fixtures are incomplete")
    return BRIDGE.replace("@INPUTS@", "\n".join(blocks))


def run(args: typing.Any) -> None:
    """Compile via the standard CPU adapter and retain every native timing row."""
    if (
        args.samples < 5
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip()
    ):
        raise ValueError("requires a clean checkout and at least five paired repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    # Capture the execution host with the run; publication must not fill in
    # historical hardware provenance using a later publishing machine.
    device = {
        "host": platform.platform(),
        "cpu": subprocess.check_output(["lscpu"], text=True, timeout=30),
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    source = args.output / "bridge.cpp"
    source.write_text(bridge_source(load_workloads(args.workload_matrix)))
    library = args.library.resolve()
    # Bind to the actual built scientific source, independent of the benchmark
    # bridge's own hashed input tables and code.
    import ctypes

    native = ctypes.CDLL(str(library))
    native.vibeqc_get_source_identity.restype = ctypes.c_char_p
    if native.vibeqc_get_source_identity().decode() != source_identity(ROOT):
        raise ValueError("native source/library identity mismatch")
    output = args.output.resolve()
    compiler = CppCompilerAdapter(Path(shutil.which("c++")))
    built = compiler.compile_shared(
        source,
        output / "bridge.so",
        includes=(ROOT / "include", ROOT / "src"),
        libraries=("vibeqc",),
        options=("-std=c++20", f"-L{library.parent}", f"-Wl,-rpath,{library.parent}"),
    )
    if built.returncode:
        raise RuntimeError(built.stderr)
    worker = (
        "import ctypes,sys; p=ctypes.CDLL(sys.argv[1]); p.measure.argtypes=[ctypes.c_char_p,ctypes.c_uint]; "
        "p.measure.restype=ctypes.c_int; sys.exit(p.measure(sys.argv[2].encode(),int(sys.argv[3])))"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            worker,
            str(output / "bridge.so"),
            str(output / "samples.csv"),
            str(args.samples),
        ],
        check=True,
        timeout=900,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    with (output / "samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2 * 2 * 2 * 2 * args.samples:
        raise ValueError("incomplete native SCF endpoint matrix")
    columns = {"case", "method", "scenario", "route"}
    integer_columns = {
        "repeat",
        "nao",
        "nocc",
        "npoint",
        "iterations",
        "density_calls",
        "orbital_calls",
        "fallbacks",
        "packed_elements",
        "factor_peak",
        "xc_peak",
        "observed_peak",
    }
    rows = [
        {
            k: v if k in columns else int(v) if k in integer_columns else float(v)
            for k, v in row.items()
        }
        for row in rows
    ]
    inputs_hash = canonical_hash(
        {
            "manifest": file_hash(args.workload_matrix / "manifest.json"),
            "cases": ["water_svp", "water_tzvp"],
            "grid": [16, 8, 16],
        }
    )
    report = new_evidence(
        tier="endpoint",
        subject="#235 native CPU RKS D/C energy-only endpoints",
        inputs_hash=inputs_hash,
    )
    report.update(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        dirty=False,
        source_identity=source_identity(ROOT),
        library_sha256=file_hash(library),
        bridge_sha256=file_hash(source),
        bridge_binary_sha256=file_hash(output / "bridge.so"),
        samples_sha256=file_hash(output / "samples.csv"),
        backend_selected="cpu",
        device=device,
    )
    report["toolchain"] = {
        "cxx": subprocess.check_output(
            [str(compiler.cxx), "--version"], text=True, timeout=30
        ),
        "flags": ["-O3", "-std=c++20"],
        "python": sys.version,
        "threads": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    }
    report["settings"] = {
        "device": "cpu",
        "scope": "complete native energy-only RKS on GridSpec(16,8,16); PBE retains production tail-v2",
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "samples_per_route": args.samples,
        "preparation": "prepared provider/basis/grid setup measured separately; each sample runs the complete existing SCF loop",
    }
    report["hashes"]["source"] = source_identity(ROOT)
    report["hashes"]["schedule"] = canonical_hash(report["settings"])
    report["hash_reasons"] = {
        "equation": "existing native RKS/XC equations identified by native source hash",
        "ir": "native SCF consumer has no separate combined method IR",
    }
    report["hardware"] = outcome("pass", backend="native CPU execution")
    report["compilation"] = {
        "seconds": built.duration_seconds,
        "scope": "test-only native invocation bridge; native library prebuilt",
    }
    report["memory"] = {
        "allocated_bytes": max(r["observed_peak"] for r in rows),
        "peak_bytes": None,
        "reason": "composed native numeric-capacity observer plus retained baseline density; not allocator/RSS peak",
    }
    for index, row in enumerate(rows):
        case = "/".join(row[k] for k in ("case", "method", "scenario"))
        report["timings"].append(
            {
                "case": case,
                "repeat": row["repeat"],
                "selection": "baseline"
                if row["route"] == "density_matrix"
                else "candidate",
                "seconds": row["seconds"],
                "workload": "energy-only",
                "synchronized": True,
                "inputs_hash": canonical_hash({"inputs": inputs_hash, "case": case}),
                "diagnostics": row,
            }
        )
        for key, tolerance in (
            ("energy_error", 1e-9),
            ("density_error", 1e-8),
            ("residual", 1e-8),
        ):
            report["block_errors"][f"{index}/{key}"] = block_error(
                [row[key]], [0.0], atol=tolerance, rtol=0
            )
        report["residuals"][str(index)] = {
            "value": row["residual"],
            "scope": "final physical commutator residual from the native solver",
        }
    report["solver_trace_reason"] = (
        "existing internal result supplies counts and final residuals, not complete per-iteration histories; no promotion"
    )
    for stage in ("representation", "source", "compilation", "numerical", "endpoint"):
        report["stages"][stage] = outcome(
            "pass",
            scope="same-loop D/C energy/density/residual gates; independent XC fixtures validated separately",
        )
    report["stages"]["production"] = outcome(
        "not-run", "complete-force endpoint and #168 promotion are unavailable"
    )
    report["performance"] = outcome(
        "not-run",
        "raw complete energy-only measurements; no complete energy-plus-force selector promotion",
    )
    report["reproduction"] = {
        "command": [
            "env",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            sys.executable,
            *sys.argv,
        ]
    }
    write_evidence(output / "verification.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-matrix", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    run(parser.parse_args())
