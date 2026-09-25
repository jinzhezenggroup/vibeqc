"""Endpoint validation evidence for #1246 xTB occupied-subspace RKS guesses."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

BRIDGE = r"""
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dft/xc.hpp"
#include "integrals/s_integrals.hpp"
#include "methods/gfn2_runtime_bridge.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/linalg.hpp"

using namespace vibeqc;
using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

core::System oxygen_hydrogens(const std::vector<std::array<double, 3>>& hydrogens, int charge) {
  core::System system;
  system.charge = charge;
  system.multiplicity = 1;
  system.basis_representation = VIBEQC_BASIS_CARTESIAN;
  system.atoms.push_back({8, {0, 0, 0}});
  for (const auto& position : hydrogens) system.atoms.push_back({1, position});
  system.shells = {
      {0, 0, {{130.70932, 0.15432897}, {23.808861, 0.53532814}, {6.4436083, 0.44463454}}},
      {0, 0, {{5.0331513, -0.09996723}, {1.1695961, 0.39951283}, {0.3803890, 0.70011547}}},
      {0, 1, {{5.0331513, 0.15591627}, {1.1695961, 0.60768372}, {0.3803890, 0.39195739}}},
  };
  for (std::size_t atom = 1; atom < system.atoms.size(); ++atom) {
    system.shells.push_back(
        {static_cast<std::uint32_t>(atom), 0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423},
          {0.168855404, 0.4446345422}}});
  }
  std::string detail;
  if (molecule::validate_and_normalize(system, detail) != VIBEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail);
  return system;
}

struct Candidate {
  scf::ScfResult result;
  double xtb_seconds{};
  double seed_seconds{};
  double scf_seconds{};
};

Candidate run_candidate(const core::System& system, const scf::PreparedFockPlan& plan,
                        const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                        const scf::ScfOptions& options) {
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  for (const auto& atom : system.atoms) {
    atomic_numbers.push_back(atom.atomic_number);
    positions.insert(positions.end(), atom.position.begin(), atom.position.end());
  }

  auto started = Clock::now();
  methods::detail::Gfn2RuntimeBridge runtime(methods::detail::Gfn2RuntimeBackend::kCpu, 0);
  methods::detail::Gfn2RuntimeRequest request;
  request.atomic_numbers = atomic_numbers;
  request.positions = positions;
  request.charge = system.charge;
  request.multiplicity = system.multiplicity;
  request.compute_orbitals = true;
  request.maximum_iterations = 100;
  request.mixer_history = 8;
  request.energy_tolerance = 1.0e-10;
  request.charge_tolerance = 1.0e-8;
  const auto xtb = runtime.execute(request);
  const double xtb_elapsed = seconds(started);
  if (xtb.status != methods::detail::Gfn2RuntimeStatus::kSuccess || !xtb.converged)
    throw std::runtime_error("GFN2 orbital seed did not converge: " + xtb.detail);
  if (!xtb.orbitals)
    throw std::runtime_error("GFN2 orbital seed did not return its occupied frame");
  if (system.multiplicity != 1 || (system.electron_count & 1) != 0 ||
      xtb.orbitals->source_system.electron_count != system.electron_count)
    throw std::runtime_error("GFN2 orbital seed requires a matched closed-shell electron space");

  const std::size_t source_n = molecule::ao_count(xtb.orbitals->source_system);
  const std::size_t occupied = static_cast<std::size_t>(system.electron_count / 2);
  if (source_n == 0 || xtb.orbitals->overlap.size() != source_n * source_n ||
      xtb.orbitals->coefficients.size() != source_n * source_n ||
      xtb.orbitals->occupations.size() != 2u * source_n)
    throw std::runtime_error("GFN2 orbital seed returned inconsistent source dimensions");

  // The bridge translates GFN2's native spherical AO convention into VibeQC's
  // public source-basis convention. Verify that translation independently with
  // the ordinary cross-overlap evaluator before timing the actual projection.
  std::vector<double> rebuilt_source_overlap(source_n * source_n);
  integrals::cross_overlap(xtb.orbitals->source_system, xtb.orbitals->source_system,
                           rebuilt_source_overlap);
  double source_overlap_error = 0.0;
  for (std::size_t i = 0; i < rebuilt_source_overlap.size(); ++i)
    source_overlap_error =
        std::max(source_overlap_error,
                 std::abs(rebuilt_source_overlap[i] - xtb.orbitals->overlap[i]));
  if (source_overlap_error > 1.0e-9)
    throw std::runtime_error("GFN2 source-basis translation changed its overlap metric");

  // For this first Phase-B slice, reject fractional/ambiguous restricted
  // occupations rather than silently choosing an occupied subspace.
  for (std::size_t orbital = 0; orbital < source_n; ++orbital) {
    const double alpha = xtb.orbitals->occupations[orbital];
    const double beta = xtb.orbitals->occupations[source_n + orbital];
    if (std::abs(alpha - beta) > 1.0e-8 ||
        (orbital < occupied && alpha <= 0.5 + 1.0e-6) ||
        (orbital >= occupied && alpha >= 0.5 - 1.0e-6))
      throw std::runtime_error("GFN2 occupied subspace is fractional or spin-ambiguous");
  }

  started = Clock::now();
  const auto& ints = plan.one_electron();
  const auto x = scf::reference::symmetric_orthogonalizer(ints.overlap, ints.nbf);
  std::vector<double> cross(ints.nbf * source_n);
  integrals::cross_overlap(system, xtb.orbitals->source_system, cross);
  const auto projection = scf::initial_guess::project_occupied_density(
      ints, x, xtb.orbitals->overlap, cross, source_n, xtb.orbitals->coefficients, occupied);
  const auto& seed = projection.density;
  const double seed_elapsed = seconds(started);

  started = Clock::now();
  auto result = scf::run_pbe_rks(plan, basis, grid, options, &seed);
  const double scf_elapsed = seconds(started);
  return {std::move(result), xtb_elapsed, seed_elapsed, scf_elapsed};
}

int main() {
  const std::vector<std::pair<std::string, core::System>> cases{
      {"water", oxygen_hydrogens({{{0, -1.43233673, 1.10715266}},
                                   {{0, 1.43233673, 1.10715266}}}, 0)},
      {"hydroxide", oxygen_hydrogens({{{0, 0, 1.8}}}, -1)},
      {"hydronium", oxygen_hydrogens({{{0, 0, 1.8}},
                                      {{1.69705627, 0, -0.6}},
                                      {{-0.84852814, 1.46969385, -0.6}}}, 1)},
  };

  std::cout << "case,repeat,baseline_iterations,candidate_iterations,baseline_seconds,"
               "xtb_seconds,seed_seconds,candidate_scf_seconds,candidate_total_seconds,"
               "energy_difference,baseline_residual,candidate_residual\n";
  for (const auto& [name, system] : cases) {
    scf::FockBuildSpec spec;
    spec.derivative_order = 0;
    spec.exchange.present = false;
    scf::PreparedFockPlan plan(
        system, nullptr, scf::resolve_fock_build(spec, scf::FockBackend::Cpu));
    dft::AoBasis basis(system);
    dft::MolecularGrid grid(system, {1, 16, 8, 16, 3, 1e-12});
    scf::ScfOptions options;
    options.compute_forces = false;
    options.max_iterations = 150;
    options.energy_tolerance = 1.0e-10;
    options.density_tolerance = 1.0e-8;

    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      scf::ScfResult baseline;
      Candidate candidate;
      double baseline_elapsed = 0.0;
      if ((repeat & 1U) == 0U) {
        auto started = Clock::now();
        baseline = scf::run_pbe_rks(plan, basis, grid, options, nullptr);
        baseline_elapsed = seconds(started);
        candidate = run_candidate(system, plan, basis, grid, options);
      } else {
        candidate = run_candidate(system, plan, basis, grid, options);
        auto started = Clock::now();
        baseline = scf::run_pbe_rks(plan, basis, grid, options, nullptr);
        baseline_elapsed = seconds(started);
      }
      if (!baseline.converged || !candidate.result.converged)
        throw std::runtime_error("PBE RKS endpoint did not converge for " + name);
      const double energy_difference = std::abs(candidate.result.energy - baseline.energy);
      std::cout << name << ',' << repeat << ',' << baseline.iterations << ','
                << candidate.result.iterations << ',' << baseline_elapsed << ','
                << candidate.xtb_seconds << ',' << candidate.seed_seconds << ','
                << candidate.scf_seconds << ','
                << candidate.xtb_seconds + candidate.seed_seconds + candidate.scf_seconds << ','
                << energy_difference << ',' << baseline.physical_residual_rms << ','
                << candidate.result.physical_residual_rms << '\n';
    }
  }
}
"""


def _compile_bridge(tmp_path: Path) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    library = Path(os.environ["VIBEQC_LIBRARY"]).resolve()
    source = tmp_path / "xtb_orbital_guess_validation.cpp"
    source.write_text(BRIDGE)
    executable = tmp_path / "xtb_orbital_guess_validation"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
            str(source),
            str(library),
            f"-Wl,-rpath,{library.parent}",
            "-o",
            str(executable),
        ],
        check=True,
        timeout=120,
    )
    return executable


def _validate_measurements(rows: list[dict[str, str]]) -> None:
    for row_index, row in enumerate(rows):
        for field, value in row.items():
            if field not in {"case", "repeat"}:
                assert math.isfinite(float(value)), (
                    f"nonfinite measurement at row {row_index}, field {field}"
                )


def test_xtb_orbital_seed_endpoint_validation(tmp_path: Path) -> None:
    executable = _compile_bridge(tmp_path)
    completed = subprocess.run(
        [str(executable)],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    if completed.returncode != 0:
        pytest.fail(
            f"orbital validation executable failed with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    rows = list(csv.DictReader(completed.stdout.splitlines()))
    assert len(rows) == 9
    _validate_measurements(rows)

    summary = {}
    for case in sorted({row["case"] for row in rows}):
        selected = [row for row in rows if row["case"] == case]
        baseline_iterations = [int(row["baseline_iterations"]) for row in selected]
        candidate_iterations = [int(row["candidate_iterations"]) for row in selected]
        baseline_seconds = [float(row["baseline_seconds"]) for row in selected]
        candidate_total = [float(row["candidate_total_seconds"]) for row in selected]
        summary[case] = {
            "baseline_iterations": baseline_iterations,
            "candidate_iterations": candidate_iterations,
            "iteration_delta": [
                candidate - baseline
                for baseline, candidate in zip(
                    baseline_iterations, candidate_iterations, strict=True
                )
            ],
            "median_baseline_seconds": statistics.median(baseline_seconds),
            "median_candidate_total_seconds": statistics.median(candidate_total),
            "median_total_ratio": statistics.median(candidate_total)
            / statistics.median(baseline_seconds),
            "median_xtb_seconds": statistics.median(
                float(row["xtb_seconds"]) for row in selected
            ),
            "median_seed_seconds": statistics.median(
                float(row["seed_seconds"]) for row in selected
            ),
            "median_candidate_scf_seconds": statistics.median(
                float(row["candidate_scf_seconds"]) for row in selected
            ),
            "max_energy_difference": max(
                float(row["energy_difference"]) for row in selected
            ),
            "max_baseline_residual": max(
                float(row["baseline_residual"]) for row in selected
            ),
            "max_candidate_residual": max(
                float(row["candidate_residual"]) for row in selected
            ),
        }

    artifact = ROOT / ".artifacts" / "xtb-orbital-guess-validation.json"
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2, sort_keys=True)
    )
    assert all(
        metrics["max_energy_difference"] < 1.0e-8 for metrics in summary.values()
    )
