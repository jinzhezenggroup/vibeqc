"""Temporary endpoint validation for #1246 xTB-informed RKS guesses."""

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
  request.compute_atomic_charges = true;
  request.maximum_iterations = 100;
  request.mixer_history = 8;
  request.energy_tolerance = 1.0e-10;
  request.charge_tolerance = 1.0e-8;
  const auto xtb = runtime.execute(request);
  const double xtb_elapsed = seconds(started);
  if (xtb.status != methods::detail::Gfn2RuntimeStatus::kSuccess || !xtb.converged)
    throw std::runtime_error("GFN2 charge seed did not converge: " + xtb.detail);

  started = Clock::now();
  const auto& ints = plan.one_electron();
  const auto x = scf::reference::symmetric_orthogonalizer(ints.overlap, ints.nbf);
  std::optional<scf::reference::EigenResult> frame;
  const auto core = scf::initial_guess::prepare_initial_density(
      system, ints, x, static_cast<std::size_t>(system.electron_count / 2), nullptr, frame);
  const auto seed = scf::initial_guess::charge_guided_lowdin_density(
      system, ints, x, core, xtb.atomic_charges);
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
    source = tmp_path / "xtb_guess_validation.cpp"
    source.write_text(BRIDGE)
    executable = tmp_path / "xtb_guess_validation"
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
            "-L",
            str(library.parent),
            f"-Wl,-rpath,{library.parent}",
            "-lvibeqc",
            "-o",
            str(executable),
        ],
        check=True,
        timeout=120,
    )
    return executable


def test_xtb_charge_seed_endpoint_validation(tmp_path: Path) -> None:
    executable = _compile_bridge(tmp_path)
    completed = subprocess.run(
        [str(executable)],
        check=True,
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
    rows = list(csv.DictReader(completed.stdout.splitlines()))
    assert len(rows) == 9
    numeric = {
        key: float(value)
        for row in rows
        for key, value in row.items()
        if key not in {"case", "repeat"}
    }
    assert all(math.isfinite(value) for value in numeric.values())

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

    artifact = ROOT / ".artifacts" / "xtb-guess-validation.json"
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2, sort_keys=True)
    )
    assert all(
        metrics["max_energy_difference"] < 1.0e-8 for metrics in summary.values()
    )
