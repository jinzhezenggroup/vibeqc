/** Manual Slurm-only probe of the exact native DF source and J/K integration.
 *
 * Text inputs describe unnormalized physical basis shells; independent Python
 * libcint references use the same input. Binary outputs are raw FP64 M/A/J/K
 * tensors, never values reconstructed from the generated recurrence itself.
 */
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using vibeqc::core::System;
using namespace vibeqc::scf;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void write_values(const std::string& path, const std::vector<double>& values) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
  require(static_cast<bool>(stream), "cannot write " + path);
}
void read_shells(std::istream& input, System& system, std::size_t count) {
  require(count > 0 && count <= 100, "invalid fixture shell count");
  for (std::size_t shell = 0; shell < count; ++shell) {
    vibeqc::core::Shell value;
    std::size_t primitives;
    input >> value.atom_index >> value.angular_momentum >> primitives;
    require(primitives > 0 && primitives <= 64, "invalid fixture contraction length");
    for (std::size_t primitive = 0; primitive < primitives; ++primitive) {
      double exponent, coefficient;
      input >> exponent >> coefficient;
      value.primitives.push_back({exponent, coefficient});
    }
    system.shells.push_back(value);
  }
  std::string detail;
  require(static_cast<bool>(input), "truncated fixture input");
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 5 || (argc == 6 && std::string(argv[5]) == "--derivatives"),
            "usage: probe input output-prefix pair-tile auxiliary-tile [--derivatives]");
    const bool derivatives = argc == 6;
    require(std::getenv("SLURM_JOB_ID") != nullptr, "native DF probe requires a Slurm allocation");
    const std::string prefix = argv[2];
    const std::size_t pair_tile = std::stoul(argv[3]), auxiliary_tile = std::stoul(argv[4]);
    require(pair_tile > 0 && auxiliary_tile > 0, "positive tile dimensions required");
    std::ifstream input(argv[1]);
    std::size_t count;
    unsigned representation;
    input >> count >> representation;
    require(count > 0 && count <= 8 && representation <= 1, "invalid fixture batch");
    std::vector<System> orbital(count), auxiliary(count);
    for (std::size_t system = 0; system < count; ++system) {
      std::size_t atoms, orbital_shells, auxiliary_shells;
      input >> atoms >> orbital_shells >> auxiliary_shells;
      require(atoms > 0 && atoms <= 100, "invalid fixture geometry");
      for (std::size_t atom = 0; atom < atoms; ++atom) {
        vibeqc::core::Atom value;
        input >> value.atomic_number >> value.position[0] >> value.position[1] >> value.position[2];
        orbital[system].atoms.push_back(value);
      }
      orbital[system].basis_representation =
          representation == 0 ? VIBEQC_BASIS_CARTESIAN : VIBEQC_BASIS_SPHERICAL;
      auxiliary[system] = orbital[system];
      read_shells(input, orbital[system], orbital_shells);
      read_shells(input, auxiliary[system], auxiliary_shells);
    }
    std::string detail;
    CudaDensityFittingIntegralSource* raw_source = nullptr;
    std::vector<double> metric;
    std::size_t nbf = 0, naux = 0;
    auto start = Clock::now();
    require(create_cuda_density_fitting_integral_source(0, orbital, auxiliary, &raw_source, metric,
                                                        nbf, naux, detail) == VIBEQC_STATUS_SUCCESS,
            detail);
    const double setup_ms = milliseconds(start);
    auto source = std::unique_ptr<CudaDensityFittingIntegralSource,
                                  decltype(&destroy_cuda_density_fitting_integral_source)>(
        raw_source, &destroy_cuda_density_fitting_integral_source);
    const auto placement = cuda_density_fitting_integral_source_diagnostic(source.get());
    const auto source_device_bytes =
        cuda_density_fitting_integral_source_device_bytes(source.get());
    const auto source_host_peak_bytes =
        cuda_density_fitting_integral_source_host_peak_bytes(source.get());
    const std::size_t pairs = nbf * nbf;
    const std::size_t tile_elements =
        std::max(std::min(pairs, pair_tile) * std::min(naux, auxiliary_tile),
                 derivatives ? naux * std::min(naux, auxiliary_tile) : 0U);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    double* device;
    check(cudaMalloc(reinterpret_cast<void**>(&device), tile_elements * sizeof(double)));
    // Exact last offsets, empty tiles, and overflowing/out-of-range requests
    // exercise the public tile guard before any kernel can touch device memory.
    require(generate_cuda_density_fitting_raw_tile(source.get(), 0, pairs, 0, naux, 0, -1, stream,
                                                   device, detail) == VIBEQC_STATUS_SUCCESS,
            "empty tile rejected");
    require(generate_cuda_density_fitting_raw_tile(
                source.get(), 0, 1, std::numeric_limits<std::size_t>::max(), 0, 1, -1, stream,
                device, detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
            "overflowing tile accepted");
    require(
        generate_cuda_density_fitting_raw_tile(source.get(), count, 0, 1, 0, 1, -1, stream, device,
                                               detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
        "invalid batch offset accepted");
    auto unsupported_auxiliary = auxiliary;
    unsupported_auxiliary[0].shells[0].angular_momentum = 4U;
    CudaDensityFittingIntegralSource* unsupported = nullptr;
    std::vector<double> unsupported_metric;
    std::size_t unsupported_nbf = 0, unsupported_naux = 0;
    require(create_cuda_density_fitting_integral_source(
                0, orbital, unsupported_auxiliary, &unsupported, unsupported_metric,
                unsupported_nbf, unsupported_naux, detail) == VIBEQC_STATUS_INVALID_ARGUMENT &&
                unsupported == nullptr && detail.find("beyond f") != std::string::npos,
            "unsupported auxiliary g shell was not explicitly rejected");
    std::vector<double> values(count * pairs * naux), tile(tile_elements);
    start = Clock::now();
    for (std::size_t system = 0; system < count; ++system) {
      for (std::size_t pair = 0; pair < pairs; pair += pair_tile) {
        const auto pcount = std::min(pair_tile, pairs - pair);
        for (std::size_t aux = 0; aux < naux; aux += auxiliary_tile) {
          const auto acount = std::min(auxiliary_tile, naux - aux);
          require(generate_cuda_density_fitting_raw_tile(source.get(), system, pair, pcount, aux,
                                                         acount, -1, stream, device,
                                                         detail) == VIBEQC_STATUS_SUCCESS,
                  detail);
          check(cudaMemcpyAsync(tile.data(), device, pcount * acount * sizeof(double),
                                cudaMemcpyDeviceToHost, stream));
          check(cudaStreamSynchronize(stream));
          for (std::size_t p = 0; p < pcount; ++p) {
            std::copy_n(tile.data() + p * acount, acount,
                        values.data() + (system * pairs + pair + p) * naux + aux);
          }
        }
      }
    }
    const double raw_ms = milliseconds(start);
    if (derivatives) {
      // Exercise both bulk and bounded public-basis APIs against independent
      // libcint center derivatives. The second system catches accidental use
      // of a local coordinate as a packed fleet-global derivative seed.
      std::vector<double> da, dm, bulk_da, bulk_dm;
      std::vector<vibeqc::integrals::DensityFittingIntegralData> bulk;
      require(build_cuda_density_fitting_integrals_batch(0, orbital, auxiliary, bulk, detail) ==
                  VIBEQC_STATUS_SUCCESS,
              detail);
      for (std::size_t system = 0; system < count; ++system) {
        const auto projected = vibeqc::integrals::transform_density_fitting_integrals(
            bulk[system], orbital[system], auxiliary[system]);
        bulk_da.insert(bulk_da.end(), projected.three_center_derivative.begin(),
                       projected.three_center_derivative.end());
        bulk_dm.insert(bulk_dm.end(), projected.metric_derivative.begin(),
                       projected.metric_derivative.end());
        const auto coordinates = 3 * orbital[system].atoms.size();
        const auto abase = da.size(), mbase = dm.size();
        da.resize(abase + coordinates * pairs * naux);
        dm.resize(mbase + coordinates * naux * naux);
        for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
          for (std::size_t pair = 0; pair < pairs; pair += pair_tile) {
            const auto pcount = std::min(pair_tile, pairs - pair);
            for (std::size_t aux = 0; aux < naux; aux += auxiliary_tile) {
              const auto acount = std::min(auxiliary_tile, naux - aux);
              require(generate_cuda_density_fitting_raw_tile(
                          source.get(), system, pair, pcount, aux, acount, coordinate, stream,
                          device, detail) == VIBEQC_STATUS_SUCCESS,
                      detail);
              check(cudaMemcpyAsync(tile.data(), device, pcount * acount * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));
              check(cudaStreamSynchronize(stream));
              for (std::size_t p = 0; p < pcount; ++p)
                std::copy_n(tile.data() + p * acount, acount,
                            da.data() + abase + (coordinate * pairs + pair + p) * naux + aux);
            }
          }
          for (std::size_t row = 0; row < naux; row += auxiliary_tile) {
            const auto rows = std::min(auxiliary_tile, naux - row);
            require(generate_cuda_density_fitting_metric_derivative_tile(
                        source.get(), system, row, rows, coordinate, stream, device, detail) ==
                        VIBEQC_STATUS_SUCCESS,
                    detail);
            check(cudaMemcpyAsync(dm.data() + mbase + (coordinate * naux + row) * naux, device,
                                  rows * naux * sizeof(double), cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
          }
        }
      }
      write_values(prefix + "-raw_derivative.bin", da);
      write_values(prefix + "-metric_derivative.bin", dm);
      write_values(prefix + "-bulk_raw_derivative.bin", bulk_da);
      write_values(prefix + "-bulk_metric_derivative.bin", bulk_dm);
    }
    check(cudaFree(device));
    check(cudaStreamDestroy(stream));
    write_values(prefix + "-metric.bin", metric);
    write_values(prefix + "-raw.bin", values);

    CudaDensityFittingJkPlan* raw_plan = nullptr;
    std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
    // Raw writes deliberately split individual AO rows. For the independent
    // J/K replay comparison, use a few partial row panels, as the production
    // planner does; one-row panels otherwise measure thousands of launches
    // and repeated integral generation rather than a bounded schedule choice.
    const std::size_t jk_pair_tile = std::min(pairs, std::max(pair_tile, nbf * (nbf / 2U + 1U)));
    start = Clock::now();
    // The plan constructor consumes the source on both success and failure.
    raw_source = source.release();
    require(create_cuda_density_fitting_jk_plan_from_source(
                0, &raw_source, count, nbf, naux, metric, 1.0e-12, auxiliary_tile, jk_pair_tile,
                &raw_plan, diagnostics, detail) == VIBEQC_STATUS_SUCCESS,
            detail);
    auto plan =
        std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)>(
            raw_plan, &destroy_cuda_density_fitting_jk_plan);
    const double plan_ms = milliseconds(start);
    std::vector<double> density(count * pairs), alpha(count * pairs), beta(count * pairs);
    for (std::size_t system = 0; system < count; ++system) {
      for (std::size_t i = 0; i < nbf; ++i) {
        for (std::size_t j = 0; j < nbf; ++j) {
          const auto index = system * pairs + i * nbf + j;
          density[index] =
              (i == j ? 0.3 : 0.01 / (1.0 + std::abs(static_cast<double>(i) - j))) * (system + 1);
          alpha[index] = 0.6 * density[index];
          beta[index] = 0.4 * density[index];
        }
      }
    }
    std::vector<double> coulomb, exchange, alpha_exchange, beta_exchange;
    std::vector<double> warm_ms;
    for (unsigned sample = 0; sample < 5; ++sample) {
      start = Clock::now();
      require(execute_cuda_density_fitting_rhf_jk(plan.get(), density, coulomb, exchange, detail) ==
                  VIBEQC_STATUS_SUCCESS,
              detail);
      if (sample > 0) warm_ms.push_back(milliseconds(start));
    }
    write_values(prefix + "-j.bin", coulomb);
    write_values(prefix + "-k.bin", exchange);
    require(execute_cuda_density_fitting_uhf_jk(plan.get(), alpha, beta, coulomb, alpha_exchange,
                                                beta_exchange, detail) == VIBEQC_STATUS_SUCCESS,
            detail);
    write_values(prefix + "-uj.bin", coulomb);
    write_values(prefix + "-ka.bin", alpha_exchange);
    write_values(prefix + "-kb.bin", beta_exchange);
    std::cout << std::setprecision(12) << "{\"nbf\":" << nbf << ",\"naux\":" << naux
              << ",\"batch\":" << count << ",\"source_setup_ms\":" << setup_ms
              << ",\"raw_reconstruction_ms\":" << raw_ms << ",\"jk_setup_ms\":" << plan_ms
              << ",\"raw_pair_tile\":" << pair_tile << ",\"jk_pair_tile\":" << jk_pair_tile
              << ",\"auxiliary_tile\":" << auxiliary_tile
              << ",\"source_device_bytes\":" << source_device_bytes
              << ",\"source_host_peak_bytes\":" << source_host_peak_bytes
              << ",\"value_backend\":" << std::quoted(placement.value_backend)
              << ",\"value_mapping\":" << std::quoted(placement.value_mapping)
              << ",\"public_transform_on_device\":"
              << (placement.public_transform_on_device ? "true" : "false")
              << ",\"metric_staged_on_host\":"
              << (placement.metric_staged_on_host ? "true" : "false")
              << ",\"slurm_job_id\":" << std::quoted(std::getenv("SLURM_JOB_ID"))
              << ",\"warm_jk_ms\":[";
    for (std::size_t i = 0; i < warm_ms.size(); ++i) std::cout << (i ? "," : "") << warm_ms[i];
    std::cout << "],\"metrics\":[";
    for (std::size_t i = 0; i < diagnostics.size(); ++i) {
      const auto& d = diagnostics[i];
      std::cout << (i ? "," : "") << "{\"rank\":" << d.effective_rank
                << ",\"condition_number\":" << d.condition_number
                << ",\"threshold\":" << d.absolute_threshold
                << ",\"device_peak_bytes\":" << d.peak_device_bytes
                << ",\"host_peak_bytes\":" << d.peak_host_bytes
                << ",\"auxiliary_tile\":" << d.auxiliary_tile << '}';
    }
    std::cout << "]}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
