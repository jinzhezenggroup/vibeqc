#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "api/ks_snapshot.hpp"
#include "dft/xc_point.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

/** Compare the actual private ABI against differentiated original PW92/PBE
 * formulas at 450 digits. References use neither the production jet nor FP64
 * finite differences, including at the extreme density tails. */
void independent_point_response() {
  std::ifstream input(VIBEQC_SOURCE_DIR "/tests/data/xc/rks_response.tsv");
  require(bool(input), "missing independent RKS response fixture");
  std::size_t count = 0;
  for (std::string line; std::getline(input, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    double method, rho, gradient[3], direction, delta_gradient[3], expected[4], magnitude[4];
    row >> method >> rho;
    for (double& value : gradient) row >> value;
    row >> direction;
    for (double& value : delta_gradient) row >> value;
    for (double& value : expected) row >> value;
    for (double& value : magnitude) row >> value;
    require(bool(row), "malformed independent RKS response fixture");
    double actual[4]{};
    const auto status =
        vibeqc_xc_rks_response_batch_v1(static_cast<std::uint32_t>(method), &rho, gradient,
                                        &direction, delta_gradient, 1, actual, 4);
    require(status == VIBEQC_STATUS_SUCCESS, "valid point response rejected");
    for (unsigned component = 0; component < 4; ++component) {
      // PBE's leading X/C gradient terms cancel at zero/small reduced gradient.
      // Bound FP64 roundoff by their independently computed component sizes,
      // with no absolute floor that could hide nonzero density-tail errors.
      const double tolerance = 3e-11 * std::abs(expected[component]) +
                               32 * std::numeric_limits<double>::epsilon() * magnitude[component] +
                               8 * std::numeric_limits<double>::denorm_min();
      if (!std::isfinite(actual[component]) ||
          std::abs(actual[component] - expected[component]) > tolerance) {
        std::cerr << std::setprecision(17) << "point " << count << " component " << component
                  << " actual " << actual[component] << " expected " << expected[component]
                  << " tolerance " << tolerance << '\n';
        throw std::runtime_error("independent directional XC reference mismatch");
      }
    }
    ++count;
  }
  require(count == 30, "incomplete independent RKS response table");
}

void response_batch_boundaries() {
  // Distinct points/directions detect transposed or broadcast ABI layouts.
  const double rho[2]{0.7, 1.1}, gradient[6]{0.2, -0.1, 0.3, -0.4, 0.2, 0.1};
  const double direction[2]{0.13, -0.07}, delta_gradient[6]{0.03, 0.01, -0.02, 0.04, -0.03, 0.01};
  double together[8]{};
  const auto call = [&](std::uint32_t method, const double* density, const double* grad,
                        const double* delta, const double* delta_grad, std::size_t n,
                        double* output, std::size_t size) {
    return vibeqc_xc_rks_response_batch_v1(method, density, grad, delta, delta_grad, n, output,
                                           size);
  };
  for (std::uint32_t method : {0U, 1U}) {
    require(call(method, rho, gradient, direction, delta_gradient, 2, together, 8) ==
                VIBEQC_STATUS_SUCCESS,
            "batched point response failed");
    for (unsigned point = 0; point < 2; ++point) {
      double alone[4]{};
      require(call(method, rho + point, gradient + 3 * point, direction + point,
                   delta_gradient + 3 * point, 1, alone, 4) == VIBEQC_STATUS_SUCCESS,
              "scalar point response failed");
      for (unsigned c = 0; c < 4; ++c)
        require(alone[c] == together[4 * point + c], "response batch layout mismatch");
    }
    double zero = 0.0, vacuum_gradient[3]{}, output[4]{};
    require(call(method, &zero, vacuum_gradient, &zero, vacuum_gradient, 1, output, 4) ==
                VIBEQC_STATUS_SUCCESS,
            "zero vacuum direction failed");
    for (double value : output) require(value == 0.0, "nonzero vacuum response");
    // A reference accepted by SCF keeps its exact zero response, including
    // subnormal gradient residues at a density rounded to zero. Normal vacuum
    // gradients and nonzero tangents retain their rejection below.
    for (double residue :
         {std::numeric_limits<double>::denorm_min(), -std::numeric_limits<double>::denorm_min(),
          std::nextafter(std::numeric_limits<double>::min(), 0.0),
          -std::nextafter(std::numeric_limits<double>::min(), 0.0)}) {
      for (unsigned axis = 0; axis < 3; ++axis) {
        double reference_gradient[3]{};
        // The ABI takes totals, while the SCF reference has two equal spins.
        reference_gradient[axis] = 2.0 * residue;
        double spin_density[2]{}, spin_gradient[2][3]{};
        spin_gradient[0][axis] = spin_gradient[1][axis] = residue;
        require(vibeqc::dft::point::evaluate(method == 1, spin_density, spin_gradient).valid,
                "test reference is outside the actual SCF point domain");
        require(call(method, &zero, reference_gradient, &zero, vacuum_gradient, 1, output, 4) ==
                    VIBEQC_STATUS_SUCCESS,
                "SCF-admitted vacuum reference rejected by RKS response");
        for (double value : output) require(value == 0.0, "vacuum residue changed zero response");
        require(call(method, &zero, vacuum_gradient, &zero, reference_gradient, 1, output, 4) ==
                    VIBEQC_STATUS_NUMERICAL_FAILURE,
                "nonzero vacuum gradient tangent admitted");
      }
    }
    const double normal_boundary = 2.0 * std::numeric_limits<double>::min();
    // The adjacent total below the nominal boundary rounds up when split into
    // equal spins, so comparing the total to 2*DBL_MIN would be incorrect.
    for (double normal : {normal_boundary, -normal_boundary, std::nextafter(normal_boundary, 0.0),
                          -std::nextafter(normal_boundary, 0.0)}) {
      double reference_gradient[3]{normal, 0.0, 0.0};
      double spin_density[2]{}, spin_gradient[2][3]{};
      spin_gradient[0][0] = spin_gradient[1][0] = normal / 2.0;
      require(!vibeqc::dft::point::evaluate(method == 1, spin_density, spin_gradient).valid,
              "test boundary is inside the actual SCF point domain");
      require(call(method, &zero, reference_gradient, &zero, vacuum_gradient, 1, output, 4) ==
                  VIBEQC_STATUS_NUMERICAL_FAILURE,
              "RKS reference gradient with normal rounded spin component admitted");
    }
    double tiny = 1e-280;
    require(call(method, &tiny, vacuum_gradient, &zero, vacuum_gradient, 1, output, 4) ==
                VIBEQC_STATUS_SUCCESS,
            "exact zero direction lost in a positive-density tail");
    for (double value : output) require(value == 0.0, "zero tail direction changed");
    double one = 1.0, nonzero_gradient[3]{1.0, 0.0, 0.0};
    require(call(method, &zero, vacuum_gradient, &one, vacuum_gradient, 1, output, 4) ==
                VIBEQC_STATUS_NUMERICAL_FAILURE,
            "undefined vacuum density direction accepted");
    require(call(method, &zero, nonzero_gradient, &zero, vacuum_gradient, 1, output, 4) ==
                VIBEQC_STATUS_NUMERICAL_FAILURE,
            "vacuum with nonzero gradient accepted");
    require(call(method, &zero, vacuum_gradient, &zero, nonzero_gradient, 1, output, 4) ==
                VIBEQC_STATUS_NUMERICAL_FAILURE,
            "undefined vacuum gradient direction accepted");
    for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
      require(call(method, &invalid, gradient, &one, delta_gradient, 1, output, 4) ==
                  VIBEQC_STATUS_NUMERICAL_FAILURE,
              "invalid point density accepted");
      require(call(method, &one, gradient, &invalid, delta_gradient, 1, output, 4) ==
                  (invalid == -1.0 ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_NUMERICAL_FAILURE),
              "point direction domain changed");
    }
    double subnormal = std::numeric_limits<double>::denorm_min();
    require(call(method, &subnormal, vacuum_gradient, &one, vacuum_gradient, 1, output, 4) ==
                VIBEQC_STATUS_NUMERICAL_FAILURE,
            "unrepresentable spin direction accepted");
  }
  for (auto status : {call(2, rho, gradient, direction, delta_gradient, 2, together, 8),
                      call(0, nullptr, gradient, direction, delta_gradient, 2, together, 8),
                      call(0, rho, nullptr, direction, delta_gradient, 2, together, 8),
                      call(0, rho, gradient, nullptr, delta_gradient, 2, together, 8),
                      call(0, rho, gradient, direction, nullptr, 2, together, 8),
                      call(0, rho, gradient, direction, delta_gradient, 2, nullptr, 8),
                      call(0, rho, gradient, direction, delta_gradient, 0, together, 0),
                      call(0, rho, gradient, direction, delta_gradient, 2, together, 7),
                      call(0, rho, gradient, direction, delta_gradient,
                           std::numeric_limits<std::size_t>::max() / 4 + 1, together, 8)})
    require(status == VIBEQC_STATUS_INVALID_ARGUMENT, "malformed point response ABI accepted");
}

/** The energy accessor must read the same successful native owner as C/F/eps.
 * Repeating the same geometry or replacing the owner cannot renew its lease. */
void snapshot_energy_lifetime() {
  vibeqc_context* raw_context{};
  const vibeqc_context_descriptor descriptor{sizeof(descriptor), VIBEQC_ABI_VERSION, 0,
                                             VIBEQC_BACKEND_CPU_REFERENCE};
  require(vibeqc_context_create(&descriptor, &raw_context) == VIBEQC_STATUS_SUCCESS,
          "CPU context creation failed");
  const std::unique_ptr<vibeqc_context, decltype(&vibeqc_context_destroy)> context(
      raw_context, &vibeqc_context_destroy);
  const vibeqc_atom atoms[2]{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}};
  const vibeqc_primitive primitives[6]{{3.425250914, 0.1543289673},  {0.6239137298, 0.5353281423},
                                       {0.168855404, 0.4446345422},  {3.425250914, 0.1543289673},
                                       {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}};
  const vibeqc_shell shells[2]{{0, 0, 0, 3}, {1, 0, 3, 3}};
  const vibeqc_system_descriptor system_descriptor{
      sizeof(system_descriptor), VIBEQC_ABI_VERSION, atoms, 2, shells, 2, primitives, 6, 0, 1,
      VIBEQC_BASIS_CARTESIAN};
  vibeqc_system* raw_system{};
  require(
      vibeqc_system_create(context.get(), &system_descriptor, &raw_system) == VIBEQC_STATUS_SUCCESS,
      "H2 creation failed");
  const std::unique_ptr<vibeqc_system, decltype(&vibeqc_system_destroy)> system(
      raw_system, &vibeqc_system_destroy);
  for (const auto method_id : {VIBEQC_METHOD_LDA_RKS, VIBEQC_METHOD_PBE_RKS}) {
    vibeqc_method_descriptor method{
        sizeof(method), VIBEQC_ABI_VERSION,          method_id, 200,   8, 1e-12, 1e-10,
        1e-12,          VIBEQC_DENSITY_FITTING_NONE, nullptr,   1e-10, 0};
    vibeqc_batch* raw_batch{};
    require(vibeqc_batch_prepare(context.get(), &raw_system, 1, &method, 0, &raw_batch) ==
                VIBEQC_STATUS_SUCCESS,
            "RKS batch creation failed");
    const std::unique_ptr<vibeqc_batch, decltype(&vibeqc_batch_destroy)> batch(
        raw_batch, &vibeqc_batch_destroy);
    const auto execute = [&]() {
      vibeqc_batch_item_result_descriptor result{};
      result.struct_size = sizeof(result);
      result.abi_version = VIBEQC_ABI_VERSION;
      require(vibeqc_batch_execute(batch.get(), nullptr, 0, &result, 1) == VIBEQC_STATUS_SUCCESS &&
                  result.status == VIBEQC_STATUS_SUCCESS && result.converged,
              "native RKS did not converge");
      return result.energy;
    };
    std::uint64_t metadata[16]{};
    vibeqc_ks_snapshot* raw_snapshot{};
    require(vibeqc_ks_snapshot_create_v1(batch.get(), 0, &raw_snapshot, metadata, 16) !=
                    VIBEQC_STATUS_SUCCESS &&
                !raw_snapshot,
            "unsolved RKS snapshot accepted");
    const double expected = execute();
    require(vibeqc_ks_snapshot_create_v1(batch.get(), 0, &raw_snapshot, metadata, 16) ==
                VIBEQC_STATUS_SUCCESS,
            "native RKS snapshot creation failed");
    const std::unique_ptr<vibeqc_ks_snapshot, decltype(&vibeqc_ks_snapshot_destroy_v1)> snapshot(
        raw_snapshot, &vibeqc_ks_snapshot_destroy_v1);
    double energy = 999.0;
    require(vibeqc_ks_snapshot_energy_v1(batch.get(), snapshot.get(), &energy) ==
                    VIBEQC_STATUS_SUCCESS &&
                energy == expected,
            "snapshot energy differs from successful RKS result");
    require(vibeqc_ks_snapshot_energy_v1(nullptr, snapshot.get(), &energy) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "null batch accepted");
    require(vibeqc_ks_snapshot_energy_v1(batch.get(), nullptr, &energy) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "null snapshot accepted");
    require(vibeqc_ks_snapshot_energy_v1(batch.get(), snapshot.get(), nullptr) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "null energy output accepted");
    execute();
    energy = 999.0;
    require(vibeqc_ks_snapshot_energy_v1(batch.get(), snapshot.get(), &energy) !=
                    VIBEQC_STATUS_SUCCESS &&
                energy == 999.0,
            "same-geometry replay published stale energy");
  }
}
}  // namespace

int main() {
  try {
    independent_point_response();
    response_batch_boundaries();
    snapshot_energy_lifetime();
    std::cout
        << "30 independent RKS point directions, ABI domains and native energy leases passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
