#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "api/handles.hpp"
#include "integrals/ecp_cuda.hpp"

// Private resource-observation ABI used by prepared Python requests as well.
extern "C" {
void* vibeqc_resource_ledger_create_v1(std::size_t bytes, int device);
void vibeqc_resource_ledger_destroy_v1(void* handle);
int vibeqc_resource_ledger_bind_v1(void* handle);
int vibeqc_resource_ledger_read_v1(void* handle, std::uint64_t* values);
int vibeqc_resource_tracking_begin_v1(unsigned workers);
int vibeqc_resource_tracking_end_v1(std::uint64_t* peak, std::uint64_t* samples);
}

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Observation {
  std::unique_ptr<void, decltype(&vibeqc_resource_ledger_destroy_v1)> ledger{
      nullptr, vibeqc_resource_ledger_destroy_v1};
  Observation(std::size_t bytes, int device)
      : ledger(vibeqc_resource_ledger_create_v1(bytes, device), vibeqc_resource_ledger_destroy_v1) {
    require(ledger != nullptr, "ledger creation failed");
    require(vibeqc_resource_tracking_begin_v1(0) == 0, "observation failed");
    if (vibeqc_resource_ledger_bind_v1(ledger.get()) != 0) {
      std::uint64_t peak{}, samples{};
      vibeqc_resource_tracking_end_v1(&peak, &samples);
      throw std::runtime_error("ledger binding failed");
    }
  }
  ~Observation() {
    std::uint64_t peak{}, samples{};
    vibeqc_resource_tracking_end_v1(&peak, &samples);
  }
  std::array<std::uint64_t, 4> read() const {
    std::array<std::uint64_t, 4> values{};
    require(vibeqc_resource_ledger_read_v1(ledger.get(), values.data()) == 0, "ledger read failed");
    return values;
  }
};

void errors_and_recovery() {
  const vibeqc_context_descriptor context_desc{sizeof(context_desc), VIBEQC_ABI_VERSION, 0,
                                               VIBEQC_BACKEND_CUDA};
  vibeqc_context* raw_context{};
  require(vibeqc_context_create(&context_desc, &raw_context) == VIBEQC_STATUS_SUCCESS,
          "CUDA context creation failed");
  std::unique_ptr<vibeqc_context, decltype(&vibeqc_context_destroy)> context(
      raw_context, vibeqc_context_destroy);
  const vibeqc_atom atom{11, 0, 0, 0};
  const vibeqc_primitive primitive{0.7, 1.0};
  const vibeqc_shell shell{0, 0, 0, 1};
  const vibeqc_system_descriptor descriptor{
      sizeof(descriptor),    VIBEQC_ABI_VERSION, &atom, 1, &shell, 1, &primitive, 1, 0, 2,
      VIBEQC_BASIS_CARTESIAN};
  const int32_t core = 10;
  const vibeqc_ecp_term term{0, -1, 2, 0.8, -2.0};
  vibeqc_system* raw_system{};
  require(vibeqc_system_create_ecp(context.get(), &descriptor, &core, &term, 1, &raw_system) ==
              VIBEQC_STATUS_SUCCESS,
          "ECP system creation failed");
  std::unique_ptr<vibeqc_system, decltype(&vibeqc_system_destroy)> system(raw_system,
                                                                          vibeqc_system_destroy);

  for (bool device_consumer : {false, true}) {
    std::array<double, 8> output;
    output.fill(123.0);
    std::string detail;
    const auto execute = [&] {
      if (device_consumer)
        return vibeqc::integrals::add_ecp_cuda(0, system->data, nullptr, nullptr, nullptr, nullptr,
                                               detail);
      const auto status = vibeqc_system_ecp_integrals(context.get(), system.get(), 160, 32, 1,
                                                      output.data(), output.size());
      detail = vibeqc_context_get_last_detail(context.get());
      return status;
    };
    {
      // Enough for several real allocations, but not the angular grid. This
      // deterministic budget rejection exercises unwinding after partial upload.
      Observation observation(256, 0);
      require(execute() == VIBEQC_STATUS_OUT_OF_MEMORY, "CUDA OOM lost its status");
      require(!detail.empty(), "OOM detail missing");
      const auto values = observation.read();
      require(values[0] == 0 && values[1] > 0 && values[2] > 1 && values[3] > 0,
              "partial ECP allocation leaked or rejection was not exercised");
      require(std::all_of(output.begin(), output.end(), [](double x) { return x == 123.0; }),
              "failed C ABI call published partial output");
    }
    {
      // Device mismatch is a non-OOM CUDA error without poisoning the GPU.
      Observation observation(8 << 20, 1);
      require(execute() == VIBEQC_STATUS_CUDA_ERROR, "CUDA runtime error lost its status");
      require(!detail.empty(), "CUDA error detail missing");
      require(observation.read()[0] == 0, "CUDA error leaked device allocations");
    }
    {
      Observation observation(8 << 20, 0);
      require(execute() == VIBEQC_STATUS_SUCCESS, "ECP execution did not recover");
      const auto values = observation.read();
      require(values[0] == 0 && values[2] > 1 && values[3] == 0,
              "successful ECP execution leaked device allocations");
      if (!device_consumer)
        require(std::isfinite(output[0]) && output[0] < 0,
                "recovered execution did not publish ECP output");
    }
  }

  std::string detail;
  double force{};
  require(vibeqc::integrals::add_ecp_cuda(0, system->data, nullptr, nullptr, nullptr, &force,
                                          detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
          "non-CUDA argument error changed category");
  system->data.ecp_terms[0].coefficient = std::numeric_limits<double>::infinity();
  Observation observation(8 << 20, 0);
  require(vibeqc::integrals::add_ecp_cuda(0, system->data, nullptr, nullptr, nullptr, nullptr,
                                          detail) == VIBEQC_STATUS_NUMERICAL_FAILURE,
          "non-CUDA convergence error changed category");
  require(observation.read()[0] == 0, "numerical failure leaked allocations");
}
}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "CUDA device unavailable\n";
    return 77;
  }
  try {
    errors_and_recovery();
    std::cout << "ECP CUDA error categories, partial cleanup and recovery PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
