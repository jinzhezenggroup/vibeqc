#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "posthf/capacity.hpp"
#include "scf/mean_field.hpp"
#include "tensor/cuda_error.hpp"

namespace vibeqc::scf::reference_detail {
inline std::size_t free_bytes(cudaStream_t stream) {
  vibeqc_tensor::cuda_check(cudaStreamSynchronize(stream));
  std::size_t available = 0, total = 0;
  vibeqc_tensor::cuda_check(cudaMemGetInfo(&available, &total));
  return available;
}
inline std::size_t retained_bytes(cudaStream_t stream, std::size_t before, std::size_t allowance) {
  const auto after = free_bytes(stream);
  const auto delta = before > after ? before - after : 0;
  if (delta > allowance)
    throw std::length_error("CUDA RHF provider exceeds retained allocation allowance");
  return delta;
}
inline std::size_t check_capacity(std::size_t base, std::size_t additional, std::size_t budget) {
  const auto peak = posthf::checked_add(base, additional);
  if (peak > budget) throw std::length_error("CUDA RHF reference exceeds numeric memory budget");
  return peak;
}
template <class Host>
std::size_t host_numeric_capacity(const Host& h) {
  std::size_t bytes = 0;
  auto count = [&](const auto& v) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    bytes = posthf::checked_add(bytes, posthf::checked_mul(v.capacity(), sizeof(T)));
  };
  count(h.atom_offsets);
  count(h.atom_systems);
  count(h.atomic_numbers);
  count(h.positions);
  count(h.system_shell_offsets);
  count(h.shell_atoms);
  count(h.shell_angular);
  count(h.shell_ao_offsets);
  count(h.shell_direct_ao_offsets);
  count(h.shell_primitive_offsets);
  count(h.system_shell_pair_offsets);
  count(h.system_shell_quartet_offsets);
  count(h.system_shell_pair_block_offsets);
  count(h.system_shell_pair_block_quartet_offsets);
  count(h.shell_pair_systems);
  count(h.shell_pair_first);
  count(h.shell_pair_second);
  count(h.shell_pair_primitive_offsets);
  count(h.psss_resident_tasks);
  count(h.psss_resident_ket_pairs);
  count(h.ao_shells);
  count(h.ao_term_counts);
  count(h.ao_term_angular);
  count(h.ao_term_coefficients);
  count(h.direct_ao_shells);
  count(h.direct_ao_angular);
  count(h.direct_ao_coefficients);
  count(h.ao_to_direct_transform);
  count(h.primitive_exponents);
  count(h.primitive_coefficients);
  count(h.occupied);
  count(h.warm_mask);
  count(h.warm_density);
  return bytes;
}

template <class Host>
std::size_t base_capacity(std::size_t arena, const Host& host, std::size_t matrices,
                          std::size_t providers, std::size_t budget) {
  // Candidate/topology copies, bounded pair-order construction and detached
  // validation matrices coexist. Runtime graphs, context/stack and allocator
  // rounding are explicitly outside numeric-buffer accounting.
  auto peak = posthf::checked_add(
      arena,
      posthf::checked_add(posthf::checked_mul(4, host_numeric_capacity(host)),
                          posthf::checked_add(posthf::checked_mul(512, matrices), 8ULL << 20)));
  return check_capacity(peak, providers, budget);
}

/** Stage the final physical P/F/C/epsilon from the existing GPU solve.
 * Matrix slots are S,h,F,C,P in column-major storage, scalar slots are energy,
 * energy change and density RMS. Nothing is published on failure. */
inline vibeqc_status download(cudaStream_t stream, std::size_t n, std::size_t occupied,
                              std::size_t capacity,
                              const std::array<const double*, 5>& matrix_sources,
                              const double* epsilon,
                              const std::array<const double*, 3>& scalar_sources,
                              const std::uint8_t* converged, const std::uint8_t* failed,
                              const std::uint32_t* iterations, ScfResult& result) {
  auto ref = std::make_shared<PhysicalReference>();
  ref->nbf = n;
  ref->nocc = occupied;
  const auto matrix_size = posthf::checked_mul(n, n);
  std::array<std::vector<double>, 5> matrices;
  for (auto& v : matrices) v.resize(matrix_size);
  ref->orbital_energies.resize(n);
  double scalars[3]{};
  std::uint8_t ok = 0, bad = 0;
  std::uint32_t count = 0;
  struct Drain {
    cudaStream_t stream;
    bool done = false;
    ~Drain() {
      if (!done) cudaStreamSynchronize(stream);
    }
  } drain{stream};
  auto copy = [&](void* to, const void* from, std::size_t bytes) {
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(to, from, bytes, cudaMemcpyDeviceToHost, stream));
  };
  for (unsigned k = 0; k < 5; ++k)
    copy(matrices[k].data(), matrix_sources[k], matrix_size * sizeof(double));
  copy(ref->orbital_energies.data(), epsilon, n * sizeof(double));
  for (unsigned k = 0; k < 3; ++k) copy(scalars + k, scalar_sources[k], sizeof(double));
  copy(&ok, converged, sizeof(ok));
  copy(&bad, failed, sizeof(bad));
  copy(&count, iterations, sizeof(count));
  vibeqc_tensor::cuda_check(cudaStreamSynchronize(stream));
  drain.done = true;
  if (bad || !ok) return bad ? VIBEQC_STATUS_NUMERICAL_FAILURE : VIBEQC_STATUS_NOT_CONVERGED;
  std::array<std::vector<double>*, 5> targets{&ref->overlap, &ref->hcore, &ref->fock,
                                              &ref->coefficients, &ref->density};
  for (unsigned k = 0; k < 5; ++k) {
    targets[k]->resize(matrix_size);
    for (std::size_t mu = 0; mu < n; ++mu)
      for (std::size_t p = 0; p < n; ++p) (*targets[k])[mu * n + p] = matrices[k][mu + p * n];
  }
  ref->energy = scalars[0];
  if (!std::isfinite(ref->energy)) throw std::runtime_error("nonfinite CUDA RHF energy");
  validate_physical_reference(*ref);
  ref->numeric_capacity_bytes = capacity;
  result.energy = ref->energy;
  result.iterations = count;
  result.energy_change = scalars[1];
  result.density_rms = scalars[2];
  result.converged = true;
  result.reference = std::move(ref);
  return VIBEQC_STATUS_SUCCESS;
}
}  // namespace vibeqc::scf::reference_detail
