#include "posthf/mp2_energy.hpp"

#include <cmath>
#include <stdexcept>

#include "posthf/mp2_cpu_generated.hpp"
#include "posthf/mp2_cuda_plan.hpp"

namespace vibeqc::mp2 {
Energy conventional_energy(const scf::PhysicalReference& ref, const posthf::RawSource& source,
                           std::size_t budget, double threshold, unsigned requested_tile, bool cuda,
                           int device) {
  if (!ref.nocc || ref.nocc >= ref.nbf || ref.orbital_energies.size() != ref.nbf ||
      !std::isfinite(threshold) || threshold <= 0 || !requested_tile)
    throw std::invalid_argument("invalid canonical MP2 reference/denominator settings");
  const auto& eps = ref.orbital_energies;
  if (!std::all_of(eps.begin(), eps.end(), [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("nonfinite MP2 orbital energies");
  const double hi = *std::max_element(eps.begin(), eps.begin() + ref.nocc);
  const double lo = *std::min_element(eps.begin() + ref.nocc, eps.end());
  const double largest = hi + hi - lo - lo;
  const double lowest_occupied = *std::min_element(eps.begin(), eps.begin() + ref.nocc);
  const double highest_virtual = *std::max_element(eps.begin() + ref.nocc, eps.end());
  const double smallest = lowest_occupied + lowest_occupied - highest_virtual - highest_virtual;
  if (!std::isfinite(largest) || !std::isfinite(smallest))
    throw std::invalid_argument("nonfinite MP2 denominator extrema");
  if (largest >= 0)
    throw std::invalid_argument("occupied MP2 energies must be below virtual energies");
  if (-largest <= threshold)
    throw std::invalid_argument("near-zero MP2 denominator; no regularization applied");
  const auto nv = ref.nbf - ref.nocc;
  unsigned tile = 1;
  while (tile < 8 && tile * 2 <= requested_tile && tile * 2 <= nv) tile *= 2;
  const auto cpu = generated::cpu_plan(tile);
  generated::CudaPlan gpu{};
  if (cuda) {
#if VIBEQC_HAS_CUDA
    gpu = generated::cuda_plan(tile, device);
#else
    throw std::runtime_error("CUDA MP2 kernels are not compiled");
#endif
  }
  posthf::NativeBlockProvider provider(source, ref, budget);
  const auto plan = provider.plan({1, tile, 1, tile}, cuda);
  // Native scalar fold, two detached MO blocks, reordered exchange, orbital
  // panels and all generated CPU tensor temporaries coexist conservatively.
  auto peak = posthf::checked_add(posthf::checked_add(plan.host_bytes, plan.device_bytes),
                                  cuda ? gpu.numeric_bytes : cpu.numeric_bytes);
  peak = posthf::checked_add(peak, 32ULL * tile * tile + 16ULL * tile + 64);
  if (peak > budget) throw std::length_error("MP2 energy phase exceeds numeric memory budget");
  Energy result;
  result.minimum_denominator = -largest;
  result.numeric_capacity_bytes = peak;
  result.equation_hash = cpu.equation_hash;
  struct KernelState {
    void* pointer{};
    generated::CudaDestroy destroy{};
    ~KernelState() {
      if (pointer) destroy(pointer);
    }
  } kernel;
  if (cuda) {
    char error[2048]{};
    kernel.destroy = gpu.destroy;
    const auto status = gpu.create(device, &kernel.pointer, error, sizeof(error));
    if (status == 2 || status == 3) throw std::bad_alloc();
    if (status) throw std::runtime_error(error);
  }
  double sum[2]{}, correction[2]{};
  for (std::size_t i = 0; i < ref.nocc; ++i)
    for (std::size_t j = 0; j < ref.nocc; ++j)
      for (std::size_t a = ref.nocc; a < ref.nbf; a += tile)
        for (std::size_t b = ref.nocc; b < ref.nbf; b += tile) {
          std::vector<std::size_t> va(tile, posthf::padded_mo), vb(tile, posthf::padded_mo);
          std::vector<double> ea(tile, eps[ref.nocc]), eb(tile, eps[ref.nocc]);
          for (unsigned k = 0; k < tile; ++k) {
            if (a + k < ref.nbf) {
              va[k] = a + k;
              ea[k] = eps[a + k];
            }
            if (b + k < ref.nbf) {
              vb[k] = b + k;
              eb[k] = eps[b + k];
            }
          }
          const auto g =
              provider.get({std::vector<std::size_t>{i}, va, std::vector<std::size_t>{j}, vb}, cuda,
                           device, &result.metrics);
          const auto exchanged =
              provider.get({std::vector<std::size_t>{i}, vb, std::vector<std::size_t>{j}, va}, cuda,
                           device, &result.metrics);
          std::vector<double> x(tile * tile);
          for (unsigned u = 0; u < tile; ++u)
            for (unsigned v = 0; v < tile; ++v) x[u * tile + v] = exchanged[v * tile + u];
          double out[2]{};
          if (cuda) {
            char error[2048]{};
            vibeqc_tensor::Metrics measured;
            const auto status = gpu.run(kernel.pointer, g.data(), x.data(), eps[i], eps[j],
                                        ea.data(), eb.data(), out, &measured, error, sizeof(error));
            if (status == 2 || status == 3) throw std::bad_alloc();
            if (status) throw std::runtime_error(error);
            if (measured.owned_device_bytes != gpu.device_bytes)
              throw std::runtime_error("MP2 tensor allocation disagrees with plan");
            result.metrics.input_ms += measured.input_ms;
            result.metrics.output_ms += measured.output_ms;
            result.metrics.kernel_ms += measured.kernel_ms;
            result.metrics.library_ms += measured.library_ms;
            result.mo_transfer_bytes =
                posthf::checked_add(result.mo_transfer_bytes, 32ULL * tile * tile);
          } else {
            cpu.run(g.data(), x.data(), eps[i], eps[j], ea.data(), eb.data(), out);
          }
          for (unsigned k = 0; k < 2; ++k) {
            const double adjusted = out[k] - correction[k], next = sum[k] + adjusted;
            correction[k] = (next - sum[k]) - adjusted;
            sum[k] = next;
            if (!std::isfinite(sum[k]))
              throw std::runtime_error("nonfinite MP2 energy accumulation");
          }
          ++result.tiles;
        }
  result.opposite_spin = sum[0];
  result.same_spin = sum[1];
  if (cuda)
    result.metrics.owned_device_bytes =
        posthf::checked_add(result.metrics.owned_device_bytes, gpu.device_bytes);
  return result;
}
}  // namespace vibeqc::mp2
