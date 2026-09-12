#include "posthf/mp2_energy.hpp"

#include <cmath>
#include <stdexcept>

#include "posthf/mp2_cpu_generated.hpp"
#include "posthf/mp2_cuda_plan.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/density_fitting.hpp"

namespace vibeqc::mp2 {
namespace {
void validate_reference(const scf::PhysicalReference& ref, double threshold,
                        unsigned requested_tile) {
  if (!ref.nocc || ref.nocc >= ref.nbf || ref.orbital_energies.size() != ref.nbf ||
      !std::isfinite(threshold) || threshold <= 0 || !requested_tile)
    throw std::invalid_argument("invalid canonical MP2 reference/denominator settings");
  if (!std::all_of(ref.orbital_energies.begin(), ref.orbital_energies.end(),
                   [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("nonfinite MP2 orbital energies");
}

std::pair<double, double> denominator_bounds(const scf::PhysicalReference& ref, double threshold) {
  const auto& eps = ref.orbital_energies;
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
  return {-largest, -smallest};
}

unsigned resolved_tile(std::size_t virtuals, unsigned requested) {
  unsigned tile = 1;
  while (tile < 8 && tile * 2 <= requested && tile * 2 <= virtuals) tile *= 2;
  return tile;
}
}  // namespace

Energy conventional_energy(const scf::PhysicalReference& ref, const posthf::RawSource& source,
                           std::size_t budget, double threshold, unsigned requested_tile, bool cuda,
                           int device) {
  validate_reference(ref, threshold, requested_tile);
  const auto& eps = ref.orbital_energies;
  const auto [minimum_denominator, maximum_denominator] = denominator_bounds(ref, threshold);
  (void)maximum_denominator;
  const auto nv = ref.nbf - ref.nocc;
  const unsigned tile = resolved_tile(nv, requested_tile);
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
  result.minimum_denominator = minimum_denominator;
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

Energy density_fitted_energy(const scf::PhysicalReference& ref, const posthf::RawSource& source,
                             std::size_t budget, double threshold, double metric_relative_threshold,
                             unsigned requested_tile, bool cuda, int device) {
  validate_reference(ref, threshold, requested_tile);
  if (!(metric_relative_threshold > 0.0) || !(metric_relative_threshold < 1.0) ||
      !std::isfinite(metric_relative_threshold) || source.naux() == 0)
    throw std::invalid_argument("invalid RI-MP2 metric or auxiliary basis");
  const auto [minimum_denominator, maximum_denominator] = denominator_bounds(ref, threshold);
  (void)maximum_denominator;
  const std::size_t n = ref.nbf, no = ref.nocc, nv = n - no, na = source.naux();
  const unsigned tile = resolved_tile(nv, requested_tile);
  const auto cpu = generated::cpu_plan(tile);
  generated::CudaPlan gpu{};
  if (cuda) {
#if VIBEQC_HAS_CUDA
    gpu = generated::cuda_plan(tile, device);
#else
    throw std::runtime_error("CUDA RI-MP2 kernels are not compiled");
#endif
  }
  const auto transformed_elements = posthf::checked_mul(posthf::checked_mul(no, nv), na);
  auto kernel_bytes = posthf::checked_add(cuda ? gpu.numeric_bytes : cpu.numeric_bytes,
                                          32ULL * tile * tile + 16ULL * tile + 64);
  const std::size_t peak =
      posthf::ri_mp2_capacity(source.orbital(), source.auxiliary(), no, kernel_bytes);
  if (peak > budget) throw std::length_error("RI-MP2 energy phase exceeds numeric memory budget");

  integrals::DensityFittingIntegralData raw;
  if (cuda) {
    std::string detail;
    std::vector<integrals::DensityFittingIntegralData> outputs;
    const auto status = scf::build_cuda_density_fitting_integrals_batch(
        device, {source.orbital()}, {source.auxiliary()}, outputs, detail, budget, false);
    if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
    if (status != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail.empty() ? "CUDA RI-MP2 integral generation failed" : detail);
    if (outputs.size() != 1) throw std::runtime_error("CUDA RI-MP2 integral output is missing");
    raw = std::move(outputs.front());
    raw = integrals::transform_density_fitting_integrals(raw, source.orbital(), source.auxiliary());
  } else {
    raw = integrals::build_density_fitting_integrals(source.orbital(), source.auxiliary(), false);
  }
  const auto factor = scf::factor_density_fitting_metric(raw.metric, na, metric_relative_threshold);
  auto whitened = scf::orthonormalize_density_fitting_three_center(raw.three_center, n, factor);
  std::vector<double> bia(transformed_elements, 0.0);
  auto bia_index = [=](std::size_t q, std::size_t i, std::size_t a) {
    return (q * no + i) * nv + a;
  };
  for (std::size_t q = 0; q < na; ++q)
    for (std::size_t i = 0; i < no; ++i)
      for (std::size_t a = 0; a < nv; ++a)
        for (std::size_t mu = 0; mu < n; ++mu)
          for (std::size_t nu = 0; nu < n; ++nu)
            bia[bia_index(q, i, a)] += ref.coefficients[mu * n + i] *
                                       ref.coefficients[nu * n + no + a] *
                                       whitened.values[(mu * n + nu) * na + q];

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
  Energy result;
  result.minimum_denominator = minimum_denominator;
  result.numeric_capacity_bytes = peak;
  result.equation_hash = cpu.equation_hash;
  double sum[2]{}, correction[2]{};
  for (std::size_t i = 0; i < no; ++i)
    for (std::size_t j = 0; j < no; ++j)
      for (std::size_t a0 = 0; a0 < nv; a0 += tile)
        for (std::size_t b0 = 0; b0 < nv; b0 += tile) {
          std::vector<double> g(tile * tile), x(tile * tile);
          std::vector<double> ea(tile, ref.orbital_energies[no]);
          std::vector<double> eb(tile, ref.orbital_energies[no]);
          for (unsigned a = 0; a < tile; ++a)
            for (unsigned b = 0; b < tile; ++b) {
              if (a0 + a >= nv || b0 + b >= nv) continue;
              ea[a] = ref.orbital_energies[no + a0 + a];
              eb[b] = ref.orbital_energies[no + b0 + b];
              for (std::size_t q = 0; q < na; ++q) {
                g[a * tile + b] += bia[bia_index(q, i, a0 + a)] * bia[bia_index(q, j, b0 + b)];
                x[a * tile + b] += bia[bia_index(q, i, b0 + b)] * bia[bia_index(q, j, a0 + a)];
              }
            }
          double out[2]{};
          if (cuda) {
            char error[2048]{};
            vibeqc_tensor::Metrics measured;
            const auto status = gpu.run(kernel.pointer, g.data(), x.data(), ref.orbital_energies[i],
                                        ref.orbital_energies[j], ea.data(), eb.data(), out,
                                        &measured, error, sizeof(error));
            if (status == 2 || status == 3) throw std::bad_alloc();
            if (status) throw std::runtime_error(error);
            result.metrics.input_ms += measured.input_ms;
            result.metrics.output_ms += measured.output_ms;
            result.metrics.kernel_ms += measured.kernel_ms;
            result.metrics.library_ms += measured.library_ms;
            result.mo_transfer_bytes =
                posthf::checked_add(result.mo_transfer_bytes, 16ULL * tile * tile);
          } else {
            cpu.run(g.data(), x.data(), ref.orbital_energies[i], ref.orbital_energies[j], ea.data(),
                    eb.data(), out);
          }
          for (unsigned k = 0; k < 2; ++k) {
            const double adjusted = out[k] - correction[k], next = sum[k] + adjusted;
            correction[k] = (next - sum[k]) - adjusted;
            sum[k] = next;
          }
          ++result.tiles;
        }
  result.opposite_spin = sum[0];
  result.same_spin = sum[1];
  if (!std::isfinite(result.opposite_spin) || !std::isfinite(result.same_spin))
    throw std::runtime_error("nonfinite RI-MP2 energy accumulation");
  result.metrics.provider_retained_bytes = posthf::checked_mul(bia.size(), sizeof(double));
  if (cuda) result.metrics.owned_device_bytes = gpu.device_bytes;
  return result;
}
}  // namespace vibeqc::mp2
