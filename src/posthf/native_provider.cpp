#include "posthf/native_provider.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "posthf/cuda_transform.hpp"

namespace vibeqc::posthf {
NativeBlockProvider::NativeBlockProvider(const RawSource& source,
                                         const scf::PhysicalReference& reference,
                                         std::size_t budget, unsigned axis_tile)
    : source_(source),
      ref_(reference),
      budget_(budget),
      source_bytes_(source_capacity(source.orbital())),
      reference_bytes_(checked_mul(
          8,
          checked_add(checked_mul(5, checked_mul(reference.nbf, reference.nbf)), reference.nbf))) {
  if (!axis_tile || ref_.nbf != source_.nbf() || ref_.coefficients.size() != ref_.nbf * ref_.nbf)
    throw std::invalid_argument("native MO provider/reference dimensions mismatch");
  std::size_t largest_shell = 0;
  for (const auto& shell : source_.orbital().shells) {
    const auto l = shell.angular_momentum;
    const auto count = source_.orbital().basis_representation == VIBEQC_BASIS_SPHERICAL
                           ? 2 * l + 1
                           : (l + 1) * (l + 2) / 2;
    largest_shell = std::max(largest_shell, static_cast<std::size_t>(count));
  }
  tile_.fill(std::min<std::size_t>(axis_tile, largest_shell));
  if (!tile_[0]) throw std::invalid_argument("empty native AO source");
}

NumericBlockPlan NativeBlockProvider::plan(const std::array<std::size_t, 4>& shape,
                                           bool cuda) const {
  for (auto n : shape)
    if (!n || n > ref_.nbf) throw std::invalid_argument("native MO block shape out of bounds");
  auto p = numeric_block_plan(ref_.nbf, reference_bytes_, source_bytes_, shape, tile_, cuda);
  if (cuda && p.stage_elements > INT32_MAX)
    throw std::length_error("MO stage exceeds cuBLAS int32 range");
  if (checked_add(p.host_bytes, p.device_bytes) > budget_)
    throw std::length_error("native MO block exceeds memory budget");
  return p;
}

std::vector<double> NativeBlockProvider::get(const MOSlots& slots, bool cuda, int device,
                                             vibeqc_tensor::Metrics* metrics) const {
  std::array<std::size_t, 4> shape{};
  for (unsigned k = 0; k < 4; ++k) shape[k] = slots[k].size();
  const auto p = plan(shape, cuda);
  std::array<std::vector<double>, 4> c;
  for (unsigned k = 0; k < 4; ++k) {
    c[k].resize(ref_.nbf * shape[k]);
    for (std::size_t mo = 0; mo < shape[k]; ++mo) {
      const auto column = slots[k][mo];
      if (column != padded_mo && column >= ref_.nbf)
        throw std::invalid_argument("native MO index out of range");
      if (column == padded_mo) continue;
      if (std::find(slots[k].begin(), slots[k].begin() + mo, column) != slots[k].begin() + mo)
        throw std::invalid_argument("native MO slots require unique real columns");
      for (std::size_t mu = 0; mu < ref_.nbf; ++mu) {
        const auto value = ref_.coefficients[mu * ref_.nbf + column];
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite MO coefficient");
        c[k][mu * shape[k] + mo] = value;
      }
    }
  }
  struct DeviceBlock {
    void* pointer{};
    ~DeviceBlock() {
#if VIBEQC_HAS_CUDA
      if (pointer) posthf_cuda_destroy_v1(pointer);
#endif
    }
  } device_block;
#if VIBEQC_HAS_CUDA
  char error[2048]{};
  auto check = [&](int status) {
    if (status == 2) throw std::bad_alloc();
    if (status) throw std::runtime_error(error);
  };
#endif
  if (cuda) {
#if VIBEQC_HAS_CUDA
    std::vector<double> panels;
    panels.reserve(p.coefficient_elements);
    for (const auto& panel : c) panels.insert(panels.end(), panel.begin(), panel.end());
    check(posthf_cuda_create_v1(device, ref_.nbf, shape.data(), tile_.data(), panels.data(),
                                p.allocation_bytes, &device_block.pointer, error, sizeof(error)));
#else
    (void)device;
    throw std::runtime_error("CUDA MO provider is not compiled");
#endif
  }
  std::vector<double> output(p.output_elements, 0), first(p.stage_elements),
      second(cuda ? 0 : p.stage_elements);
  for (std::size_t u = 0; u < ref_.nbf; u += tile_[0])
    for (std::size_t v = 0; v < ref_.nbf; v += tile_[1])
      for (std::size_t w = 0; w < ref_.nbf; w += tile_[2])
        for (std::size_t x = 0; x < ref_.nbf; x += tile_[3]) {
          const std::array<std::size_t, 4> begin{u, v, w, x};
          std::array<std::size_t, 4> current{};
          std::size_t elements = 1;
          for (unsigned k = 0; k < 4; ++k) {
            current[k] = std::min(tile_[k], ref_.nbf - begin[k]);
            elements *= current[k];
          }
          source_.read(RawSource::Operator::eri, begin, current, first.data(), elements);
          if (cuda) {
#if VIBEQC_HAS_CUDA
            check(posthf_cuda_add_v1(device_block.pointer, first.data(), begin.data(),
                                     current.data(), error, sizeof(error)));
#endif
            continue;
          }
          for (unsigned k = 0; k < 4; ++k) {
            const auto ao = current[0], rest = elements / ao, columns = shape[k];
            std::fill_n(second.begin(), rest * columns, 0.0);
            // Identical cyclic order to CG10 transform_tile and cuBLAS:
            // input[AO,rest]^T @ C[AO,MO] -> output[rest,MO].
            for (std::size_t r = 0; r < rest; ++r)
              for (std::size_t a = 0; a < ao; ++a)
                for (std::size_t m = 0; m < columns; ++m)
                  second[r * columns + m] +=
                      first[a * rest + r] * c[k][(begin[k] + a) * columns + m];
            elements = rest * columns;
            for (unsigned j = 0; j < 3; ++j) current[j] = current[j + 1];
            current[3] = columns;
            first.swap(second);
          }
          for (std::size_t q = 0; q < output.size(); ++q) {
            output[q] += first[q];
            if (!std::isfinite(output[q]))
              throw std::runtime_error("nonfinite native MO transformation");
          }
        }
  if (cuda) {
#if VIBEQC_HAS_CUDA
    check(posthf_cuda_download_v1(device_block.pointer, output.data(), output.size(), error,
                                  sizeof(error)));
    if (metrics) {
      vibeqc_tensor::Metrics m;
      check(posthf_cuda_metrics_v1(device_block.pointer, &m, error, sizeof(error)));
      metrics->owned_device_bytes = std::max(metrics->owned_device_bytes, m.owned_device_bytes);
      metrics->provider_retained_bytes =
          std::max(metrics->provider_retained_bytes, m.provider_retained_bytes);
      metrics->input_ms += m.input_ms;
      metrics->output_ms += m.output_ms;
      metrics->library_ms += m.library_ms;
      metrics->kernel_ms += m.kernel_ms;
    }
#else
    (void)metrics;
#endif
  }
  return output;
}
}  // namespace vibeqc::posthf
