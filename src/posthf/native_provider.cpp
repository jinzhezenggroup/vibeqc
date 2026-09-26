#include "posthf/native_provider.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "integrals/density_fitting_metric.hpp"
#include "integrals/s_integrals.hpp"
#include "posthf/cuda_transform.hpp"

namespace vibeqc::posthf {
NativeBlockProvider::NativeBlockProvider(const integrals::ElectronInteractionSource& source,
                                         const scf::PhysicalReference& reference,
                                         std::size_t budget, unsigned axis_tile)
    : source_(source),
      ref_(reference),
      budget_(budget),
      source_bytes_(source.retained_numeric_bytes()),
      reference_bytes_(checked_mul(
          8,
          checked_add(checked_mul(5, checked_mul(reference.nbf, reference.nbf)), reference.nbf))) {
  if (!source_.supports(integrals::ElectronInteractionOperator::eri))
    throw std::invalid_argument("native MO provider requires AO ERI source capability");
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

std::size_t NativeBlockProvider::common_host_bytes() const {
  std::size_t tile_elements = 1;
  for (const auto extent : tile_) tile_elements = checked_mul(tile_elements, extent);
  return checked_add(checked_add(checked_add(reference_bytes_, source_bytes_), 8388608ULL),
                     checked_mul(32ULL, tile_elements));
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

std::size_t NativeBlockProvider::batch_bytes(const std::array<std::size_t, 4>& shape,
                                             std::size_t requests, bool cuda) const {
  if (!requests && !cuda) return common_host_bytes();
  const auto p = plan(shape, cuda);
  const auto host_common = common_host_bytes();
  if (p.host_bytes < host_common)
    throw std::logic_error("native MO batch host accounting underflow");

  auto common = host_common;
  auto per_request = p.host_bytes - host_common;
  if (cuda) {
    if (p.device_bytes < p.aligned_numeric)
      throw std::logic_error("native MO batch device accounting underflow");
    common = checked_add(common, p.device_bytes - p.aligned_numeric);
    per_request = checked_add(per_request, p.aligned_numeric);
  }
  return checked_add(common, checked_mul(requests, per_request));
}

std::size_t NativeBlockProvider::batch_capacity(const std::array<std::size_t, 4>& shape,
                                                bool cuda) const {
  const auto common = batch_bytes(shape, 0, cuda);
  const auto one = batch_bytes(shape, 1, cuda);
  if (one < common) throw std::logic_error("native MO batch accounting underflow");
  const auto per_request = one - common;
  if (!per_request || budget_ <= common) return 0;
  return (budget_ - common) / per_request;
}

std::vector<std::vector<double>> NativeBlockProvider::get_many(const std::vector<MOSlots>& requests,
                                                               bool cuda, int device,
                                                               vibeqc_tensor::Metrics* metrics,
                                                               ProviderWork* work) const {
  if (requests.empty()) return {};

  std::vector<std::array<std::size_t, 4>> shapes;
  std::vector<NumericBlockPlan> plans;
  shapes.reserve(requests.size());
  plans.reserve(requests.size());
  const auto host_common = common_host_bytes();
  auto batch_memory = host_common;
  std::size_t shared_device_fixed = 0;
  bool shared_device_fixed_set = false;
  for (const auto& slots : requests) {
    std::array<std::size_t, 4> shape{};
    for (unsigned k = 0; k < 4; ++k) shape[k] = slots[k].size();
    const auto p = plan(shape, cuda);
    if (p.host_bytes < host_common)
      throw std::logic_error("native MO batch host accounting underflow");
    auto incremental = p.host_bytes - host_common;
    if (cuda) {
      if (p.device_bytes < p.aligned_numeric)
        throw std::logic_error("native MO batch device accounting underflow");
      const auto fixed = p.device_bytes - p.aligned_numeric;
      if (!shared_device_fixed_set) {
        shared_device_fixed = fixed;
        shared_device_fixed_set = true;
        batch_memory = checked_add(batch_memory, shared_device_fixed);
      } else if (fixed != shared_device_fixed) {
        throw std::logic_error("native MO batch fixed device allowance changed by request shape");
      }
      incremental = checked_add(incremental, p.aligned_numeric);
    }
    batch_memory = checked_add(batch_memory, incremental);
    shapes.push_back(shape);
    plans.push_back(p);
  }
  if (batch_memory > budget_) throw std::length_error("native MO batch exceeds memory budget");

  struct State {
    std::array<std::size_t, 4> shape{};
    NumericBlockPlan plan{};
    std::array<std::vector<double>, 4> coefficients;
    std::vector<double> output;
    std::vector<double> first;
    std::vector<double> second;
  };
  std::vector<State> states;
  states.reserve(requests.size());
  for (std::size_t request = 0; request < requests.size(); ++request) {
    State state;
    state.shape = shapes[request];
    state.plan = plans[request];
    const auto& slots = requests[request];
    for (unsigned k = 0; k < 4; ++k) {
      auto& c = state.coefficients[k];
      c.resize(ref_.nbf * state.shape[k]);
      for (std::size_t mo = 0; mo < state.shape[k]; ++mo) {
        const auto column = slots[k][mo];
        if (column != padded_mo && column >= ref_.nbf)
          throw std::invalid_argument("native MO index out of range");
        if (column == padded_mo) continue;
        if (std::find(slots[k].begin(), slots[k].begin() + mo, column) != slots[k].begin() + mo)
          throw std::invalid_argument("native MO slots require unique real columns");
        for (std::size_t mu = 0; mu < ref_.nbf; ++mu) {
          const auto value = ref_.coefficients[mu * ref_.nbf + column];
          if (!std::isfinite(value)) throw std::invalid_argument("nonfinite MO coefficient");
          c[mu * state.shape[k] + mo] = value;
        }
      }
    }
    state.output.assign(state.plan.output_elements, 0.0);
    if (!cuda) {
      state.first.resize(state.plan.stage_elements);
      state.second.resize(state.plan.stage_elements);
    }
    states.push_back(std::move(state));
  }

  struct DeviceBatch {
    void* pointer{};
    ~DeviceBatch() {
#if VIBEQC_HAS_CUDA
      if (pointer) posthf_cuda_batch_destroy_v1(pointer);
#endif
    }
  } device_batch;
#if VIBEQC_HAS_CUDA
  char error[2048]{};
  auto check = [&](int status) {
    if (status == 2) throw std::bad_alloc();
    if (status) throw std::runtime_error(error);
  };
#endif
  if (cuda) {
#if VIBEQC_HAS_CUDA
    std::vector<std::size_t> batch_shapes;
    std::vector<double> panels;
    batch_shapes.reserve(4 * states.size());
    std::size_t coefficient_elements = 0;
    std::size_t maximum_allocation_bytes = 0;
    for (const auto& state : states) {
      batch_shapes.insert(batch_shapes.end(), state.shape.begin(), state.shape.end());
      coefficient_elements = checked_add(coefficient_elements, state.plan.coefficient_elements);
      maximum_allocation_bytes = checked_add(maximum_allocation_bytes, state.plan.allocation_bytes);
    }
    panels.reserve(coefficient_elements);
    for (const auto& state : states)
      for (const auto& panel : state.coefficients)
        panels.insert(panels.end(), panel.begin(), panel.end());
    if (panels.size() != coefficient_elements)
      throw std::logic_error("native MO batch coefficient accounting mismatch");
    check(posthf_cuda_batch_create_v1(device, ref_.nbf, states.size(), batch_shapes.data(),
                                      tile_.data(), panels.data(), maximum_allocation_bytes,
                                      &device_batch.pointer, error, sizeof(error)));
    if (work)
      work->h2d_bytes =
          checked_add(work->h2d_bytes, checked_mul(coefficient_elements, sizeof(double)));
#else
    (void)device;
    throw std::runtime_error("CUDA MO provider is not compiled");
#endif
  }

  std::size_t raw_elements = 1;
  for (const auto extent : tile_) raw_elements = checked_mul(raw_elements, extent);
  std::vector<double> raw(raw_elements);
  if (work) {
    work->source_scans = checked_add(work->source_scans, 1);
    work->mo_blocks = checked_add(work->mo_blocks, requests.size());
  }
  for (std::size_t u = 0; u < ref_.nbf; u += tile_[0])
    for (std::size_t v = 0; v < ref_.nbf; v += tile_[1])
      for (std::size_t w = 0; w < ref_.nbf; w += tile_[2])
        for (std::size_t x = 0; x < ref_.nbf; x += tile_[3]) {
          const std::array<std::size_t, 4> begin{u, v, w, x};
          std::array<std::size_t, 4> current{};
          std::size_t elements = 1;
          for (unsigned k = 0; k < 4; ++k) {
            current[k] = std::min(tile_[k], ref_.nbf - begin[k]);
            elements = checked_mul(elements, current[k]);
          }
          source_.read(integrals::ElectronInteractionOperator::eri, begin, current, raw.data(),
                       elements);
          if (work) {
            work->source_reads = checked_add(work->source_reads, 1);
            work->source_values = checked_add(work->source_values, elements);
            for (const auto& state : states) {
              auto work_shape = current;
              auto work_elements = elements;
              for (unsigned k = 0; k < 4; ++k) {
                const auto ao = work_shape[0];
                const auto rest = work_elements / ao;
                const auto columns = state.shape[k];
                work->transform_fmas =
                    checked_add(work->transform_fmas, checked_mul(checked_mul(rest, ao), columns));
                work_elements = checked_mul(rest, columns);
                for (unsigned j = 0; j < 3; ++j) work_shape[j] = work_shape[j + 1];
                work_shape[3] = columns;
              }
            }
          }
          if (cuda) {
#if VIBEQC_HAS_CUDA
            if (work) {
              work->cuda_transform_calls = checked_add(work->cuda_transform_calls, states.size());
              work->cuda_batch_calls = checked_add(work->cuda_batch_calls, 1);
              work->h2d_bytes = checked_add(work->h2d_bytes, checked_mul(elements, sizeof(double)));
            }
            check(posthf_cuda_batch_add_v1(device_batch.pointer, raw.data(), begin.data(),
                                           current.data(), error, sizeof(error)));
#endif
            continue;
          }

          for (auto& state : states) {
            auto transformed_shape = current;
            auto transformed_elements = elements;
            const double* input = raw.data();
            bool write_first = true;
            for (unsigned k = 0; k < 4; ++k) {
              auto& output = write_first ? state.first : state.second;
              const auto ao = transformed_shape[0];
              const auto rest = transformed_elements / ao;
              const auto columns = state.shape[k];
              std::fill_n(output.begin(), rest * columns, 0.0);
              // Identical cyclic order to CG10 transform_tile and cuBLAS:
              // input[AO,rest]^T @ C[AO,MO] -> output[rest,MO].
              for (std::size_t r = 0; r < rest; ++r)
                for (std::size_t a = 0; a < ao; ++a)
                  for (std::size_t m = 0; m < columns; ++m)
                    output[r * columns + m] +=
                        input[a * rest + r] * state.coefficients[k][(begin[k] + a) * columns + m];
              transformed_elements = rest * columns;
              for (unsigned j = 0; j < 3; ++j) transformed_shape[j] = transformed_shape[j + 1];
              transformed_shape[3] = columns;
              input = output.data();
              write_first = !write_first;
            }
            for (std::size_t q = 0; q < state.output.size(); ++q) {
              state.output[q] += input[q];
              if (!std::isfinite(state.output[q]))
                throw std::runtime_error("nonfinite native MO transformation");
            }
          }
        }

  if (cuda) {
#if VIBEQC_HAS_CUDA
    std::vector<double*> output_pointers;
    std::vector<std::size_t> output_sizes;
    output_pointers.reserve(states.size());
    output_sizes.reserve(states.size());
    for (auto& state : states) {
      output_pointers.push_back(state.output.data());
      output_sizes.push_back(state.output.size());
      if (work)
        work->d2h_bytes =
            checked_add(work->d2h_bytes, checked_mul(state.output.size(), sizeof(double)));
    }
    check(posthf_cuda_batch_download_v1(device_batch.pointer, output_pointers.data(),
                                        output_sizes.data(), output_sizes.size(), error,
                                        sizeof(error)));
    if (metrics) {
      vibeqc_tensor::Metrics measured{};
      check(posthf_cuda_batch_metrics_v1(device_batch.pointer, &measured, error, sizeof(error)));
      metrics->input_ms += measured.input_ms;
      metrics->output_ms += measured.output_ms;
      metrics->library_ms += measured.library_ms;
      metrics->kernel_ms += measured.kernel_ms;
      metrics->owned_device_bytes = std::max<decltype(metrics->owned_device_bytes)>(
          metrics->owned_device_bytes, measured.owned_device_bytes);
      metrics->provider_retained_bytes = std::max<decltype(metrics->provider_retained_bytes)>(
          metrics->provider_retained_bytes, measured.provider_retained_bytes);
    }
#else
    (void)metrics;
#endif
  }

  std::vector<std::vector<double>> outputs;
  outputs.reserve(states.size());
  for (auto& state : states) outputs.push_back(std::move(state.output));
  return outputs;
}

std::vector<double> NativeBlockProvider::get(const MOSlots& slots, bool cuda, int device,
                                             vibeqc_tensor::Metrics* metrics,
                                             ProviderWork* work) const {
  std::vector<MOSlots> requests{slots};
  auto outputs = get_many(requests, cuda, device, metrics, work);
  return std::move(outputs.front());
}

DensityFittedBlockProvider::DensityFittedBlockProvider(const RawSource& source,
                                                       const scf::PhysicalReference& reference,
                                                       std::size_t budget,
                                                       double relative_threshold)
    : source_(source),
      ref_(reference),
      budget_(budget),
      n_(reference.nbf),
      naux_(source.naux()),
      relative_threshold_(relative_threshold) {
  if (!n_ || !ref_.nocc || ref_.nocc >= n_ || source_.nbf() != n_ || !naux_ ||
      ref_.coefficients.size() != checked_mul(n_, n_) || !budget_ ||
      !std::isfinite(relative_threshold_) || relative_threshold_ <= 0.0 ||
      relative_threshold_ >= 1.0)
    throw std::invalid_argument("invalid density-fitted MO provider request");

  // This bound includes source owners, value generation/factorization and the
  // returned RI-MP2 value state. Preflight it before any dense factor is built.
  const auto preparation_peak = ri_mp2_capacity(source_.orbital(), source_.auxiliary(), ref_.nocc);
  if (preparation_peak > budget_)
    throw std::length_error("density-fitted MO provider exceeds memory budget");

  auto raw =
      integrals::build_density_fitting_integrals(source_.orbital(), source_.auxiliary(), false);
  if (raw.nbf != n_ || raw.naux != naux_ || raw.metric.size() != checked_mul(naux_, naux_) ||
      raw.three_center.size() != checked_mul(checked_mul(n_, n_), naux_))
    throw std::runtime_error("density-fitted MO provider integral dimensions changed");
  const auto factor =
      integrals::factor_density_fitting_metric(raw.metric, naux_, relative_threshold_);
  metric_ = std::move(raw.metric);
  inverse_square_root_ = factor.inverse_square_root;

  const auto pair_count = checked_mul(n_, n_);
  const auto three_center_elements = checked_mul(pair_count, naux_);
  transformed_.assign(three_center_elements, 0.0);
  whitened_.assign(three_center_elements, 0.0);
  auto idx = [this](std::size_t p, std::size_t q, std::size_t aux) {
    return (p * n_ + q) * naux_ + aux;
  };
  for (std::size_t p = 0; p < n_; ++p)
    for (std::size_t q = 0; q < n_; ++q)
      for (std::size_t mu = 0; mu < n_; ++mu) {
        const double left = ref_.coefficients[mu * n_ + p];
        for (std::size_t nu = 0; nu < n_; ++nu) {
          const double coefficient = left * ref_.coefficients[nu * n_ + q];
          if (coefficient == 0.0) continue;
          const auto raw_pair = (mu * n_ + nu) * naux_;
          for (std::size_t aux = 0; aux < naux_; ++aux)
            transformed_[idx(p, q, aux)] += coefficient * raw.three_center[raw_pair + aux];
        }
      }
  for (std::size_t p = 0; p < n_; ++p)
    for (std::size_t q = 0; q < n_; ++q)
      for (std::size_t target = 0; target < naux_; ++target)
        for (std::size_t source_aux = 0; source_aux < naux_; ++source_aux)
          whitened_[idx(p, q, target)] += transformed_[idx(p, q, source_aux)] *
                                          inverse_square_root_[source_aux * naux_ + target];

  const auto reference_elements = checked_add(checked_mul(5, pair_count), n_);
  auto bytes =
      checked_add(source_capacity(source_.orbital()), source_capacity(source_.auxiliary()));
  bytes = checked_add(bytes, checked_mul(reference_elements, sizeof(double)));
  bytes = checked_add(bytes, checked_mul(metric_.size(), sizeof(double)));
  bytes = checked_add(bytes, checked_mul(inverse_square_root_.size(), sizeof(double)));
  bytes = checked_add(bytes, checked_mul(transformed_.size(), sizeof(double)));
  bytes = checked_add(bytes, checked_mul(whitened_.size(), sizeof(double)));
  provider_bytes_ = bytes;
  if (provider_bytes_ > budget_)
    throw std::length_error("density-fitted MO provider retained state exceeds memory budget");

  auto finite = [](const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  if (!finite(metric_) || !finite(inverse_square_root_) || !finite(transformed_) ||
      !finite(whitened_))
    throw std::runtime_error("density-fitted MO provider produced nonfinite state");
}

std::vector<double> DensityFittedBlockProvider::get(const MOSlots& slots, bool cuda, int device,
                                                    vibeqc_tensor::Metrics* metrics) const {
  (void)device;
  (void)metrics;
  if (cuda) throw std::runtime_error("CUDA density-fitted MO block execution is not implemented");
  std::array<std::size_t, 4> shape{};
  std::size_t output_elements = 1;
  for (unsigned axis = 0; axis < 4; ++axis) {
    shape[axis] = slots[axis].size();
    if (!shape[axis]) throw std::invalid_argument("density-fitted MO block has an empty axis");
    output_elements = checked_mul(output_elements, shape[axis]);
    for (std::size_t item = 0; item < shape[axis]; ++item) {
      const auto orbital = slots[axis][item];
      if (orbital != padded_mo && orbital >= n_)
        throw std::invalid_argument("density-fitted MO index out of range");
      if (orbital != padded_mo && std::find(slots[axis].begin(), slots[axis].begin() + item,
                                            orbital) != slots[axis].begin() + item)
        throw std::invalid_argument("density-fitted MO slots require unique real columns");
    }
  }
  const auto output_bytes = checked_mul(output_elements, sizeof(double));
  if (checked_add(provider_bytes_, output_bytes) > budget_)
    throw std::length_error("density-fitted MO block exceeds memory budget");

  std::vector<double> output(output_elements, 0.0);
  auto pair_offset = [this](std::size_t p, std::size_t q) { return (p * n_ + q) * naux_; };
  std::size_t out = 0;
  for (const auto p : slots[0])
    for (const auto q : slots[1])
      for (const auto r : slots[2])
        for (const auto s : slots[3]) {
          if (p == padded_mo || q == padded_mo || r == padded_mo || s == padded_mo) {
            ++out;
            continue;
          }
          const auto left = pair_offset(p, q), right = pair_offset(r, s);
          double value = 0.0;
          for (std::size_t aux = 0; aux < naux_; ++aux)
            value += whitened_[left + aux] * whitened_[right + aux];
          if (!std::isfinite(value)) throw std::runtime_error("nonfinite density-fitted MO block");
          output[out++] = value;
        }
  return output;
}

}  // namespace vibeqc::posthf
