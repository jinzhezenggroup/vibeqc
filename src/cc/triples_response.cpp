#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cc/triples_response_internal.hpp"
#include "generated_rccsd_cpu.hpp"

namespace generativeqc::cc {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b) { return generated::checked_add(a, b); }

std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("RCCSD(T) response size overflow");
  return a * b;
}

std::size_t bytes(std::size_t elements) { return checked_mul(elements, sizeof(double)); }

double degeneracy(std::size_t a, std::size_t b, std::size_t c) {
  if (a == c) return 6.0;
  if (a == b || b == c) return 2.0;
  return 1.0;
}

void accumulate(double& target, double contribution) {
  const double updated = target + contribution;
  if (!std::isfinite(updated)) throw std::runtime_error("nonfinite RCCSD(T) response accumulation");
  target = updated;
}

void add_block(std::vector<double>& target, const double* source) {
  for (std::size_t i = 0; i < target.size(); ++i) accumulate(target[i], source[i]);
}

void add_symmetric_matrix_block(std::vector<double>& target, const double* source, std::size_t n) {
  if (target.size() != checked_mul(n, n))
    throw std::logic_error("RCCSD(T) symmetric response block shape mismatch");
  for (std::size_t left = 0; left < n; ++left)
    for (std::size_t right = 0; right <= left; ++right) {
      const auto forward = left * n + right;
      const auto reverse = right * n + left;
      const double contribution = 0.5 * (source[forward] + source[reverse]);
      accumulate(target[forward], contribution);
      if (reverse != forward) accumulate(target[reverse], contribution);
    }
}

void add_last_two_symmetric_block(std::vector<double>& target, const double* source,
                                  std::size_t outer, std::size_t n) {
  const auto matrix_size = checked_mul(n, n);
  if (target.size() != checked_mul(outer, matrix_size))
    throw std::logic_error("RCCSD(T) tail-symmetric response block shape mismatch");
  for (std::size_t prefix = 0; prefix < outer; ++prefix) {
    const auto base = prefix * matrix_size;
    for (std::size_t left = 0; left < n; ++left)
      for (std::size_t right = 0; right <= left; ++right) {
        const auto forward = base + left * n + right;
        const auto reverse = base + right * n + left;
        const double contribution = 0.5 * (source[forward] + source[reverse]);
        accumulate(target[forward], contribution);
        if (reverse != forward) accumulate(target[reverse], contribution);
      }
  }
}

}  // namespace

double detail::validate_triples_response(const Problem& p, const SolverResult& cc,
                                         const std::vector<double>& eps_o,
                                         const std::vector<double>& eps_v,
                                         const TriplesResponseOptions& options) {
  validate_problem(p);
  if (!cc.converged())
    throw std::invalid_argument("RCCSD(T) response requires converged RCCSD amplitudes");
  if (eps_o.size() != p.nocc || eps_v.size() != p.nvir)
    throw std::invalid_argument("RCCSD(T) response orbital-energy shape mismatch");
  if (!std::isfinite(options.denominator_threshold) || options.denominator_threshold <= 0.0 ||
      !options.max_bytes || !options.batch_capacity)
    throw std::invalid_argument("invalid RCCSD(T) response options");
  if (cc.t1.size() != checked_mul(p.nocc, p.nvir) ||
      cc.t2.size() != checked_mul(checked_mul(p.nocc, p.nocc), checked_mul(p.nvir, p.nvir)))
    throw std::invalid_argument("RCCSD(T) response amplitude shape mismatch");

  for (const auto* amplitudes : {&cc.t1, &cc.t2})
    for (double value : *amplitudes)
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite RCCSD(T) amplitude");

  for (double value : eps_o)
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite occupied orbital energy");
  for (double value : eps_v)
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite virtual orbital energy");
  const double max_occ = *std::max_element(eps_o.begin(), eps_o.end());
  const double min_vir = *std::min_element(eps_v.begin(), eps_v.end());
  if (!(max_occ < min_vir))
    throw std::invalid_argument("noncanonical RCCSD(T) orbital-energy ordering");
  const double minimum_denominator = 3.0 * (min_vir - max_occ);
  if (!std::isfinite(minimum_denominator))
    throw std::invalid_argument("nonfinite RCCSD(T) denominator range");
  if (minimum_denominator <= options.denominator_threshold)
    throw std::invalid_argument("near-zero RCCSD(T) denominator");

  return minimum_denominator;
}
detail::TriplesResponseLayout detail::triples_response_layout(std::size_t o, std::size_t v,
                                                              std::size_t batch_capacity,
                                                              bool cuda) {
  if (!o || !v || !batch_capacity)
    throw std::invalid_argument("invalid triples response dimensions");
  const std::array<std::size_t, 8> output_sizes{checked_mul(checked_mul(o, v), checked_mul(v, v)),
                                                checked_mul(checked_mul(o, v), checked_mul(o, o)),
                                                checked_mul(checked_mul(o, v), checked_mul(o, v)),
                                                checked_mul(o, v),
                                                checked_mul(o, v),
                                                checked_mul(checked_mul(o, o), checked_mul(v, v)),
                                                o,
                                                v};
  std::size_t output_elements = 0;
  for (const auto size : output_sizes) output_elements = checked_add(output_elements, size);
  const std::size_t total_triples = checked_mul(v, checked_mul(v + 1, v + 2)) / 6;
  TriplesResponseLayout result;
  result.sizes = output_sizes;
  result.outputs = output_elements;
  result.q = std::min(batch_capacity, total_triples);
  result.arena = generated::triples_response_arena_elements(o, v, result.q);
  const auto controls = checked_mul(result.q, 3 * sizeof(std::int64_t) + 2 * sizeof(double));
  result.host_bytes = checked_add(bytes(output_elements), controls);
  if (cuda) {
    // Resident scientific inputs and accumulated cotangents each have the same
    // eight shapes. The energy seed and asynchronous error flag are also owned.
    const auto elements = checked_add(checked_mul(2, output_elements), result.arena);
    const auto raw =
        checked_add(bytes(elements), checked_add(controls, sizeof(double) + sizeof(int)));
    result.device_bytes = checked_mul(checked_add(raw, 255) / 256, 256);
    (void)checked_add(result.host_bytes, result.device_bytes);
  } else {
    result.host_bytes = checked_add(result.host_bytes, bytes(result.arena));
  }
  return result;
}

TriplesResponseResult triples_response_cpu(const Problem& p, const SolverResult& cc,
                                           const std::vector<double>& eps_o,
                                           const std::vector<double>& eps_v,
                                           const TriplesResponseOptions& options) {
  const double minimum_denominator =
      detail::validate_triples_response(p, cc, eps_o, eps_v, options);

  TriplesResponseResult result;
  result.minimum_absolute_denominator = minimum_denominator;
  const std::array<std::size_t, 8> output_sizes{
      checked_mul(checked_mul(p.nocc, p.nvir), checked_mul(p.nvir, p.nvir)),
      checked_mul(checked_mul(p.nocc, p.nvir), checked_mul(p.nocc, p.nocc)),
      checked_mul(checked_mul(p.nocc, p.nvir), checked_mul(p.nocc, p.nvir)),
      checked_mul(p.nocc, p.nvir),
      checked_mul(p.nocc, p.nvir),
      checked_mul(checked_mul(p.nocc, p.nocc), checked_mul(p.nvir, p.nvir)),
      p.nocc,
      p.nvir};
  std::size_t output_elements = 0;
  for (const auto size : output_sizes) output_elements = checked_add(output_elements, size);
  const std::size_t total_triples = checked_mul(p.nvir, checked_mul(p.nvir + 1, p.nvir + 2)) / 6;
  std::size_t q = std::min(options.batch_capacity, total_triples);
  for (; q; --q) {
    const auto arena_elements = generated::triples_response_arena_elements(p.nocc, p.nvir, q);
    const auto control_bytes = checked_mul(q, 5 * sizeof(double));
    const auto required =
        checked_add(bytes(output_elements), checked_add(bytes(arena_elements), control_bytes));
    if (required <= options.max_bytes) break;
  }
  if (!q) throw std::length_error("RCCSD(T) response exceeds the host memory budget");

  const auto arena_elements = generated::triples_response_arena_elements(p.nocc, p.nvir, q);
  result.arena_bytes = bytes(arena_elements);
  result.numeric_capacity_bytes = checked_add(
      bytes(output_elements), checked_add(result.arena_bytes, checked_mul(q, 5 * sizeof(double))));
  // Allocate response outputs only after the complete scratch/output admission.
  const std::array<std::vector<double>*, 8> output_vectors{
      &result.ovvv, &result.ovoo, &result.ovov,  &result.fov,
      &result.t1,   &result.t2,   &result.eps_o, &result.eps_v};
  for (std::size_t index = 0; index < output_vectors.size(); ++index)
    output_vectors[index]->assign(output_sizes[index], 0.0);
  std::vector<double> arena(arena_elements);
  std::vector<std::int64_t> a_map(q), b_map(q), c_map(q);
  std::vector<double> active(q), weights(q, 1.0);
  const double energy_seed = 1.0;

  generated::TriplesResponseInputs inputs{};
  inputs.ovvv = p.ovvv.data();
  inputs.ovoo = p.ovoo.data();
  inputs.ovov = p.ovov.data();
  inputs.fov = p.fov.data();
  inputs.t1 = cc.t1.data();
  inputs.t2 = cc.t2.data();
  inputs.eps_o = eps_o.data();
  inputs.eps_v = eps_v.data();
  inputs.a_map = a_map.data();
  inputs.b_map = b_map.data();
  inputs.c_map = c_map.data();
  inputs.active = active.data();
  inputs.degeneracy = weights.data();
  inputs.bar_triples_energy = &energy_seed;

  auto run_page = [&](std::size_t count) {
    if (!count) return;
    for (std::size_t page_lane = 0; page_lane < q; ++page_lane) {
      active[page_lane] = page_lane < count ? 1.0 : 0.0;
      if (page_lane >= count) weights[page_lane] = 1.0;
    }
    const auto response =
        generated::run_triples_response_cpu(p.nocc, p.nvir, q, inputs, arena.data(), arena.size());
    const auto ov = checked_mul(p.nocc, p.nvir);
    add_last_two_symmetric_block(result.ovvv, response.ovvv, ov, p.nvir);
    add_last_two_symmetric_block(result.ovoo, response.ovoo, ov, p.nocc);
    add_symmetric_matrix_block(result.ovov, response.ovov, ov);
    add_block(result.fov, response.fov);
    add_block(result.t1, response.t1);
    add_block(result.t2, response.t2);
    add_block(result.eps_o, response.eps_o);
    add_block(result.eps_v, response.eps_v);
    ++result.pages;
  };

  std::size_t lane = 0;
  for (std::size_t a = 0; a < p.nvir; ++a) {
    for (std::size_t b = 0; b <= a; ++b) {
      for (std::size_t c = 0; c <= b; ++c) {
        a_map[lane] = static_cast<std::int64_t>(a);
        b_map[lane] = static_cast<std::int64_t>(b);
        c_map[lane] = static_cast<std::int64_t>(c);
        weights[lane] = degeneracy(a, b, c);
        ++lane;
        if (lane == q) {
          run_page(lane);
          lane = 0;
        }
      }
    }
  }
  run_page(lane);
  // The generated arena is dead before diagnostic strings are published.
  // Release it first so metadata cannot extend the exact numeric peak when
  // the triples phase becomes the largest force phase at larger dimensions.
  std::vector<double>().swap(arena);
  result.program_hash = generated::triples_response_program_hash;
  result.reason =
      "runtime-indexed generated standard-(T) response completed with fused parameter-source "
      "projection";
  return result;
}

}  // namespace generativeqc::cc
