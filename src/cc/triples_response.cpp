#include "cc/triples_response.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "generated_rccsd_cpu.hpp"

namespace vibeqc::cc {
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

void add_block(std::vector<double>& target, const double* source) {
  for (std::size_t i = 0; i < target.size(); ++i) {
    const double updated = target[i] + source[i];
    if (!std::isfinite(updated))
      throw std::runtime_error("nonfinite RCCSD(T) response accumulation");
    target[i] = updated;
  }
}

}  // namespace

TriplesResponseResult triples_response_cpu(const Problem& p, const SolverResult& cc,
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

  TriplesResponseResult result;
  result.minimum_absolute_denominator = minimum_denominator;
  result.program_hash = generated::triples_response_program_hash;
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
    add_block(result.ovvv, response.ovvv);
    add_block(result.ovoo, response.ovoo);
    add_block(result.ovov, response.ovov);
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
  result.reason = "runtime-indexed generated standard-(T) response completed";
  return result;
}

}  // namespace vibeqc::cc
