#include "cc/triples_fock_response.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "cc/triples_response_internal.hpp"
#include "generated_triples_fock_response_cpu.hpp"

namespace generativeqc::cc {
namespace {
using generated::checked_add;
using generated::checked_product;
std::size_t bytes(std::size_t count) { return checked_product({count, sizeof(double)}); }
void accumulate(double& target, double value) {
  target += value;
  if (!std::isfinite(target)) throw std::runtime_error("nonfinite triples Fock moment");
}
}  // namespace

detail::TriplesFockLayout detail::triples_fock_response_layout(std::size_t o, std::size_t v,
                                                               std::size_t capacity, bool cuda) {
  if (!o || !v || !capacity) throw std::invalid_argument("invalid triples Fock dimensions");
  TriplesFockLayout l;
  l.q = std::min(capacity, v);
  l.input_sizes = {checked_product({o, v, v, v}),
                   checked_product({o, v, o, o}),
                   checked_product({o, v, o, v}),
                   checked_product({o, v}),
                   checked_product({o, v}),
                   checked_product({o, o, v, v}),
                   o,
                   v};
  for (auto count : l.input_sizes) l.inputs = checked_add(l.inputs, count);
  l.outputs = checked_add(checked_product({o, o}), checked_product({v, v}));
  l.vector_elements = checked_product({l.q, o, o, o});
  l.arena = std::max({generated::triples_resolvent_arena_elements(o, v, l.q),
                      generated::triples_oo_moment_arena_elements(o, v, l.q),
                      generated::triples_vv_moment_arena_elements(o, v, l.q)});
  const auto controls = checked_product({l.q, 3 * sizeof(std::int64_t) + sizeof(double)});
  const auto scratch = bytes(checked_add(l.arena, checked_product({4, l.vector_elements})));
  l.host_bytes = checked_add(bytes(l.outputs), controls);
  if (cuda) {
    const auto raw = checked_add(checked_add(bytes(checked_add(l.inputs, l.outputs)), scratch),
                                 checked_add(controls, sizeof(int)));
    l.device_bytes = checked_product({checked_add(raw, 255) / 256, 256});
  } else {
    l.host_bytes = checked_add(l.host_bytes, scratch);
  }
  (void)checked_add(l.host_bytes, l.device_bytes);
  return l;
}

TriplesFockResponseResult triples_fock_response_cpu(const Problem& p, const SolverResult& cc,
                                                    const std::vector<double>& eo,
                                                    const std::vector<double>& ev,
                                                    const TriplesResponseOptions& options) {
  (void)detail::validate_triples_response(p, cc, eo, ev, options);
  auto l = detail::triples_fock_response_layout(p.nocc, p.nvir, options.batch_capacity, false);
  while (l.numeric_bytes() > options.max_bytes && l.q > 1)
    l = detail::triples_fock_response_layout(p.nocc, p.nvir, l.q - 1, false);
  if (l.numeric_bytes() > options.max_bytes)
    throw std::length_error("triples Fock response exceeds host memory budget");
  const auto o = p.nocc, v = p.nvir, q = l.q;
  TriplesFockResponseResult result;
  result.page_capacity = q;
  result.numeric_capacity_bytes = l.numeric_bytes();
  result.foo.assign(o * o, 0.0);
  result.fvv.assign(v * v, 0.0);
  std::vector<double> arena(l.arena), xl(l.vector_elements), yl(l.vector_elements),
      xr(l.vector_elements), yr(l.vector_elements), active(q);
  std::vector<std::int64_t> amap(q), bmap(q), cmap(q);
  generated::TriplesResolventInputs in{};
  in.ovvv = p.ovvv.data();
  in.ovoo = p.ovoo.data();
  in.ovov = p.ovov.data();
  in.fov = p.fov.data();
  in.t1 = cc.t1.data();
  in.t2 = cc.t2.data();
  in.eps_o = eo.data();
  in.eps_v = ev.data();
  in.a_map = amap.data();
  in.b_map = bmap.data();
  in.c_map = cmap.data();
  in.active = active.data();
  auto vectors = [&](std::size_t start, std::vector<double>& x, std::vector<double>& y) {
    const auto count = std::min(q, v - start);
    for (std::size_t lane = 0; lane < q; ++lane) {
      amap[lane] = lane < count ? start + lane : 0;
      active[lane] = lane < count ? 1.0 : 0.0;
    }
    const auto out = generated::run_triples_resolvent_cpu(o, v, q, in, arena.data(), arena.size());
    std::copy_n(out.x, l.vector_elements, x.data());
    std::copy_n(out.y, l.vector_elements, y.data());
    ++result.vector_pages;
    return count;
  };
  for (std::size_t b = 0; b < v; ++b)
    for (std::size_t c = 0; c <= b; ++c) {
      std::fill(bmap.begin(), bmap.end(), b);
      std::fill(cmap.begin(), cmap.end(), c);
      const double weight = b == c ? 1.0 : 2.0;
      ++result.pair_panels;
      for (std::size_t left = 0; left < v; left += q) {
        const auto nl = vectors(left, xl, yl);
        generated::TriplesFockMomentInputs moments{xl.data(), yl.data(), nullptr, nullptr};
        const auto occupied =
            generated::run_triples_oo_moment_cpu(o, v, q, moments, arena.data(), arena.size());
        for (std::size_t k = 0; k < result.foo.size(); ++k)
          accumulate(result.foo[k], weight * occupied.values[k]);
        ++result.occupied_moments;
        for (std::size_t right = left; right < v; right += q) {
          const bool same = right == left;
          const auto nr = same ? nl : vectors(right, xr, yr);
          moments.x_right = same ? xl.data() : xr.data();
          moments.y_right = same ? yl.data() : yr.data();
          const auto virt =
              generated::run_triples_vv_moment_cpu(o, v, q, moments, arena.data(), arena.size());
          for (std::size_t i = 0; i < nl; ++i)
            for (std::size_t j = 0; j < nr; ++j) {
              const double value = weight * virt.values[i * q + j];
              accumulate(result.fvv[(left + i) * v + right + j], value);
              if (!same) accumulate(result.fvv[(right + j) * v + left + i], value);
            }
          ++result.virtual_moments;
        }
      }
    }
  result.program_hash = generated::triples_resolvent_program_hash;
  return result;
}

}  // namespace generativeqc::cc
