#pragma once

#include <algorithm>
#include <cstddef>

#include "generated_df_ccsd_hoisted_cpu.hpp"

namespace generativeqc::cc {

// Complete per-evaluation DF storage/work choice. The caller adds histories,
// inputs, replay arena and retained host state before allocation. A rejected
// fast candidate can be replanned with allow_hoisting=false under the same
// budget, without materializing any omitted four-index virtual block.
struct DFIterationPlan {
  bool hoisted{};
  std::size_t iteration{}, auxiliary{}, preparation{}, accumulation{};
  std::size_t preparation_terms{}, auxiliary_terms{}, core_terms{};
};

inline DFIterationPlan df_iteration_plan(std::size_t o, std::size_t v, std::size_t q, bool cuda,
                                         bool allow_hoisting) {
  using generated::df::checked_add;
  using generated::df::checked_mul;
  namespace fast = generated::dfhoist;
  const auto n1 = checked_mul(o, v), n2 = checked_mul(n1, n1);
  const auto old_terms = cuda ? fast::fallback_virtual_cuda_contraction_terms(o, v)
                              : fast::fallback_virtual_cpu_contraction_terms(o, v);
  DFIterationPlan plan{false,
                       generated::dfcore::iteration_arena_elements(o, v),
                       cuda ? generated::df::virtual_cuda_arena_elements(o, v)
                            : generated::df::virtual_cpu_arena_elements(o, v),
                       0,
                       checked_add(n1, n2),
                       0,
                       old_terms,
                       fast::fallback_core_contraction_terms(o, v)};
  if (!allow_hoisting) return plan;
  const auto prepare_work = fast::prepare_contraction_terms(o, v);
  const auto q_work = fast::auxiliary_contraction_terms(o, v);
  const auto core_work = fast::iteration_contraction_terms(o, v);
  const auto old_work = checked_add(checked_mul(q, old_terms), plan.core_terms);
  const auto new_work = checked_add(prepare_work, checked_add(checked_mul(q, q_work), core_work));
  if (new_work >= old_work) return plan;
  plan.hoisted = true;
  plan.iteration = fast::iteration_arena_elements(o, v);
  // Independent convergence replay still uses the original expanded one-Q
  // action. Its arena can be reused because all accumulated outputs are owned.
  plan.auxiliary = std::max(plan.auxiliary, fast::auxiliary_arena_elements(o, v));
  plan.preparation = fast::prepare_arena_elements(o, v);
  plan.accumulation =
      checked_add(plan.accumulation, checked_add(checked_mul(3, n2), checked_mul(v, v)));
  plan.preparation_terms = prepare_work;
  plan.auxiliary_terms = q_work;
  plan.core_terms = core_work;
  return plan;
}

}  // namespace generativeqc::cc
