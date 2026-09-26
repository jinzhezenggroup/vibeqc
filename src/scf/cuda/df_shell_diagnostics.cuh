#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace vibeqc::scf {
/** Source-level operation counts, never hardware instruction/traffic counters.
 * The small-argument count is a subset of the positive-series branch. Stage
 * durations come from the separate class/signature CUDA/Nsight intervals.
 */
enum class DfShellWork : unsigned {
  shell_tasks,
  active_shell_tasks,
  primitive_products,
  geometry_preparations,
  boys_evaluations,
  boys_order_sum,
  boys_series_iterations,
  boys_series,
  boys_small_argument,
  boys_large_argument,
  axis_polynomial_calls,
  specialized_prepare_axis_calls,
  cache_coefficient_values,
  convolution_iterations,
  active_component_products,
  public_weight_loads,
  public_nonzero_weights,
  expansion_term_products,
  folding_shared_atomics,
  folding_direct_stores,
  gradient_atomics_a,
  gradient_atomics_b,
  gradient_atomics_c,
  gradient_atomics_shared_atom,
  gradient_atomics_distinct_atom,
  subgroup_rendezvous,
  orbital_local_accumulations,
  rys_evaluations,
  rys_roots,
  recurrence_states,
  count
};
inline constexpr std::array<const char*, static_cast<unsigned>(DfShellWork::count)>
    df_shell_work_names{"shell_tasks",
                        "active_shell_tasks",
                        "primitive_products",
                        "geometry_preparations",
                        "boys_evaluations",
                        "boys_order_sum",
                        "boys_series_iterations",
                        "boys_series",
                        "boys_small_argument",
                        "boys_large_argument",
                        "axis_polynomial_calls",
                        "specialized_prepare_axis_calls",
                        "cache_coefficient_values",
                        "convolution_iterations",
                        "active_component_products",
                        "public_weight_loads",
                        "public_nonzero_weights",
                        "expansion_term_products",
                        "folding_shared_atomics",
                        "folding_direct_stores",
                        "gradient_atomics_a",
                        "gradient_atomics_b",
                        "gradient_atomics_c",
                        "gradient_atomics_shared_atom",
                        "gradient_atomics_distinct_atom",
                        "subgroup_rendezvous",
                        "orbital_local_accumulations",
                        "rys_evaluations",
                        "rys_roots",
                        "recurrence_states"};

/** Fixed diagnostic storage reused after each bounded signature packet.
 * Shards distribute diagnostic atomics; each owner publishes aggregated work
 * rather than one counter update per coefficient or primitive operation.
 * The caller owns both arrays through the stream drain. A non-null sink opts
 * into explicit readback/drains and must never accompany clean timing.
 */
struct DfShellDiagnostics {
  static constexpr unsigned packet_capacity = 24;
  static constexpr unsigned shards = 16;
  static constexpr unsigned metrics = static_cast<unsigned>(DfShellWork::count);
  static constexpr unsigned row_elements = shards * metrics;
  static constexpr unsigned elements = packet_capacity * row_elements;
  using Value = unsigned long long;
  using Observer = void (*)(unsigned, unsigned, unsigned, std::size_t, std::size_t, std::size_t,
                            std::span<const Value>, void*);
  Value* device{};
  Value* host{};
  Observer observer{};
  void* observer_context{};
  std::size_t readback_bytes{}, stream_drains{};
};
}  // namespace vibeqc::scf
