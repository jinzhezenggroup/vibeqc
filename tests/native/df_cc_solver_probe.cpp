// Internal supplied-Hamiltonian solver qualification. Binary input keeps large
// DF factors out of command arguments; no reference state enters production.
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cc/solver.hpp"

int main() {
  try {
    std::array<std::uint64_t, 7> header{};
    std::cin.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    generativeqc::cc::Problem p;
    p.nocc = header[0];
    p.nvir = header[1];
    p.naux = header[2];
    generativeqc::cc::SolverOptions options;
    options.max_bytes = header[3];
    options.max_iterations = header[4];
    options.diis_size = header[5];
    options.df_auxiliary_reduction = !(header[6] & 4);
    options.energy_tolerance = 1e-12;
    options.residual_tolerance = 1e-10;
    const auto o = p.nocc, v = p.nvir;
    const std::array<std::size_t, 16> sizes{o * o,
                                            o * v,
                                            v * v,
                                            o * v * o * v,
                                            o * v * v * o,
                                            o * o * v * v,
                                            p.naux ? 0 : o * v * v * v,
                                            o * v * o * o,
                                            o * o * o * o,
                                            p.naux ? 0 : v * v * v * v,
                                            o * v,
                                            o * o * v * v,
                                            o * v,
                                            o * o * v * v,
                                            p.naux * o * v,
                                            p.naux * v * v};
    const std::array<std::vector<double>*, 16> fields{
        &p.foo,  &p.fov,  &p.fvv, &p.ovov, &p.ovvo,       &p.oovv,       &p.ovvv,   &p.ovoo,
        &p.oooo, &p.vvvv, &p.d1,  &p.d2,   &p.initial_t1, &p.initial_t2, &p.df_bov, &p.df_bvv};
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      fields[i]->resize(sizes[i]);
      std::cin.read(reinterpret_cast<char*>(fields[i]->data()), sizes[i] * sizeof(double));
    }
    if (!std::cin) throw std::invalid_argument("truncated solver probe input");
    // Mode 2 probes the default admission used by conventional response owners.
    if (header[6] == 2) {
      generativeqc::cc::validate_problem(p);
      throw std::runtime_error("conventional admission accepted DF");
    }
    const auto result = (header[6] & 1) ? generativeqc::cc::solve_cuda(p, options, 0)
                                        : generativeqc::cc::solve_cpu(p, options);
    const auto& d = result.diagnostic;
    std::cout << std::setprecision(17) << static_cast<int>(result.status) << ' '
              << result.correlation_energy << ' ' << d.iterations << ' ' << d.replay_r1_max << ' '
              << d.replay_r2_max << ' ' << d.numeric_capacity_bytes << ' ' << d.owned_device_bytes
              << ' ' << d.setup_h2d_bytes << ' ' << d.scalar_d2h_bytes << ' '
              << d.amplitude_d2h_bytes << ' ' << d.iteration_graph_calls << ' '
              << d.replay_graph_calls << ' ' << d.df_auxiliary_slices << ' '
              << d.df_virtual_operations << ' ' << d.df_accumulation_calls << ' '
              << d.tensor_seconds << ' ' << d.df_hoisted_evaluations << ' '
              << d.df_preparation_calls << ' ' << d.df_contraction_terms << '\n';
    for (double x : result.t1) std::cout << x << ' ';
    for (double x : result.t2) std::cout << x << ' ';
    std::cout << '\n' << result.reason << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
