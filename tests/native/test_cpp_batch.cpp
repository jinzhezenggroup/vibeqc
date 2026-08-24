#include "qce/qce.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    qce_context_descriptor context_descriptor{
        sizeof(qce_context_descriptor), QCE_ABI_VERSION, 0,
        QCE_BACKEND_CPU_REFERENCE};
    qce::Context context(context_descriptor);

    const std::array<qce_primitive, 6> h_primitives{{
        {3.42525091, 0.15432897}, {0.62391373, 0.53532814},
        {0.16885540, 0.44463454}, {3.42525091, 0.15432897},
        {0.62391373, 0.53532814}, {0.16885540, 0.44463454},
    }};
    const std::array<qce_atom, 2> h_atoms{{
        {1, 0.0, 0.0, -0.7}, {1, 0.0, 0.0, 0.7},
    }};
    const std::array<qce_shell, 2> h_shells{{{0, 0, 0, 3}, {1, 0, 3, 3}}};
    qce_system_descriptor h2_descriptor{
        sizeof(qce_system_descriptor), QCE_ABI_VERSION,
        h_atoms.data(), h_atoms.size(), h_shells.data(), h_shells.size(),
        h_primitives.data(), h_primitives.size(), 0, 1};
    qce::System h2(context, h2_descriptor);

    const std::array<qce_atom, 1> he_atoms{{{2, 0.0, 0.0, 0.0}}};
    const std::array<qce_primitive, 3> he_primitives{{
        {6.36242139, 0.15432897},
        {1.15892300, 0.53532814},
        {0.31364979, 0.44463454},
    }};
    const std::array<qce_shell, 1> he_shells{{{0, 0, 0, 3}}};
    qce_system_descriptor he_descriptor{
        sizeof(qce_system_descriptor), QCE_ABI_VERSION,
        he_atoms.data(), he_atoms.size(), he_shells.data(), he_shells.size(),
        he_primitives.data(), he_primitives.size(), 0, 1};
    qce::System helium(context, he_descriptor);

    qce_method_descriptor method{
        sizeof(qce_method_descriptor), QCE_ABI_VERSION, QCE_METHOD_RHF,
        100, 8, 1.0e-12, 1.0e-10, 1.0e-14};
    const std::array<const qce::System*, 2> systems{{&h2, &helium}};
    qce::Batch batch(context, systems, method);
    const auto cold = batch.execute();
    const auto warm = batch.execute();
    require(cold.size() == 2 && warm.size() == 2,
            "C++ batch result count is incorrect");
    require(std::abs(cold[0].energy - (-1.11671432506255)) < 2.0e-9,
            "C++ batch H2 energy is incorrect");
    require(std::abs(cold[1].energy - (-2.807783957539976)) < 2.0e-10,
            "C++ batch helium energy is incorrect");
    require(!cold[0].warm_start_used && warm[0].warm_start_used,
            "C++ batch did not expose warm-start state");
    require(cold[0].forces.size() == 6 && cold[1].forces.size() == 3,
            "C++ batch introduced padded force storage");
    std::cout << "C++ ragged batch API: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
