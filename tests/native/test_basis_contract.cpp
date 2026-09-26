#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "api/handles.hpp"
#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "posthf/raw_source.hpp"

namespace {
vibeqc::core::System system() {
  vibeqc::core::System result;
  result.atoms = {{26, {0, 0, 0}}};
  result.shells = {{0, 0, {{80, 1}, {5, 0}}}};
  result.charge = 24;
  return result;
}
void require(bool condition, const char* text) {
  if (!condition) throw std::runtime_error(text);
}

template <typename Exception, typename Function>
void require_error(Function operation, const char* detail) {
  try {
    operation();
  } catch (const Exception& error) {
    require(std::string(error.what()).find(detail) != std::string::npos,
            "capability failure must identify its boundary");
    return;
  }
  throw std::runtime_error("unsupported input was unexpectedly accepted");
}

void source_capability_boundaries() {
  for (unsigned l : {5U, std::numeric_limits<unsigned>::max()}) {
    require_error<std::invalid_argument>([=] { (void)vibeqc::molecule::cartesian_components(l); },
                                         "l<=4");
    for (auto representation : {VIBEQC_BASIS_CARTESIAN, VIBEQC_BASIS_SPHERICAL}) {
      require_error<std::invalid_argument>(
          [=] { (void)vibeqc::molecule::ao_expansions(l, representation); }, "l<=4");
    }
  }
  for (auto representation : {VIBEQC_BASIS_CARTESIAN, VIBEQC_BASIS_SPHERICAL}) {
    auto orbital = system();
    orbital.basis_representation = representation;
    orbital.shells[0].angular_momentum = 4;
    std::string detail;
    require(vibeqc::molecule::validate_and_normalize(orbital, detail) == VIBEQC_STATUS_SUCCESS,
            "valid g source must normalize");
    auto auxiliary = orbital;
    vibeqc::posthf::RawSource source(orbital, &auxiliary);
    const std::size_t expected = representation == VIBEQC_BASIS_CARTESIAN ? 15 : 9;
    require(source.nbf() == expected && source.naux() == expected,
            "raw orbital and auxiliary sources must retain every public g AO");
    auxiliary.shells[0].angular_momentum = 5;
    require_error<std::invalid_argument>(
        [&] { vibeqc::posthf::RawSource rejected(orbital, &auxiliary); },
        "raw auxiliary source supports through g");
    orbital.shells[0].angular_momentum = 5;
    require_error<std::invalid_argument>([&] { vibeqc::posthf::RawSource rejected(orbital); },
                                         "raw post-HF source supports through g");
  }
}

void tensor_extent_boundary() {
  auto oversized = system();
  // O(65536) small shell records on 64-bit hosts, but nbf^4 overflows size_t.
  // The guard must throw before any dense tensor or AO recurrence allocation.
  constexpr unsigned exponent = (std::numeric_limits<std::size_t>::digits + 3) / 4;
  oversized.shells.resize(std::size_t{1} << exponent, oversized.shells.front());
  for (bool derivatives : {false, true}) {
    require_error<std::overflow_error>(
        [&] { vibeqc::integrals::build_integrals(oversized, derivatives); },
        "CPU integral tensor extent overflows size_t");
  }
}

void system_backend_boundary() {
  // This is the host-side ABI preflight only: no device or CUDA execution is
  // initialized. A resolved backend fixture exercises rejection in CPU CI.
  vibeqc_context context;
  vibeqc_atom atom{2, 0, 0, 0};
  vibeqc_primitive primitive{1, 1};
  vibeqc_shell shell{0, 4, 0, 1};
  vibeqc_system_descriptor descriptor{
      sizeof(descriptor), VIBEQC_ABI_VERSION, &atom, 1, &shell, 1, &primitive, 1, 0, 1};
  for (auto representation : {VIBEQC_BASIS_CARTESIAN, VIBEQC_BASIS_SPHERICAL}) {
    descriptor.basis_representation = representation;
    for (auto backend : {VIBEQC_BACKEND_CPU_REFERENCE, VIBEQC_BACKEND_CUDA}) {
      context.state.executed_backend = backend;
      for (unsigned l : {3U, 4U, 5U}) {
        shell.angular_momentum = l;
        vibeqc_system sentinel;
        auto* result = &sentinel;
        const auto status = vibeqc_system_create(&context, &descriptor, &result);
        const bool supported = l <= (backend == VIBEQC_BACKEND_CPU_REFERENCE ? 4U : 3U);
        if (supported) {
          require(status == VIBEQC_STATUS_SUCCESS && result != nullptr && result != &sentinel,
                  "supported shell must publish a system");
          vibeqc_system_destroy(result);
        } else {
          require(status == VIBEQC_STATUS_NOT_IMPLEMENTED && result == nullptr,
                  "unsupported shell must fail without publishing a system");
          if (backend == VIBEQC_BACKEND_CUDA)
            require(
                context.last_detail.find("CUDA basis execution supports l<=3") != std::string::npos,
                "CUDA rejection must identify its backend capability boundary");
        }
      }
    }
  }
}
}  // namespace
int main() {
  try {
    source_capability_boundaries();
    tensor_extent_boundary();
    system_backend_boundary();
    std::string detail;
    auto valid = system();
    require(vibeqc::molecule::validate_and_normalize(valid, detail) == VIBEQC_STATUS_SUCCESS,
            "all-electron Fe ion with explicit small basis must be accepted");
    require(valid.electron_count == 2, "ionic charge must remain distinct from nuclear Z");
    require(
        valid.shells[0].primitives.size() == 2 && valid.shells[0].primitives[1].coefficient == 0,
        "zero primitives must be retained");
    for (double exponent : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
      auto input = system();
      input.shells[0].primitives[0].exponent = exponent;
      require(
          vibeqc::molecule::validate_and_normalize(input, detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
          "nonpositive or nonfinite exponent must fail before normalization");
    }
    auto overflow = system();
    overflow.charge = std::numeric_limits<int>::min();
    require(vibeqc::molecule::validate_and_normalize(overflow, detail) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "electron count must not overflow signed ABI storage");
    auto underflow = system();
    underflow.shells[0].primitives[1] = {0.01, std::numeric_limits<double>::denorm_min()};
    require(vibeqc::molecule::validate_and_normalize(underflow, detail) ==
                VIBEQC_STATUS_NUMERICAL_FAILURE,
            "native normalization must not silently erase a nonzero primitive");
    auto spin_overflow = system();
    spin_overflow.multiplicity = std::numeric_limits<unsigned>::max();
    require(vibeqc::molecule::validate_and_normalize(spin_overflow, detail) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "multiplicity must not wrap signed native occupation storage");
    spin_overflow.charge = 26 - std::numeric_limits<int>::max();
    spin_overflow.multiplicity = 2;
    require(vibeqc::molecule::validate_and_normalize(spin_overflow, detail) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "electron plus spin occupation intermediate must not overflow");
    auto unsupported = system();
    auto g = system();
    g.shells[0].angular_momentum = 4;
    require(vibeqc::molecule::validate_and_normalize(g, detail) == VIBEQC_STATUS_SUCCESS,
            "CPU g normalization must be accepted");
    require(vibeqc::molecule::cartesian_components(4).size() == 15,
            "g shell must have all fifteen Cartesian components");
    const auto harmonics = vibeqc::molecule::ao_expansions(4, VIBEQC_BASIS_SPHERICAL);
    require(harmonics.size() == 9 && harmonics[4].size() == 6,
            "g spherical expansion must retain all nine AOs and six m=0 terms");
    unsupported.shells[0].angular_momentum = 5;
    require(vibeqc::molecule::validate_and_normalize(unsupported, detail) ==
                    VIBEQC_STATUS_NOT_IMPLEMENTED &&
                detail.find("l=5") != std::string::npos,
            "high angular momentum must identify the missing shell");
    auto invalid_atom = system();
    invalid_atom.atoms[0].atomic_number = 119;
    require(vibeqc::molecule::validate_and_normalize(invalid_atom, detail) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "invalid atomic number must not become a nuclear charge");
    auto invalid_position = system();
    invalid_position.atoms[0].position[1] = std::numeric_limits<double>::infinity();
    require(vibeqc::molecule::validate_and_normalize(invalid_position, detail) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "nonfinite coordinates must fail before integral execution");
    std::cout << "basis contract gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
