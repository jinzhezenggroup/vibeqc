// Standalone allocation audit, never linked into the production library.
// Executable-level new/delete interposition observes actual requested C++ heap
// bytes during native preparation and two serialized CPU fleet executions.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/fleet.hpp"

namespace {
struct Header {
  void* allocation;
  std::size_t bytes;
  bool tracked;
};
thread_local bool recording = false;
std::atomic<std::size_t> live{}, peak{}, allocations{};

void* allocate(std::size_t bytes, std::size_t alignment) {
  alignment = std::max(alignment, alignof(std::max_align_t));
  if (bytes > std::numeric_limits<std::size_t>::max() - sizeof(Header) - alignment)
    throw std::bad_alloc();
  void* raw = std::malloc(std::max<std::size_t>(bytes, 1) + sizeof(Header) + alignment);
  if (raw == nullptr) throw std::bad_alloc();
  auto address = reinterpret_cast<std::uintptr_t>(raw) + sizeof(Header);
  address = (address + alignment - 1) & ~(alignment - 1);
  auto* header = reinterpret_cast<Header*>(address) - 1;
  *header = {raw, bytes, recording};
  if (recording) {
    const auto now = live.fetch_add(bytes) + bytes;
    auto previous = peak.load();
    while (previous < now && !peak.compare_exchange_weak(previous, now)) {
    }
    ++allocations;
  }
  return reinterpret_cast<void*>(address);
}

void release(void* pointer) noexcept {
  if (pointer == nullptr) return;
  auto* header = static_cast<Header*>(pointer) - 1;
  if (header->tracked) live.fetch_sub(header->bytes);
  std::free(header->allocation);
}
}  // namespace

void* operator new(std::size_t n) { return allocate(n, alignof(std::max_align_t)); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
  return allocate(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p) noexcept { release(p); }
void operator delete[](void* p) noexcept { release(p); }
void operator delete(void* p, std::size_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t) noexcept { release(p); }
void operator delete(void* p, std::align_val_t) noexcept { release(p); }
void operator delete[](void* p, std::align_val_t) noexcept { release(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { release(p); }

extern "C" int vibeqc_resource_tracking_begin_v1(unsigned);
extern "C" int vibeqc_resource_tracking_end_v1(std::uint64_t*, std::uint64_t*);

int main() {
  std::size_t count = 0;
  int unrestricted = 0, fitted = 0;
  if (!(std::cin >> count >> unrestricted >> fitted) || count == 0 || count > 64) return 2;
  std::vector<vibeqc::core::System> inputs(count);
  for (auto& system : inputs) {
    std::size_t atoms = 0, shells = 0;
    std::cin >> atoms >> shells >> system.charge >> system.multiplicity >> system.electron_count;
    if (atoms == 0 || shells == 0 || atoms > 10000 || shells > 10000) return 2;
    system.atoms.resize(atoms);
    system.shells.resize(shells);
    for (auto& atom : system.atoms)
      std::cin >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    for (auto& shell : system.shells) {
      std::size_t primitives = 0;
      std::cin >> shell.atom_index >> shell.angular_momentum >> primitives;
      if (primitives == 0 || primitives > 10000) return 2;
      shell.primitives.resize(primitives);
      for (auto& primitive : shell.primitives)
        std::cin >> primitive.exponent >> primitive.coefficient;
    }
  }
  if (!std::cin) return 2;
  for (auto& system : inputs) {
    // Match the public system-construction boundary before supplying caller
    // inputs to FleetPlan; raw BSE coefficients are not native coefficients.
    std::string detail;
    if (vibeqc::molecule::validate_and_normalize(system, detail) != VIBEQC_STATUS_SUCCESS) return 2;
  }
  const auto method = unrestricted ? VIBEQC_METHOD_UHF : VIBEQC_METHOD_RHF;
  vibeqc::scf::ScfOptions options;
  options.density_fitting_mode =
      fitted ? VIBEQC_DENSITY_FITTING_CPU_REFERENCE : VIBEQC_DENSITY_FITTING_NONE;
  std::array<double, 64> energies{};
  std::array<unsigned, 64> iterations{};
  bool success = true;
  if (vibeqc_resource_tracking_begin_v1(1) != 0) return 2;
  recording = true;
  {
    // Pass by value so the prepared fleet owns a measured copy of caller
    // topology. The input parser and caller-retained inputs are outside scope.
    vibeqc::scf::FleetPlan fleet(inputs, method, options, true, false, false, false, 0);
    for (unsigned replay = 0; replay < 2; ++replay) {
      const auto results = fleet.execute({});
      for (std::size_t i = 0; i < count; ++i) {
        success = success && results[i].status == VIBEQC_STATUS_SUCCESS;
        energies[i] = results[i].scf.energy;
        iterations[i] = results[i].scf.iterations;
      }
    }
  }
  recording = false;
  std::uint64_t sampled = 0, samples = 0;
  if (vibeqc_resource_tracking_end_v1(&sampled, &samples) != 0) return 2;
  std::cout << "{\"peak_host_bytes\":" << peak << ",\"live_after_destroy\":" << live
            << ",\"allocations\":" << allocations << ",\"sampled_host_bytes\":" << sampled
            << ",\"energies\":[" << std::setprecision(17);
  for (std::size_t i = 0; i < count; ++i) std::cout << (i ? "," : "") << energies[i];
  std::cout << "],\"iterations\":[";
  for (std::size_t i = 0; i < count; ++i) std::cout << (i ? "," : "") << iterations[i];
  std::cout << "],\"converged\":" << (success ? "true" : "false") << "}\n";
  return success && live == 0 ? 0 : 1;
}
