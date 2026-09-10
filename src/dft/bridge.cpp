// Private source-checkout ABI. It adds no executable DFT method or force API.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>

#include "api/handles.hpp"
#include "dft/ao_grid.hpp"

namespace {
template <class F>
int guarded(char* error, std::size_t size, F operation) noexcept {
  try {
    operation();
    return 0;
  } catch (const std::exception& e) {
    if (error && size) std::snprintf(error, size, "%s", e.what());
    return 1;
  } catch (...) {
    if (error && size) std::snprintf(error, size, "unknown AO grid failure");
    return 1;
  }
}
using vibeqc::dft::AoBasis;
}  // namespace

extern "C" {
int vibeqc_grid_basis_create_v1(const vibeqc_system* system, void** output, std::size_t* dimensions,
                                char* error, std::size_t size) {
  return guarded(error, size, [&] {
    if (!output) throw std::invalid_argument("null AO basis output");
    *output = nullptr;
    if (!system || !dimensions) throw std::invalid_argument("null AO basis input");
    auto basis = std::make_unique<AoBasis>(system->data);
    dimensions[0] = basis->natom;
    dimensions[1] = basis->nprimitive;
    dimensions[2] = basis->nao;
    *output = basis.release();
  });
}
void vibeqc_grid_basis_destroy_v1(void* basis) { delete static_cast<AoBasis*>(basis); }
int vibeqc_grid_basis_pack_v1(const void* handle, double* output, std::size_t elements, char* error,
                              std::size_t size) {
  return guarded(error, size, [&] {
    if (!handle || !output) throw std::invalid_argument("null AO packing input");
    const auto& basis = *static_cast<const AoBasis*>(handle);
    if (elements != basis.packed.size()) throw std::invalid_argument("AO packed size mismatch");
    std::copy(basis.packed.begin(), basis.packed.end(), output);
  });
}
int vibeqc_grid_ao_v1(const void* handle, const double* points, std::size_t npoint, unsigned order,
                      std::size_t begin, std::size_t count, double* output, std::size_t elements,
                      char* error, std::size_t size) {
  return guarded(error, size, [&] {
    if (!handle) throw std::invalid_argument("null AO basis");
    static_cast<const AoBasis*>(handle)->evaluate(points, npoint, order, begin, count, output,
                                                  elements);
  });
}
/** Selected columns share the normalized AO evaluator; no global AO tile is
 * materialized. The sorted unique map and its count are explicit ABI inputs. */
int vibeqc_grid_ao_selected_v1(const void* handle, const double* points, std::size_t npoint,
                               unsigned order, const std::size_t* ao_ids, std::size_t count,
                               double* output, std::size_t elements, char* error,
                               std::size_t size) {
  return guarded(error, size, [&] {
    if (!handle || (count && !ao_ids)) throw std::invalid_argument("null selected AO input");
    static_cast<const AoBasis*>(handle)->evaluate(points, npoint, order, 0, count, output, elements,
                                                  ao_ids);
  });
}
}
