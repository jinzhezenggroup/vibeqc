#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

#include "runtime/resource_cuda.cuh"
#include "scf/cuda/df_source_kernels.hpp"
#include "scf/cuda/metadata_upload.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf::cuda_execution {

/** Private DF source ownership shared by setup and replay. These allocations contain
 * metadata/transforms, independently of SCF state. */
/**
 * Device-side metadata and public-basis transforms for budgeted DF replay.
 *
 * This object is intentionally separate from the ordinary HF bucket plan:
 * the latter owns a large arena of Fock/SCF state, while this source owns only
 * the immutable basis description needed to regenerate a requested DF tile.
 */
struct CudaDensityFittingIntegralSourceImpl {
  int device_id{-1};
  // Freeze the generated schedule so a warm plan never mixes mapping policies.
  unsigned value_mapping{};
  unsigned value_math{};  // Frozen with mapping; unsupported angular classes use generic Rys.
  std::size_t batch_size{};
  std::size_t public_nbf{};
  std::size_t public_naux{};
  std::size_t cartesian_nbf{};
  std::size_t cartesian_naux{};
  std::size_t dummy_index{};
  DeviceBatch batch{};
  const DfPublicAoExpansion* orbital_to_cartesian{};
  const DfPublicAoExpansion* auxiliary_to_cartesian{};
  // Host mirror used only to translate a public per-system derivative index;
  // the packed DeviceBatch pointer cannot be dereferenced by host code.
  std::vector<std::int64_t> host_atom_offsets;
  std::vector<void*> allocations;
  std::size_t device_bytes{};
  std::size_t host_bytes{};
  std::size_t host_peak_bytes{};

  ~CudaDensityFittingIntegralSourceImpl() {
    if (device_id >= 0) (void)cudaSetDevice(device_id);
    for (void* pointer : allocations) (void)runtime::resource_cuda_free(pointer);
  }
};

/** Reject unsupported physical shells before packing either source or exported tensors. */
bool cuda_df_shell_domain(const core::System& system, const char* role, std::string& detail);

/** Build a source transactionally and return its current metric; ownership transfers only on
 * success. */
vibeqc_status create_cuda_density_fitting_integral_source_impl(
    int device_id, const std::vector<core::System>& orbital_systems,
    const std::vector<core::System>& auxiliary_systems,
    CudaDensityFittingIntegralSourceImpl** source, std::vector<double>& metrics, std::size_t& nbf,
    std::size_t& naux, std::string& detail);

}  // namespace vibeqc::scf::cuda_execution
