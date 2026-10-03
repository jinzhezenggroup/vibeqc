#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/df_source_internal.hpp"
#include "scf/cuda/df_source_kernels.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda/topology.hpp"

namespace generativeqc::scf::cuda_execution {

/** Host topology validation, public-basis transforms, metadata uploads and metric setup. The
 * generated launch boundary alone requires CUDA compilation. */
bool cuda_df_shell_domain(const core::System& system, const char* role, std::string& detail) {
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > 3U) {
      detail = std::string("CUDA DF ") + role + " shells beyond f (l > 3) are unsupported";
      return false;
    }
  }
  return true;
}

bool cuda_df_value_domain(const core::System& orbital, const core::System& auxiliary,
                          std::string& detail) {
  if (!cuda_df_shell_domain(orbital, "orbital", detail)) return false;
  for (const auto& shell : auxiliary.shells) {
    if (shell.angular_momentum > 4U) {
      detail = "CUDA DF auxiliary value shells beyond g (l > 4) are unsupported";
      return false;
    }
  }
  return true;
}

namespace {

/** Pack only nonzero normalized expansion terms, preserving Cartesian order.
 * Sorting reproduces the old dense scan's summation order even when the
 * scientific expansion lists its terms in a different order.
 */
std::vector<DfPublicAoExpansion> make_public_to_cartesian_transform(const core::System& system) {
  const auto public_count = molecule::ao_count(system);
  std::vector<DfPublicAoExpansion> transform(public_count);
  std::size_t public_offset = 0, cartesian_offset = 0;
  for (const auto& shell : system.shells) {
    const auto components = molecule::cartesian_components(shell.angular_momentum);
    const auto expansions =
        molecule::ao_expansions(shell.angular_momentum, system.basis_representation);
    for (const auto& expansion : expansions) {
      auto& packed = transform.at(public_offset++);
      for (std::size_t component = 0; component < components.size(); ++component) {
        const auto term = std::find_if(expansion.begin(), expansion.end(), [&](const auto& entry) {
          return entry.component == components[component];
        });
        if (term == expansion.end() || term->coefficient == 0.0) continue;
        if (packed.count == kDfPublicAoExpansionTerms)
          throw std::invalid_argument("DF public AO expansion exceeds normalized basis bound");
        packed.cartesian[packed.count] = static_cast<std::int32_t>(cartesian_offset + component);
        packed.coefficients[packed.count++] = term->coefficient;
      }
      if (!packed.count) throw std::invalid_argument("empty DF public AO expansion");
    }
    cartesian_offset += components.size();
  }
  if (public_offset != public_count || cartesian_offset != molecule::cartesian_ao_count(system))
    throw std::invalid_argument("DF public-basis transform dimensions are inconsistent");
  return transform;
}

}  // namespace

generativeqc_status create_cuda_density_fitting_integral_source_impl(
    int device_id, const std::vector<core::System>& orbital_systems,
    const std::vector<core::System>& auxiliary_systems,
    CudaDensityFittingIntegralSourceImpl** source, std::vector<double>& metrics, std::size_t& nbf,
    std::size_t& naux, std::string& detail) {
  detail.clear();
  metrics.clear();
  nbf = 0U;
  naux = 0U;
  if (source == nullptr || device_id < 0 || orbital_systems.empty() ||
      orbital_systems.size() != auxiliary_systems.size()) {
    detail = "bounded DF source dimensions are invalid";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  *source = nullptr;
  const std::size_t batch_size = orbital_systems.size();
  for (std::size_t system = 0; system < batch_size; ++system) {
    if (!cuda_df_value_domain(orbital_systems[system], auxiliary_systems[system], detail)) {
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
  }
  const std::size_t public_nbf = molecule::ao_count(orbital_systems.front());
  const std::size_t public_naux = molecule::ao_count(auxiliary_systems.front());
  const std::size_t cartesian_nbf = molecule::cartesian_ao_count(orbital_systems.front());
  const std::size_t cartesian_naux = molecule::cartesian_ao_count(auxiliary_systems.front());
  std::size_t metric_elements = 0;
  std::size_t metric_total_elements = 0;
  if (!generativeqc::runtime::checked_multiply(public_naux, public_naux, metric_elements) ||
      !generativeqc::runtime::checked_multiply(batch_size, metric_elements,
                                               metric_total_elements)) {
    detail = "bounded DF source metric dimensions overflow size_t";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }
  if (public_nbf == 0U || public_naux == 0U || cartesian_nbf == 0U || cartesian_naux == 0U ||
      batch_size > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    detail = "bounded DF source basis dimensions are invalid";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  std::vector<core::System> combined;
  try {
    combined.reserve(batch_size);
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for bounded DF source systems";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }
  for (std::size_t system = 0; system < batch_size; ++system) {
    const core::System& orbital = orbital_systems[system];
    const core::System& auxiliary = auxiliary_systems[system];
    if (molecule::ao_count(orbital) != public_nbf || molecule::ao_count(auxiliary) != public_naux ||
        molecule::cartesian_ao_count(orbital) != cartesian_nbf ||
        molecule::cartesian_ao_count(auxiliary) != cartesian_naux ||
        orbital.atoms.size() != auxiliary.atoms.size()) {
      detail = "bounded DF source requires homogeneous AO dimensions";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t atom = 0; atom < orbital.atoms.size(); ++atom) {
      if (orbital.atoms[atom].atomic_number != auxiliary.atoms[atom].atomic_number ||
          orbital.atoms[atom].position != auxiliary.atoms[atom].position) {
        detail = "bounded DF source orbital and auxiliary geometries differ";
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      }
    }
    core::System item;
    item.atoms = orbital.atoms;
    item.shells = orbital.shells;
    item.shells.insert(item.shells.end(), auxiliary.shells.begin(), auxiliary.shells.end());
    item.shells.push_back({0, 0, {{0.0, 1.0}}});
    item.charge = orbital.charge;
    item.multiplicity = 1;
    item.electron_count = 2;
    item.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
    try {
      combined.push_back(std::move(item));
    } catch (const std::bad_alloc&) {
      detail = "host allocation failed for bounded DF source systems";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
  }

  HostBatch host;
  std::vector<const std::vector<double>*> no_warm(batch_size, nullptr);
  try {
    // Reuse normalized basis packing, explicitly omitting SCF density and
    // pair/quartet task construction. Public transforms are owned below.
    if (!pack_host_batch(combined, no_warm, host, false, true, false, HostBasisPacking::DfValues) ||
        host.nbf != cartesian_nbf + cartesian_naux + 1U) {
      detail = "bounded DF source Cartesian packing failed";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed while packing bounded DF source";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t int32_max = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  if (host.nbf > int32_max || host.direct_nbf > int32_max || cartesian_nbf > int32_max ||
      cartesian_naux > int32_max || cartesian_nbf > int32_max - cartesian_naux) {
    detail = "bounded DF source dimensions exceed CUDA int32 indexing";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) return source_cuda_status(cuda_error);
  auto candidate = std::unique_ptr<CudaDensityFittingIntegralSourceImpl>(
      new (std::nothrow) CudaDensityFittingIntegralSourceImpl{});
  if (!candidate) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  candidate->device_id = device_id;
  const auto requested_mapping = cuda_policy::df_value_mapping_requested();
  candidate->value_mapping = resolve_cuda_df_source_value_mapping(requested_mapping, true);
  candidate->raw_value_mapping = resolve_cuda_df_source_value_mapping(requested_mapping, false);
  if (!cuda_policy::df_value_math_requested(candidate->value_math)) {
    detail = "GENERATIVEQC_DF_VALUE_MATH must be auto, generic, polynomial rys or candidate";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  // Freeze the value-only capability with this immutable basis owner. All
  // derivative entry points reject it before launching or changing outputs.
  for (const auto& auxiliary : auxiliary_systems)
    for (const auto& shell : auxiliary.shells)
      candidate->auxiliary_g_values_only |= shell.angular_momentum == 4U;
  if (candidate->auxiliary_g_values_only && candidate->value_math != 0U) {
    detail = "g auxiliary DF values require the generic generated math policy";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  candidate->batch_size = batch_size;
  candidate->public_nbf = public_nbf;
  candidate->public_naux = public_naux;
  candidate->cartesian_nbf = cartesian_nbf;
  candidate->cartesian_naux = cartesian_naux;
  candidate->dummy_index = cartesian_nbf + cartesian_naux;
  candidate->batch.batch_size = static_cast<std::int32_t>(batch_size);
  candidate->batch.nbf = static_cast<std::int32_t>(host.nbf);
  candidate->batch.direct_nbf = static_cast<std::int32_t>(host.direct_nbf);
  candidate->batch.total_atoms = static_cast<std::int64_t>(host.atomic_numbers.size());
  candidate->batch.total_shells = static_cast<std::int64_t>(host.shell_atoms.size());
  try {
    candidate->host_atom_offsets = host.atom_offsets;
    candidate->orbital_identities.reserve(batch_size);
    candidate->auxiliary_identities.reserve(batch_size);
    for (std::size_t system = 0; system < batch_size; ++system) {
      candidate->orbital_identities.emplace_back(orbital_systems[system]);
      candidate->auxiliary_identities.emplace_back(auxiliary_systems[system]);
    }
    // This early charge covers the atom-prefix mirror. The complete retained
    // identity/allocation ledger is finalized after upload below. Include the
    // owning object and vector capacity in the retained-host diagnostic so a
    // positive-budget plan cannot silently omit this metadata allocation.
    std::size_t atom_offset_bytes = 0U;
    if (!generativeqc::runtime::checked_multiply(candidate->host_atom_offsets.capacity(),
                                                 sizeof(std::int64_t), atom_offset_bytes) ||
        !generativeqc::runtime::checked_add(sizeof(*candidate), atom_offset_bytes,
                                            candidate->host_bytes)) {
      detail = "bounded DF source host metadata bytes overflow size_t";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for bounded DF source atom offsets";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }

  const auto source_vector_bytes_valid = [](const auto& values) {
    return values.size() <= std::numeric_limits<std::size_t>::max() / sizeof(values[0]);
  };
  if (!source_vector_bytes_valid(host.atom_offsets) ||
      !source_vector_bytes_valid(host.atom_systems) ||
      !source_vector_bytes_valid(host.atomic_numbers) ||
      !source_vector_bytes_valid(host.positions) || !source_vector_bytes_valid(host.shell_atoms) ||
      !source_vector_bytes_valid(host.shell_angular) ||
      !source_vector_bytes_valid(host.shell_ao_offsets) ||
      !source_vector_bytes_valid(host.shell_direct_ao_offsets) ||
      !source_vector_bytes_valid(host.shell_primitive_offsets) ||
      !source_vector_bytes_valid(host.ao_shells) ||
      !source_vector_bytes_valid(host.ao_term_counts) ||
      !source_vector_bytes_valid(host.ao_term_angular) ||
      !source_vector_bytes_valid(host.ao_term_coefficients) ||
      !source_vector_bytes_valid(host.direct_ao_shells) ||
      !source_vector_bytes_valid(host.direct_ao_angular) ||
      !source_vector_bytes_valid(host.direct_ao_coefficients) ||
      !source_vector_bytes_valid(host.primitive_exponents) ||
      !source_vector_bytes_valid(host.primitive_coefficients)) {
    detail = "bounded DF source metadata size overflows size_t";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }

#define GENERATIVEQC_UPLOAD_SOURCE_FIELD(field, values)                                         \
  do {                                                                                          \
    void* uploaded = nullptr;                                                                   \
    const generativeqc_status upload_status = source_upload(                                    \
        *candidate, (values).data(), (values).size() * sizeof((values)[0]), &uploaded, detail); \
    if (upload_status != GENERATIVEQC_STATUS_SUCCESS) return upload_status;                     \
    candidate->batch.field = static_cast<decltype(candidate->batch.field)>(uploaded);           \
  } while (false)

  GENERATIVEQC_UPLOAD_SOURCE_FIELD(atom_offsets, host.atom_offsets);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(atom_systems, host.atom_systems);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(atomic_numbers, host.atomic_numbers);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(positions, host.positions);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(shell_atoms, host.shell_atoms);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(shell_angular, host.shell_angular);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(shell_ao_offsets, host.shell_ao_offsets);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(shell_direct_ao_offsets, host.shell_direct_ao_offsets);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(shell_primitive_offsets, host.shell_primitive_offsets);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(ao_shells, host.ao_shells);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(ao_term_counts, host.ao_term_counts);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(ao_term_angular, host.ao_term_angular);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(ao_term_coefficients, host.ao_term_coefficients);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(direct_ao_shells, host.direct_ao_shells);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(direct_ao_angular, host.direct_ao_angular);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(direct_ao_coefficients, host.direct_ao_coefficients);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(primitive_exponents, host.primitive_exponents);
  GENERATIVEQC_UPLOAD_SOURCE_FIELD(primitive_coefficients, host.primitive_coefficients);
#undef GENERATIVEQC_UPLOAD_SOURCE_FIELD

  // Keep a transform per batch item.  Equal AO counts do not imply equal
  // shell layouts (for example, two hetero-nuclear systems can have the same
  // dimension but different contraction order), so a single front-item
  // transform would silently corrupt every later source replay.
  std::size_t orbital_transform_elements = 0;
  std::size_t auxiliary_transform_elements = 0;
  if (!generativeqc::runtime::checked_multiply(public_nbf, sizeof(DfPublicAoExpansion),
                                               orbital_transform_elements) ||
      !generativeqc::runtime::checked_multiply(public_naux, sizeof(DfPublicAoExpansion),
                                               auxiliary_transform_elements) ||
      !generativeqc::runtime::checked_multiply(batch_size, orbital_transform_elements,
                                               orbital_transform_elements) ||
      !generativeqc::runtime::checked_multiply(batch_size, auxiliary_transform_elements,
                                               auxiliary_transform_elements)) {
    detail = "bounded DF source transform dimensions overflow size_t";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }
  std::vector<DfPublicAoExpansion> orbital_transform;
  std::vector<DfPublicAoExpansion> auxiliary_transform;
  try {
    orbital_transform.resize(orbital_transform_elements / sizeof(DfPublicAoExpansion));
    auxiliary_transform.resize(auxiliary_transform_elements / sizeof(DfPublicAoExpansion));
    for (std::size_t system = 0; system < batch_size; ++system) {
      const auto orbital_item = make_public_to_cartesian_transform(orbital_systems[system]);
      const auto auxiliary_item = make_public_to_cartesian_transform(auxiliary_systems[system]);
      std::copy(orbital_item.begin(), orbital_item.end(),
                orbital_transform.begin() + system * public_nbf);
      std::copy(auxiliary_item.begin(), auxiliary_item.end(),
                auxiliary_transform.begin() + system * public_naux);
    }
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for bounded DF source transforms";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  void* orbital_transform_device = nullptr;
  generativeqc_status status = source_upload(*candidate, orbital_transform.data(),
                                             orbital_transform.size() * sizeof(DfPublicAoExpansion),
                                             &orbital_transform_device, detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  void* auxiliary_transform_device = nullptr;
  status = source_upload(*candidate, auxiliary_transform.data(),
                         auxiliary_transform.size() * sizeof(DfPublicAoExpansion),
                         &auxiliary_transform_device, detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  candidate->orbital_to_cartesian =
      static_cast<const DfPublicAoExpansion*>(orbital_transform_device);
  candidate->auxiliary_to_cartesian =
      static_cast<const DfPublicAoExpansion*>(auxiliary_transform_device);

  try {
    metrics.resize(metric_total_elements);
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for bounded DF metric";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }
  cudaStream_t stream = nullptr;
  cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) return source_cuda_status(cuda_error);
  double* metric_device = nullptr;
  const unsigned metric_outputs_per_block = candidate->value_mapping == 2U ? 4U : 128U;
  if (metric_elements >
      static_cast<std::size_t>(std::numeric_limits<unsigned>::max()) * metric_outputs_per_block) {
    (void)cudaStreamDestroy(stream);
    detail = "bounded DF metric launch exceeds CUDA grid limits";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  cuda_error = runtime::resource_cuda_malloc(&metric_device, metric_elements * sizeof(double));
  if (cuda_error != cudaSuccess) {
    (void)cudaStreamDestroy(stream);
    return source_cuda_status(cuda_error);
  }
  for (std::size_t system = 0; system < batch_size && cuda_error == cudaSuccess; ++system) {
    launch_build_cuda_df_metric_source_kernel(
        false,
        static_cast<unsigned>((metric_elements + metric_outputs_per_block - 1U) /
                              metric_outputs_per_block),
        128U, 0, stream, candidate->batch, cartesian_nbf, cartesian_naux, public_naux,
        candidate->dummy_index, system, 0, public_naux, -1, candidate->auxiliary_to_cartesian,
        metric_device, candidate->value_mapping);
    cuda_error = cudaGetLastError();
    if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(metrics.data() + system * metric_elements, metric_device,
                              metric_elements * sizeof(double), cudaMemcpyDeviceToHost);
    }
  }
  (void)runtime::resource_cuda_free(metric_device);
  (void)cudaStreamDestroy(stream);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA bounded DF metric generation failed";
    return source_cuda_status(cuda_error);
  }
  // Record the transient host setup footprint while the packed basis metadata,
  // public-basis transforms, and generated metric are all alive.  Only the
  // compact atom-offset mirror is retained after this function returns, but a
  // positive-budget diagnostic must still expose the larger construction peak.
  const auto capacity_bytes = [](const auto& values) -> long double {
    return static_cast<long double>(values.capacity()) *
           static_cast<long double>(sizeof(values[0]));
  };
  const auto system_capacity_bytes = [&](const core::System& system) {
    long double bytes = static_cast<long double>(sizeof(system)) + capacity_bytes(system.atoms) +
                        capacity_bytes(system.shells);
    for (const core::Shell& shell : system.shells) {
      bytes += capacity_bytes(shell.primitives);
    }
    return bytes;
  };
  long double host_peak = static_cast<long double>(sizeof(*candidate));
#define GENERATIVEQC_SOURCE_HOST_FIELD(field) host_peak += capacity_bytes(host.field)
  GENERATIVEQC_SOURCE_HOST_FIELD(atom_offsets);
  GENERATIVEQC_SOURCE_HOST_FIELD(atom_systems);
  GENERATIVEQC_SOURCE_HOST_FIELD(atomic_numbers);
  GENERATIVEQC_SOURCE_HOST_FIELD(positions);
  GENERATIVEQC_SOURCE_HOST_FIELD(system_shell_offsets);
  GENERATIVEQC_SOURCE_HOST_FIELD(shell_atoms);
  GENERATIVEQC_SOURCE_HOST_FIELD(shell_angular);
  GENERATIVEQC_SOURCE_HOST_FIELD(shell_ao_offsets);
  GENERATIVEQC_SOURCE_HOST_FIELD(shell_direct_ao_offsets);
  GENERATIVEQC_SOURCE_HOST_FIELD(shell_primitive_offsets);
  GENERATIVEQC_SOURCE_HOST_FIELD(system_shell_pair_offsets);
  GENERATIVEQC_SOURCE_HOST_FIELD(system_shell_quartet_offsets);
  GENERATIVEQC_SOURCE_HOST_FIELD(system_shell_pair_block_offsets);
  GENERATIVEQC_SOURCE_HOST_FIELD(system_shell_pair_block_quartet_offsets);
  GENERATIVEQC_SOURCE_HOST_FIELD(shell_pair_systems);
  GENERATIVEQC_SOURCE_HOST_FIELD(shell_pair_first);
  GENERATIVEQC_SOURCE_HOST_FIELD(shell_pair_second);
  GENERATIVEQC_SOURCE_HOST_FIELD(shell_pair_primitive_offsets);
  GENERATIVEQC_SOURCE_HOST_FIELD(psss_resident_tasks);
  GENERATIVEQC_SOURCE_HOST_FIELD(psss_resident_ket_pairs);
  GENERATIVEQC_SOURCE_HOST_FIELD(ao_shells);
  GENERATIVEQC_SOURCE_HOST_FIELD(ao_term_counts);
  GENERATIVEQC_SOURCE_HOST_FIELD(ao_term_angular);
  GENERATIVEQC_SOURCE_HOST_FIELD(ao_term_coefficients);
  GENERATIVEQC_SOURCE_HOST_FIELD(direct_ao_shells);
  GENERATIVEQC_SOURCE_HOST_FIELD(direct_ao_angular);
  GENERATIVEQC_SOURCE_HOST_FIELD(direct_ao_coefficients);
  GENERATIVEQC_SOURCE_HOST_FIELD(ao_to_direct_transform);
  GENERATIVEQC_SOURCE_HOST_FIELD(primitive_exponents);
  GENERATIVEQC_SOURCE_HOST_FIELD(primitive_coefficients);
  GENERATIVEQC_SOURCE_HOST_FIELD(occupied);
  GENERATIVEQC_SOURCE_HOST_FIELD(warm_mask);
  GENERATIVEQC_SOURCE_HOST_FIELD(warm_density);
#undef GENERATIVEQC_SOURCE_HOST_FIELD
  host_peak += capacity_bytes(orbital_transform);
  host_peak += capacity_bytes(auxiliary_transform);
  host_peak += capacity_bytes(metrics);
  host_peak += capacity_bytes(combined);
  host_peak += capacity_bytes(no_warm);
  host_peak += capacity_bytes(candidate->host_atom_offsets);
  host_peak += capacity_bytes(candidate->allocations);
  long double identity_bytes = capacity_bytes(candidate->orbital_identities) +
                               capacity_bytes(candidate->auxiliary_identities);
  for (const auto& identity : candidate->orbital_identities)
    identity_bytes += identity.storage_bytes();
  for (const auto& identity : candidate->auxiliary_identities)
    identity_bytes += identity.storage_bytes();
  host_peak += identity_bytes;
  for (const core::System& system : combined) {
    host_peak += system_capacity_bytes(system);
  }
  // Each transform helper briefly materializes one public-to-Cartesian item
  // before copying it into the packed arrays. Charge that per-item temporary
  // in addition to the retained batch transforms.
  host_peak += static_cast<long double>(public_nbf) * sizeof(DfPublicAoExpansion);
  host_peak += static_cast<long double>(public_naux) * sizeof(DfPublicAoExpansion);
  // Recompute retained bytes after all metadata uploads have populated the
  // device-allocation pointer vector.  The dynamic pointer array is small but
  // is still host state owned by the source and must not disappear from the
  // resident-byte diagnostic.
  const long double retained_host = static_cast<long double>(sizeof(*candidate)) +
                                    capacity_bytes(candidate->host_atom_offsets) +
                                    capacity_bytes(candidate->allocations) + identity_bytes;
  const long double size_limit = static_cast<long double>(std::numeric_limits<std::size_t>::max());
  candidate->host_bytes = retained_host >= size_limit ? std::numeric_limits<std::size_t>::max()
                                                      : static_cast<std::size_t>(retained_host);
  candidate->host_peak_bytes = host_peak >= size_limit ? std::numeric_limits<std::size_t>::max()
                                                       : static_cast<std::size_t>(host_peak);
  *source = candidate.release();
  nbf = public_nbf;
  naux = public_naux;
  return GENERATIVEQC_STATUS_SUCCESS;
}

}  // namespace generativeqc::scf::cuda_execution
