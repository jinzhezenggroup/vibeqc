#pragma once

#include <cstddef>
#include <cstdint>

#include "vibeqc/vibeqc.h"

/** Private ctypes bridge for #163; deliberately absent from the installed API.
 * A snapshot owns its token and arrays, but borrows no calculation pointer.
 * Reads/checks require a live batch and compare its current #162 token. */
struct vibeqc_ks_snapshot;
struct vibeqc_ks_xc_response;

extern "C" {
vibeqc_status vibeqc_ks_snapshot_create_v1(vibeqc_batch* batch, std::size_t index,
                                           vibeqc_ks_snapshot** output, std::uint64_t* metadata,
                                           std::size_t metadata_count);
vibeqc_status vibeqc_ks_snapshot_check_v1(const vibeqc_batch* batch,
                                          const vibeqc_ks_snapshot* snapshot);
vibeqc_status vibeqc_ks_snapshot_copy_v1(const vibeqc_batch* batch,
                                         const vibeqc_ks_snapshot* snapshot, double* values,
                                         std::size_t count);
void vibeqc_ks_snapshot_destroy_v1(vibeqc_ks_snapshot* snapshot);
vibeqc_status vibeqc_ks_snapshot_energy_v1(const vibeqc_batch* batch,
                                           const vibeqc_ks_snapshot* snapshot, double* energy);
/** Live native proof: 0=all-electron, 1=ECP; never inferred from electron count. */
vibeqc_status vibeqc_ks_snapshot_hamiltonian_v1(const vibeqc_batch* batch,
                                                const vibeqc_ks_snapshot* snapshot,
                                                std::uint32_t* kind);
/** Private bounded CUDA XC response owner. It copies the successful state's
 * exact density/basis/grid and retains its token. Every execute requires the
 * live batch; no snapshot pointer is borrowed by the native owner. */
vibeqc_status vibeqc_ks_xc_response_create_v1(vibeqc_batch* batch,
                                              const vibeqc_ks_snapshot* snapshot,
                                              std::size_t tile_points, std::size_t budget_bytes,
                                              vibeqc_ks_xc_response** output);
vibeqc_status vibeqc_ks_xc_response_apply_v1(vibeqc_batch* batch, vibeqc_ks_xc_response* response,
                                             const double* direction, std::size_t count,
                                             double* output, std::size_t output_count);
/** Twelve uint64 values: device bytes, setup H2D, action H2D, D2H, syncs,
 * enqueues, spin blocks, AO count, grid points, preparation snapshot-export
 * D2H/reads/syncs. Enqueues count submitted actions, including later numerical
 * rejection; syncs also include the owner's explicit failure-cleanup fences. */
vibeqc_status vibeqc_ks_xc_response_diagnostic_v1(const vibeqc_ks_xc_response* response,
                                                  std::uint64_t* values, std::size_t count);
void vibeqc_ks_xc_response_destroy_v1(vibeqc_ks_xc_response* response);
/** Private CPU RKS directional potential bridge; rho/gradient use total density.
 * On failure, callers must discard the output buffer, including completed rows. */
vibeqc_status vibeqc_xc_rks_response_batch_v1(std::uint32_t pbe, const double* rho,
                                              const double* gradient, const double* delta_rho,
                                              const double* delta_gradient, std::size_t point_count,
                                              double* values, std::size_t value_count);
/** Spin-major rho[2,n] and gradient[2,n,3]; outputs point-major rho[2], gradient[2,3].
 * As for the restricted bridge, discard the complete output after any failure. */
vibeqc_status vibeqc_xc_uks_response_batch_v1(std::uint32_t pbe, const double* rho,
                                              const double* gradient, const double* delta_rho,
                                              const double* delta_gradient, std::size_t point_count,
                                              double* values, std::size_t value_count);
vibeqc_status vibeqc_ks_snapshot_ecp_derivatives_v1(vibeqc_batch* batch,
                                                    const vibeqc_ks_snapshot* snapshot,
                                                    double* values, std::size_t count);
}
