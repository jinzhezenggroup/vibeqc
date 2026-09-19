#pragma once

#include <cstddef>
#include <cstdint>

#include "vibeqc/vibeqc.h"

/** Private ctypes bridge for #163; deliberately absent from the installed API.
 * A snapshot owns its token and arrays, but borrows no calculation pointer.
 * Reads/checks require a live batch and compare its current #162 token. */
struct vibeqc_ks_snapshot;

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
vibeqc_status vibeqc_ks_snapshot_ecp_derivatives_v1(vibeqc_batch* batch,
                                                    const vibeqc_ks_snapshot* snapshot,
                                                    double* values, std::size_t count);
}
