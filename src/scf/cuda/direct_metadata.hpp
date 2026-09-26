#pragma once

#include <cstdint>
#include <limits>

#include "scf/cuda_batch.hpp"
#include "scf/generated_shell_task.hpp"

namespace vibeqc::scf::cuda_execution {

/** POD queue and diagnostic records shared by host planning and device consumers; these own no
 * allocations. */
using GeneratedShellTask = detail::GeneratedShellTask;
using GeneratedPppsResidentTask = detail::GeneratedPppsResidentTask;
using GeneratedShellPairStream = detail::GeneratedShellPairStream;

/** Geometry-dependent direct-J/K work emitted by shell-bound compaction. */
struct ActiveShellQuartetTile {
  std::uint32_t first_pair;
  std::uint32_t second_pair;
  std::uint32_t tile;
};

static_assert(sizeof(ActiveShellQuartetTile) == 3 * sizeof(std::uint32_t));

/**
 * First invalid descriptor found by the optional post-compaction validator.
 *
 * The validator writes one record with an atomic first-writer-wins protocol,
 * so it can run on the same stream as compaction without device printf or a
 * host synchronization in the captured graph.  ``error`` is initialized to
 * ``kDirectTileValidationNoError`` before each replay.
 */
struct DirectTileValidationRecord {
  std::uint32_t error;
  std::uint32_t angular_order;
  std::uint32_t slot;
  std::uint32_t tile;
  std::uint32_t first_pair;
  std::uint32_t second_pair;
  std::int32_t shell[4];
  std::uint32_t direct_nbf;
  std::uint32_t first_pair_count;
  std::uint32_t second_pair_count;
  std::uint32_t i;
  std::uint32_t j;
  std::uint32_t k;
  std::uint32_t l;
  std::uint32_t active_tile_count;
  std::uint32_t partition_capacity;
  std::uint32_t partition_begin;
};

static_assert(sizeof(DirectTileValidationRecord) == 20 * sizeof(std::uint32_t));
constexpr std::uint32_t kDirectTileValidationNoError = std::numeric_limits<std::uint32_t>::max();
enum class DirectTileValidationError : std::uint32_t {
  count_exceeds_capacity = 1,
  pair_out_of_bounds = 2,
  shell_out_of_bounds = 3,
  tile_out_of_bounds = 4,
  ao_range_invalid = 5,
};

/** One static resident-bra task over a compact contiguous ket-pair chunk. */
struct PsssResidentTask {
  std::uint32_t bra_pair;
  std::uint32_t ket_begin;
  std::uint32_t ket_count;
};

static_assert(sizeof(PsssResidentTask) == 3 * sizeof(std::uint32_t));

/** Optional final-density profiling counters; never touched in normal runs. */
struct DeviceShellClassProfileEntry {
  unsigned long long shell_quartets;
  unsigned long long tiles;
  unsigned long long ao_quartets;
  unsigned long long primitive_quartets;
};

static_assert(sizeof(DeviceShellClassProfileEntry) == sizeof(CudaRhfShellClassProfileEntry));

/** Raw spin-resolved density magnitudes for one direct-AO shell block. */
struct ShellPairDensityBounds {
  double coulomb;
  double exchange_alpha;
  double exchange_beta;
};

static_assert(sizeof(ShellPairDensityBounds) == 3 * sizeof(double));
static_assert(sizeof(ShellPairDensityBounds) == sizeof(detail::GeneratedShellPairDensityBounds));
static_assert(alignof(ShellPairDensityBounds) == alignof(detail::GeneratedShellPairDensityBounds));

/** Select the density gate applied while compacting direct shell quartets. */
enum class DirectScreeningPurpose : std::uint8_t {
  Fock,
  Force,
};

}  // namespace vibeqc::scf::cuda_execution
