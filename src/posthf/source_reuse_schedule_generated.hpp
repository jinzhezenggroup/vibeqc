// Generated from vibeqc_compiler.common.source_reuse; do not edit.
#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "posthf/capacity.hpp"
namespace vibeqc::posthf::generated {
struct UniformSourceReusePlan {
  bool shared_scan;
  std::size_t jobs_per_batch;
  std::size_t provider_requests;
};
inline UniformSourceReusePlan uniform_source_reuse_plan(
    std::size_t request_capacity, std::size_t requests_per_job,
    std::size_t total_jobs) {
  if (!requests_per_job || !total_jobs)
    throw std::invalid_argument("source-reuse schedule requires nonzero job dimensions");
  if (request_capacity < requests_per_job) return {false, 1, 1};
  const auto jobs = std::min(request_capacity / requests_per_job, total_jobs);
  return {true, jobs, posthf::checked_mul(requests_per_job, jobs)};
}
struct SourceReuseRequest {
  std::size_t live_bytes;
  std::size_t retained_bytes;
};
struct SourceReuseBatch {
  std::size_t begin;
  std::size_t end;
  std::size_t peak_bytes;
};
struct OrderedSourceReusePlan {
  std::vector<SourceReuseBatch> batches;
  std::size_t peak_bytes;
};
inline OrderedSourceReusePlan ordered_source_reuse_plan(
    std::size_t common_bytes, std::size_t fixed_live_bytes,
    std::size_t budget_bytes, const std::vector<SourceReuseRequest>& requests) {
  if (requests.empty())
    throw std::invalid_argument("source-reuse schedule requires at least one request");
  for (const auto& request : requests)
    if (request.live_bytes < request.retained_bytes)
      throw std::invalid_argument(
          "source-reuse request live bytes must include its retained output");
  OrderedSourceReusePlan result;
  std::size_t retained = fixed_live_bytes;
  result.peak_bytes = retained;
  std::size_t begin = 0;
  while (begin < requests.size()) {
    std::size_t batch_live = common_bytes;
    std::size_t end = begin;
    std::size_t batch_peak = 0;
    while (end < requests.size()) {
      const auto candidate_batch =
          posthf::checked_add(batch_live, requests[end].live_bytes);
      const auto candidate_peak =
          posthf::checked_add(retained, candidate_batch);
      if (candidate_peak > budget_bytes) break;
      batch_live = candidate_batch;
      batch_peak = candidate_peak;
      ++end;
    }
    if (end == begin)
      throw std::length_error("source-reuse request exceeds numeric memory budget");
    result.batches.push_back({begin, end, batch_peak});
    result.peak_bytes = std::max(result.peak_bytes, batch_peak);
    for (std::size_t index = begin; index < end; ++index)
      retained = posthf::checked_add(retained, requests[index].retained_bytes);
    if (retained > budget_bytes)
      throw std::length_error(
          "retained source-reuse outputs exceed numeric memory budget");
    result.peak_bytes = std::max(result.peak_bytes, retained);
    begin = end;
  }
  return result;
}

struct SourceTileCandidate {
  std::size_t axis_tile;
  std::size_t source_scans;
  std::size_t peak_bytes;
};
struct SourceTilePlan {
  std::size_t axis_tile;
  std::size_t source_reads;
  std::size_t source_scans;
  std::size_t peak_bytes;
};
inline std::size_t source_reads_per_scan(std::size_t nbf, std::size_t axis_tile) {
  if (!nbf || !axis_tile || axis_tile > nbf)
    throw std::invalid_argument("invalid source-tile dimensions");
  const auto tiles = 1 + (nbf - 1) / axis_tile;
  const auto squared = posthf::checked_mul(tiles, tiles);
  return posthf::checked_mul(squared, squared);
}
inline SourceTilePlan select_source_tile(
    std::size_t nbf, const std::vector<SourceTileCandidate>& candidates) {
  if (!nbf || candidates.empty())
    throw std::invalid_argument("source-tile selection requires AO count and candidates");
  SourceTilePlan best{};
  bool selected = false;
  for (const auto& candidate : candidates) {
    if (!candidate.axis_tile || !candidate.source_scans || candidate.axis_tile > nbf)
      throw std::invalid_argument("invalid source-tile candidate");
    const auto reads =
        posthf::checked_mul(source_reads_per_scan(nbf, candidate.axis_tile),
                            candidate.source_scans);
    const bool better =
        !selected || reads < best.source_reads ||
        (reads == best.source_reads && candidate.source_scans < best.source_scans) ||
        (reads == best.source_reads && candidate.source_scans == best.source_scans &&
         candidate.peak_bytes < best.peak_bytes) ||
        (reads == best.source_reads && candidate.source_scans == best.source_scans &&
         candidate.peak_bytes == best.peak_bytes && candidate.axis_tile > best.axis_tile);
    if (better) {
      best = {candidate.axis_tile, reads, candidate.source_scans, candidate.peak_bytes};
      selected = true;
    }
  }
  return best;
}
}  // namespace vibeqc::posthf::generated
