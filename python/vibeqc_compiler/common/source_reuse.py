"""Compiler-owned bounded source-reuse schedules for provider-produced blocks.

The scientific consumer decides which blocks it needs and in what deterministic
order. This module only decides how many requests may share one immutable source
traversal under an explicit numeric-memory budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from vibeqc_compiler.common.resources import MAX_BYTES, checked_bytes


def _add(a: int, b: int) -> int:
    value = a + b
    if value > MAX_BYTES:
        raise OverflowError("source-reuse schedule byte count overflow")
    return value


def _mul(a: int, b: int) -> int:
    value = a * b
    if value > MAX_BYTES:
        raise OverflowError("source-reuse schedule byte count overflow")
    return value


@dataclass(frozen=True)
class UniformSourceReusePlan:
    """Uniform jobs whose requests may share one source traversal."""

    shared_scan: bool
    jobs_per_batch: int
    provider_requests: int


def uniform_source_reuse_plan(
    request_capacity: int, requests_per_job: int, total_jobs: int
) -> UniformSourceReusePlan:
    """Maximize complete jobs in one scan without changing job/request order."""

    for name, value in (
        ("provider request capacity", request_capacity),
        ("requests per job", requests_per_job),
        ("total jobs", total_jobs),
    ):
        checked_bytes(value, name)
    if not requests_per_job or not total_jobs:
        raise ValueError("source-reuse schedule requires nonzero job dimensions")
    if request_capacity < requests_per_job:
        return UniformSourceReusePlan(False, 1, 1)
    jobs = min(request_capacity // requests_per_job, total_jobs)
    return UniformSourceReusePlan(True, jobs, _mul(requests_per_job, jobs))


@dataclass(frozen=True)
class SourceReuseRequest:
    """One request's concurrent provider bytes and retained output bytes."""

    live_bytes: int
    retained_bytes: int

    def __post_init__(self) -> None:
        checked_bytes(self.live_bytes, "source-reuse request live bytes")
        checked_bytes(self.retained_bytes, "source-reuse request retained bytes")
        if self.live_bytes < self.retained_bytes:
            raise ValueError(
                "source-reuse request live bytes must include its retained output"
            )


@dataclass(frozen=True)
class SourceReuseBatch:
    begin: int
    end: int
    peak_bytes: int


@dataclass(frozen=True)
class OrderedSourceReusePlan:
    batches: tuple[SourceReuseBatch, ...]
    peak_bytes: int

    @property
    def source_scans(self) -> int:
        return len(self.batches)


def ordered_source_reuse_plan(
    common_bytes: int,
    fixed_live_bytes: int,
    budget_bytes: int,
    requests: tuple[SourceReuseRequest, ...] | list[SourceReuseRequest],
) -> OrderedSourceReusePlan:
    """Greedily form maximal ordered batches under monotone live-byte costs.

    Earlier outputs stay retained while later requests execute. Since all byte
    costs are nonnegative and request order is fixed, taking the longest feasible
    prefix at every step minimizes the number of source traversals.
    """

    for name, value in (
        ("common provider bytes", common_bytes),
        ("fixed live bytes", fixed_live_bytes),
        ("numeric memory budget", budget_bytes),
    ):
        checked_bytes(value, name)
    requests = tuple(requests)
    if not requests:
        raise ValueError("source-reuse schedule requires at least one request")
    if not all(isinstance(request, SourceReuseRequest) for request in requests):
        raise TypeError("source-reuse requests must use SourceReuseRequest")

    retained = fixed_live_bytes
    peak = retained
    batches: list[SourceReuseBatch] = []
    begin = 0
    while begin < len(requests):
        batch_live = common_bytes
        end = begin
        batch_peak = 0
        while end < len(requests):
            candidate_batch = _add(batch_live, requests[end].live_bytes)
            candidate_peak = _add(retained, candidate_batch)
            if candidate_peak > budget_bytes:
                break
            batch_live = candidate_batch
            batch_peak = candidate_peak
            end += 1
        if end == begin:
            raise ValueError("source-reuse request exceeds numeric memory budget")
        batches.append(SourceReuseBatch(begin, end, batch_peak))
        peak = max(peak, batch_peak)
        for request in requests[begin:end]:
            retained = _add(retained, request.retained_bytes)
        if retained > budget_bytes:
            raise ValueError(
                "retained source-reuse outputs exceed numeric memory budget"
            )
        peak = max(peak, retained)
        begin = end

    return OrderedSourceReusePlan(tuple(batches), peak)


@dataclass(frozen=True)
class SourceTileCandidate:
    """One feasible AO source-tile choice and its complete scan count."""

    axis_tile: int
    source_scans: int
    peak_bytes: int

    def __post_init__(self) -> None:
        checked_bytes(self.axis_tile, "source axis tile")
        checked_bytes(self.source_scans, "source scan count")
        checked_bytes(self.peak_bytes, "source-tile peak bytes")
        if not self.axis_tile or not self.source_scans:
            raise ValueError(
                "source-tile candidate requires nonzero tile and scan count"
            )


@dataclass(frozen=True)
class SourceTilePlan:
    axis_tile: int
    source_reads: int
    source_scans: int
    peak_bytes: int


def source_reads_per_scan(nbf: int, axis_tile: int) -> int:
    """Return the exact rectangular AO-tile read count for one four-axis scan."""

    checked_bytes(nbf, "source AO count")
    checked_bytes(axis_tile, "source axis tile")
    if not nbf or not axis_tile or axis_tile > nbf:
        raise ValueError("invalid source-tile dimensions")
    tiles = 1 + (nbf - 1) // axis_tile
    squared = _mul(tiles, tiles)
    return _mul(squared, squared)


def select_source_tile(
    nbf: int, candidates: tuple[SourceTileCandidate, ...] | list[SourceTileCandidate]
) -> SourceTilePlan:
    """Minimize semantic source reads, then scans and admitted peak bytes."""

    checked_bytes(nbf, "source AO count")
    candidates = tuple(candidates)
    if not nbf or not candidates:
        raise ValueError("source-tile selection requires AO count and candidates")
    if not all(isinstance(candidate, SourceTileCandidate) for candidate in candidates):
        raise TypeError("source-tile candidates must use SourceTileCandidate")

    scored: list[tuple[tuple[int, int, int, int], SourceTilePlan]] = []
    for candidate in candidates:
        reads = _mul(
            source_reads_per_scan(nbf, candidate.axis_tile), candidate.source_scans
        )
        plan = SourceTilePlan(
            candidate.axis_tile, reads, candidate.source_scans, candidate.peak_bytes
        )
        scored.append(
            (
                (
                    reads,
                    candidate.source_scans,
                    candidate.peak_bytes,
                    -candidate.axis_tile,
                ),
                plan,
            )
        )
    return min(scored, key=lambda item: item[0])[1]


def native_header() -> str:
    """Emit the native runtime transcription of the generic schedule."""

    return r"""// Generated from vibeqc_compiler.common.source_reuse; do not edit.
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
"""
