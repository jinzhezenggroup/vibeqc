#pragma once

#include <mutex>

namespace vibeqc::runtime {
/** Serialize native provider construction/destruction while allocation deltas
 * are observed. Independent-stream numerical execution does not take this lock. */
inline std::mutex allocation_measurement_mutex;
}  // namespace vibeqc::runtime
