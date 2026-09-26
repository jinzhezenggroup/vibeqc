#pragma once

#include "scf/cuda/df_shell_derivatives.cuh"

namespace vibeqc::scf {
/** Host-only launch arguments. No numerical implementation or policy is included. */
struct DfShellLaunch {
  const double* positions;
  std::size_t begin, count;
  const double* weights;
  double* gradient;
  unsigned long long* counters;
  cudaStream_t stream;
  unsigned variant;
  DfDerivativePairs pairs;
  DfShellDiagnostics* diagnostics;
  bool rys = false;
};

/** Every generated angular class provides the same three public launch forms. */
struct DfShellDispatch {
  cudaError_t (*panel)(DfShellBasisView, DfShellBasisView, const DfShellLaunch&);
  cudaError_t (*group)(DfShellBasisView, DfShellBasisView, DfShellBasisView, bool,
                       const DfShellLaunch&);
  cudaError_t (*packets)(std::span<const DfShellBasisView>, std::span<const DfShellBasisView>,
                         const DfShellLaunch&);
};
}  // namespace vibeqc::scf
