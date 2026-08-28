#include "scf/cuda_eigensolver_policy.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    using vibeqc::scf::XsyevBatchedEligibilityReason;
    using vibeqc::scf::xsyev_batched_api_eligibility;

    require(!xsyev_batched_api_eligibility(0, 0, 1).eligible,
            "XsyevBatched accepted a zero matrix dimension");
    require(xsyev_batched_api_eligibility(64, 63, 1).reason ==
                XsyevBatchedEligibilityReason::invalid_leading_dimension,
            "XsyevBatched accepted lda smaller than n");

    const auto largest_documented =
        xsyev_batched_api_eligibility(32768, 32768, 1);
    require(largest_documented.eligible &&
                largest_documented.matrix_batch_product == 1073741824ULL,
            "XsyevBatched rejected the documented n=32768 boundary");
    require(xsyev_batched_api_eligibility(32769, 32769, 1).reason ==
                XsyevBatchedEligibilityReason::documented_dimension_limit,
            "XsyevBatched treated n>32768 as API-eligible");

    const auto last_512_batch =
        xsyev_batched_api_eligibility(512, 512, 8191);
    require(last_512_batch.eligible &&
                last_512_batch.matrix_batch_product == 2147221504ULL,
            "checked product rejected the last eligible 512-AO batch");
    require(xsyev_batched_api_eligibility(512, 512, 8192).reason ==
                XsyevBatchedEligibilityReason::documented_product_limit,
            "checked product accepted a signature above INT32_MAX");

    // The same physical fleet can be eligible for RHF but not UHF because the
    // latter submits separate alpha and beta matrices to cuSOLVER.
    require(xsyev_batched_api_eligibility(1024, 1024, 1024).eligible,
            "RHF solver-batch accounting rejected an eligible fleet");
    require(xsyev_batched_api_eligibility(1024, 1024, 2048).reason ==
                XsyevBatchedEligibilityReason::documented_product_limit,
            "UHF solver-batch accounting ignored spin expansion");

    vibeqc::scf::XsyevBatchedGraphProbeResult rejected_capture;
    rejected_capture.ordinary_execution_passed = true;
    rejected_capture.graph_eligible = false;
    const auto fallback =
        vibeqc::scf::select_xsyev_batched_dispatch(rejected_capture);
    require(fallback.ordinary_stream_provider &&
                !fallback.device_launch_graph_provider,
            "capture rejection disabled valid ordinary-stream cuSOLVER");
    rejected_capture.graph_eligible = true;
    const auto qualified =
        vibeqc::scf::select_xsyev_batched_dispatch(rejected_capture);
    require(qualified.ordinary_stream_provider &&
                qualified.device_launch_graph_provider,
            "qualified provider was not selected for both execution modes");

#if VIBEQC_HAS_CUDA
    const auto device_probe =
        vibeqc::scf::probe_xsyev_batched_device_launch_graph(0, 512, 1);
    if (device_probe.failure_stage ==
            vibeqc::scf::XsyevBatchedGraphProbeStage::select_device ||
        device_probe.failure_stage ==
            vibeqc::scf::XsyevBatchedGraphProbeStage::device_identity) {
      std::cout << "CUDA XsyevBatched graph probe skipped: no allocated "
                   "device\n";
    } else {
      const std::array<std::uint64_t, 3> dimensions{512, 513, 768};
      for (const std::uint64_t dimension : dimensions) {
        const auto probe =
            dimension == 512
                ? device_probe
                : vibeqc::scf::probe_xsyev_batched_device_launch_graph(
                      0, dimension, 1);
        std::cout << "XsyevBatched n=" << dimension
                  << " ordinary="
                  << (probe.ordinary_execution_passed ? "pass" : "fail")
                  << " graph="
                  << (probe.graph_eligible ? "pass" : "fallback")
                  << " stage=" << static_cast<unsigned>(probe.failure_stage)
                  << " cuda=" << probe.cuda_error
                  << " cusolver=" << probe.cusolver_error
                  << " error=" << probe.maximum_eigenvalue_error << "/"
                  << probe.maximum_residual << "/"
                  << probe.maximum_orthogonality_error
                  << " workspace=" << probe.device_workspace_bytes << "/"
                  << probe.host_workspace_bytes << " bytes\n";
        require(probe.api.eligible,
                "documented XsyevBatched signature was not API-eligible");
        require(probe.n == dimension && probe.solver_batch == 1 &&
                    probe.device_id == 0 && probe.device_name.front() != '\0' &&
                    probe.compute_capability_major > 0 &&
                    probe.cuda_runtime_version >= 12090 &&
                    probe.cuda_driver_version > 0 &&
                    probe.cusolver_version > 0,
                "graph probe omitted exact stack/signature identity");
        require(probe.ordinary_execution_passed,
                "ordinary XsyevBatched execution above the old 512 cutoff "
                "failed");
        if (dimension == 512) {
          require(probe.graph_eligible,
                  "validated 512-AO device-launch provider path regressed");
        }
        if (probe.graph_eligible) {
          require(probe.graph_capture_passed &&
                      probe.host_graph_replay_passed &&
                      probe.device_tail_replay_passed,
                  "graph-eligible probe omitted a required replay check");
        } else {
          require(probe.failure_stage !=
                      vibeqc::scf::XsyevBatchedGraphProbeStage::none,
                  "rejected graph probe did not retain a failure stage");
        }
      }
    }
#endif

    std::cout << "validated XsyevBatched API eligibility boundaries\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
