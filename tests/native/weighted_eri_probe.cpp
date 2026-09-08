/** Slurm-only probe of the exact native arbitrary-weight primitive boundary.
 *
 * Input records carry contracted/public-basis weights prepared independently
 * by the validation driver. Outputs are small weighted tile scalars/gradients;
 * no HF density or molecular four-index response tensor is passed to CUDA.
 */
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "scf/cuda_weighted_eri.hpp"

using namespace vibeqc::scf;

int main(int argc, char** argv) {
  try {
    if (argc != 5 || std::getenv("SLURM_JOB_ID") == nullptr) {
      throw std::runtime_error("usage inside Slurm: probe input output budget generated(0|1)");
    }
    const std::size_t budget = std::stoull(argv[3]);
    const unsigned generated = std::stoul(argv[4]);
    if (generated > 1U) throw std::runtime_error("generated must be zero or one");
    std::ifstream input(argv[1], std::ios::binary);
    char magic[8];
    std::uint64_t count{}, tiles{};
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    input.read(reinterpret_cast<char*>(&tiles), sizeof(tiles));
    if (!input || std::string(magic, 8) != "VQWE1441" || count > 1000000 || tiles > 100000) {
      throw std::runtime_error("invalid weighted fixture dimensions");
    }
    std::vector<CudaWeightedEriPrimitive> records(count);
    input.read(reinterpret_cast<char*>(records.data()), count * sizeof(records[0]));
    if (!input || input.peek() != std::char_traits<char>::eof()) {
      throw std::runtime_error("truncated or oversized weighted fixture");
    }
    std::vector<CudaWeightedEriResult> output;
    CudaWeightedEriDiagnostic diagnostic;
    std::string detail;
    // Exercise checked failure/empty boundaries without requiring another GPU
    // context or accepting partial output from an invalid primitive request.
    if (count) {
      auto invalid = records[0];
      invalid.output_tile = static_cast<std::uint32_t>(tiles);
      if (contract_cuda_weighted_eri_primitives(0, &invalid, 1, tiles, budget, generated, output,
                                                diagnostic,
                                                detail) != VIBEQC_STATUS_INVALID_ARGUMENT) {
        throw std::runtime_error("out-of-range output tile was accepted");
      }
      invalid = records[0];
      invalid.angular[0][0] = 4;
      if (contract_cuda_weighted_eri_primitives(0, &invalid, 1, tiles, budget, generated, output,
                                                diagnostic,
                                                detail) != VIBEQC_STATUS_INVALID_ARGUMENT) {
        throw std::runtime_error("unsupported angular component was accepted");
      }
      if (contract_cuda_weighted_eri_primitives(0, records.data(), count, tiles, 0, generated,
                                                output, diagnostic,
                                                detail) != VIBEQC_STATUS_INVALID_ARGUMENT) {
        throw std::runtime_error("insufficient numeric budget was accepted");
      }
    }
    if (contract_cuda_weighted_eri_primitives(0, nullptr, 0, tiles, budget, generated, output,
                                              diagnostic, detail) != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error("valid empty primitive stream was rejected: " + detail);
    }
    for (const auto& result : output) {
      if (result.value != 0.0) throw std::runtime_error("nonzero empty result");
      for (const auto& center : result.center) {
        for (double value : center) {
          if (value != 0.0) throw std::runtime_error("nonzero empty gradient");
        }
      }
    }
    const auto begin = std::chrono::steady_clock::now();
    if (contract_cuda_weighted_eri_primitives(0, records.data(), count, tiles, budget, generated,
                                              output, diagnostic,
                                              detail) != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(detail);
    }
    const double milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (diagnostic.host_peak_bytes + diagnostic.device_peak_bytes > budget) {
      throw std::runtime_error("reported host/device numeric storage exceeds budget");
    }
    std::ofstream stream(argv[2], std::ios::binary);
    stream.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(output[0]));
    if (!stream) throw std::runtime_error("could not write complete weighted output");
    std::cout << std::setprecision(12) << "{\"records\":" << count << ",\"tiles\":" << tiles
              << ",\"milliseconds\":" << milliseconds
              << ",\"host_peak_bytes\":" << diagnostic.host_peak_bytes
              << ",\"device_peak_bytes\":" << diagnostic.device_peak_bytes
              << ",\"primitive_capacity\":" << diagnostic.primitive_capacity
              << ",\"generated_records\":" << diagnostic.generated_records
              << ",\"reference_records\":" << diagnostic.reference_records
              << ",\"slurm_job_id\":" << std::quoted(std::getenv("SLURM_JOB_ID")) << "}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
