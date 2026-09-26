#include "dft/cuda_ks.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "dft/cuda_ks_kernels.hpp"
#include "dft/cuda_xc.hpp"
#include "dft/xc.hpp"
#include "runtime/compiled_execution_region.hpp"
#include "runtime/host_component_trace.hpp"
#include "runtime/resource_cuda.cuh"
#include "runtime/solver_region_cuda.cuh"
#include "scf/cuda/eigensolver.hpp"
#include "scf/cuda/mean_field_setup.hpp"
#include "scf/cuda/scf_constants.hpp"
#include "scf/cuda/scf_density_kernels.hpp"
#include "scf/cuda/scf_diis_kernels.hpp"
#include "scf/cuda/scf_matrix_kernels.hpp"
#include "scf/cuda_density_fitting_device.hpp"
#include "scf/cuda_fock_execution.hpp"
#include "scf/eigensolver_workspace.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/proposal_control.hpp"
#include "vibeqc/vibeqc.hpp"
#include "xc_cpu_generated.hpp"

#if defined(VIBEQC_TEST_HOOKS)
namespace {
// One-shot injection uses the real status mapper without poisoning the CUDA
// context, allowing the public API to verify explicit recovery and seed reuse.
thread_local bool fail_next_ks_runtime = false;
}  // namespace
extern "C" void ks_cuda_fail_next_runtime_for_test_v1() { fail_next_ks_runtime = true; }
#endif

namespace vibeqc::dft {
namespace {
using namespace scf::cuda_execution;
constexpr unsigned kMaximumFinalCorrections = 4;
constexpr unsigned kCudaKsChunkCapacity = 2;
void check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw vibeqc::Error(VIBEQC_STATUS_CUDA_ERROR, cudaGetErrorString(status));
}
void check(vibeqc_status status, const std::string& detail) {
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status == VIBEQC_STATUS_CUDA_ERROR) throw vibeqc::Error(status, detail);
  if (status == VIBEQC_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(detail);
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
template <class Function>
void run_resident_nonlocal_cuda(Function function) {
  try {
    function();
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::invalid_argument&) {
    throw;
  } catch (const std::overflow_error&) {
    throw;
  } catch (const vibeqc::Error&) {
    throw;
  } catch (const std::runtime_error& error) {
    // The resident VV10 seam uses runtime::cuda_resource_check internally.
    // Translate its untyped CUDA runtime failures at the KS owner boundary;
    // allocation failures already arrive as std::bad_alloc.
    throw vibeqc::Error(VIBEQC_STATUS_CUDA_ERROR, error.what());
  }
}
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("CUDA KS storage overflow");
  return a * b;
}
std::size_t sum(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::overflow_error("CUDA KS storage overflow");
  return a + b;
}
/** Numeric arena view shared by allocation and metadata-only planning. */
struct KsStateStorage {
  double *hcore{}, *overlap{}, *x{}, *j{}, *exchange{}, *range_exchange{}, *density{}, *proposal{},
      *warm{}, *warm_orbitals{}, *fock{}, *residual{}, *tmp1{}, *tmp2{}, *effective{},
      *fock_history{}, *residual_history{}, *gram{}, *weights{}, *eigenvalues{},
      *final_coefficients{}, *final_eigenvalues{}, *cold_seed{};
  std::int32_t* occupied{};
  std::uint8_t *enabled{}, *spin_enabled{};
  std::uint32_t *history_count{}, *history_head{};
  int *solver_info{}, *final_solver_info{}, *jk_error{}, *range_jk_error{}, *staged_xc_error{};
  double* staged_xc_totals{};
  std::uint8_t *final_spin_enabled{}, *final_enabled{};
  cuda_ks_detail::Control* control{};
  cuda_ks_detail::Scalars* scalar_records{};
  /** The dry run and actual partition share one checked, typed layout. All
   * persistent and phase-local numeric buffers are explicitly charged. */
  std::size_t partition(std::size_t n, unsigned spins, unsigned history, bool exact_exchange,
                        bool range_correction, void* storage) {
    const auto matrix = product(n, n), elements = product(spins, matrix);
    std::size_t bytes = 0;
    const auto reserve = [&](auto*& pointer, std::size_t count) {
      using T = std::remove_pointer_t<std::remove_reference_t<decltype(pointer)>>;
      const auto remainder = bytes % alignof(T);
      if (remainder) bytes = sum(bytes, alignof(T) - remainder);
      pointer = storage ? reinterpret_cast<T*>(static_cast<char*>(storage) + bytes) : nullptr;
      bytes = sum(bytes, product(count, sizeof(T)));
    };
    for (auto** pointer : {&hcore, &overlap, &x, &j}) reserve(*pointer, matrix);
    if (exact_exchange)
      reserve(exchange, elements);
    else
      exchange = nullptr;
    if (range_correction)
      reserve(range_exchange, elements);
    else
      range_exchange = nullptr;
    for (auto** pointer :
         {&density, &proposal, &warm, &warm_orbitals, &fock, &residual, &tmp1, &tmp2, &effective})
      reserve(*pointer, elements);
    // The generated cold guess stays resident across cold retries. It cannot
    // alias proposal/warm storage, which changes during every SCF trajectory.
    reserve(cold_seed, elements);
    reserve(fock_history, product(history, elements));
    reserve(residual_history, product(history, elements));
    reserve(gram, product(history + 1, history + 1));
    reserve(weights, history + 1);
    reserve(eigenvalues, product(spins, n));
    reserve(final_coefficients, elements);
    reserve(final_eigenvalues, product(spins, n));
    reserve(occupied, spins);
    reserve(enabled, 1);
    reserve(spin_enabled, spins);
    reserve(history_count, 1);
    reserve(history_head, 1);
    reserve(solver_info, spins);
    reserve(final_solver_info, spins);
    reserve(jk_error, 1);
    if (range_correction)
      reserve(range_jk_error, 1);
    else
      range_jk_error = nullptr;
    reserve(staged_xc_totals, 3);
    reserve(staged_xc_error, 1);
    reserve(final_spin_enabled, spins);
    reserve(final_enabled, 1);
    reserve(control, 1);
    reserve(scalar_records, kCudaKsChunkCapacity);
    return bytes;
  }
};

std::uint64_t next_ks_owner() noexcept {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  while (value && value != std::numeric_limits<std::uint64_t>::max()) {
    if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
  }
  return 0;
}
}  // namespace

std::size_t cuda_ks_state_bytes(std::size_t n, unsigned spins, unsigned history,
                                bool exact_exchange, bool range_correction) {
  if (!n || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (spins != 1 && spins != 2) || history > 64)
    throw std::invalid_argument("invalid CUDA KS resource shape");
  KsStateStorage layout;
  return sum(
      layout.partition(n, spins, std::max(1U, history), exact_exchange, range_correction, nullptr),
      n <= kSmallEigensolverLimit ? 0 : scf::ordinary_eigensolver_workspace_allowance(n));
}

struct CudaKsPlan::Impl : KsStateStorage {
  const scf::PreparedFockPlan& provider;
  const AoBasis& basis;
  const MolecularGrid& grid;
  scf::ScfOptions options;
  scf::PreparedCudaFockBinding fock_binding{};
  scf::CudaDensityFittingJkPlan* fitted{};
  cudaStream_t stream{};
  int device{};
  std::size_t n{}, matrix{}, elements{};
  unsigned spins{}, history{};
  std::array<std::size_t, 2> occupations{};
  std::vector<double> host_xc_density, host_xc_alpha, host_xc_beta, host_xc_potential;
  // Async H2D copies retain these controls through the existing stream drain.
  std::array<double, 3> host_xc_totals{};
  int host_xc_error{};
  std::array<std::int32_t, 2> host_spin_counts{};
  std::array<std::uint8_t, 2> host_selected{}, host_all_spins{1, 1};
  std::uint8_t host_one{1};
  GridSpec grid_spec;
  CudaXcLayout xc_layout;
  CudaKsResources resource;
  CudaKsTransfers movement;
  void *arena{}, *xc_arena{}, *nonlocal_arena{};
  std::size_t ks_arena_bytes{}, nonlocal_arena_bytes{};
  double *nonlocal_raw_density{}, *nonlocal_raw_gradient{}, *nonlocal_effective_weights{},
      *nonlocal_effective_density{}, *nonlocal_effective_gradient{}, *nonlocal_vrho{},
      *nonlocal_vsigma{}, *nonlocal_workspace{};
  int *nonlocal_domain_error{}, *nonlocal_pair_error{};
  nlc::Vv10CudaDeviceLayout nonlocal_layout{};
  std::unique_ptr<CudaXcPlan> xc;
  std::unique_ptr<OrdinaryStreamEigensolver> eigensolver;
  scf::ScfResult output;
  bool is_active{}, is_pending{}, is_failed{}, warm_ready{}, warm_orbitals_ready{}, started{};
  bool warm_updates{true}, device_chunk_mode{};
  bool stabilize_occupations{}, final_closure{}, has_exchange{}, has_range_correction{};
  bool mixed_j{}, strict_refinement{}, pending_mixed_j{}, mixed_j_executed{}, device_nonlocal{};
  double exchange_coefficient{}, range_exchange_coefficient{};
  std::optional<scf::ResolvedFockBuild> range_correction;
  nlc::Vv10Plan* nonlocal_correlation{};
  nlc::Vv10DensityDomain nonlocal_domain{nlc::Vv10DensityDomain::StrictPositive};
  unsigned final_corrections{}, refinement_iterations{};
  SemilocalFamily functional{SemilocalFamily::Lda};
  bool final_state_ready{}, final_frame_ready{};
  std::uint64_t owner{next_ks_owner()}, solve_epoch{}, generation{}, final_generation{};
  double previous_energy{std::numeric_limits<double>::infinity()};
  unsigned pending_iterations{};
  std::array<std::uint64_t, kCudaKsChunkCapacity> pending_generations{};
  runtime::SolverRegionCudaExecutor solver_region_executor;
  runtime::CompiledExecutionRegion device_chunk_region;

  runtime::CompiledExecutionBinding device_chunk_binding() const {
    return {"cuda-ks-device-chunk-v1:" + std::to_string(n) + ":" + std::to_string(spins) + ":" +
                std::to_string(semilocal_family_code(functional)) + ":" + std::to_string(history) +
                ":" + std::to_string(xc_layout.tile_points),
            // The prepared facade owns provider lifetime and replay identity;
            // device chunks are admitted only for its direct-Fock binding.
            device, stream, arena, fock_binding.source_identity};
  }

  void invalidate_warm_orbitals() noexcept {
    if (warm_orbitals_ready) ++movement.warm_orbital_frame_invalidations;
    warm_orbitals_ready = false;
  }

  void clear_warm_state() noexcept {
    warm_ready = false;
    invalidate_warm_orbitals();
  }

  void current_device() const {
    // Prepared owners select their bound device on every entry, as the common
    // Fock provider does. Another context may have changed this thread's device
    // between calls; borrowed XC views still enforce their own device identity.
    check(cudaSetDevice(device));
  }

  std::size_t partition_nonlocal(void* storage) {
    if (!device_nonlocal) {
      nonlocal_raw_density = nonlocal_raw_gradient = nonlocal_effective_weights =
          nonlocal_effective_density = nonlocal_effective_gradient = nonlocal_vrho =
              nonlocal_vsigma = nonlocal_workspace = nullptr;
      nonlocal_domain_error = nonlocal_pair_error = nullptr;
      return 0;
    }
    if (nonlocal_layout.workspace_bytes % sizeof(double))
      throw std::logic_error("resident VV10 workspace is not double-aligned");
    const auto points = xc_layout.npoint;
    const auto workspace_doubles = nonlocal_layout.workspace_bytes / sizeof(double);
    const auto doubles = sum(product(11, points), sum(workspace_doubles, std::size_t{2}));
    if (!storage) return product(doubles, sizeof(double));
    auto* cursor = static_cast<double*>(storage);
    auto take = [&](std::size_t count) {
      auto* out = cursor;
      cursor += count;
      return out;
    };
    nonlocal_raw_density = take(points);
    nonlocal_raw_gradient = take(product(3, points));
    nonlocal_effective_weights = take(points);
    nonlocal_effective_density = take(points);
    nonlocal_effective_gradient = take(product(3, points));
    nonlocal_vrho = take(points);
    nonlocal_vsigma = take(points);
    nonlocal_workspace = take(workspace_doubles);
    nonlocal_domain_error = reinterpret_cast<int*>(take(1));
    nonlocal_pair_error = reinterpret_cast<int*>(take(1));
    if (cursor != static_cast<double*>(storage) + doubles)
      throw std::logic_error("resident VV10 KS arena partition mismatch");
    return product(doubles, sizeof(double));
  }

  std::vector<double> seed(const std::vector<double>* input) const {
    using namespace scf::reference;
    if (!input) throw std::logic_error("CUDA cold guesses must use resident setup state");
    const auto& ints = provider.one_electron();
    if (options.strict_initial_density && input) {
      const std::vector<unsigned> counts =
          spins == 2 ? std::vector<unsigned>{static_cast<unsigned>(occupations[0]),
                                             static_cast<unsigned>(occupations[1])}
                     : std::vector<unsigned>{static_cast<unsigned>(occupations[0])};
      scf::solver::validate_seed(ints.overlap, *input, n, counts, spins == 2 ? 1.0 : 2.0);
      return *input;
    }
    if (spins == 2) {
      const auto pair = scf::initial_guess::normalized_warm_uhf_density(ints, occupations[0],
                                                                        occupations[1], *input);
      return concatenate(pair.first, pair.second);
    }
    return scf::initial_guess::normalized_warm_density(provider.system(), ints, *input);
  }

  /** Construct X and the historical core guess with the ordinary GPU solver.
   * All working matrices borrow the existing arena before iteration begins;
   * only the immutable cold density adds retained storage. Host warm-input
   * normalization remains an explicit input-boundary operation, never a
   * reference eigen fallback. */
  void prepare_initial_state() {
    runtime::host_trace::Region trace("cuda_ks_initial_state", n);
    const auto matrix_blocks = (matrix + 127) / 128;
    if (matrix_blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("CUDA KS setup matrix launch exceeds the device grid domain");
    const auto blocks = static_cast<unsigned>(matrix_blocks);
    const auto multiply = [&](const double* a, bool transpose, const double* b, double* c) {
      launch_matrix_product_kernel(blocks, 128, 0, stream, 1, n, a, transpose, b, final_enabled, c,
                                   1.0);
      check(cudaGetLastError());
    };
    const auto solve = [&] {
      check(eigensolver->launch(1, tmp2, effective, eigenvalues, solver_info, final_enabled),
            "CUDA KS setup eigensolver launch failed");
    };
    const auto read_status = [&](bool metric) {
      // Stack-backed downloads are drained before any error escapes this scope.
      std::array<int, 2> status{};
      try {
        check(
            cudaMemcpyAsync(&status[0], solver_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
        if (metric)
          check(cudaMemcpyAsync(&status[1], jk_error, sizeof(int), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
      } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
      }
      movement.scalar_d2h_bytes += sizeof(int) * (metric ? 2 : 1);
      ++movement.synchronizations;
      if (status[0]) throw std::runtime_error("CUDA KS setup eigensolver did not converge");
      if (status[1])
        throw std::runtime_error("overlap matrix is singular or failed its metric identity check");
    };
    check(
        cudaMemcpyAsync(tmp2, overlap, matrix * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    solve();
    check(cudaMemsetAsync(jk_error, 0, sizeof(int), stream));
    form_overlap_weights(stream, n, eigenvalues, jk_error);
    check(cudaGetLastError());
    form_weighted_projector(stream, n, 1, tmp2, eigenvalues, x);
    check(cudaGetLastError());
    multiply(overlap, false, x, tmp1);
    multiply(x, true, tmp1, residual);
    check_overlap_metric(stream, n, residual, jk_error);
    check(cudaGetLastError());
    read_status(true);  // Reject singular S before attempting a core solve.

    multiply(hcore, false, x, tmp1);
    multiply(x, true, tmp1, tmp2);
    solve();
    multiply(x, false, tmp2, tmp1);
    if (spins == 2) {
      check(cudaMemcpyAsync(tmp1 + matrix, tmp1, matrix * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream));
      // Reuse the common UHF frontier rotation, including beta=0/equal-spin
      // branches; the scientific seed policy is identical to HF and CPU KS.
      launch_mix_open_shell_guess_kernel(static_cast<unsigned>((n + 127) / 128), 128, 0, stream, 1,
                                         n, occupied, final_enabled, tmp1);
      check(cudaGetLastError());
    }
    form_occupation_weights(stream, n, spins, occupations[0], occupations[1], eigenvalues);
    check(cudaGetLastError());
    form_weighted_projector(stream, n, spins, tmp1, eigenvalues, cold_seed);
    check(cudaGetLastError());
    read_status(false);
  }

  Impl(const scf::PreparedFockPlan& plan, const AoBasis& basis, const MolecularGrid& grid,
       const scf::ScfOptions& control, SemilocalFamily functional, std::size_t tile,
       const scf::ResolvedFockBuild* range, nlc::Vv10Plan* nonlocal, nlc::Vv10DensityDomain domain)
      : provider(plan),
        basis(basis),
        grid(grid),
        options(control),
        grid_spec(grid.spec()),
        functional(functional),
        range_correction(range ? std::optional<scf::ResolvedFockBuild>(*range) : std::nullopt),
        nonlocal_correlation(nonlocal),
        nonlocal_domain(domain) {
    if (!semilocal_family_has_cuda_ks(functional))
      throw std::invalid_argument(
          "CUDA KS semilocal family has no qualified device implementation");
    const auto& strategy = provider.strategy();
    scf::validate_resolved_fock_build(strategy);
    has_exchange = strategy.spec.exchange.present;
    exchange_coefficient = has_exchange ? strategy.spec.exchange.coefficient : 0.0;
    has_range_correction = range_correction.has_value();
    if (has_range_correction) {
      scf::validate_resolved_fock_build(*range_correction);
      range_exchange_coefficient = range_correction->spec.exchange.coefficient;
    }
    fock_binding = scf::prepared_cuda_fock_binding(provider);
    fitted = provider.cuda_fitted_source();
    if (!owner || strategy.backend != scf::FockBackend::Cuda ||
        strategy.spec.derivative_order != 0 || !strategy.spec.coulomb.present ||
        strategy.spec.coulomb.coefficient != 1.0 ||
        (strategy.spec.coulomb.approximation != scf::FockApproximation::Exact &&
         strategy.spec.coulomb.approximation != scf::FockApproximation::DensityFitted) ||
        (has_exchange && (strategy.spec.exchange.approximation != scf::FockApproximation::Exact ||
                          strategy.spec.exchange.op != scf::FockOperator::FullRange)) ||
        (has_exchange && fitted) || (!fock_binding && !fitted) || (fock_binding && fitted))
      throw std::invalid_argument(
          "CUDA KS requires prepared Coulomb and optional exact full-range exchange");
    if (has_range_correction) {
      const auto& correction = *range_correction;
      const auto& spec = correction.spec;
      if (fitted || !fock_binding || correction.backend != scf::FockBackend::Cuda ||
          spec.spin != strategy.spec.spin || spec.derivative_order != 0 || spec.coulomb.present ||
          !spec.exchange.present || spec.exchange.approximation != scf::FockApproximation::Exact ||
          spec.exchange.op != scf::FockOperator::LongRange || spec.exchange.omega <= 0.0 ||
          correction.screening_tolerance != strategy.screening_tolerance)
        throw std::invalid_argument(
            "CUDA KS range correction must be one compatible exact long-range exchange term");
    }
    if (options.compute_forces || options.hooks || options.export_physical_reference ||
        options.xc_density_route != XcDensityRoute::DensityMatrix ||
        (options.precision_mode && *options.precision_mode != VIBEQC_PRECISION_FP64 &&
         *options.precision_mode != VIBEQC_PRECISION_AUTO))
      throw std::invalid_argument("CUDA KS received an unsupported execution policy");
    mixed_j = options.precision_mode && *options.precision_mode == VIBEQC_PRECISION_AUTO;
    if (mixed_j && fitted) throw std::invalid_argument("CUDA fitted KS requires strict FP64");
    if (mixed_j && (has_exchange || has_range_correction))
      throw std::invalid_argument("CUDA exact-exchange KS currently requires strict FP64");
    if (mixed_j && (functional == SemilocalFamily::R2scan || functional == SemilocalFamily::Wb97mv))
      throw std::invalid_argument("meta-GGA CUDA KS currently requires strict FP64");
    if (nonlocal_correlation) {
      if (functional != SemilocalFamily::Pbe && functional != SemilocalFamily::Wb97mv)
        throw std::invalid_argument("CUDA KS nonlocal composition requires a PBE-family graph");
      if (mixed_j)
        throw std::invalid_argument("CUDA KS nonlocal composition currently requires strict FP64");
      device_nonlocal =
          options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused;
      if (device_nonlocal &&
          (functional != SemilocalFamily::Wb97mv ||
           nonlocal_domain != nlc::Vv10DensityDomain::MolecularV1 ||
           nonlocal_correlation->parameters().variant != nlc::Vv10Variant::vv10 || fitted))
        throw std::invalid_argument(
            "device-resident CUDA nonlocal KS is qualified only for WB97M-V MolecularV1");
      if (!device_nonlocal &&
          options.xc_execution_schedule != scf::ScfOptions::XcExecutionSchedule::HostUnfused)
        throw std::invalid_argument("unknown CUDA KS nonlocal execution schedule");
    }
    if (!options.max_iterations || !std::isfinite(options.energy_tolerance) ||
        !std::isfinite(options.density_tolerance) || options.energy_tolerance <= 0.0 ||
        options.density_tolerance <= 0.0 || options.diis_history > 64)
      throw std::invalid_argument("invalid CUDA KS convergence, DIIS or precision controls");
    if (!provider.matches_system(grid.system()) ||
        basis.packed != AoBasis(provider.system()).packed)
      throw std::invalid_argument(
          "CUDA KS refuses a stale geometry, basis, charge or spin binding");
    n = provider.one_electron().nbf;
    if (!n || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) || basis.nao != n)
      throw std::invalid_argument("invalid CUDA KS AO dimension");
    matrix = product(n, n);
    spins = strategy.spec.spin == scf::FockSpin::Unrestricted ? 2 : 1;
    elements = product(spins, matrix);
    const auto counts = scf::initial_guess::spin_occupations(provider.system());
    occupations = {counts.first, counts.second};
    if (!provider.system().electron_count || occupations[0] > n || occupations[1] > n ||
        (spins == 1 && (occupations[0] != occupations[1] || provider.system().multiplicity != 1)))
      throw std::invalid_argument("CUDA KS occupations do not match the spin/orbital space");
    // A valid overlap does not imply finite physical data: unlike two equal
    // H centers, coincident O/H centers can retain a nonsingular AO metric
    // while nuclear repulsion is infinite. Fail before staging a cold seed.
    const auto& integrals = provider.one_electron();
    const auto finite = [](double value) { return std::isfinite(value); };
    if (!finite(integrals.nuclear_repulsion) ||
        !std::all_of(integrals.overlap.begin(), integrals.overlap.end(), finite) ||
        !std::all_of(integrals.hcore.begin(), integrals.hcore.end(), finite))
      throw std::runtime_error("nonfinite CUDA KS one-electron or nuclear energy");
    history = std::max(1U, options.diis_history);
    device = fitted ? scf::cuda_density_fitting_device(fitted) : fock_binding.device_id;
    stream = fitted ? scf::cuda_density_fitting_stream(fitted) : fock_binding.stream;
    if (nonlocal_correlation &&
        (nonlocal_correlation->backend() != VIBEQC_BACKEND_CUDA ||
         nonlocal_correlation->device_id() != device ||
         nonlocal_correlation->resources().point_count != grid.point_count()))
      throw std::invalid_argument("CUDA KS nonlocal owner is incompatible with the grid or device");
    current_device();
    xc_layout = cuda_xc_layout(basis, grid, semilocal_family_code(functional), spins == 2, tile,
                               CudaXcAoPrecision::Fp64, options.semilocal_exchange_scale,
                               options.semilocal_correlation_scale);
    const bool host_unfused =
        options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::HostUnfused;
    if (host_unfused &&
        (options.semilocal_exchange_scale != 1.0 || options.semilocal_correlation_scale != 1.0))
      throw std::invalid_argument("scaled CUDA XC requires device-fused execution");
    if (host_unfused) {
      host_xc_density.resize(elements);
      host_xc_potential.resize(elements);
      if (spins == 2) {
        host_xc_alpha.resize(matrix);
        host_xc_beta.resize(matrix);
      }
    }
    if (device_nonlocal) {
      nonlocal_layout =
          nlc::vv10_cuda_device_layout(xc_layout.npoint, xc_layout.tile_points, true, false);
      nonlocal_arena_bytes = partition_nonlocal(nullptr);
      if (nonlocal_arena_bytes > nonlocal_correlation->resources().device_workspace_bytes)
        throw std::invalid_argument(
            "resident CUDA KS nonlocal workspace exceeds the prepared VV10 device bound");
    }
    ks_arena_bytes = partition(n, spins, history, has_exchange, has_range_correction, nullptr);
    resource.state_device_bytes = sum(ks_arena_bytes, nonlocal_arena_bytes);
    resource.xc_device_bytes = host_unfused ? 0 : xc_layout.device_bytes;
    resource.provider_device_bytes = provider.diagnostic().device_bytes;
    const auto diagnostic_iterations =
        mixed_j ? sum(product(options.max_iterations, 2U), kMaximumFinalCorrections)
                : options.max_iterations;
    output.dft_diagnostic.history.reserve(diagnostic_iterations);
    resource.retained_host_numeric_bytes =
        (host_xc_density.capacity() + host_xc_alpha.capacity() + host_xc_beta.capacity() +
         host_xc_potential.capacity()) *
            sizeof(double) +
        output.dft_diagnostic.history.capacity() * sizeof(ScfIteration) + sizeof(host_xc_totals) +
        sizeof(host_xc_error) + sizeof(host_spin_counts) + sizeof(host_selected) +
        sizeof(host_all_spins) + sizeof(host_one);
    try {
      check(runtime::resource_cuda_malloc(&arena, ks_arena_bytes));
      partition(n, spins, history, has_exchange, has_range_correction, arena);
      if (resource.xc_device_bytes)
        check(runtime::resource_cuda_malloc(&xc_arena, resource.xc_device_bytes));
      if (nonlocal_arena_bytes) {
        check(runtime::resource_cuda_malloc(&nonlocal_arena, nonlocal_arena_bytes));
        partition_nonlocal(nonlocal_arena);
      }
      check(cudaMemsetAsync(arena, 0, ks_arena_bytes, stream));
      if (nonlocal_arena) check(cudaMemsetAsync(nonlocal_arena, 0, nonlocal_arena_bytes, stream));
      const auto upload = [&](void* destination, const void* source, std::size_t bytes) {
        check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
        movement.setup_h2d_bytes += bytes;
      };
      upload(hcore, provider.one_electron().hcore.data(), matrix * sizeof(double));
      upload(overlap, provider.one_electron().overlap.data(), matrix * sizeof(double));
      host_spin_counts = {static_cast<std::int32_t>(occupations[0]),
                          static_cast<std::int32_t>(occupations[1])};
      host_selected = {static_cast<std::uint8_t>(occupations[0] > 0),
                       static_cast<std::uint8_t>(occupations[1] > 0)};
      upload(occupied, host_spin_counts.data(), spins * sizeof(std::int32_t));
      upload(spin_enabled, host_selected.data(), spins * sizeof(std::uint8_t));
      upload(final_spin_enabled, host_all_spins.data(), spins * sizeof(std::uint8_t));
      upload(final_enabled, &host_one, sizeof(host_one));
      if (!host_unfused) {
        // Device-fused XC setup drains this same stream.
        xc = std::make_unique<CudaXcPlan>(
            basis, grid, semilocal_family_code(functional), spins == 2, tile, xc_arena,
            resource.xc_device_bytes, stream, CudaXcAoPrecision::Fp64,
            options.semilocal_exchange_scale, options.semilocal_correlation_scale);
      }
      // This owner uses ordinary stream execution. Reuse the common provider
      // instead of forcing the graph-safe maximum-pivot fallback at every size.
      eigensolver = std::make_unique<OrdinaryStreamEigensolver>(stream, n, tmp2, eigenvalues);
      resource.state_device_bytes = sum(resource.state_device_bytes, eigensolver->device_bytes());
      resource.retained_host_numeric_bytes =
          sum(resource.retained_host_numeric_bytes, eigensolver->host_bytes());
      prepare_initial_state();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  void cleanup() noexcept {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream);
    xc.reset();
    eigensolver.reset();
    if (nonlocal_arena) runtime::resource_cuda_free(nonlocal_arena);
    if (xc_arena) runtime::resource_cuda_free(xc_arena);
    if (arena) runtime::resource_cuda_free(arena);
    nonlocal_arena = xc_arena = arena = nullptr;
    cudaSetDevice(previous);
  }
  ~Impl() { cleanup(); }

  void begin(const std::vector<double>* input, bool reuse_warm) {
    if (is_pending) throw std::logic_error("cannot replace a pending CUDA KS iteration");
    final_state_ready = final_frame_ready = false;
    final_generation = 0;
    // #991's first KS slice is deliberately intra-trajectory only. A changed
    // geometry may reuse the last-good density, but its previous orthonormal
    // orbital frame is not projected across metrics until that route is
    // independently qualified.
    invalidate_warm_orbitals();
    if (solve_epoch == std::numeric_limits<std::uint64_t>::max()) {
      is_active = false;
      is_failed = true;
      throw std::overflow_error("CUDA KS solve epoch exhausted");
    }
    ++solve_epoch;
    is_active = false;
    is_failed = true;
    current_device();
#if defined(VIBEQC_TEST_HOOKS)
    if (fail_next_ks_runtime) {
      fail_next_ks_runtime = false;
      check(cudaErrorUnknown);
    }
#endif
    const bool use_warm = !input && reuse_warm && warm_ready;
    std::vector<double> prepared;
    if (input) prepared = seed(input);  // Validate before replacing current state.
    auto retained_history = std::move(output.dft_diagnostic.history);
    retained_history.clear();
    output = {};
    output.dft_diagnostic.history = std::move(retained_history);
    output.dft_diagnostic.occupations = occupations;
    output.dft_diagnostic.grid_points = xc_layout.npoint;
    output.dft_diagnostic.tile_points = xc_layout.tile_points;
    output.dft_diagnostic.ao_order = functional == SemilocalFamily::Lda ? 0 : 1;
    output.dft_diagnostic.scf_domain_version = semilocal_family_domain_version(functional);
    output.initial_density_used = input != nullptr || use_warm;
    is_active = false;
    started = true;
    is_failed = false;
    stabilize_occupations = false;
    final_closure = false;
    strict_refinement = false;
    pending_mixed_j = false;
    mixed_j_executed = false;
    final_corrections = 0;
    refinement_iterations = 0;
    output.precision.requested_mode = options.precision_mode.value_or(VIBEQC_PRECISION_FP64);
    pending_iterations = 0;
    // The bounded device-control prototype is qualified only for strict-FP64
    // direct all-electron RKS. AUTO must stay on the legacy host-controlled
    // path so its FP32 mixed-J stage and independent FP64 refinement cannot be
    // bypassed by an opt-in two-iteration device chunk.
    device_chunk_mode =
        options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused &&
        !fitted && !has_exchange && !has_range_correction && !nonlocal_correlation && !mixed_j &&
        spins == 1 && functional != SemilocalFamily::Wb97mv &&
        options.semilocal_exchange_scale == 1.0 && options.semilocal_correlation_scale == 1.0 &&
        provider.system().ecp_terms.empty() && configured_chunk_width() == kCudaKsChunkCapacity;
    if (device_chunk_mode) {
      const auto binding = device_chunk_binding();
      if (!device_chunk_region.matches(binding))
        device_chunk_region.bind(binding);
      else if (device_chunk_region.failed())
        device_chunk_region.recover();
    } else if (device_chunk_region.bound()) {
      device_chunk_region.invalidate();
    }
    try {
      check(cudaMemsetAsync(history_count, 0, sizeof(*history_count), stream));
      check(cudaMemsetAsync(history_head, 0, sizeof(*history_head), stream));
      cuda_ks_detail::reset_control(stream, spins, static_cast<int>(occupations[0]),
                                    static_cast<int>(occupations[1]), control, enabled,
                                    spin_enabled);
      check(cudaGetLastError());
      if (use_warm) {
        check(cudaMemcpyAsync(density, warm, elements * sizeof(double), cudaMemcpyDeviceToDevice,
                              stream));
      } else if (input) {
        check(cudaMemcpyAsync(density, prepared.data(), elements * sizeof(double),
                              cudaMemcpyHostToDevice, stream));
        // Explicit initial-guess staging, never an iteration matrix transfer.
        check(cudaStreamSynchronize(stream));
        movement.density_h2d_bytes += elements * sizeof(double);
        ++movement.synchronizations;
      } else {
        check(cudaMemcpyAsync(density, cold_seed, elements * sizeof(double),
                              cudaMemcpyDeviceToDevice, stream));
      }
    } catch (...) {
      cudaStreamSynchronize(stream);
      is_failed = true;
      if (device_chunk_mode && device_chunk_region.bound())
        device_chunk_region.mark_failure("CUDA KS device region preparation failed");
      throw;
    }
    previous_energy = std::numeric_limits<double>::infinity();
    is_active = true;
  }

  unsigned configured_chunk_width() const noexcept {
    const char* selection = std::getenv("VIBEQC_CUDA_KS_CHUNK");
    if (selection != nullptr) {
      if (std::strcmp(selection, "0") == 0 || std::strcmp(selection, "1") == 0 ||
          std::strcmp(selection, "off") == 0 || std::strcmp(selection, "none") == 0)
        return 1;
      if (std::strcmp(selection, "2") == 0) return kCudaKsChunkCapacity;
    }
    // Complete cold/warm/changed-geometry endpoint measurements did not
    // establish a reproducible benefit for automatic promotion.
    return 1;
  }

  runtime::SolverRegionCudaBinding solver_region_binding() const {
    return {{"cuda-ks-rks-solver-region-v1", device, stream, arena, fock_binding.source_identity},
            kCudaKsChunkCapacity,
            runtime::SolverRegionCompletionMode::Scalar,
            false};
  }

  unsigned submission_width() const noexcept {
    unsigned width = configured_chunk_width();
    if (width == 1 || output.iterations >= options.max_iterations) return 1;
    width = std::min<unsigned>(width, options.max_iterations - output.iterations);
    if (!output.dft_diagnostic.history.empty()) {
      const auto& last = output.dft_diagnostic.history.back();
      const double residual_gate = std::min(1e-9, options.density_tolerance);
      if (last.energy_change < 32.0 * options.energy_tolerance ||
          last.density_change < 32.0 * options.density_tolerance ||
          last.physical_residual < 32.0 * residual_gate)
        width = 1;
    }
    return std::max(1U, width);
  }

  void enqueue_one(unsigned slot) {
    if (slot >= kCudaKsChunkCapacity) throw std::logic_error("CUDA KS chunk slot overflow");
    if (generation == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CUDA KS density generation exhausted");
    std::string detail;
    check(scf::enqueue_prepared_cuda_fock(provider, density, nullptr, matrix, j, exchange, nullptr,
                                          jk_error, false, detail),
          detail);
    if (has_range_correction)
      check(scf::enqueue_prepared_cuda_exchange_correction(provider, *range_correction, density,
                                                           nullptr, matrix, range_exchange, nullptr,
                                                           range_jk_error, detail),
            detail);
    xc->enqueue(density, elements, ++generation);
    pending_generations[slot] = generation;
    ++movement.submitted_iterations;
    const auto potential = xc->view(generation);
    cuda_ks_detail::assemble_fock(stream, n, spins, hcore, j, exchange, exchange_coefficient,
                                  range_exchange, range_exchange_coefficient, potential.potential,
                                  enabled, fock);
    check(cudaGetLastError());
    const auto blocks = static_cast<unsigned>((elements + 127) / 128);
    const auto multiply = [&](const double* a, bool a_spin, bool transpose, const double* b,
                              bool b_spin, const std::uint8_t* mask, double* c) {
      launch_spin_matrix_product_kernel(blocks, 128, 0, stream, 1, spins, n, a, a_spin, transpose,
                                        b, b_spin, mask, c);
      check(cudaGetLastError());
    };
    multiply(fock, true, false, density, true, enabled, tmp1);
    multiply(tmp1, true, false, overlap, false, enabled, residual);
    multiply(overlap, false, false, density, true, enabled, tmp1);
    multiply(tmp1, true, false, fock, true, enabled, tmp2);
    launch_subtract_matrix_batches_kernel(blocks, 128, 0, stream, 1, spins, n, tmp2, enabled,
                                          residual);
    check(cudaGetLastError());
    launch_update_diis_kernel(1, 32, 0, stream, 1, n, spins, history, fock, residual, enabled,
                              fock_history, residual_history, gram, weights, history_count,
                              history_head, effective, true);
    check(cudaGetLastError());
    multiply(effective, true, false, x, false, enabled, tmp1);
    multiply(x, false, true, tmp1, true, enabled, tmp2);
    check(eigensolver->launch(spins, tmp2, effective, eigenvalues, solver_info, spin_enabled),
          "CUDA KS eigensolver launch failed");
    multiply(x, false, false, tmp2, true, enabled, tmp1);
    launch_build_density_kernel(blocks, 128, 0, stream, 1, n, occupied, tmp1, enabled, proposal);
    check(cudaGetLastError());
    auto* record = scalar_records + slot;
    cuda_ks_detail::diagnostics(
        stream, n, spins, density, proposal, residual, hcore, overlap, j, exchange,
        exchange_coefficient, range_exchange, range_exchange_coefficient, potential.totals,
        potential.error, jk_error, range_jk_error,
        device_nonlocal ? nonlocal_domain_error : nullptr,
        device_nonlocal ? nonlocal_pair_error : nullptr, solver_info, enabled, record);
    check(cudaGetLastError());
    cuda_ks_detail::advance(stream, n, spins, provider.one_electron().nuclear_repulsion,
                            static_cast<int>(occupations[0]), static_cast<int>(occupations[1]),
                            options.energy_tolerance, options.density_tolerance,
                            options.max_iterations, warm_updates, record, control, proposal,
                            density, warm, enabled, spin_enabled);
    check(cudaGetLastError());
  }

  void enqueue_device() {
    current_device();
    if (!is_active || is_pending) throw std::logic_error("CUDA KS iteration state is not ready");
    is_pending = true;
    pending_iterations = 0;
    try {
      const unsigned width = submission_width();
      const unsigned remaining = options.max_iterations - output.iterations;
      pending_iterations =
          solver_region_executor.submit(solver_region_binding(), width, remaining, false,
                                        [&](unsigned slot) { enqueue_one(slot); });
    } catch (...) {
      cudaStreamSynchronize(stream);
      ++movement.synchronizations;
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
      device_chunk_region.mark_failure("CUDA KS device chunk submission failed");
      throw;
    }
  }

  bool finish_device() {
    current_device();
    if (!is_pending || pending_iterations == 0)
      throw std::logic_error("no pending CUDA KS iteration chunk");
    std::array<cuda_ks_detail::Scalars, kCudaKsChunkCapacity> physical{};
    cuda_ks_detail::Control device_control{};
    const unsigned submitted = pending_iterations;
    try {
      check(cudaMemcpyAsync(physical.data(), scalar_records, submitted * sizeof(physical[0]),
                            cudaMemcpyDeviceToHost, stream));
      check(cudaMemcpyAsync(&device_control, control, sizeof(device_control),
                            cudaMemcpyDeviceToHost, stream));
      check(cudaStreamSynchronize(stream));
    } catch (...) {
      cudaStreamSynchronize(stream);
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
      device_chunk_region.mark_failure("CUDA KS device chunk completion failed");
      throw;
    }
    movement.scalar_d2h_bytes += submitted * sizeof(physical[0]) + sizeof(device_control);
    ++movement.synchronizations;
    ++movement.iteration_synchronizations;
    ++movement.iteration_chunks;
    solver_region_executor.checkpoint();
    if (device_control.iterations <= output.iterations ||
        device_control.iterations > output.iterations + submitted) {
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
      device_chunk_region.mark_failure("CUDA KS device chunk returned an invalid iteration count");
      throw std::runtime_error("CUDA KS device chunk returned an invalid iteration count");
    }
    const unsigned completed = device_control.iterations - output.iterations;
    movement.iterations += completed;
    auto& diagnostic = output.dft_diagnostic;
    for (unsigned slot = 0; slot < completed; ++slot) {
      const auto& item = physical[slot];
      ++output.iterations;
      ++output.fock_builds;
      diagnostic.components = {provider.one_electron().nuclear_repulsion, item.one_electron,
                               item.hartree, item.xc, item.exact_exchange};
      diagnostic.physical_residual = item.residual;
      diagnostic.electrons = {item.electrons[0], item.electrons[1]};
      diagnostic.density_change = item.density_change;
      output.physical_residual_rms = item.residual_rms;
      output.energy = diagnostic.components.total();
      output.energy_change = item.energy_change;
      output.density_rms = item.density_rms;
      diagnostic.history.push_back({output.iterations, diagnostic.components, item.energy_change,
                                    item.density_change, item.residual, diagnostic.electrons,
                                    false});
    }
    is_pending = false;
    pending_iterations = 0;
    is_failed = device_control.failed != 0;
    is_active = device_control.active != 0;
    output.converged = device_control.converged != 0;
    if (output.converged && warm_updates) warm_ready = true;
    if (output.converged) {
      final_state_ready = true;
      final_generation = pending_generations[completed - 1U];
    }
    if (is_failed)
      device_chunk_region.mark_failure("CUDA KS device control reported failure");
    else
      device_chunk_region.mark_success();
    return is_active;
  }

  void enqueue() {
    try {
      if (device_chunk_mode)
        enqueue_device();
      else
        enqueue_legacy();
    } catch (...) {
      invalidate_warm_orbitals();
      throw;
    }
  }
  bool finish() {
    // A partial density/frame copy or rejected iteration cannot lend the old
    // intermediate frame. Preserve the independent last-good warm density.
    try {
      const bool active = device_chunk_mode ? finish_device() : finish_legacy();
      if (is_failed) invalidate_warm_orbitals();
      return active;
    } catch (...) {
      invalidate_warm_orbitals();
      throw;
    }
  }

  CudaXcView stage_xc(std::uint64_t next_generation,
                      CudaXcDensityPrecision precision = CudaXcDensityPrecision::Fp64) {
    if (options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused) {
      if (!xc) throw std::logic_error("device-fused XC owner is unavailable");
      if (!device_nonlocal) {
        xc->enqueue(density, elements, next_generation, precision);
        return xc->view(next_generation);
      }
      xc->enqueue_density_features(density, elements, next_generation, nonlocal_raw_density,
                                   nonlocal_raw_gradient);
      const auto quadrature = xc->grid_view();
      run_resident_nonlocal_cuda([&] {
        nlc::enqueue_vv10_molecular_domain_cuda(
            stream, xc_layout.npoint, generated::kMolecularVv10DensityThreshold, quadrature.weights,
            nonlocal_raw_density, nonlocal_raw_gradient, nonlocal_effective_weights,
            nonlocal_effective_density, nonlocal_effective_gradient, nonlocal_domain_error);
      });
      run_resident_nonlocal_cuda([&] {
        nlc::enqueue_vv10_cuda_device(
            nonlocal_layout, nonlocal_correlation->parameters(), device, stream, quadrature.points,
            nonlocal_effective_weights, nonlocal_effective_density, nonlocal_effective_gradient,
            nonlocal_workspace, nonlocal_layout.workspace_bytes, nonlocal_workspace, nonlocal_vrho,
            nonlocal_vsigma, nullptr, nullptr, nonlocal_pair_error);
      });
      xc->enqueue_nonlocal_potential(next_generation, nonlocal_effective_weights,
                                     nonlocal_effective_gradient, nonlocal_vrho, nonlocal_vsigma,
                                     nonlocal_workspace);
      return xc->view(next_generation);
    }
    const auto bytes = elements * sizeof(double);
    check(cudaMemcpyAsync(host_xc_density.data(), density, bytes, cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    movement.xc_host_d2h_bytes += bytes;
    ++movement.xc_host_synchronizations;
    ++movement.synchronizations;

    host_xc_totals.fill(0.0);
    if (spins == 1) {
      XcIntegral value;
      if (functional == SemilocalFamily::Lda)
        value = integrate_lda_xc_pw_rks(basis, grid, host_xc_density, xc_layout.tile_points);
      else if (functional == SemilocalFamily::Pbe)
        value = integrate_pbe_rks_with_tail(basis, grid, host_xc_density, xc_layout.tile_points);
      else if (functional == SemilocalFamily::Wb97mv)
        value = integrate_wb97mv_rks(basis, grid, host_xc_density, xc_layout.tile_points);
      else
        value = integrate_r2scan_rks(basis, grid, host_xc_density, xc_layout.tile_points);
      if (value.potential.size() != matrix)
        throw std::runtime_error("host-unfused RKS XC potential size changed");
      std::copy(value.potential.begin(), value.potential.end(), host_xc_potential.begin());
      host_xc_totals = {value.energy, 0.5 * value.electrons, 0.5 * value.electrons};
      if (nonlocal_correlation) {
        const auto nonlocal =
            nlc::integrate_vv10_rks(basis, grid, host_xc_density, *nonlocal_correlation,
                                    xc_layout.tile_points, {}, nonlocal_domain);
        if (nonlocal.potential.size() != matrix)
          throw std::runtime_error("host-unfused RKS nonlocal potential size changed");
        for (std::size_t i = 0; i < matrix; ++i) host_xc_potential[i] += nonlocal.potential[i];
        host_xc_totals[0] += nonlocal.energy;
      }
    } else {
      std::copy_n(host_xc_density.begin(), matrix, host_xc_alpha.begin());
      std::copy_n(host_xc_density.begin() + matrix, matrix, host_xc_beta.begin());
      SpinXcIntegral value;
      if (functional == SemilocalFamily::Lda)
        value = integrate_lda_xc_pw_uks(basis, grid, host_xc_alpha, host_xc_beta,
                                        xc_layout.tile_points);
      else if (functional == SemilocalFamily::Pbe)
        value = integrate_pbe_uks(basis, grid, host_xc_alpha, host_xc_beta, xc_layout.tile_points);
      else if (functional == SemilocalFamily::Wb97mv)
        value =
            integrate_wb97mv_uks(basis, grid, host_xc_alpha, host_xc_beta, xc_layout.tile_points);
      else
        value =
            integrate_r2scan_uks(basis, grid, host_xc_alpha, host_xc_beta, xc_layout.tile_points);
      if (value.potential[0].size() != matrix || value.potential[1].size() != matrix)
        throw std::runtime_error("host-unfused UKS XC potential size changed");
      std::copy(value.potential[0].begin(), value.potential[0].end(), host_xc_potential.begin());
      std::copy(value.potential[1].begin(), value.potential[1].end(),
                host_xc_potential.begin() + matrix);
      host_xc_totals = {value.energy, value.electrons[0], value.electrons[1]};
      if (nonlocal_correlation) {
        const auto nonlocal =
            nlc::integrate_vv10_uks(basis, grid, host_xc_alpha, host_xc_beta, *nonlocal_correlation,
                                    xc_layout.tile_points, nonlocal_domain);
        if (nonlocal.potential[0].size() != matrix || nonlocal.potential[1].size() != matrix)
          throw std::runtime_error("host-unfused UKS nonlocal potential size changed");
        for (std::size_t i = 0; i < matrix; ++i) {
          host_xc_potential[i] += nonlocal.potential[0][i];
          host_xc_potential[matrix + i] += nonlocal.potential[1][i];
        }
        host_xc_totals[0] += nonlocal.energy;
      }
    }

    check(cudaMemcpyAsync(tmp1, host_xc_potential.data(), bytes, cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(staged_xc_totals, host_xc_totals.data(), sizeof(host_xc_totals),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(staged_xc_error, &host_xc_error, sizeof(host_xc_error),
                          cudaMemcpyHostToDevice, stream));
    movement.xc_host_h2d_bytes += bytes + sizeof(host_xc_totals) + sizeof(host_xc_error);
    return {next_generation, n, spins, tmp1, staged_xc_totals, staged_xc_error, stream};
  }

  void enqueue_legacy() {
    current_device();
    if (!is_active || is_pending) throw std::logic_error("CUDA KS iteration state is not ready");
    if (generation == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CUDA KS density generation exhausted");
    is_pending = true;  // Any partial CUDA submission is drained on failure.
    try {
      std::string detail;
      pending_mixed_j = mixed_j && !strict_refinement;
      // DF retains its qualified resident adapter until the prepared execution
      // seam supports fitted providers. Both routes use their owner's stream;
      // the exact route never exposes its concrete Direct-J/K handle here.
      if (fitted) check(cudaMemsetAsync(jk_error, 0, sizeof(*jk_error), stream));
      const auto jk_status =
          fitted ? (spins == 2 ? scf::execute_cuda_density_fitting_uhf_jk_device(
                                     fitted, density, density + matrix, j, nullptr, nullptr, detail,
                                     {true, false}, scf::FockMatrixLayout::RowMajor)
                               : scf::execute_cuda_density_fitting_rhf_jk_device(
                                     fitted, density, j, nullptr, detail, {true, false},
                                     scf::FockMatrixLayout::RowMajor))
                 : scf::enqueue_prepared_cuda_fock(
                       provider, density, spins == 2 ? density + matrix : nullptr, matrix, j,
                       exchange, has_exchange && spins == 2 ? exchange + matrix : nullptr, jk_error,
                       pending_mixed_j, detail);
      check(jk_status, detail);
      if (has_range_correction)
        check(scf::enqueue_prepared_cuda_exchange_correction(
                  provider, *range_correction, density, spins == 2 ? density + matrix : nullptr,
                  matrix, range_exchange, spins == 2 ? range_exchange + matrix : nullptr,
                  range_jk_error, detail),
              detail);
      mixed_j_executed = mixed_j_executed || pending_mixed_j;
      const auto potential =
          stage_xc(++generation, pending_mixed_j ? CudaXcDensityPrecision::Fp32ComputeFp64Accumulate
                                                 : CudaXcDensityPrecision::Fp64);
      pending_generations[0] = generation;
      ++movement.submitted_iterations;
      pending_iterations = 1;
      cuda_ks_detail::assemble_fock(stream, n, spins, hcore, j, exchange, exchange_coefficient,
                                    range_exchange, range_exchange_coefficient, potential.potential,
                                    enabled, fock);
      check(cudaGetLastError());
      const auto blocks = static_cast<unsigned>((elements + 127) / 128);
      const auto multiply = [&](const double* a, bool a_spin, bool transpose, const double* b,
                                bool b_spin, double* c) {
        launch_spin_matrix_product_kernel(blocks, 128, 0, stream, 1, spins, n, a, a_spin, transpose,
                                          b, b_spin, enabled, c);
        check(cudaGetLastError());
      };
      // Physical residual is FDS-SDF, using the unchanged CURRENT density.
      multiply(fock, true, false, density, true, tmp1);
      multiply(tmp1, true, false, overlap, false, residual);
      multiply(overlap, false, false, density, true, tmp1);
      multiply(tmp1, true, false, fock, true, tmp2);
      launch_subtract_matrix_batches_kernel(blocks, 128, 0, stream, 1, spins, n, tmp2, enabled,
                                            residual);
      check(cudaGetLastError());
      if (final_closure) {
        // Discard DIIS history and rebuild the closure proposal from F[D].
        // A stationary UKS occupation cycle still needs its virtual-space
        // shift, as on CPU. Export separately validates the unshifted F[D].
        check(cudaMemcpyAsync(effective, fock, elements * sizeof(double), cudaMemcpyDeviceToDevice,
                              stream));
      } else {
        launch_update_diis_kernel(1, 32, 0, stream, 1, n, spins, history, fock, residual, enabled,
                                  fock_history, residual_history, gram, weights, history_count,
                                  history_head, effective, true);
        check(cudaGetLastError());
      }
      if (stabilize_occupations) {
        // Match the CPU stationary-cycle policy. The unit-occupation virtual
        // projector is S-SDS for each spin. Shift only the DIIS proposal;
        // physical F/D/residual and the history above remain unmodified.
        multiply(overlap, false, false, density, true, tmp1);
        multiply(tmp1, true, false, overlap, false, tmp2);
        cuda_ks_detail::stabilize_uks_proposal(stream, n, overlap, tmp2, enabled, effective);
        check(cudaGetLastError());
        ++movement.occupation_stabilized_proposals;
      }
      multiply(effective, true, false, x, false, tmp1);
      multiply(x, false, true, tmp1, true, tmp2);
      check(eigensolver->launch(spins, tmp2, effective, eigenvalues, solver_info, spin_enabled),
            "CUDA KS eigensolver launch failed");
      multiply(x, false, false, tmp2, true, tmp1);
      if (spins == 1)
        launch_build_density_kernel(blocks, 128, 0, stream, 1, n, occupied, tmp1, enabled,
                                    proposal);
      else
        launch_build_spin_density_kernel(blocks, 128, 0, stream, 1, spins, n, occupied, tmp1,
                                         enabled, proposal);
      check(cudaGetLastError());
      cuda_ks_detail::diagnostics(
          stream, n, spins, density, proposal, residual, hcore, overlap, j, exchange,
          exchange_coefficient, range_exchange, range_exchange_coefficient, potential.totals,
          potential.error, jk_error, range_jk_error,
          device_nonlocal ? nonlocal_domain_error : nullptr,
          device_nonlocal ? nonlocal_pair_error : nullptr, solver_info, enabled, scalar_records);
      check(cudaGetLastError());
    } catch (...) {
      cudaStreamSynchronize(stream);
      ++movement.synchronizations;
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
      throw;
    }
  }

  bool finish_legacy() {
    current_device();
    if (!is_pending) throw std::logic_error("no pending CUDA KS iteration");
    cuda_ks_detail::Scalars physical{};
    try {
      check(cudaMemcpyAsync(&physical, scalar_records, sizeof(physical), cudaMemcpyDeviceToHost,
                            stream));
      check(cudaStreamSynchronize(stream));
    } catch (...) {
      cudaStreamSynchronize(stream);
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
      throw;
    }
    movement.scalar_d2h_bytes += sizeof(physical);
    ++movement.synchronizations;
    ++movement.iteration_synchronizations;
    ++movement.iteration_chunks;
    ++movement.iterations;
    is_pending = false;
    pending_iterations = 0;
    ++output.iterations;
    ++output.fock_builds;
    if (mixed_j_executed && !pending_mixed_j) ++refinement_iterations;
    output.precision.requested_mode = options.precision_mode.value_or(VIBEQC_PRECISION_FP64);
    output.precision.effective_bits = mixed_j_executed ? 32U : 64U;
    output.precision.strict_refinement_applied = mixed_j_executed && refinement_iterations > 0;
    output.precision.refinement_iterations = refinement_iterations;
    auto& diagnostic = output.dft_diagnostic;
    diagnostic.components = {provider.one_electron().nuclear_repulsion, physical.one_electron,
                             physical.hartree, physical.xc, physical.exact_exchange};
    diagnostic.physical_residual = physical.residual;
    diagnostic.electrons = {physical.electrons[0], physical.electrons[1]};
    diagnostic.density_change = physical.density_change;
    output.physical_residual_rms = physical.residual_rms;
    output.energy = diagnostic.components.total();
    output.energy_change = std::abs(output.energy - previous_energy);
    output.density_rms = physical.density_rms;
    diagnostic.history.push_back({output.iterations, diagnostic.components, output.energy_change,
                                  physical.density_change, physical.residual, diagnostic.electrons,
                                  stabilize_occupations});
    // The kernel validates electronic components; their host-side sum with
    // the nuclear term must also be finite before any convergence/cache gate.
    is_failed = physical.failure != 0 || !std::isfinite(output.energy);
    for (unsigned s = 0; s < 2; ++s)
      if (std::abs(physical.electrons[s] - occupations[s]) > 1e-8) is_failed = true;
    if (is_failed) {
      if (pending_mixed_j) {
        // Any failed low-precision attempt retries the same density with the
        // strict target operator. Do not publish or cache the failed proposal.
        strict_refinement = true;
        stabilize_occupations = false;
        final_closure = false;
        final_corrections = 0;
        is_failed = false;
        is_active = true;
        check(cudaMemsetAsync(history_count, 0, sizeof(*history_count), stream));
        check(cudaMemsetAsync(history_head, 0, sizeof(*history_head), stream));
        previous_energy = std::numeric_limits<double>::infinity();
        return true;
      }
      is_active = false;
      return false;
    }
    // A stationary physical state can still alternate integer occupations.
    // Enable the same 0.1-Eh proposal shift as CPU UKS only after both physical
    // gates pass. A subsequent density-change gate must still pass to finish.
    if (!final_closure && spins == 2 && output.iterations > 1 &&
        output.energy_change < options.energy_tolerance &&
        physical.residual < std::min(1e-9, options.density_tolerance) &&
        physical.density_change >= options.density_tolerance)
      stabilize_occupations = true;
    const bool converged = output.iterations > 1 &&
                           output.energy_change < options.energy_tolerance &&
                           physical.density_change < options.density_tolerance &&
                           physical.residual < std::min(1e-9, options.density_tolerance) &&
                           physical.maximum_residual < std::min(1e-9, options.density_tolerance);
    // Keep the existing RMS diagnostic, but do not publish an energy-only state
    // that the shared final-state validator will reject on the AO maximum norm.
    const bool strict_final_closure = spins == 2 || !provider.system().ecp_terms.empty();
    const bool mixed_stage = mixed_j && !strict_refinement;
    const bool enter_strict_refinement =
        mixed_stage && (converged || output.iterations >= options.max_iterations);
    if (enter_strict_refinement) {
      // AUTO may use FP32 only as an iterative accelerator. Reset nonlinear
      // history and give refinement its own full budget for the FP64 target.
      strict_refinement = true;
      final_closure = false;
      final_corrections = 0;
      stabilize_occupations = false;
      output.converged = false;
      is_active = true;
      check(cudaMemsetAsync(history_count, 0, sizeof(*history_count), stream));
      check(cudaMemsetAsync(history_head, 0, sizeof(*history_head), stream));
    } else if (strict_final_closure && converged && !final_closure) {
      // A DIIS proposal can satisfy the SCF gate before a fresh F[D] proposal
      // does. Preserve any established UKS occupation stabilization through
      // this bounded correction, just as CPU UKS does; clearing it restarts
      // the stationary occupation cycle. Physical energy/residual gates and
      // the separate unshifted final-state export validator stay unchanged.
      final_closure = true;
      final_corrections = 0;
      output.converged = false;
      is_active = true;
    } else if (final_closure) {
      ++final_corrections;
      output.converged = converged;
      is_active = !output.converged && final_corrections < kMaximumFinalCorrections;
    } else {
      output.converged = converged;
      const bool refinement_budget = strict_refinement && mixed_j
                                         ? refinement_iterations < options.max_iterations
                                         : output.iterations < options.max_iterations;
      is_active = !output.converged && refinement_budget;
    }
    try {
      if (output.converged && warm_updates) {
        // E, F, residual and retained D all belong to this same generation.
        // A failed/unfinished solve can never overwrite the last-good cache.
        check(cudaMemcpyAsync(warm, density, elements * sizeof(double), cudaMemcpyDeviceToDevice,
                              stream));
        warm_ready = true;
      } else if (is_active) {
        check(cudaMemcpyAsync(density, proposal, elements * sizeof(double),
                              cudaMemcpyDeviceToDevice, stream));
        // tmp2 still owns the ordinary eigensolver's orthonormal-basis orbital
        // frame. Retain it beside the accepted proposal so the next RKS/UKS
        // iteration can evaluate the same #996 occupied-subspace gate without
        // a matrix D2H. No solver routing changes in this slice.
        check(cudaMemcpyAsync(warm_orbitals, tmp2, elements * sizeof(double),
                              cudaMemcpyDeviceToDevice, stream));
        warm_orbitals_ready = true;
        ++movement.warm_orbital_frames_retained;
      } else {
        invalidate_warm_orbitals();
      }
      if (output.converged) {
        final_state_ready = true;
        final_generation = generation;
      }
    } catch (...) {
      cudaStreamSynchronize(stream);
      is_active = false;
      is_failed = true;
      output.converged = false;
      throw;
    }
    previous_energy = output.energy;
    return is_active;
  }

  KsFinalStateIdentity final_identity() const {
    KsFinalStateIdentity identity;
    identity.determinant.factor = {owner, 1, final_generation, final_generation};
    identity.determinant.solve_epoch = solve_epoch;
    identity.determinant.model = provider.strategy();
    identity.determinant.occupied = {occupations[0]};
    if (spins == 2) identity.determinant.occupied.push_back(occupations[1]);
    identity.model = {1,
                      semilocal_family_domain_version(functional),
                      grid_spec,
                      xc_layout.tile_points,
                      semilocal_family_code(functional),
                      spins,
                      device,
                      owner};
    if (range_correction) identity.model.range_correction = *range_correction;
    if (nonlocal_correlation) {
      identity.model.nonlocal_correlation = nonlocal_correlation->parameters();
      identity.model.nonlocal_density_domain = nonlocal_domain;
    }
    // A snapshot must describe the same XC composition that built its Fock matrix.
    identity.model.semilocal_exchange_scale = options.semilocal_exchange_scale;
    identity.model.semilocal_correlation_scale = options.semilocal_correlation_scale;
    return identity;
  }

  CudaKsFinalStateToken token() const {
    if (!final_state_ready || !output.converged || is_active || is_pending || is_failed ||
        !solve_epoch || !final_generation)
      throw std::invalid_argument("CUDA KS owner has no successful current final state");
    return {1, final_identity()};
  }

  VerifiedKsFinalState read_final(const CudaKsFinalStateToken& expected,
                                  bool compute_weighted_density, std::string& detail) {
    const auto current = token();
    if (expected.version != 1 || expected != current)
      throw std::invalid_argument(
          "CUDA KS final-state token has stale owner, epoch, generation, model or occupations");
    current_device();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument(
          "CUDA KS final-state read requires an ordinary noncapturing stream");

    if (!final_frame_ready) {
      const auto blocks = static_cast<unsigned>((elements + 127) / 128);
      const auto multiply = [&](const double* a, bool a_spin, bool transpose, const double* b,
                                bool b_spin, double* c) {
        launch_spin_matrix_product_kernel(blocks, 128, 0, stream, 1, spins, n, a, a_spin, transpose,
                                          b, b_spin, final_enabled, c);
        check(cudaGetLastError());
      };
      multiply(fock, true, false, x, false, tmp1);
      multiply(x, false, true, tmp1, true, tmp2);
      check(eigensolver->launch(spins, tmp2, effective, final_eigenvalues, final_solver_info,
                                final_spin_enabled),
            "CUDA KS final-state eigensolver launch failed");
      multiply(x, false, false, tmp2, true, final_coefficients);
    }

    KsPhysicalState physical;
    KsFinalStateCandidate candidate;
    physical.identity = candidate.identity = current.identity;
    physical.physical = candidate.physical_origin = true;
    physical.components = output.dft_diagnostic.components;
    physical.reported_energy = output.energy;
    physical.physical_residual = output.dft_diagnostic.physical_residual;
    candidate.fock_density_generation = final_generation;
    physical.density.resize(spins);
    physical.fock.resize(spins);
    candidate.spins.resize(spins);
    for (unsigned spin = 0; spin < spins; ++spin) {
      physical.density[spin].resize(matrix);
      physical.fock[spin].resize(matrix);
      candidate.spins[spin].vectors.resize(matrix);
      candidate.spins[spin].values.resize(n);
    }
    int info[2]{};
    cudaError_t error = cudaSuccess;
    const auto copy = [&](void* target, const void* source, std::size_t bytes) {
      if (error == cudaSuccess)
        error = cudaMemcpyAsync(target, source, bytes, cudaMemcpyDeviceToHost, stream);
    };
    for (unsigned spin = 0; spin < spins; ++spin) {
      const auto offset = static_cast<std::size_t>(spin) * matrix;
      copy(physical.density[spin].data(), density + offset, matrix * sizeof(double));
      copy(physical.fock[spin].data(), fock + offset, matrix * sizeof(double));
      copy(candidate.spins[spin].vectors.data(), final_coefficients + offset,
           matrix * sizeof(double));
      copy(candidate.spins[spin].values.data(),
           final_eigenvalues + static_cast<std::size_t>(spin) * n, n * sizeof(double));
      copy(&info[spin], final_solver_info + spin, sizeof(int));
    }
    const auto drained = cudaStreamSynchronize(stream);
    if (error == cudaSuccess) error = drained;
    check(error);
    ++movement.synchronizations;
    ++movement.final_state_reads;
    const auto numeric_bytes = spins * (3 * matrix + n) * sizeof(double);
    movement.matrix_d2h_bytes += spins * 3 * matrix * sizeof(double);
    movement.final_state_d2h_bytes += numeric_bytes + spins * sizeof(int);
    final_frame_ready = true;
    for (unsigned spin = 0; spin < spins; ++spin)
      if (info[spin] != 0) {
        final_state_ready = false;
        throw std::runtime_error("CUDA KS final-state eigensolver reported failure");
      }
    // CUDA matrix products/eigensolvers store columns contiguously, whereas
    // the detached reference frame uses row-major C[ao, orbital]. Symmetric
    // D/F need no conversion; copying C verbatim would validate its transpose
    // and reject even a converged two-orbital state.
    for (auto& frame : candidate.spins)
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = row + 1; column < n; ++column)
          std::swap(frame.vectors[row * n + column], frame.vectors[column * n + row]);
    if (token() != current)
      throw std::invalid_argument("CUDA KS final-state eligibility changed during export");

    scf::solver::FinalStateLimits limits;
    limits.density_tolerance = options.density_tolerance;
    limits.energy_tolerance = options.energy_tolerance;
    limits.maximum_corrections = 0;
    limits.require_canonicality = true;
    VerifiedKsFinalState verified;
    if (!validate_ks_final_state(current.identity, provider.one_electron().overlap,
                                 provider.one_electron().hcore, physical, candidate, limits,
                                 compute_weighted_density, verified, detail)) {
      final_state_ready = false;
      throw std::runtime_error(detail.empty() ? "CUDA KS final-state validation failed" : detail);
    }
    return verified;
  }

  std::vector<double> download(const double* source) {
    current_device();
    std::vector<double> data(elements);
    try {
      check(cudaMemcpyAsync(data.data(), source, elements * sizeof(double), cudaMemcpyDeviceToHost,
                            stream));
      check(cudaStreamSynchronize(stream));
    } catch (...) {
      cudaStreamSynchronize(stream);
      throw;
    }
    movement.matrix_d2h_bytes += elements * sizeof(double);
    ++movement.synchronizations;
    return data;
  }
};

CudaKsPlan::CudaKsPlan(const scf::PreparedFockPlan& fock, const AoBasis& basis,
                       const MolecularGrid& grid, const scf::ScfOptions& options,
                       SemilocalFamily functional, std::size_t tile_points,
                       const scf::ResolvedFockBuild* range_correction,
                       nlc::Vv10Plan* nonlocal_correlation, nlc::Vv10DensityDomain nonlocal_domain)
    : impl_(std::make_unique<Impl>(fock, basis, grid, options, functional, tile_points,
                                   range_correction, nonlocal_correlation, nonlocal_domain)) {}
CudaKsPlan::~CudaKsPlan() = default;
void CudaKsPlan::begin(const std::vector<double>* seed, bool reuse_warm) {
  impl_->begin(seed, reuse_warm);
}
bool CudaKsPlan::active() const noexcept { return impl_->is_active; }
bool CudaKsPlan::pending() const noexcept { return impl_->is_pending; }
bool CudaKsPlan::failed() const noexcept { return impl_->is_failed; }
void CudaKsPlan::enqueue_iteration() { impl_->enqueue(); }
bool CudaKsPlan::finish_iteration() { return impl_->finish(); }
scf::ScfResult CudaKsPlan::result(bool export_density) {
  if (!impl_->started || impl_->is_active || impl_->is_pending)
    throw std::logic_error("CUDA KS result is not terminal");
  auto result = impl_->output;
  if (export_density) result.density = impl_->download(impl_->density);
  return result;
}
scf::ScfResult CudaKsPlan::run(const std::vector<double>* seed, bool reuse_warm,
                               bool export_density) {
  begin(seed, reuse_warm);
  while (active()) {
    enqueue_iteration();
    finish_iteration();
  }
  return result(export_density);
}
std::vector<double> CudaKsPlan::warm_density() {
  if (impl_->is_pending)
    throw std::logic_error("cannot export warm state during a pending iteration");
  return impl_->warm_ready ? impl_->download(impl_->warm) : std::vector<double>{};
}
void CudaKsPlan::set_warm_start_updates(bool enabled) noexcept { impl_->warm_updates = enabled; }
void CudaKsPlan::clear_warm_start() noexcept { impl_->clear_warm_state(); }
void CudaKsPlan::invalidate_final_state() noexcept {
  impl_->final_state_ready = impl_->final_frame_ready = false;
  impl_->final_generation = 0;
}
vibeqc_status CudaKsPlan::final_state_token(CudaKsFinalStateToken& token,
                                            std::string& detail) const {
  token = {};
  detail.clear();
  try {
    token = impl_->token();
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA KS final-state token failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
}
vibeqc_status CudaKsPlan::read_final_state(const CudaKsFinalStateToken& expected,
                                           bool compute_weighted_density,
                                           VerifiedKsFinalState& state, std::string& detail) {
  state = {};
  detail.clear();
  try {
    state = impl_->read_final(expected, compute_weighted_density, detail);
    detail.clear();
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "host allocation for detached CUDA KS final state failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const vibeqc::Error& error) {
    impl_->final_state_ready = false;
    detail = error.what();
    return error.status();
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    impl_->final_state_ready = false;
    if (detail.empty()) detail = error.what();
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
}
const CudaKsResources& CudaKsPlan::resources() const noexcept { return impl_->resource; }
CudaKsTransfers CudaKsPlan::transfers() const noexcept {
  auto out = impl_->movement;
  const auto& region = impl_->device_chunk_region.metrics();
  out.execution_region_bindings = region.bindings;
  out.execution_region_invalidations = region.invalidations;
  out.execution_region_executions = region.executions;
  out.execution_region_failures = region.failures;
  out.execution_region_recoveries = region.recoveries;
  if (impl_->xc) {
    const auto& xc = impl_->xc->transfers();
    out.setup_h2d_bytes += xc.setup_h2d_bytes;
    out.synchronizations += xc.synchronizations;
  }
  return out;
}
}  // namespace vibeqc::dft
