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
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/eigensolver.hpp"
#include "scf/cuda/scf_constants.hpp"
#include "scf/cuda/scf_density_kernels.hpp"
#include "scf/cuda/scf_diis_kernels.hpp"
#include "scf/cuda/scf_matrix_kernels.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/proposal_control.hpp"
#include "vibeqc/vibeqc.hpp"

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
  double *hcore{}, *overlap{}, *x{}, *j{}, *density{}, *proposal{}, *warm{}, *fock{}, *residual{},
      *tmp1{}, *tmp2{}, *effective{}, *fock_history{}, *residual_history{}, *gram{}, *weights{},
      *eigenvalues{}, *final_coefficients{}, *final_eigenvalues{};
  std::int32_t* occupied{};
  std::uint8_t *enabled{}, *spin_enabled{};
  std::uint32_t *history_count{}, *history_head{};
  int *solver_info{}, *final_solver_info{}, *jk_error{};
  std::uint8_t *final_spin_enabled{}, *final_enabled{};
  cuda_ks_detail::Control* control{};
  cuda_ks_detail::Scalars* scalar_records{};
  /** The dry run and actual partition share one checked, typed layout. All
   * persistent and phase-local numeric buffers are explicitly charged. */
  std::size_t partition(std::size_t n, unsigned spins, unsigned history, void* storage) {
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
    for (auto** pointer : {&density, &proposal, &warm, &fock, &residual, &tmp1, &tmp2, &effective})
      reserve(*pointer, elements);
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

std::size_t cuda_ks_state_bytes(std::size_t n, unsigned spins, unsigned history) {
  if (!n || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (spins != 1 && spins != 2) || history > 64)
    throw std::invalid_argument("invalid CUDA KS resource shape");
  KsStateStorage layout;
  return layout.partition(n, spins, std::max(1U, history), nullptr);
}

struct CudaKsPlan::Impl : KsStateStorage {
  const scf::PreparedFockPlan& provider;
  scf::ScfOptions options;
  scf::CudaDirectJkPlan* direct{};
  cudaStream_t stream{};
  int device{};
  std::size_t n{}, matrix{}, elements{};
  unsigned spins{}, history{};
  std::array<std::size_t, 2> occupations{};
  std::vector<double> orthogonalizer, cold_density;
  GridSpec grid_spec;
  CudaKsResources resource;
  CudaKsTransfers movement;
  void *arena{}, *xc_arena{};
  std::unique_ptr<CudaXcPlan> xc;
  scf::ScfResult output;
  bool is_active{}, is_pending{}, is_failed{}, warm_ready{}, started{};
  bool warm_updates{true}, device_chunk_mode{};
  bool stabilize_occupations{}, final_closure{};
  unsigned final_corrections{};
  bool pbe{}, final_state_ready{}, final_frame_ready{};
  std::uint64_t owner{next_ks_owner()}, solve_epoch{}, generation{}, final_generation{};
  double previous_energy{std::numeric_limits<double>::infinity()};
  unsigned pending_iterations{};
  std::array<std::uint64_t, kCudaKsChunkCapacity> pending_generations{};

  void current_device() const {
    // Prepared owners select their bound device on every entry, as the common
    // Fock provider does. Another context may have changed this thread's device
    // between calls; borrowed XC views still enforce their own device identity.
    check(cudaSetDevice(device));
  }

  std::vector<double> seed(const std::vector<double>* input) const {
    using namespace scf::reference;
    const auto& ints = provider.one_electron();
    if (options.strict_initial_density && input) {
      const std::vector<unsigned> counts =
          spins == 2 ? std::vector<unsigned>{static_cast<unsigned>(occupations[0]),
                                             static_cast<unsigned>(occupations[1])}
                     : std::vector<unsigned>{static_cast<unsigned>(occupations[0])};
      scf::solver::validate_seed(ints.overlap, *input, n, counts, spins == 2 ? 1.0 : 2.0);
      return *input;
    }
    std::optional<EigenResult> a, b;
    if (spins == 2) {
      const auto pair = scf::initial_guess::prepare_initial_uhf_density(
          ints, orthogonalizer, occupations[0], occupations[1], input, a, b);
      return concatenate(pair.first, pair.second);
    }
    return scf::initial_guess::prepare_initial_density(provider.system(), ints, orthogonalizer,
                                                       occupations[0], input, a);
  }

  Impl(const scf::PreparedFockPlan& plan, const AoBasis& basis, const MolecularGrid& grid,
       const scf::ScfOptions& control, bool pbe, std::size_t tile)
      : provider(plan), options(control), grid_spec(grid.spec()), pbe(pbe) {
    const auto& strategy = provider.strategy();
    scf::validate_resolved_fock_build(strategy);
    if (!owner || strategy.backend != scf::FockBackend::Cuda ||
        strategy.spec.derivative_order != 0 || !strategy.spec.coulomb.present ||
        strategy.spec.coulomb.coefficient != 1.0 ||
        strategy.spec.coulomb.approximation != scf::FockApproximation::Exact ||
        strategy.spec.exchange.present || !(direct = provider.cuda_direct_source()))
      throw std::invalid_argument(
          "CUDA KS requires the prepared conventional Coulomb-only strategy");
    if (options.compute_forces || options.hooks || options.export_physical_reference ||
        options.xc_density_route != XcDensityRoute::DensityMatrix ||
        (options.precision_mode && *options.precision_mode != VIBEQC_PRECISION_FP64))
      throw std::invalid_argument("CUDA KS supports FP64 density-matrix energy execution only");
    if (!options.max_iterations || !std::isfinite(options.energy_tolerance) ||
        !std::isfinite(options.density_tolerance) || options.energy_tolerance <= 0.0 ||
        options.density_tolerance <= 0.0 || options.diis_history > 64)
      throw std::invalid_argument("invalid CUDA KS convergence or DIIS controls");
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
    device = scf::cuda_direct_jk_device(direct);
    stream = scf::cuda_direct_jk_stream(direct);
    current_device();
    orthogonalizer = scf::reference::symmetric_orthogonalizer(provider.one_electron().overlap, n);
    cold_density = seed(nullptr);
    resource.state_device_bytes = partition(n, spins, history, nullptr);
    resource.xc_device_bytes = cuda_xc_layout(basis, grid, pbe, spins == 2, tile).device_bytes;
    resource.provider_device_bytes = provider.diagnostic().device_bytes;
    output.dft_diagnostic.history.reserve(options.max_iterations);
    resource.retained_host_numeric_bytes =
        (orthogonalizer.capacity() + cold_density.capacity()) * sizeof(double) +
        output.dft_diagnostic.history.capacity() * sizeof(ScfIteration);
    try {
      check(runtime::resource_cuda_malloc(&arena, resource.state_device_bytes));
      partition(n, spins, history, arena);
      check(runtime::resource_cuda_malloc(&xc_arena, resource.xc_device_bytes));
      check(cudaMemsetAsync(arena, 0, resource.state_device_bytes, stream));
      const auto upload = [&](void* destination, const void* source, std::size_t bytes) {
        check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
        movement.setup_h2d_bytes += bytes;
      };
      upload(hcore, provider.one_electron().hcore.data(), matrix * sizeof(double));
      upload(overlap, provider.one_electron().overlap.data(), matrix * sizeof(double));
      upload(x, orthogonalizer.data(), matrix * sizeof(double));
      const std::int32_t spin_counts[]{static_cast<std::int32_t>(occupations[0]),
                                       static_cast<std::int32_t>(occupations[1])};
      const std::uint8_t selected[]{static_cast<std::uint8_t>(occupations[0] > 0),
                                    static_cast<std::uint8_t>(occupations[1] > 0)};
      upload(occupied, spin_counts, spins * sizeof(std::int32_t));
      upload(spin_enabled, selected, spins * sizeof(std::uint8_t));
      const std::uint8_t all_spins[]{1, 1};
      const std::uint8_t one = 1;
      upload(final_spin_enabled, all_spins, spins * sizeof(std::uint8_t));
      upload(final_enabled, &one, sizeof(one));
      // XC setup drains this same stream, including the small stack inputs.
      xc = std::make_unique<CudaXcPlan>(basis, grid, pbe, spins == 2, tile, xc_arena,
                                        resource.xc_device_bytes, stream);
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
    if (xc_arena) runtime::resource_cuda_free(xc_arena);
    if (arena) runtime::resource_cuda_free(arena);
    xc_arena = arena = nullptr;
    cudaSetDevice(previous);
  }
  ~Impl() { cleanup(); }

  void begin(const std::vector<double>* input, bool reuse_warm) {
    if (is_pending) throw std::logic_error("cannot replace a pending CUDA KS iteration");
    final_state_ready = final_frame_ready = false;
    final_generation = 0;
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
    output.dft_diagnostic.grid_points = xc->layout().npoint;
    output.dft_diagnostic.tile_points = xc->layout().tile_points;
    output.dft_diagnostic.ao_order = xc->layout().pbe ? 1 : 0;
    output.initial_density_used = input != nullptr || use_warm;
    is_active = false;
    started = true;
    is_failed = false;
    stabilize_occupations = false;
    final_closure = false;
    final_corrections = 0;
    pending_iterations = 0;
    // The bounded device-control prototype is qualified only for direct
    // all-electron RKS. UKS keeps occupation/final-closure host control, and
    // ECP RKS keeps the strict physical final closure required by #586.
    device_chunk_mode = spins == 1 && provider.system().ecp_terms.empty() &&
                        configured_chunk_width() == kCudaKsChunkCapacity;
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
      } else {
        const auto& initial = input ? prepared : cold_density;
        check(cudaMemcpyAsync(density, initial.data(), elements * sizeof(double),
                              cudaMemcpyHostToDevice, stream));
        // Explicit initial-guess staging, never an iteration matrix transfer.
        check(cudaStreamSynchronize(stream));
        movement.density_h2d_bytes += elements * sizeof(double);
        ++movement.synchronizations;
      }
    } catch (...) {
      cudaStreamSynchronize(stream);
      is_failed = true;
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
    check(scf::enqueue_cuda_direct_jk_device(direct, provider.strategy().spec, density, nullptr,
                                             matrix, j, nullptr, nullptr, jk_error, detail),
          detail);
    xc->enqueue(density, elements, ++generation);
    pending_generations[slot] = generation;
    ++movement.submitted_iterations;
    const auto potential = xc->view(generation);
    cuda_ks_detail::assemble_fock(stream, n, spins, hcore, j, potential.potential, enabled, fock);
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
    EigensolverResources solver{};
    solver.stream_ = stream;
    const auto family = n <= kSmallEigensolverLimit ? scf::CudaEigensolverFamily::small_native
                                                    : scf::CudaEigensolverFamily::graph_native;
    check(launch_solver(solver, family, n, spins, tmp2, effective, eigenvalues, 0, solver_info,
                        spin_enabled),
          "CUDA KS eigensolver launch failed");
    multiply(x, false, false, tmp2, true, enabled, tmp1);
    launch_build_density_kernel(blocks, 128, 0, stream, 1, n, occupied, tmp1, enabled, proposal);
    check(cudaGetLastError());
    auto* record = scalar_records + slot;
    cuda_ks_detail::diagnostics(stream, n, spins, density, proposal, residual, hcore, overlap, j,
                                potential.totals, potential.error, jk_error, solver_info, enabled,
                                record);
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
      for (unsigned slot = 0; slot < width; ++slot) {
        enqueue_one(slot);
        ++pending_iterations;
      }
    } catch (...) {
      cudaStreamSynchronize(stream);
      ++movement.synchronizations;
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
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
      throw;
    }
    movement.scalar_d2h_bytes += submitted * sizeof(physical[0]) + sizeof(device_control);
    ++movement.synchronizations;
    ++movement.iteration_synchronizations;
    ++movement.iteration_chunks;
    if (device_control.iterations <= output.iterations ||
        device_control.iterations > output.iterations + submitted) {
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
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
                               item.hartree, item.xc};
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
    return is_active;
  }

  void enqueue() {
    if (device_chunk_mode)
      enqueue_device();
    else
      enqueue_legacy();
  }
  bool finish() { return device_chunk_mode ? finish_device() : finish_legacy(); }

  void enqueue_legacy() {
    current_device();
    if (!is_active || is_pending) throw std::logic_error("CUDA KS iteration state is not ready");
    if (generation == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CUDA KS density generation exhausted");
    is_pending = true;  // Any partial CUDA submission is drained on failure.
    try {
      std::string detail;
      check(scf::enqueue_cuda_direct_jk_device(direct, provider.strategy().spec, density,
                                               spins == 2 ? density + matrix : nullptr, matrix, j,
                                               nullptr, nullptr, jk_error, detail),
            detail);
      xc->enqueue(density, elements, ++generation);
      pending_generations[0] = generation;
      ++movement.submitted_iterations;
      pending_iterations = 1;
      const auto potential = xc->view(generation);
      cuda_ks_detail::assemble_fock(stream, n, spins, hcore, j, potential.potential, enabled, fock);
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
        // The public derivative state is validated against the unshifted
        // physical F[D], not the preceding DIIS/stabilized proposal. During
        // bounded final closure, diagonalize exactly that physical operator.
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
      EigensolverResources solver{};
      solver.stream_ = stream;
      // Both families are existing native solvers and work on ordinary
      // streams. The larger family has no fixed AO bound or opaque workspace.
      const auto family = n <= kSmallEigensolverLimit ? scf::CudaEigensolverFamily::small_native
                                                      : scf::CudaEigensolverFamily::graph_native;
      check(launch_solver(solver, family, n, spins, tmp2, effective, eigenvalues, 0, solver_info,
                          spin_enabled),
            "CUDA KS eigensolver launch failed");
      multiply(x, false, false, tmp2, true, tmp1);
      if (spins == 1)
        launch_build_density_kernel(blocks, 128, 0, stream, 1, n, occupied, tmp1, enabled,
                                    proposal);
      else
        launch_build_spin_density_kernel(blocks, 128, 0, stream, 1, spins, n, occupied, tmp1,
                                         enabled, proposal);
      check(cudaGetLastError());
      cuda_ks_detail::diagnostics(stream, n, spins, density, proposal, residual, hcore, overlap, j,
                                  potential.totals, potential.error, jk_error, solver_info, enabled,
                                  scalar_records);
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
    auto& diagnostic = output.dft_diagnostic;
    diagnostic.components = {provider.one_electron().nuclear_repulsion, physical.one_electron,
                             physical.hartree, physical.xc};
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
                           physical.residual < std::min(1e-9, options.density_tolerance);
    constexpr unsigned maximum_final_corrections = 4;
    const bool strict_final_closure = spins == 2 || !provider.system().ecp_terms.empty();
    if (strict_final_closure && converged && !final_closure) {
      // A DIIS proposal can satisfy the ordinary SCF density-change gate while
      // the canonical density of the unshifted physical Fock is microscopically
      // outside the derivative-state tolerance. UKS already requires this closure;
      // ECP RKS needs the same physical fixed point for strict derivative snapshots.
      final_closure = true;
      final_corrections = 0;
      stabilize_occupations = false;
      output.converged = false;
      is_active = true;
    } else if (final_closure) {
      ++final_corrections;
      output.converged = converged;
      is_active = !output.converged && final_corrections < maximum_final_corrections;
    } else {
      output.converged = converged;
      is_active = !output.converged && output.iterations < options.max_iterations;
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
    identity.model = {1, 1, grid_spec, xc->layout().tile_points, pbe, spins, device, owner};
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
      EigensolverResources solver{};
      solver.stream_ = stream;
      const auto family = n <= kSmallEigensolverLimit ? scf::CudaEigensolverFamily::small_native
                                                      : scf::CudaEigensolverFamily::graph_native;
      check(launch_solver(solver, family, n, spins, tmp2, effective, final_eigenvalues, 0,
                          final_solver_info, final_spin_enabled),
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
                       const MolecularGrid& grid, const scf::ScfOptions& options, bool pbe,
                       std::size_t tile_points)
    : impl_(std::make_unique<Impl>(fock, basis, grid, options, pbe, tile_points)) {}
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
void CudaKsPlan::clear_warm_start() noexcept { impl_->warm_ready = false; }
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
  const auto& xc = impl_->xc->transfers();
  out.setup_h2d_bytes += xc.setup_h2d_bytes;
  out.synchronizations += xc.synchronizations;
  return out;
}
}  // namespace vibeqc::dft
