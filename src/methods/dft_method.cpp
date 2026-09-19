#include "methods/dft_method.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/types.hpp"
#include "vibeqc/vibeqc.hpp"

#if VIBEQC_HAS_CUDA
#include "dft/cuda_ks.hpp"
#include "scf/cuda_direct_jk.hpp"
#endif

namespace vibeqc::methods::detail {
namespace {

std::uint64_t next_cpu_ks_owner() {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  do {
    if (value == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CPU KS owner identity exhausted");
  } while (!next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed));
  return value;
}

bool is_uks(vibeqc_method method) noexcept {
  return method == VIBEQC_METHOD_LDA_UKS || method == VIBEQC_METHOD_PBE_UKS;
}

bool is_supported_dft(vibeqc_method method) noexcept {
  return method == VIBEQC_METHOD_LDA_RKS || method == VIBEQC_METHOD_PBE_RKS || is_uks(method);
}

bool field_present(const vibeqc_method_descriptor& descriptor, std::size_t offset,
                   std::size_t width) noexcept {
  return descriptor.struct_size >= offset && descriptor.struct_size - offset >= width;
}

scf::ScfOptions dft_options(const vibeqc_method_descriptor& descriptor, vibeqc_backend backend) {
  if (!std::isfinite(descriptor.energy_tolerance) || !std::isfinite(descriptor.density_tolerance) ||
      !std::isfinite(descriptor.screening_tolerance))
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "DFT tolerances must be finite");
  scf::ScfOptions options;
  options.max_iterations = descriptor.max_iterations == 0 ? 100 : descriptor.max_iterations;
  options.diis_history = descriptor.diis_history == 0 ? 8 : descriptor.diis_history;
  options.energy_tolerance =
      descriptor.energy_tolerance > 0.0 ? descriptor.energy_tolerance : 1.0e-10;
  options.density_tolerance =
      descriptor.density_tolerance > 0.0 ? descriptor.density_tolerance : 1.0e-8;
  options.screening_tolerance =
      descriptor.screening_tolerance > 0.0 ? descriptor.screening_tolerance : 1.0e-12;
  if (field_present(descriptor, offsetof(vibeqc_method_descriptor, density_fitting_mode),
                    sizeof(descriptor.density_fitting_mode))) {
    const auto mode = descriptor.density_fitting_mode;
    if (mode != VIBEQC_DENSITY_FITTING_NONE && mode != VIBEQC_DENSITY_FITTING_CPU_REFERENCE &&
        mode != VIBEQC_DENSITY_FITTING_CUDA && mode != VIBEQC_DENSITY_FITTING_AUTO)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown density-fitting execution mode");
    if (mode != VIBEQC_DENSITY_FITTING_NONE)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT supports conventional Coulomb only");
  }
  if (field_present(descriptor, offsetof(vibeqc_method_descriptor, density_fitting_auxiliary_basis),
                    sizeof(descriptor.density_fitting_auxiliary_basis)) &&
      descriptor.density_fitting_auxiliary_basis != nullptr)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "DFT does not accept an unused auxiliary basis");
  if (field_present(descriptor, offsetof(vibeqc_method_descriptor, precision_mode),
                    sizeof(descriptor.precision_mode))) {
    if (descriptor.precision_mode != VIBEQC_PRECISION_FP64 &&
        descriptor.precision_mode != VIBEQC_PRECISION_AUTO)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown floating-point precision mode");
    if (descriptor.precision_mode == VIBEQC_PRECISION_AUTO)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT supports explicit FP64 precision only");
  }

  scf::FockBuildSpec fock;
  fock.spin =
      (descriptor.method == VIBEQC_METHOD_LDA_UKS || descriptor.method == VIBEQC_METHOD_PBE_UKS)
          ? scf::FockSpin::Unrestricted
          : scf::FockSpin::Restricted;
  fock.derivative_order = 0;
  fock.exchange.present = false;
  options.resolved_fock_build = scf::resolve_fock_build(
      fock, backend == VIBEQC_BACKEND_CUDA ? scf::FockBackend::Cuda : scf::FockBackend::Cpu,
      options.screening_tolerance);
  options.compute_forces = false;
  return options;
}

/** Copy every pointee before constructing scientific owners. Legacy method
 * descriptors retain the original unit-radius GridSpec and 256-point tiles. */
dft::GridSpec ks_grid_options(const vibeqc_method_descriptor& descriptor,
                              scf::ScfOptions& options) {
  dft::GridSpec grid;
  if (!field_present(descriptor, offsetof(vibeqc_method_descriptor, ks_options),
                     sizeof(descriptor.ks_options)) ||
      !descriptor.ks_options)
    return grid;
  const auto& input = *descriptor.ks_options;
  if (input.struct_size < sizeof(vibeqc_ks_options) || input.abi_version != VIBEQC_ABI_VERSION)
    throw MethodError(VIBEQC_STATUS_ABI_MISMATCH, "KS options ABI mismatch");
  if (input.scf_domain_version != 1)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "unsupported KS tail/spin domain policy");
  if (!input.tile_points || input.tile_points > static_cast<std::uint64_t>(INT_MAX))
    throw std::invalid_argument("invalid KS XC tile points");
  options.xc_tile_points = input.tile_points;
  grid.version = input.grid_version;
  grid.radial_points = input.radial_points;
  grid.angular_polar = input.angular_polar;
  grid.angular_azimuth = input.angular_azimuth;
  grid.partition_iterations = input.partition_iterations;
  grid.coincident_tolerance = input.coincident_tolerance;
  if ((input.element_radii == nullptr) != (input.element_radius_count == 0) ||
      (input.element_radii && input.element_radius_count != grid.element_radii.size()))
    throw std::invalid_argument("KS element radii require 119 entries or NULL/zero");
  if (input.element_radii) {
    for (std::size_t z = 1; z < grid.element_radii.size(); ++z) {
      const double radius = input.element_radii[z];
      if (!std::isfinite(radius) || radius <= 0.0)
        throw std::invalid_argument("KS element radii must be positive finite");
      grid.element_radii[z] = radius == 1.0 ? 0.0 : radius;
    }
  }
  dft::validate_grid_spec(grid);
  return grid;
}

Result adapt_result(scf::ScfResult native, vibeqc_backend backend) {
  Result result;
  result.energy = native.energy;
  result.convergence.iterations = native.iterations;
  result.convergence.energy_change = native.energy_change;
  result.convergence.residual_rms = native.density_rms;
  result.physical_residual_rms = native.physical_residual_rms;
  result.convergence.converged = native.converged;
  result.executed_backend = backend;
  result.fock_builds = native.fock_builds;
  native.dft_diagnostic.fock_builds = native.fock_builds;
  native.dft_diagnostic.initial_density_used = native.initial_density_used;
  // Move the snapshot instead of retaining another max-iteration history.
  result.ks_diagnostic = std::move(native.dft_diagnostic);
  return result;
}

/** The method's global ledger supplies the budget. Size the common direct
 * source explicitly so its standalone default cap is not a second KS limit. */
std::size_t ks_provider_bytes(const core::System& system, vibeqc_backend backend) {
#if VIBEQC_HAS_CUDA
  if (backend == VIBEQC_BACKEND_CUDA) {
    std::size_t primitives = 0;
    for (const auto& shell : system.shells)
      primitives = runtime::add_capacity(primitives, shell.primitives.size());
    return scf::cuda_direct_jk_device_bytes(1, molecule::ao_count(system), system.atoms.size(),
                                            system.shells.size(), primitives, 0);
  }
#endif
  return 0;
}

#if VIBEQC_HAS_CUDA
KsTransportDiagnostic adapt_transfers(const dft::CudaKsTransfers& value) {
  return {value.setup_h2d_bytes,
          value.density_h2d_bytes,
          value.scalar_d2h_bytes,
          value.matrix_d2h_bytes,
          value.final_state_d2h_bytes,
          value.final_state_reads,
          value.synchronizations,
          value.iterations,
          value.occupation_stabilized_proposals};
}

void add_transfers(dft::CudaKsTransfers& target, const dft::CudaKsTransfers& value) {
  const auto add = [](std::uint64_t& destination, std::uint64_t increment) {
    if (increment > std::numeric_limits<std::uint64_t>::max() - destination)
      throw std::overflow_error("CUDA KS transport counter overflow");
    destination += increment;
  };
  add(target.setup_h2d_bytes, value.setup_h2d_bytes);
  add(target.density_h2d_bytes, value.density_h2d_bytes);
  add(target.scalar_d2h_bytes, value.scalar_d2h_bytes);
  add(target.matrix_d2h_bytes, value.matrix_d2h_bytes);
  add(target.final_state_d2h_bytes, value.final_state_d2h_bytes);
  add(target.final_state_reads, value.final_state_reads);
  add(target.synchronizations, value.synchronizations);
  add(target.iterations, value.iterations);
  add(target.occupation_stabilized_proposals, value.occupation_stabilized_proposals);
}
#endif

class KsPreparedCalculation final : public PreparedCalculation {
 public:
  KsPreparedCalculation(Capabilities capabilities, core::System system, vibeqc_method method,
                        scf::ScfOptions options, dft::GridSpec grid, vibeqc_backend backend,
                        int device)
      : capabilities_(capabilities),
        system_(std::move(system)),
        method_(method),
        options_(std::move(options)),
        backend_(backend),
        fock_(system_, nullptr, *options_.resolved_fock_build, device,
              ks_provider_bytes(system_, backend)),
        basis_(system_),
        grid_(system_, grid) {
    options_.retain_ks_state = backend_ != VIBEQC_BACKEND_CUDA;
#if VIBEQC_HAS_CUDA
    if (backend_ == VIBEQC_BACKEND_CUDA)
      cuda_ = std::make_unique<dft::CudaKsPlan>(
          fock_, basis_, grid_, options_,
          method_ == VIBEQC_METHOD_PBE_RKS || method_ == VIBEQC_METHOD_PBE_UKS,
          options_.xc_tile_points);
#endif
    runtime::sample_cpu_capacity(host_numeric_capacity());
  }

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }
  const core::System& system() const noexcept { return system_; }

  /** Explicit retained vectors; object metadata and transient setup are not
   * inferred from this lower-bound observation. Grid/basis buffers are owned. */
  std::size_t host_numeric_capacity() const noexcept {
    auto bytes =
        runtime::add_capacity(fock_.cpu_observation_capacity(),
                              runtime::vector_capacities(basis_.packed, grid_.points(),
                                                         grid_.weights(), grid_.owners(), warm_));
    if (cpu_physical_)
      for (const auto* matrices : {&cpu_physical_->density, &cpu_physical_->fock})
        for (const auto& matrix : *matrices)
          bytes = runtime::add_capacity(bytes, runtime::vector_bytes(matrix));
#if VIBEQC_HAS_CUDA
    if (cuda_) bytes = runtime::add_capacity(bytes, cuda_->resources().retained_host_numeric_bytes);
#endif
    return bytes;
  }

  /** Explicit output/rebuild export. Ordinary CUDA replays keep this on device. */
  std::vector<double> warm_density() {
#if VIBEQC_HAS_CUDA
    if (cuda_) return cuda_->warm_density();
#endif
    return warm_;
  }

  void clear_warm_start() noexcept {
    warm_.clear();
#if VIBEQC_HAS_CUDA
    if (cuda_) cuda_->clear_warm_start();
#endif
  }

  void invalidate_result() override { invalidate_final_state(); }

  void invalidate_final_state() noexcept {
    cpu_physical_.reset();
#if VIBEQC_HAS_CUDA
    if (cuda_) cuda_->invalidate_final_state();
#endif
  }

#if VIBEQC_HAS_CUDA
  dft::CudaKsPlan* cuda_plan() noexcept { return cuda_.get(); }
#endif

  std::optional<KsTransportDiagnostic> ks_transport_diagnostic() const override {
#if VIBEQC_HAS_CUDA
    if (cuda_) return adapt_transfers(cuda_->transfers());
#endif
    return std::nullopt;
  }

  vibeqc_status final_state_token(dft::CudaKsFinalStateToken& token, std::string& detail) const {
#if VIBEQC_HAS_CUDA
    if (cuda_) return cuda_->final_state_token(token, detail);
#endif
    token = {};
    if (!system_.ecp_terms.empty()) {
      detail = "CPU ECP final-state handoff is not qualified";
      return VIBEQC_STATUS_NOT_IMPLEMENTED;
    }
    if (!cpu_physical_) {
      detail = "CPU KS owner has no successful current final state";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    token = {1, cpu_physical_->identity};
    detail.clear();
    return VIBEQC_STATUS_SUCCESS;
  }

  vibeqc_status read_final_state(const dft::CudaKsFinalStateToken& expected,
                                 bool compute_weighted_density, dft::VerifiedKsFinalState& state,
                                 std::string& detail) {
#if VIBEQC_HAS_CUDA
    if (cuda_) return cuda_->read_final_state(expected, compute_weighted_density, state, detail);
#endif
    state = {};
    dft::CudaKsFinalStateToken current;
    const auto status = final_state_token(current, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    if (expected != current) {
      detail = "CPU KS final-state token is stale";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    // The SCF frame predates the last F[D] rebuild. Diagonalize that actual
    // retained physical F here; do not relabel the lagged orbital energies.
    // Explicit export costs one eigen solve and validation, zero Fock builds.
    const auto& ints = fock_.one_electron();
    const auto x = scf::reference::symmetric_orthogonalizer(ints.overlap, ints.nbf);
    std::vector<scf::reference::EigenResult> spins;
    spins.reserve(current.identity.model.spins);
    for (const auto& fock : cpu_physical_->fock)
      spins.push_back(scf::reference::generalized_eigen(fock, x, ints.nbf));
    dft::KsFinalStateCandidate candidate{current.identity,
                                         current.identity.determinant.factor.density_generation,
                                         true, std::move(spins)};
    scf::solver::FinalStateLimits limits{options_.density_tolerance, options_.energy_tolerance, 0,
                                         true};
    if (!dft::validate_ks_final_state(current.identity, ints.overlap, ints.hcore, *cpu_physical_,
                                      candidate, limits, compute_weighted_density, state, detail)) {
      invalidate_final_state();
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    return VIBEQC_STATUS_SUCCESS;
  }

  vibeqc_status read_derivative_state(const dft::CudaKsFinalStateToken& expected,
                                      KsDerivativeSnapshot& output, std::string& detail) {
    output = {};
    dft::VerifiedKsFinalState state;
#if VIBEQC_HAS_CUDA
    const auto before = cuda_ ? cuda_->transfers() : dft::CudaKsTransfers{};
#endif
    const auto status = read_final_state(expected, true, state, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    // These are the provider's actual metric and the collocation/grid sources
    // used by this immutable KS owner, not caller-supplied identity labels.
    output = {std::move(state), system_,        fock_.one_electron().overlap,
              basis_.packed,    grid_.points(), grid_.weights(),
              grid_.owners()};
    // Both backends collocate this owner's exact host-built quadrature. Export
    // its raw measures directly; dividing partitioned weights loses tail data.
    output.atomic_weights = grid_.atomic_weights();
#if VIBEQC_HAS_CUDA
    if (cuda_) {
      const auto after = cuda_->transfers();
      output.export_d2h_bytes = after.final_state_d2h_bytes - before.final_state_d2h_bytes;
      output.export_reads = after.final_state_reads - before.final_state_reads;
      output.export_synchronizations = after.synchronizations - before.synchronizations;
    }
#endif
    return VIBEQC_STATUS_SUCCESS;
  }

  Result execute(bool compute_forces) override {
    invalidate_final_state();
    const char* method_name =
        (method_ == VIBEQC_METHOD_PBE_RKS || method_ == VIBEQC_METHOD_PBE_UKS) ? "PBE" : "LDA";
    if (compute_forces) {
      throw MethodError(
          VIBEQC_STATUS_NOT_IMPLEMENTED,
          std::string(method_name) + " KS nuclear gradients are tracked separately in issue #163");
    }
    return adapt_result(run(nullptr, true, true), backend_);
  }

  /** Single-system and native batch paths share the same scientific owner. */
  scf::ScfResult run(const std::vector<double>* initial_density, bool reuse_warm,
                     bool update_warm) {
    invalidate_final_state();
#if VIBEQC_HAS_CUDA
    if (cuda_) {
      // Native iterations read only scalar diagnostics. The public energy
      // result does not require a final AO matrix download; warm D stays resident.
      cuda_->set_warm_start_updates(update_warm);
      auto native = cuda_->run(initial_density, reuse_warm, false);
      if (cuda_->failed())
        throw MethodError(VIBEQC_STATUS_NUMERICAL_FAILURE, "CUDA KS physical evaluation failed");
      return native;
    }
#endif
    if (cpu_epoch_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CPU KS solve epoch exhausted");
    ++cpu_epoch_;
    const auto* seed =
        initial_density ? initial_density : (reuse_warm && !warm_.empty() ? &warm_ : nullptr);
    // The CPU driver already samples its provider/grid. Add only the retained
    // last-good density, which coexists with its current/proposed densities.
    runtime::CpuRetainedCapacity retained_warm(runtime::vector_bytes(warm_));
    scf::ScfResult native;
    if (method_ == VIBEQC_METHOD_LDA_UKS || method_ == VIBEQC_METHOD_PBE_UKS)
      native = scf::run_uks(fock_, basis_, grid_, options_, method_ == VIBEQC_METHOD_PBE_UKS, seed);
    else if (method_ == VIBEQC_METHOD_PBE_RKS)
      native = scf::run_pbe_rks(fock_, basis_, grid_, options_, seed);
    else
      native = scf::run_lda_rks(fock_, basis_, grid_, options_, seed);
    // This owner has immutable model/geometry/spin identity. Only successful
    // executions may replace its compatible last-good density; DIIS is fresh.
    if (native.converged && options_.retain_ks_state) {
      const auto spins = is_uks(method_) ? 2U : 1U;
      const auto matrix = fock_.one_electron().nbf * fock_.one_electron().nbf;
      if (native.density.size() != spins * matrix ||
          native.ks_physical_fock.size() != spins * matrix ||
          (spins == 2 && native.dft_diagnostic.occupations.size() != spins))
        throw std::runtime_error("CPU KS retained spin-state shape mismatch");
      std::vector<scf::reference::Matrix> densities, focks;
      densities.reserve(spins);
      focks.reserve(spins);
      for (unsigned spin = 0; spin < spins; ++spin) {
        const auto begin = spin * matrix;
        densities.emplace_back(native.density.begin() + begin,
                               native.density.begin() + begin + matrix);
        focks.emplace_back(native.ks_physical_fock.begin() + begin,
                           native.ks_physical_fock.begin() + begin + matrix);
      }
      std::vector<std::size_t> occupied;
      if (spins == 1)
        occupied = {static_cast<std::size_t>(system_.electron_count / 2)};
      else
        occupied.assign(native.dft_diagnostic.occupations.begin(),
                        native.dft_diagnostic.occupations.end());
      dft::KsFinalStateIdentity identity;
      identity.determinant = {
          {cpu_owner_, 1, 1, 1}, cpu_epoch_, fock_.strategy(), std::move(occupied)};
      identity.model = {1,
                        1,
                        grid_.spec(),
                        options_.xc_tile_points,
                        method_ == VIBEQC_METHOD_PBE_RKS || method_ == VIBEQC_METHOD_PBE_UKS,
                        spins,
                        -1,
                        cpu_owner_};
      dft::KsPhysicalState physical{identity,
                                    true,
                                    std::move(densities),
                                    std::move(focks),
                                    native.dft_diagnostic.components,
                                    native.energy,
                                    native.dft_diagnostic.physical_residual};
      cpu_physical_ = std::move(physical);
    }
    if (native.converged && update_warm) warm_ = std::move(native.density);
    runtime::sample_cpu_capacity(host_numeric_capacity());
    return native;
  }

 private:
  Capabilities capabilities_;
  core::System system_;
  vibeqc_method method_{};
  scf::ScfOptions options_;
  vibeqc_backend backend_;
  scf::PreparedFockPlan fock_;
  dft::AoBasis basis_;
  dft::MolecularGrid grid_;
  std::vector<double> warm_;
  const std::uint64_t cpu_owner_{next_cpu_ks_owner()};
  std::uint64_t cpu_epoch_{};
  std::optional<dft::KsPhysicalState> cpu_physical_;
#if VIBEQC_HAS_CUDA
  std::unique_ptr<dft::CudaKsPlan> cuda_;
#endif
};

std::vector<double> positions(const core::System& system) {
  std::vector<double> out;
  out.reserve(3 * system.atoms.size());
  for (const auto& atom : system.atoms)
    out.insert(out.end(), atom.position.begin(), atom.position.end());
  return out;
}

bool valid_positions(const std::vector<double>& coordinates, const core::System& system) {
  return coordinates.size() == 3 * system.atoms.size() &&
         std::all_of(coordinates.begin(), coordinates.end(),
                     [](double value) { return std::isfinite(value); });
}

void set_positions(core::System& system, const std::vector<double>& coordinates) {
  for (std::size_t i = 0; i < system.atoms.size(); ++i)
    std::copy_n(coordinates.begin() + 3 * i, 3, system.atoms[i].position.begin());
}

vibeqc_status item_exception_status() {
  try {
    throw;
  } catch (const MethodError& error) {
    return error.status();
  } catch (const vibeqc::Error& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
}

/** Independent native KS owners, with ordinary-stream round-robin CUDA work.
 * Geometry is rebuilt per item, while model/basis/charge/spin remain immutable.
 * No HF graph or Python calculation loop participates in this schedule. */
class KsPreparedBatch final : public PreparedBatch {
 public:
  KsPreparedBatch(Capabilities capabilities, std::vector<core::System> systems,
                  vibeqc_method method, scf::ScfOptions options, dft::GridSpec grid,
                  vibeqc_backend backend, int device, bool warm_enabled)
      : capabilities_(capabilities),
        systems_(std::move(systems)),
        method_(method),
        options_(std::move(options)),
        grid_spec_(std::move(grid)),
        backend_(backend),
        device_(device),
        warm_enabled_(warm_enabled),
        items_(systems_.size()) {
    for (std::size_t i = 0; i < size(); ++i) {
      runtime::CpuRetainedCapacity neighbors(host_numeric_capacity());
      items_[i].plan = make_plan(systems_[i]);
    }
  }

  std::size_t size() const noexcept override { return systems_.size(); }

  void invalidate_result() override {
    for (auto& item : items_)
      if (item.plan) item.plan->invalidate_final_state();
  }

  std::vector<BatchItemResult> execute(const Coordinates& coordinates,
                                       bool compute_forces) override {
    invalidate_result();
    if (compute_forces)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "KS nuclear gradients are tracked separately in issue #163");
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument("KS batch coordinates do not match system count");
    std::vector<BatchItemResult> results(size());
    std::vector<bool> ready(size(), false);
    // Allocate source-geometry metadata before launching any item. The success
    // path can then publish its last-good identity without a coordinate copy.
    std::vector<scf::HfWarmState> candidates(size());
    for (std::size_t i = 0; i < size(); ++i) {
      auto& result = results[i];
      result.bucket_id = i;  // One ordinary stream/owner per stable input slot.
      result.calculation.executed_backend = backend_;
      result.calculation.energy = std::numeric_limits<double>::quiet_NaN();
      try {
        auto target = systems_[i];
        if (!coordinates.empty() && coordinates[i]) {
          if (!valid_positions(*coordinates[i], target))
            throw std::invalid_argument("invalid KS batch item coordinates");
          set_positions(target, *coordinates[i]);
        }
        auto& item = items_[i];
        candidates[i].coordinates = positions(target);
        if (!item.plan || positions(item.plan->system()) != candidates[i].coordinates) {
          // Preserve the last GOOD seed before freeing its device owner. This
          // explicit rebuild download is never part of routine SCF iterations.
          materialize_warm(i);
#if VIBEQC_HAS_CUDA
          if (auto* cuda = item.plan ? item.plan->cuda_plan() : nullptr)
            add_transfers(item.retired_transfers, cuda->transfers());
#endif
          item.plan.reset();
          item.resident_warm = false;
          item.plan = make_plan(target);
        }
        result.warm_start_used = warm_enabled_ && item.warm.has_value();
        ready[i] = true;
      } catch (...) {
        result.status = item_exception_status();
      }
    }

    const auto finish = [&](std::size_t i, scf::ScfResult native) {
      auto& result = results[i];
      result.calculation = adapt_result(std::move(native), backend_);
      const auto& calculation = result.calculation;
      result.status =
          calculation.convergence.converged ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_NOT_CONVERGED;
      if (calculation.convergence.converged && warm_enabled_ && warm_updates_) {
        auto& state = candidates[i];
        state.energy = calculation.energy;
        state.energy_change = calculation.convergence.energy_change;
        state.density_rms = calculation.convergence.residual_rms;
        state.iterations = calculation.convergence.iterations;
        items_[i].warm = std::move(state);
        items_[i].resident_warm = true;
      }
    };

    // A rejected/nonconverged warm solve gets one cold retry. CUDA retries
    // retain the same per-item scheduler; a failed neighbor never halts it.
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
      std::vector<bool> running(size(), false);
      for (std::size_t i = 0; i < size(); ++i) {
        auto& result = results[i];
        // Retry only seed-related failures. Resource/driver failures preserve
        // their first status and leave the last-good seed for explicit replay.
        const bool seed_failure = result.status == VIBEQC_STATUS_NOT_CONVERGED ||
                                  result.status == VIBEQC_STATUS_NUMERICAL_FAILURE ||
                                  result.status == VIBEQC_STATUS_INVALID_ARGUMENT;
        if (!ready[i] || (attempt && (!result.warm_start_used || !seed_failure))) continue;
        if (attempt) {
          result.warm_start_fallback = true;
          // Release the failed attempt's exported history before starting another
          // solve, preserving the two-history resource bound.
          result.calculation.ks_diagnostic.reset();
        }
        auto& item = items_[i];
        const bool reuse = !attempt && result.warm_start_used;
        const auto* seed = reuse && !item.resident_warm ? &item.warm->density : nullptr;
        try {
#if VIBEQC_HAS_CUDA
          if (auto* cuda = item.plan->cuda_plan()) {
            cuda->set_warm_start_updates(warm_enabled_ && warm_updates_);
            cuda->begin(seed, reuse && item.resident_warm);
            running[i] = true;
            continue;
          }
#endif
          runtime::CpuRetainedCapacity neighbors(host_numeric_capacity(i));
          finish(i,
                 item.plan->run(seed, reuse && item.resident_warm, warm_enabled_ && warm_updates_));
        } catch (...) {
          result.status = item_exception_status();
        }
      }
#if VIBEQC_HAS_CUDA
      while (std::any_of(running.begin(), running.end(), [](bool value) { return value; })) {
        // Submit ALL active streams before synchronizing any scalar record.
        for (std::size_t i = 0; i < size(); ++i) {
          if (!running[i]) continue;
          try {
            items_[i].plan->cuda_plan()->enqueue_iteration();
          } catch (...) {
            results[i].status = item_exception_status();
            running[i] = false;
          }
        }
        for (std::size_t i = 0; i < size(); ++i) {
          if (!running[i]) continue;
          try {
            auto* cuda = items_[i].plan->cuda_plan();
            if (cuda->finish_iteration()) continue;
            running[i] = false;
            if (cuda->failed())
              throw MethodError(VIBEQC_STATUS_NUMERICAL_FAILURE,
                                "CUDA KS physical evaluation failed");
            finish(i, cuda->result(false));
          } catch (...) {
            results[i].status = item_exception_status();
            running[i] = false;
          }
        }
      }
#endif
    }
    runtime::sample_cpu_capacity(host_numeric_capacity());
    return results;
  }

  void clear_warm_starts() override {
    for (auto& item : items_) {
      item.warm.reset();
      item.resident_warm = false;
      if (item.plan) item.plan->clear_warm_start();
    }
  }

  std::size_t warm_density_size(std::size_t index) const override {
    const auto n = molecule::ao_count(systems_.at(index));
    const std::size_t spins = is_uks(method_) ? 2 : 1;
    if (!n || n > std::numeric_limits<std::size_t>::max() / n / spins / sizeof(double))
      throw std::invalid_argument("KS warm density dimensions overflow");
    return spins * n * n;
  }

  const std::optional<scf::HfWarmState>& warm_state(std::size_t index) const override {
    materialize_warm(index);
    return items_.at(index).warm;
  }

  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> states) override {
    if (!warm_enabled_ || states.size() != size())
      throw std::invalid_argument("KS seed restore requires a matching warm-enabled batch");
    for (std::size_t i = 0; i < size(); ++i) {
      if (!states[i]) continue;
      const auto& state = *states[i];
      if (state.density.size() != warm_density_size(i) ||
          !valid_positions(state.coordinates, systems_[i]) || state.iterations < 0 ||
          !std::isfinite(state.energy) || !std::isfinite(state.energy_change) ||
          !std::isfinite(state.density_rms) || state.density_rms < 0)
        throw std::invalid_argument("invalid KS seed dimensions or diagnostics");
      auto source = systems_[i];
      set_positions(source, state.coordinates);
      // This common validation reads only source S and checks the shared
      // spin-density convention. It performs no HF Fock/energy evaluation.
      scf::validate_hf_warm_density(source, is_uks(method_) ? VIBEQC_METHOD_UHF : VIBEQC_METHOD_RHF,
                                    state.density);
    }
    // All source-metric validation precedes the no-throw commit. Missing
    // entries preserve neighbors, including their resident density ownership.
    for (std::size_t i = 0; i < size(); ++i) {
      if (!states[i]) continue;
      auto& item = items_[i];
      item.warm.swap(states[i]);
      item.resident_warm = false;
      if (item.plan) item.plan->clear_warm_start();
    }
  }

  void set_warm_start_updates(bool enabled) override { warm_updates_ = enabled; }

  std::optional<KsTransportDiagnostic> ks_transport_diagnostic(std::size_t index) const override {
    const auto& item = items_.at(index);
#if VIBEQC_HAS_CUDA
    if (auto* cuda = item.plan ? item.plan->cuda_plan() : nullptr) {
      auto cumulative = item.retired_transfers;
      add_transfers(cumulative, cuda->transfers());
      return adapt_transfers(cumulative);
    }
#endif
    return std::nullopt;
  }

  vibeqc_status final_state_token(std::size_t index, dft::CudaKsFinalStateToken& token,
                                  std::string& detail) const {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->final_state_token(token, detail);
    token = {};
    detail = "KS batch item has no prepared final-state owner";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  vibeqc_status read_final_state(std::size_t index, const dft::CudaKsFinalStateToken& expected,
                                 bool compute_weighted_density, dft::VerifiedKsFinalState& state,
                                 std::string& detail) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->read_final_state(expected, compute_weighted_density, state,
                                                  detail);
    state = {};
    detail = "KS batch item has no prepared final-state owner";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  vibeqc_status read_derivative_state(std::size_t index, const dft::CudaKsFinalStateToken& expected,
                                      KsDerivativeSnapshot& output, std::string& detail) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->read_derivative_state(expected, output, detail);
    output = {};
    detail = "KS batch item has no prepared final-state owner";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  // These profiles describe HF graph/provider layouts, not this method's
  // ordinary-stream schedule. Absence is explicit at the common interface.
  std::optional<std::vector<DirectShellClassProfileEntry>> last_direct_shell_class_profile()
      const override {
    return std::nullopt;
  }
  std::optional<DirectPppsQueueProfile> last_direct_ppps_queue_profile() const override {
    return std::nullopt;
  }
  std::vector<EigensolverDiagnostic> last_eigensolver_diagnostics() const override { return {}; }
  std::vector<scf::CudaDensityFittingMetricDiagnostic> last_density_fitting_metric_diagnostics()
      const override {
    return {};
  }
  std::vector<InactiveEigensolverProfileEntry> last_inactive_eigensolver_profile() const override {
    return {};
  }

 private:
  /** All other owners remain alive while one CPU item executes. The selected
   * item's externally materialized seed is also distinct from its plan. */
  std::size_t host_numeric_capacity(std::size_t exclude_plan = SIZE_MAX) const noexcept {
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < items_.size(); ++i) {
      const auto& item = items_[i];
      if (item.plan && i != exclude_plan)
        bytes = runtime::add_capacity(bytes, item.plan->host_numeric_capacity());
      if (item.warm)
        bytes = runtime::add_capacity(
            bytes, runtime::vector_capacities(item.warm->density, item.warm->coordinates));
    }
    return bytes;
  }

  struct Item {
    std::unique_ptr<KsPreparedCalculation> plan;
    // Density is materialized only for explicit output, import, or rebuilding
    // an owner. Empty density with resident_warm=true is a valid lazy snapshot.
    mutable std::optional<scf::HfWarmState> warm;
    bool resident_warm{};
#if VIBEQC_HAS_CUDA
    dft::CudaKsTransfers retired_transfers;
#endif
  };
  std::unique_ptr<KsPreparedCalculation> make_plan(const core::System& system) const {
    return std::make_unique<KsPreparedCalculation>(capabilities_, system, method_, options_,
                                                   grid_spec_, backend_, device_);
  }
  void materialize_warm(std::size_t i) const {
    const auto& item = items_.at(i);
    if (item.warm && item.warm->density.empty() && item.resident_warm)
      item.warm->density = item.plan->warm_density();
  }

  Capabilities capabilities_;
  std::vector<core::System> systems_;
  vibeqc_method method_;
  scf::ScfOptions options_;
  dft::GridSpec grid_spec_;
  vibeqc_backend backend_;
  int device_;
  bool warm_enabled_, warm_updates_{true};
  std::vector<Item> items_;
};

}  // namespace

vibeqc_status dft_final_state_token(const PreparedCalculation& calculation,
                                    dft::CudaKsFinalStateToken& token, std::string& detail) {
  const auto* ks = dynamic_cast<const KsPreparedCalculation*>(&calculation);
  if (ks) return ks->final_state_token(token, detail);
  token = {};
  detail = "prepared calculation is not a KS final-state owner";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

vibeqc_status read_dft_final_state(PreparedCalculation& calculation,
                                   const dft::CudaKsFinalStateToken& expected,
                                   bool compute_weighted_density, dft::VerifiedKsFinalState& state,
                                   std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedCalculation*>(&calculation);
  if (ks) return ks->read_final_state(expected, compute_weighted_density, state, detail);
  state = {};
  detail = "prepared calculation is not a KS final-state owner";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

vibeqc_status dft_final_state_token(const PreparedBatch& batch, std::size_t index,
                                    dft::CudaKsFinalStateToken& token, std::string& detail) {
  const auto* ks = dynamic_cast<const KsPreparedBatch*>(&batch);
  if (ks) return ks->final_state_token(index, token, detail);
  token = {};
  detail = "prepared batch is not a KS final-state owner";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

vibeqc_status read_dft_final_state(PreparedBatch& batch, std::size_t index,
                                   const dft::CudaKsFinalStateToken& expected,
                                   bool compute_weighted_density, dft::VerifiedKsFinalState& state,
                                   std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks) return ks->read_final_state(index, expected, compute_weighted_density, state, detail);
  state = {};
  detail = "prepared batch is not a KS final-state owner";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

vibeqc_status read_dft_derivative_state(PreparedBatch& batch, std::size_t index,
                                        const dft::CudaKsFinalStateToken& expected,
                                        KsDerivativeSnapshot& output, std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks) return ks->read_derivative_state(index, expected, output, detail);
  output = {};
  detail = "prepared batch is not a KS final-state owner";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

vibeqc_status validate_dft_system(vibeqc_method method, const core::System& system,
                                  std::string& detail) {
  if (!is_supported_dft(method)) {
    detail = "requested DFT method is reserved but not implemented";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  const char* functional =
      method == VIBEQC_METHOD_PBE_RKS || method == VIBEQC_METHOD_PBE_UKS ? "PBE" : "LDA";
  if (!is_uks(method)) {
    if (system.electron_count > 0 && system.electron_count % 2 == 0 && system.multiplicity == 1)
      return VIBEQC_STATUS_SUCCESS;
    detail = std::string(functional) +
             " RKS requires a positive even electron count and spin multiplicity 1";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  if (system.electron_count > 0 && spin_excess >= 0 && spin_excess <= system.electron_count &&
      (system.electron_count - spin_excess) % 2 == 0)
    return VIBEQC_STATUS_SUCCESS;
  detail = std::string(functional) +
           " UKS requires electron count and multiplicity to define integer nonnegative spin "
           "occupations";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

std::unique_ptr<PreparedCalculation> prepare_dft_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor) {
#if !VIBEQC_HAS_CUDA
  if (context.requested_backend == VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT CUDA backend is not built");
#endif
  if (descriptor.method != VIBEQC_METHOD_LDA_RKS && descriptor.method != VIBEQC_METHOD_PBE_RKS &&
      descriptor.method != VIBEQC_METHOD_LDA_UKS && descriptor.method != VIBEQC_METHOD_PBE_UKS)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "requested DFT method is reserved but not implemented");
  auto options = dft_options(descriptor, context.requested_backend);
  auto grid = ks_grid_options(descriptor, options);
  return std::make_unique<KsPreparedCalculation>(capabilities, system, descriptor.method,
                                                 std::move(options), std::move(grid),
                                                 context.requested_backend, context.device_id);
}

std::unique_ptr<PreparedBatch> prepare_dft_batch(const Capabilities& capabilities,
                                                 core::ContextState& context,
                                                 std::vector<core::System> systems,
                                                 const vibeqc_method_descriptor& descriptor,
                                                 vibeqc_batch_flags flags) {
  if ((flags & ~VIBEQC_BATCH_ENABLE_WARM_STARTS) != 0)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "KS batches support warm starts but not HF-specific profiling flags");
#if !VIBEQC_HAS_CUDA
  if (context.requested_backend == VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT CUDA backend is not built");
#endif
  auto options = dft_options(descriptor, context.requested_backend);
  auto grid = ks_grid_options(descriptor, options);
  return std::make_unique<KsPreparedBatch>(
      capabilities, std::move(systems), descriptor.method, std::move(options), std::move(grid),
      context.requested_backend, context.device_id, (flags & VIBEQC_BATCH_ENABLE_WARM_STARTS) != 0);
}

}  // namespace vibeqc::methods::detail
