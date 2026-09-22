"""Device-resident RCCSD iteration over the generated #148 equations.

The primary TensorIR plan owns amplitudes, physical residuals, denominators,
DIIS histories and integral inputs for the solve lifetime. Large T/R arrays do
not cross the host boundary per iteration: host control reads only energy and
physical residual maxima. Final amplitudes are downloaded once for independent
expanded-equation acceptance and the serializable result.

This is #149-B single-system solver infrastructure, not native method
registration, homogeneous batches, or a CCSD force implementation.
"""

from __future__ import annotations

import ctypes
import time
import typing
from contextlib import ExitStack
from dataclasses import asdict, fields
from hashlib import sha256
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_resident import PreparedResident, compile_resident

from tools.vibeqc_posthf.reference import ReferenceSnapshot, immutable

from .gpu_state import AmplitudeSnapshot, solver_plans
from .solver import CCSDResult, PreparedCCSD, SolverOptions


def _offsets(plan: typing.Any) -> typing.Any:
    inputs = {
        plan.steps[index].node.attrs["name"]: plan.steps[index].offset
        for index in plan.inputs
    }
    outputs = {name: plan.steps[index].offset for name, index in plan.outputs}
    return inputs, outputs


def _resident_extension(
    plan: typing.Any,
    segments: typing.Any,
    n1: typing.Any,
    n2: typing.Any,
    diis_size: typing.Any,
) -> typing.Any:
    inputs, outputs = _offsets(plan)
    required_inputs = {"t1", "t2"}
    required_outputs = {
        "correlation_energy",
        "singles_residual",
        "doubles_residual",
        "next_t1",
        "next_t2",
    }
    if not required_inputs <= inputs.keys() or not required_outputs <= outputs.keys():
        raise ValueError("resident CC plan is missing required pinned state spans")
    reserve = plan.arena_bytes + plan.panel_bytes + plan.library_bytes
    n = n1 + n2
    p1 = segments["r1_partials"]["bytes"] // 8
    p2 = segments["r2_partials"]["bytes"] // 8

    def seg(name: typing.Any) -> typing.Any:
        return reserve + segments[name]["offset"]

    return f"""// __VIBEQC_RESIDENT_POST_RUN_DECL__
#include "cuda_state.cuh"
namespace vibeqc_cc_resident {{
constexpr size_t kT1={inputs["t1"]}ULL,kT2={inputs["t2"]}ULL;
constexpr size_t kNextT1={outputs["next_t1"]}ULL,kNextT2={outputs["next_t2"]}ULL;
constexpr size_t kR1={outputs["singles_residual"]}ULL,kR2={outputs["doubles_residual"]}ULL;
constexpr size_t kEnergy={outputs["correlation_energy"]}ULL;
constexpr size_t kLastT1={seg("last_t1")}ULL,kLastT2={seg("last_t2")}ULL;
constexpr size_t kVectors={seg("diis_vectors")}ULL,kErrors={seg("diis_errors")}ULL;
constexpr size_t kGram={seg("gram")}ULL,kSystem={seg("system")}ULL;
constexpr size_t kCoefficients={seg("coefficients")}ULL;
constexpr size_t kR1Partials={seg("r1_partials")}ULL,kR2Partials={seg("r2_partials")}ULL;
constexpr size_t kScalars={seg("scalars")}ULL,kReservation={reserve}ULL;
constexpr size_t kReservationBytes={plan.reservations.diis}ULL;
constexpr long long kN1={n1}LL,kN2={n2}LL,kElements={n}LL;
constexpr int kDiis={diis_size},kR1PartialCount={p1},kR2PartialCount={p2};
inline int fail(char* error,size_t size,const char* text){{vibeqc_tensor::error_text(error,size,text);return 1;}}
inline vibeqc_tensor::Context& context(void* pointer){{if(!pointer)throw std::runtime_error("null resident CC owner");return *static_cast<vibeqc_tensor::Context*>(pointer);}}
}}
int vibeqc_resident_post_run(void* pointer,int,Metrics*,char*,size_t){{
  using namespace vibeqc_cc_resident; auto& ctx=context(pointer); auto* p=ctx.arena;
  auto* scalars=reinterpret_cast<double*>(p+kScalars);
  auto* arithmetic=reinterpret_cast<int*>(p+kScalars+2*sizeof(double)+sizeof(int));
  cuda_check(cudaMemsetAsync(arithmetic,0,sizeof(int),ctx.stream));
  vibeqc::cc::residual_partials<<<kR1PartialCount,256,0,ctx.stream>>>(reinterpret_cast<const double*>(p+kR1),kN1,reinterpret_cast<double*>(p+kR1Partials),arithmetic);
  vibeqc::cc::residual_partials<<<kR2PartialCount,256,0,ctx.stream>>>(reinterpret_cast<const double*>(p+kR2),kN2,reinterpret_cast<double*>(p+kR2Partials),arithmetic);
  cuda_check(cudaGetLastError());
  vibeqc::cc::residual_finish<<<1,1,0,ctx.stream>>>(reinterpret_cast<const double*>(p+kR1Partials),kR1PartialCount,scalars);
  vibeqc::cc::residual_finish<<<1,1,0,ctx.stream>>>(reinterpret_cast<const double*>(p+kR2Partials),kR2PartialCount,scalars+1);
  cuda_check(cudaGetLastError()); return 0;
}}
extern "C" int resident_cc_initialize(void* pointer,char* error,size_t size){{using namespace vibeqc_cc_resident;try{{auto& ctx=context(pointer);std::unique_lock<std::mutex> lock(ctx.mutex,std::try_to_lock);if(!lock.owns_lock())return fail(error,size,"resident CC owner is busy");ctx.check_device();auto* p=ctx.arena;cuda_check(cudaMemsetAsync(p+kReservation,0,kReservationBytes,ctx.stream));cuda_check(cudaMemcpyAsync(p+kLastT1,p+kT1,kN1*sizeof(double),cudaMemcpyDeviceToDevice,ctx.stream));cuda_check(cudaMemcpyAsync(p+kLastT2,p+kT2,kN2*sizeof(double),cudaMemcpyDeviceToDevice,ctx.stream));cuda_check(cudaStreamSynchronize(ctx.stream));return 0;}}catch(const std::exception& e){{return fail(error,size,e.what());}}}}
extern "C" int resident_cc_status(void* pointer,double* host,size_t count,char* error,size_t size){{using namespace vibeqc_cc_resident;try{{if(!host||count!=3)return fail(error,size,"resident CC status needs three scalars");auto& ctx=context(pointer);std::unique_lock<std::mutex> lock(ctx.mutex,std::try_to_lock);if(!lock.owns_lock())return fail(error,size,"resident CC owner is busy");ctx.check_device();auto* p=ctx.arena;int arithmetic=0;cuda_check(cudaMemcpyAsync(host,p+kEnergy,sizeof(double),cudaMemcpyDeviceToHost,ctx.stream));cuda_check(cudaMemcpyAsync(host+1,p+kScalars,2*sizeof(double),cudaMemcpyDeviceToHost,ctx.stream));cuda_check(cudaMemcpyAsync(&arithmetic,p+kScalars+2*sizeof(double)+sizeof(int),sizeof(int),cudaMemcpyDeviceToHost,ctx.stream));cuda_check(cudaStreamSynchronize(ctx.stream));if(arithmetic)return fail(error,size,"resident CC residual reduction is nonfinite");return 0;}}catch(const std::exception& e){{return fail(error,size,e.what());}}}}
extern "C" int resident_cc_advance_trial(void* pointer,char* error,size_t size){{using namespace vibeqc_cc_resident;try{{auto& ctx=context(pointer);std::unique_lock<std::mutex> lock(ctx.mutex,std::try_to_lock);if(!lock.owns_lock())return fail(error,size,"resident CC owner is busy");ctx.check_device();auto* p=ctx.arena;cuda_check(cudaMemcpyAsync(p+kLastT1,p+kT1,kN1*sizeof(double),cudaMemcpyDeviceToDevice,ctx.stream));cuda_check(cudaMemcpyAsync(p+kLastT2,p+kT2,kN2*sizeof(double),cudaMemcpyDeviceToDevice,ctx.stream));cuda_check(cudaMemcpyAsync(p+kT1,p+kNextT1,kN1*sizeof(double),cudaMemcpyDeviceToDevice,ctx.stream));cuda_check(cudaMemcpyAsync(p+kT2,p+kNextT2,kN2*sizeof(double),cudaMemcpyDeviceToDevice,ctx.stream));return 0;}}catch(const std::exception& e){{return fail(error,size,e.what());}}}}
extern "C" int resident_cc_diis(void* pointer,int history_count,int* new_count,int* restarts_added,int* solve_attempts,char* error,size_t size){{using namespace vibeqc_cc_resident;try{{if(!new_count||!restarts_added||!solve_attempts||history_count<0||history_count>kDiis)return fail(error,size,"invalid resident CC DIIS state");*new_count=history_count;*restarts_added=0;*solve_attempts=0;auto& ctx=context(pointer);std::unique_lock<std::mutex> lock(ctx.mutex,std::try_to_lock);if(!lock.owns_lock())return fail(error,size,"resident CC owner is busy");ctx.check_device();auto* p=ctx.arena;if constexpr(kDiis==0)return 0;auto* vectors=reinterpret_cast<double*>(p+kVectors);auto* errors=reinterpret_cast<double*>(p+kErrors);auto* status=reinterpret_cast<int*>(p+kScalars+2*sizeof(double));auto* arithmetic=status+1;int count=history_count;if(count==kDiis){{vibeqc::cc::history_shift<<<vibeqc_tensor::blocks(kElements,256),256,0,ctx.stream>>>(vectors,kElements,count);vibeqc::cc::history_shift<<<vibeqc_tensor::blocks(kElements,256),256,0,ctx.stream>>>(errors,kElements,count);cuda_check(cudaGetLastError());--count;}}const int slot=count++;cuda_check(cudaMemcpyAsync(vectors+static_cast<long long>(slot)*kElements,p+kT1,kN1*sizeof(double),cudaMemcpyDeviceToDevice,ctx.stream));cuda_check(cudaMemcpyAsync(vectors+static_cast<long long>(slot)*kElements+kN1,p+kT2,kN2*sizeof(double),cudaMemcpyDeviceToDevice,ctx.stream));cuda_check(cudaMemcpyAsync(errors+static_cast<long long>(slot)*kElements,p+kR1,kN1*sizeof(double),cudaMemcpyDeviceToDevice,ctx.stream));cuda_check(cudaMemcpyAsync(errors+static_cast<long long>(slot)*kElements+kN1,p+kR2,kN2*sizeof(double),cudaMemcpyDeviceToDevice,ctx.stream));while(count>1){{vibeqc::cc::diis_gram(ctx,errors,static_cast<int>(kElements),count,reinterpret_cast<double*>(p+kGram));vibeqc::cc::diis_coefficients<<<1,1,0,ctx.stream>>>(reinterpret_cast<const double*>(p+kGram),count,reinterpret_cast<double*>(p+kSystem),reinterpret_cast<double*>(p+kCoefficients),status);cuda_check(cudaGetLastError());int host_status=1;cuda_check(cudaMemcpyAsync(&host_status,status,sizeof(int),cudaMemcpyDeviceToHost,ctx.stream));cuda_check(cudaStreamSynchronize(ctx.stream));++*solve_attempts;if(host_status==2)break;if(!host_status){{cuda_check(cudaMemsetAsync(arithmetic,0,sizeof(int),ctx.stream));vibeqc::cc::diis_combine_slice<<<vibeqc_tensor::blocks(kN1,256),256,0,ctx.stream>>>(vectors,reinterpret_cast<const double*>(p+kCoefficients),kElements,0,kN1,count,status,reinterpret_cast<double*>(p+kT1),arithmetic);vibeqc::cc::diis_combine_slice<<<vibeqc_tensor::blocks(kN2,256),256,0,ctx.stream>>>(vectors,reinterpret_cast<const double*>(p+kCoefficients),kElements,kN1,kN2,count,status,reinterpret_cast<double*>(p+kT2),arithmetic);cuda_check(cudaGetLastError());int host_arithmetic=0;cuda_check(cudaMemcpyAsync(&host_arithmetic,arithmetic,sizeof(int),cudaMemcpyDeviceToHost,ctx.stream));cuda_check(cudaStreamSynchronize(ctx.stream));if(host_arithmetic)return fail(error,size,"resident CC DIIS extrapolation is nonfinite");break;}}vibeqc::cc::history_shift<<<vibeqc_tensor::blocks(kElements,256),256,0,ctx.stream>>>(vectors,kElements,count);vibeqc::cc::history_shift<<<vibeqc_tensor::blocks(kElements,256),256,0,ctx.stream>>>(errors,kElements,count);cuda_check(cudaGetLastError());--count;++*restarts_added;}}*new_count=count;return 0;}}catch(const std::exception& e){{return fail(error,size,e.what());}}}}
extern "C" int resident_cc_download_amplitudes(void* pointer,int last,double* t1,size_t n1,double* t2,size_t n2,char* error,size_t size){{using namespace vibeqc_cc_resident;try{{if(!t1||!t2||n1!=kN1||n2!=kN2)return fail(error,size,"resident CC amplitude download shape mismatch");auto& ctx=context(pointer);std::unique_lock<std::mutex> lock(ctx.mutex,std::try_to_lock);if(!lock.owns_lock())return fail(error,size,"resident CC owner is busy");ctx.check_device();auto* p=ctx.arena;cuda_check(cudaMemcpyAsync(t1,p+(last?kLastT1:kT1),kN1*sizeof(double),cudaMemcpyDeviceToHost,ctx.stream));cuda_check(cudaMemcpyAsync(t2,p+(last?kLastT2:kT2),kN2*sizeof(double),cudaMemcpyDeviceToHost,ctx.stream));cuda_check(cudaStreamSynchronize(ctx.stream));return 0;}}catch(const std::exception& e){{return fail(error,size,e.what());}}}}
"""


class _ResidentCCOwner(PreparedResident):
    def __init__(
        self,
        plan: typing.Any,
        artifact: typing.Any,
        *,
        n1: typing.Any,
        n2: typing.Any,
        diis_size: typing.Any,
        device: typing.Any = 0,
    ) -> None:
        super().__init__(plan, artifact, device=device)
        self._n1, self._n2, self._diis_size = n1, n2, diis_size
        self._t1_shape = plan.steps[plan.inputs[self._names["t1"]]].node.spec.shape
        self._t2_shape = plan.steps[plan.inputs[self._names["t2"]]].node.spec.shape
        self._history_count = self._restarts = 0
        self.control_transfers = {
            "scalar_d2h_bytes": 0,
            "amplitude_d2h_bytes": 0,
            "scalar_synchronizations": 0,
            "amplitude_synchronizations": 0,
            "diis_attempts": 0,
        }
        lib = self._library
        lib.resident_cc_initialize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.resident_cc_initialize.restype = ctypes.c_int
        lib.resident_cc_status.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.resident_cc_status.restype = ctypes.c_int
        lib.resident_cc_advance_trial.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.resident_cc_advance_trial.restype = ctypes.c_int
        lib.resident_cc_diis.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.resident_cc_diis.restype = ctypes.c_int
        lib.resident_cc_download_amplitudes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.resident_cc_download_amplitudes.restype = ctypes.c_int

    def _call_cc(self, name: typing.Any, *args: typing.Any) -> None:
        error = ctypes.create_string_buffer(2048)
        if getattr(self._library, name)(self._pointer, *args, error, len(error)):
            raise RuntimeError(error.value.decode())

    def initialize_cc(self) -> None:
        with self._lock:
            self._call_cc("resident_cc_initialize")
            self._history_count = self._restarts = 0
            self._invalidate()

    def status(self) -> typing.Any:
        with self._lock:
            if not self._ready:
                raise RuntimeError(
                    "resident CC status requires a completed current-state run"
                )
            values = np.empty(3, dtype=np.float64)
            self._call_cc(
                "resident_cc_status",
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                values.size,
            )
            if not np.isfinite(values).all():
                raise FloatingPointError("nonfinite resident CC scalar status")
            self.control_transfers["scalar_d2h_bytes"] += values.nbytes + 4
            self.control_transfers["scalar_synchronizations"] += 1
            return tuple(float(v) for v in values)

    def advance_trial(self) -> None:
        with self._lock:
            if not self._ready:
                raise RuntimeError(
                    "resident CC trial advance requires a completed current-state run"
                )
            self._call_cc("resident_cc_advance_trial")
            self._invalidate()

    def diis_update(self) -> None:
        with self._lock:
            if not self._ready:
                raise RuntimeError(
                    "resident CC DIIS requires a completed trial-state run"
                )
            new_count, restarts, attempts = (
                ctypes.c_int(),
                ctypes.c_int(),
                ctypes.c_int(),
            )
            self._call_cc(
                "resident_cc_diis",
                self._history_count,
                ctypes.byref(new_count),
                ctypes.byref(restarts),
                ctypes.byref(attempts),
            )
            self._history_count = new_count.value
            self._restarts += restarts.value
            self.control_transfers["diis_attempts"] += attempts.value
            self.control_transfers["scalar_d2h_bytes"] += 4 * attempts.value
            self.control_transfers["scalar_synchronizations"] += attempts.value
            if attempts.value and self._history_count > 1:
                self.control_transfers["scalar_d2h_bytes"] += 4
                self.control_transfers["scalar_synchronizations"] += 1
            self._invalidate()

    def amplitudes(self, *, last: typing.Any = False) -> typing.Any:
        with self._lock:
            t1 = np.empty(self._t1_shape, dtype=np.float64)
            t2 = np.empty(self._t2_shape, dtype=np.float64)
            self._call_cc(
                "resident_cc_download_amplitudes",
                int(last),
                t1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                t1.size,
                t2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                t2.size,
            )
            self.control_transfers["amplitude_d2h_bytes"] += t1.nbytes + t2.nbytes
            self.control_transfers["amplitude_synchronizations"] += 1
            return immutable(t1), immutable(t2)


class PreparedResidentCCSD:
    """Own one resident physical iteration plan plus independent replay."""

    def __init__(
        self,
        snapshot: typing.Any,
        provider: typing.Any,
        compiler: typing.Any,
        cache: typing.Any,
        *,
        options: typing.Any = None,
        t1: typing.Any = None,
        t2: typing.Any = None,
        warm_start: typing.Any = None,
        device: typing.Any = 0,
        provider_peak_bytes: typing.Any = 0,
    ) -> None:
        if not isinstance(compiler, CudaCompilerAdapter):
            raise TypeError("resident GPU RCCSD requires a CudaCompilerAdapter")
        if not isinstance(cache, Path):
            raise TypeError("resident GPU RCCSD cache must be a pathlib.Path")
        if not isinstance(snapshot, ReferenceSnapshot):
            raise TypeError("resident CCSD requires a validated RHF ReferenceSnapshot")
        if type(device) is not int or device < 0:
            raise ValueError("device must be a nonnegative visible CUDA ordinal")
        self.options = SolverOptions() if options is None else options
        if warm_start is not None:
            if not isinstance(warm_start, AmplitudeSnapshot):
                raise TypeError("resident CC warm_start must be AmplitudeSnapshot")
            if t1 is not None or t2 is not None:
                raise ValueError(
                    "resident CC warm_start cannot be combined with raw t1/t2"
                )
            t1, t2 = warm_start.for_reference(snapshot)
        self.snapshot, self.device = snapshot, device
        o, v = snapshot.nocc, snapshot.nmo - snapshot.nocc
        primary, replay, diagnostic = solver_plans(
            o, v, compiler.target, self.options, provider_peak_bytes=provider_peak_bytes
        )
        self.cpu = PreparedCCSD(snapshot, provider, self.options, t1, t2)
        n1, n2 = (layout.spec.size for layout in self.cpu.layouts)
        extension = _resident_extension(
            primary, diagnostic["state_segments"], n1, n2, self.options.diis_size
        )
        dependency = asset_path("src/cc/cuda_state.cuh")
        with ExitStack() as cleanup:
            self.primary = cleanup.enter_context(
                _ResidentCCOwner(
                    primary,
                    compile_resident(
                        primary,
                        compiler,
                        cache,
                        extension=extension,
                        dependencies=(dependency,),
                    ),
                    n1=n1,
                    n2=n2,
                    diis_size=self.options.diis_size,
                    device=device,
                )
            )
            self.replay = cleanup.enter_context(
                PreparedCuda(
                    replay, compile_cuda(replay, compiler, cache), device=device
                )
            )
            feeds = {
                **self.cpu.feeds,
                "t1": self.cpu.initial[0],
                "t2": self.cpu.initial[1],
                "d1": self.cpu.denominators[0],
                "d2": self.cpu.denominators[1],
            }
            self.primary.upload(feeds)
            self.primary.initialize_cc()
            cleanup.pop_all()
        self.diagnostic = diagnostic
        self._static_upload_bytes = sum(
            np.asarray(value).nbytes for value in feeds.values()
        )
        self.owner_identity = canonical_hash(
            {
                "reference": snapshot.identity,
                "integrals": self.cpu.integral_hash,
                "iteration": primary.program.logical_hash,
                "replay": replay.program.logical_hash,
                "resident_artifact": self.primary.artifact.metadata["key"],
                "device": device,
            }
        )
        # Compatibility name means immutable owner identity, never a claim that
        # the current amplitudes have passed convergence/replay qualification.
        self.state_identity = self.owner_identity
        self._solved_state_identity = None

    def _run(self) -> typing.Any:
        try:
            return self.primary.run()
        except RuntimeError as error:
            if str(error).startswith(
                ("non-finite tensor at step ", "tensor division by zero at step ")
            ):
                raise FloatingPointError(str(error)) from error
            raise

    def current_status(self) -> typing.Any:
        _, metrics = self._run()
        return (*self.primary.status(), metrics)

    def amplitudes(self, *, last: typing.Any = False) -> typing.Any:
        return self.primary.amplitudes(last=last)

    def amplitude_snapshot(self) -> typing.Any:
        """Detach the current amplitudes with exact-reference warm-start identity."""
        return AmplitudeSnapshot(self.snapshot.identity, *self.amplitudes())

    def independent(self, t1: typing.Any, t2: typing.Any) -> typing.Any:
        return self.replay.execute({**self.cpu.feeds, "t1": t1, "t2": t2}).outputs

    @property
    def solved_state_identity(self) -> typing.Any:
        """Exact qualified amplitude state; absent during mutation/nonconvergence."""
        if self._solved_state_identity is None:
            raise RuntimeError("resident CC owner has no qualified solved state")
        return self._solved_state_identity

    def close(self) -> None:
        self.primary.close()
        self.replay.close()

    def __enter__(self) -> typing.Any:
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()

    def solve(self) -> typing.Any:
        # Any new solve may mutate amplitudes; downstream solved-state bindings
        # fail closed until a fresh expanded replay qualifies the final T again.
        self._solved_state_identity = None
        options, history, previous_energy = self.options, [], None
        status, reason = "not_converged", "maximum CCSD iterations reached"
        energy, final = None, None
        start = time.perf_counter()
        for iteration in range(options.max_iterations + 1):
            try:
                energy, r1_max, r2_max, _ = self.current_status()
                residual = max(r1_max, r2_max)
                delta = (
                    None if previous_energy is None else abs(energy - previous_energy)
                )
                row = {
                    "iteration": iteration,
                    "correlation_energy": energy,
                    "energy_change": delta,
                    "r1_max": r1_max,
                    "r2_max": r2_max,
                    "elapsed_seconds": time.perf_counter() - start,
                    "backend": "cuda-fp64-resident",
                }
                history.append(row)
                if (
                    delta is not None
                    and delta <= options.energy_tolerance
                    and residual <= options.residual_tolerance
                ):
                    current = self.amplitudes()
                    independent = self.independent(*current)
                    row["independent_r1_max"] = float(
                        np.max(np.abs(independent["singles_residual"]))
                    )
                    row["independent_r2_max"] = float(
                        np.max(np.abs(independent["doubles_residual"]))
                    )
                    if (
                        max(row["independent_r1_max"], row["independent_r2_max"])
                        <= options.residual_tolerance
                        and abs(float(independent["correlation_energy"]) - energy)
                        <= options.energy_tolerance
                    ):
                        status, reason, final = (
                            "converged",
                            "energy change and freshly expanded physical R1/R2 passed on GPU",
                            current,
                        )
                        break
                if iteration == options.max_iterations:
                    final = self.amplitudes()
                    break
                self.primary.advance_trial()
                self._run()  # trial residual stays device resident
                self.primary.diis_update()
                previous_energy = energy
            except (FloatingPointError, OverflowError) as error:
                status, reason, final = (
                    "nonfinite",
                    str(error),
                    self.amplitudes(last=True),
                )
                break
            except RuntimeError as error:
                if (
                    "nonfinite" not in str(error).lower()
                    and "non-finite" not in str(error).lower()
                ):
                    raise
                status, reason, final = (
                    "nonfinite",
                    str(error),
                    self.amplitudes(last=True),
                )
                break
        if final is None:
            final = self.amplitudes(last=status == "nonfinite")
        if status == "converged":
            self._solved_state_identity = canonical_hash(
                {
                    "owner": self.owner_identity,
                    "t1_sha256": sha256(
                        np.ascontiguousarray(final[0]).tobytes()
                    ).hexdigest(),
                    "t2_sha256": sha256(
                        np.ascontiguousarray(final[1]).tobytes()
                    ).hexdigest(),
                    "correlation_energy": energy,
                    "qualification": "expanded-physical-replay",
                }
            )
        transfers = {
            **dict(self.primary.transfers),
            "scalar_control_d2h_bytes": self.primary.control_transfers[
                "scalar_d2h_bytes"
            ],
            "scalar_control_synchronizations": self.primary.control_transfers[
                "scalar_synchronizations"
            ],
            "amplitude_d2h_bytes_total": self.primary.control_transfers[
                "amplitude_d2h_bytes"
            ],
            "amplitude_synchronizations": self.primary.control_transfers[
                "amplitude_synchronizations"
            ],
            "diis_solve_attempts": self.primary.control_transfers["diis_attempts"],
            "initial_large_h2d_bytes": self._static_upload_bytes,
            "per_iteration_large_h2d_bytes": 0,
            "per_iteration_large_d2h_bytes": 0,
            "final_amplitude_d2h_bytes": final[0].nbytes + final[1].nbytes,
        }
        provenance = {
            "reference_id": self.snapshot.identity,
            "hamiltonian_id": self.snapshot.hamiltonian_id,
            "integral_hash": self.cpu.integral_hash,
            "equation_hash": self.primary.plan.program.provenance.get(
                "physical_equation"
            ),
            "iteration_equation_hash": self.primary.plan.program.logical_hash,
            "independent_equation_hash": self.replay.plan.program.logical_hash,
            "options": asdict(options),
            "diis_restarts": self.primary._restarts,
            "logical_required_bytes": self.cpu.logical_required_bytes,
            "reference_energy": self.snapshot.reference_energy,
            "backend": "cuda-fp64-resident",
            "device": self.device,
            "resident_owner_identity": self.owner_identity,
            "resident_state_identity": self.owner_identity,
            "resident_solved_state_identity": self._solved_state_identity,
            "primary_peak_bytes": self.primary.plan.peak_bytes,
            "independent_replay_peak_bytes": self.replay.plan.peak_bytes,
            "combined_peak_bytes": self.diagnostic["combined_peak_bytes"],
            "transfer": transfers,
            "residency": "T/R/integrals/denominators/DIIS remain device resident across iterations; host reads scalar convergence control only",
            "replay": "expanded physical TensorIR executes on GPU after one final amplitude download",
            "solver_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        replay = {k: v.tolist() for k, v in self.cpu.feeds.items()}
        replay.update(
            initial_t1=self.cpu.initial[0].tolist(),
            initial_t2=self.cpu.initial[1].tolist(),
            orbital_energies=self.snapshot.orbital_energies.tolist(),
            snapshot={
                field.name: (
                    getattr(self.snapshot, field.name).tolist()
                    if isinstance(getattr(self.snapshot, field.name), np.ndarray)
                    else getattr(self.snapshot, field.name)
                )
                for field in fields(self.snapshot)
                if field.init
            },
        )
        return CCSDResult(
            status,
            reason,
            energy,
            None if energy is None else self.snapshot.reference_energy + energy,
            immutable(final[0]),
            immutable(final[1]),
            tuple(history),
            provenance,
            replay,
        )


def solve_gpu_resident(
    snapshot: typing.Any,
    provider: typing.Any,
    *,
    compiler: typing.Any,
    cache: typing.Any,
    options: typing.Any = None,
    t1: typing.Any = None,
    t2: typing.Any = None,
    warm_start: typing.Any = None,
    device: typing.Any = 0,
    provider_peak_bytes: typing.Any = 0,
) -> typing.Any:
    """Convenience owner for the #149-B resident single-system solver."""
    with PreparedResidentCCSD(
        snapshot,
        provider,
        compiler,
        cache,
        options=options,
        t1=t1,
        t2=t2,
        warm_start=warm_start,
        device=device,
        provider_peak_bytes=provider_peak_bytes,
    ) as prepared:
        return prepared.solve()
