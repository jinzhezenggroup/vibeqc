"""Optional OpenCL 1.2 compiler/runtime adapter using the installed vendor ICD.

Importing this module requires no SDK, creates no context and queries no device.
The selected GPU executes ordinary host-submitted kernels. No numerical CPU
fallback, CUDA syntax rewriting, graph emulation or implicit remote binary load
is provided. The manual execution gate must run inside a Slurm GPU allocation.
"""

from __future__ import annotations

import ctypes as c
import ctypes.util
import re
from dataclasses import dataclass, field

from .backend import TargetInfo
from .runtime_backend import (
    ExecutionShape,
    RuntimeCapabilities,
    UnsupportedBackendFeature,
    UnsupportedLibraryProvider,
)


class OpenCLError(RuntimeError):
    """Retain the vendor error code and complete build log without reclassification."""

    def __init__(self, operation, code, log=""):
        self.operation, self.code, self.log = operation, int(code), log
        super().__init__(
            f"{operation} failed with OpenCL status {code}"
            + (f":\n{log}" if log else "")
        )


@dataclass(eq=False, frozen=True)
class Resource:
    """An immutable typed native handle; events retain submitted resources.

    Caller mutation of a kind/handle could send a buffer pointer to a queue API.
    Only the owning runtime updates completion and invalidates released handles.
    """

    kind: str
    handle: int
    owner: object = field(repr=False)
    nbytes: int = 0
    dependencies: tuple[Resource, ...] = ()
    completed: bool = False


@dataclass(frozen=True)
class LocalMemory:
    """Dynamic per-workgroup memory passed as a local-address-space argument."""

    nbytes: int


def _check(code, operation, log=""):
    if code:
        raise OpenCLError(operation, code, log)


class OpenCLRuntime:
    """One selected GPU/context with explicit compile/link, queues, memory and events.

    With multiple visible OpenCL GPUs, callers must select an exact queried
    UUID. OpenCL ICDs need not honor CUDA_VISIBLE_DEVICES, so the manual Slurm
    runner additionally rejects ambiguous device inventories. This adapter
    preserves the environment and never silently selects a CPU OpenCL device.
    """

    def __init__(self, *, library=None, device_uuid=None):
        path = library or ctypes_library()
        self.api = c.CDLL(path)
        self.library = path
        self._resources = {}
        self._context = None
        self._closed = False
        self._bind()
        inventory = self._devices()
        selected = [
            row
            for row in inventory
            if device_uuid is None or row["uuid"] == device_uuid
        ]
        if len(selected) != 1:
            raise UnsupportedBackendFeature(
                "select one exact OpenCL GPU UUID; inventory is ambiguous or empty"
            )
        self.device = selected[0]
        self._device = c.c_void_p(self.device["handle"])
        if not self.device["compiler_available"] or not self.device["linker_available"]:
            raise UnsupportedBackendFeature(
                "selected OpenCL device has no online compiler/linker"
            )
        error = c.c_int()
        self._context = self.api.clCreateContext(
            None, 1, c.byref(self._device), None, None, c.byref(error)
        )
        _check(error.value, "clCreateContext")
        try:
            self.queue = self.create_stream()
        except BaseException:
            self.close()
            raise

    def _bind(self):
        """Declare OpenCL 1.2 ABI signatures, using pointer-sized handles explicitly."""
        P, U, I, S, B = c.c_void_p, c.c_uint, c.c_int, c.c_size_t, c.c_uint64
        PP, UP, IP, SP = c.POINTER(P), c.POINTER(U), c.POINTER(I), c.POINTER(S)
        declarations = {
            "clGetPlatformIDs": (I, [U, PP, UP]),
            "clGetDeviceIDs": (I, [P, B, U, PP, UP]),
            "clGetDeviceInfo": (I, [P, U, S, P, SP]),
            "clCreateContext": (P, [P, U, PP, P, P, IP]),
            "clCreateCommandQueue": (P, [P, P, B, IP]),
            "clCreateBuffer": (P, [P, B, S, P, IP]),
            "clEnqueueWriteBuffer": (I, [P, P, U, S, S, P, U, PP, PP]),
            "clEnqueueReadBuffer": (I, [P, P, U, S, S, P, U, PP, PP]),
            "clCreateProgramWithSource": (P, [P, U, c.POINTER(c.c_char_p), SP, IP]),
            "clCompileProgram": (I, [P, U, PP, c.c_char_p, U, PP, P, P, P]),
            "clLinkProgram": (P, [P, U, PP, c.c_char_p, U, PP, P, P, IP]),
            "clGetProgramBuildInfo": (I, [P, P, U, S, P, SP]),
            "clGetProgramInfo": (I, [P, U, S, P, SP]),
            "clCreateProgramWithBinary": (P, [P, U, PP, SP, PP, IP, IP]),
            "clBuildProgram": (I, [P, U, PP, c.c_char_p, P, P]),
            "clCreateKernel": (P, [P, c.c_char_p, IP]),
            "clSetKernelArg": (I, [P, U, S, P]),
            "clEnqueueNDRangeKernel": (I, [P, P, U, SP, SP, SP, U, PP, PP]),
            "clGetKernelWorkGroupInfo": (I, [P, P, U, S, P, SP]),
            "clWaitForEvents": (I, [U, PP]),
            "clGetEventProfilingInfo": (I, [P, U, S, P, SP]),
            "clFinish": (I, [P]),
            "clFlush": (I, [P]),
            **{
                f"clRelease{name}": (I, [P])
                for name in (
                    "Context",
                    "CommandQueue",
                    "MemObject",
                    "Program",
                    "Kernel",
                    "Event",
                )
            },
        }
        for name, (result, args) in declarations.items():
            function = getattr(self.api, name)
            function.restype, function.argtypes = result, args

    def _device_info(self, device, key, scalar=None):
        size = c.c_size_t()
        _check(
            self.api.clGetDeviceInfo(device, key, 0, None, c.byref(size)),
            "clGetDeviceInfo",
        )
        value = c.create_string_buffer(size.value)
        _check(
            self.api.clGetDeviceInfo(device, key, size, value, None), "clGetDeviceInfo"
        )
        return (
            scalar.from_buffer(value).value
            if scalar
            else value.raw.rstrip(b"\0").decode()
        )

    def _devices(self):
        count = c.c_uint()
        _check(self.api.clGetPlatformIDs(0, None, c.byref(count)), "clGetPlatformIDs")
        platforms = (c.c_void_p * count.value)()
        _check(self.api.clGetPlatformIDs(count, platforms, None), "clGetPlatformIDs")
        rows = []
        for platform in platforms:
            total = c.c_uint()
            code = self.api.clGetDeviceIDs(
                platform, 4, 0, None, c.byref(total)
            )  # CL_DEVICE_TYPE_GPU
            if code == -1:
                continue
            _check(code, "clGetDeviceIDs")
            devices = (c.c_void_p * total.value)()
            _check(
                self.api.clGetDeviceIDs(platform, 4, total, devices, None),
                "clGetDeviceIDs",
            )
            for device in devices:
                info = {
                    name: self._device_info(device, key)
                    for name, key in (
                        ("name", 0x102B),
                        ("vendor", 0x102C),
                        ("driver", 0x102D),
                        ("version", 0x102F),
                        ("language", 0x103D),
                        ("extensions", 0x1030),
                    )
                }
                for name, key, scalar in (
                    ("compiler_available", 0x1028, c.c_uint),
                    ("linker_available", 0x103E, c.c_uint),
                    ("maximum_workgroup_threads", 0x1004, c.c_size_t),
                    ("local_memory_bytes", 0x1023, c.c_uint64),
                    ("maximum_allocation_bytes", 0x1010, c.c_uint64),
                ):
                    info[name] = self._device_info(device, key, scalar)
                extensions = set(info["extensions"].split())
                info["fp64"] = (
                    "cl_khr_fp64" in extensions
                    and self._device_info(device, 0x1032, c.c_uint64) != 0
                )
                info["subgroup_size"] = (
                    self._device_info(device, 0x4003, c.c_uint)
                    if "cl_nv_device_attribute_query" in extensions
                    else None
                )
                info["uuid"] = None
                if "cl_khr_device_uuid" in extensions:
                    uuid = (c.c_ubyte * 16)()
                    _check(
                        self.api.clGetDeviceInfo(device, 0x106A, 16, uuid, None),
                        "device UUID",
                    )
                    info["uuid"] = bytes(uuid).hex()
                info["handle"] = device
                rows.append(info)
        return rows

    def capabilities(self):
        return RuntimeCapabilities(
            "opencl",
            self.device["fp64"],
            self.device["maximum_workgroup_threads"],
            self.device["local_memory_bytes"],
            self.device["subgroup_size"],
        )

    def target_info(self):
        """Expose queried limits through the established generic target contract."""
        return TargetInfo(
            "opencl",
            self.device["name"],
            self.device["subgroup_size"],
            self.device["maximum_workgroup_threads"],
            None,
        )

    def library_provider(self):
        """OpenCL core supplies no BLAS/eigensolver/factorization provider."""
        return UnsupportedLibraryProvider(
            "opencl", "no native GEMM/eigensolver/Cholesky library adapter configured"
        )

    def _own(self, kind, handle, nbytes=0, dependencies=()):
        if not handle:
            raise OpenCLError(f"create {kind}", -1, "vendor returned a null handle")
        resource = Resource(kind, int(handle), self, nbytes, tuple(dependencies))
        self._resources[id(resource)] = resource
        return resource

    def _require(self, resource, kind):
        if (
            self._closed
            or not isinstance(resource, Resource)
            or resource.owner is not self
            or not resource.handle
            or resource.kind != kind
            or self._resources.get(id(resource)) is not resource
        ):
            raise ValueError(f"live {kind} from this OpenCL context required")
        return resource.handle

    def create_stream(self):
        """Create an in-order queue with native event profiling enabled."""
        if self._closed or not self._context:
            raise ValueError("OpenCL context is closed")
        error = c.c_int()
        handle = self.api.clCreateCommandQueue(
            self._context, self._device, 2, c.byref(error)
        )
        _check(error.value, "clCreateCommandQueue")
        return self._own("queue", handle)

    def allocate(self, nbytes):
        """Allocate on the selected GPU, enforcing the queried per-buffer limit."""
        if (
            self._closed
            or type(nbytes) is not int
            or not 0 < nbytes <= self.device["maximum_allocation_bytes"]
        ):
            raise ValueError(
                "positive allocation within the live device limit required"
            )
        error = c.c_int()
        handle = self.api.clCreateBuffer(self._context, 1, nbytes, None, c.byref(error))
        _check(error.value, "clCreateBuffer")
        return self._own("buffer", handle, nbytes)

    def _bounds(self, buffer, offset, nbytes):
        handle = self._require(buffer, "buffer")
        if (
            type(offset) is not int
            or type(nbytes) is not int
            or min(offset, nbytes) < 0
            or offset > buffer.nbytes
            or nbytes > buffer.nbytes - offset
        ):
            raise ValueError("transfer exceeds the owned buffer")
        return handle

    def write(self, buffer, data, *, offset=0, stream=None):
        """A blocking transfer owns the host input until the ICD has consumed it."""
        data = bytes(data)
        handle = self._bounds(buffer, offset, len(data))
        queue = self._require(stream or self.queue, "queue")
        if data:
            host = c.create_string_buffer(data)
            _check(
                self.api.clEnqueueWriteBuffer(
                    queue, handle, 1, offset, len(data), host, 0, None, None
                ),
                "clEnqueueWriteBuffer",
            )

    def read(self, buffer, nbytes, *, offset=0, stream=None):
        handle = self._bounds(buffer, offset, nbytes)
        queue = self._require(stream or self.queue, "queue")
        if not nbytes:
            return b""
        host = c.create_string_buffer(nbytes)
        _check(
            self.api.clEnqueueReadBuffer(
                queue, handle, 1, offset, nbytes, host, 0, None, None
            ),
            "clEnqueueReadBuffer",
        )
        return host.raw

    def build_log(self, program):
        handle = self._require(program, "program")
        size = c.c_size_t()
        _check(
            self.api.clGetProgramBuildInfo(
                handle, self._device, 0x1183, 0, None, c.byref(size)
            ),
            "build log size",
        )
        value = c.create_string_buffer(size.value)
        _check(
            self.api.clGetProgramBuildInfo(
                handle, self._device, 0x1183, size, value, None
            ),
            "build log",
        )
        return value.value.decode(errors="replace")

    def compile(self, source, *, options=("-cl-std=CL1.2",)):
        """Use clCompileProgram and preserve the vendor compiler's full diagnostics."""
        if self._closed or not isinstance(source, str) or not source:
            raise ValueError("nonempty source and a live context required")
        data = source.encode()
        text, length, error = c.c_char_p(data), c.c_size_t(len(data)), c.c_int()
        handle = self.api.clCreateProgramWithSource(
            self._context, 1, c.byref(text), c.byref(length), c.byref(error)
        )
        _check(error.value, "clCreateProgramWithSource")
        program = self._own("program", handle)
        code = self.api.clCompileProgram(
            handle,
            1,
            c.byref(self._device),
            " ".join(options).encode(),
            0,
            None,
            None,
            None,
            None,
        )
        if code:
            log = self.build_log(program)
            self.release(program)
            raise OpenCLError("clCompileProgram", code, log)
        return program

    def link(self, programs):
        """Link compatible compiled objects in this same live context/device."""
        if not programs:
            raise ValueError("at least one compiled program required")
        handles = (c.c_void_p * len(programs))(
            *(self._require(p, "program") for p in programs)
        )
        error = c.c_int()
        handle = self.api.clLinkProgram(
            self._context,
            1,
            c.byref(self._device),
            b"",
            len(programs),
            handles,
            None,
            None,
            c.byref(error),
        )
        if error.value:
            log = ""
            if handle:
                failed = self._own("program", handle)
                log = self.build_log(failed)
                self.release(failed)
            raise OpenCLError("clLinkProgram", error.value, log)
        return self._own("program", handle)

    def binary(self, program):
        """Export this context's single-device executable for an explicit local cache."""
        handle = self._require(program, "program")
        size = c.c_size_t()
        _check(
            self.api.clGetProgramInfo(
                handle, 0x1165, c.sizeof(size), c.byref(size), None
            ),
            "program binary size",
        )
        if not 0 < size.value <= 64 << 20:
            raise ValueError(
                "program binary is empty or exceeds the bounded cache format"
            )
        data = c.create_string_buffer(size.value)
        buffers = (c.c_void_p * 1)(c.cast(data, c.c_void_p).value)
        _check(
            self.api.clGetProgramInfo(handle, 0x1166, c.sizeof(buffers), buffers, None),
            "program binary",
        )
        return data.raw

    def load_binary(self, binary, identity):
        """Load explicitly supplied, validated local bytes only for this runtime identity.

        The cache caller must first verify the planned scientific/source/schedule
        identity. Runtime checks additionally prevent loading an artifact into
        a different ICD, language, driver or device. No remote profile is fetched.
        """
        device = self.device
        expected = (
            "opencl",
            device["uuid"] or device["vendor"] + "/" + device["name"],
            device["version"],
            "vendor ICD online compiler " + device["driver"],
            device["driver"],
            device["language"],
        )
        actual = (
            identity.backend,
            identity.device,
            identity.runtime,
            identity.compiler,
            identity.driver,
            identity.language,
        )
        if self._closed or actual != expected:
            raise ValueError("binary is incompatible with this live OpenCL runtime")
        if not isinstance(binary, bytes) or not 0 < len(binary) <= 64 << 20:
            raise ValueError("bounded nonempty binary bytes required")
        data = c.create_string_buffer(binary)
        pointers = (c.c_void_p * 1)(c.cast(data, c.c_void_p).value)
        size, status, error = c.c_size_t(len(binary)), c.c_int(), c.c_int()
        handle = self.api.clCreateProgramWithBinary(
            self._context,
            1,
            c.byref(self._device),
            c.byref(size),
            pointers,
            c.byref(status),
            c.byref(error),
        )
        if error.value or status.value:
            if handle:
                self.api.clReleaseProgram(handle)
            raise OpenCLError("clCreateProgramWithBinary", error.value or status.value)
        program = self._own("program", handle)
        code = self.api.clBuildProgram(
            handle,
            1,
            c.byref(self._device),
            " ".join(identity.options).encode(),
            None,
            None,
        )
        if code:
            log = self.build_log(program)
            self.release(program)
            raise OpenCLError("clBuildProgram(binary)", code, log)
        return program

    def _kernel(self, program, name):
        self._require(program, "program")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise ValueError("kernel name must be an identifier")
        error = c.c_int()
        handle = self.api.clCreateKernel(program.handle, name.encode(), c.byref(error))
        _check(error.value, "clCreateKernel")
        return self._own("kernel", handle)

    def resources(self, program, name):
        """Report only resources exposed by OpenCL; registers/spills remain unknown."""
        kernel = self._kernel(program, name)
        result = {}
        try:
            for label, key, scalar in (
                ("maximum_workgroup_threads", 0x11B0, c.c_size_t),
                ("local_bytes", 0x11B2, c.c_uint64),
                ("preferred_workgroup_multiple", 0x11B3, c.c_size_t),
                ("private_bytes", 0x11B4, c.c_uint64),
            ):
                value = scalar()
                _check(
                    self.api.clGetKernelWorkGroupInfo(
                        kernel.handle,
                        self._device,
                        key,
                        c.sizeof(value),
                        c.byref(value),
                        None,
                    ),
                    "kernel resource query",
                )
                result[label] = value.value
        finally:
            self.release(kernel)
        return {
            **result,
            "registers": None,
            "spills": None,
            "unavailable_reason": "OpenCL 1.2 has no portable register/spill query",
        }

    def launch(
        self,
        program,
        kernel,
        arguments,
        *,
        items,
        shape: ExecutionShape,
        stream=None,
        wait_for=(),
    ):
        """Submit a padded NDRange and retain referenced objects until explicit wait."""
        shape.validate_for(self.capabilities())
        if type(items) is not int or items < 1:
            raise ValueError("positive logical item count required")
        queue = stream or self.queue
        self._require(queue, "queue")
        fn = self._kernel(program, kernel)
        dependencies = [program, queue]
        try:
            local_bytes = 0
            for index, argument in enumerate(arguments):
                if isinstance(argument, Resource):
                    value = c.c_void_p(self._require(argument, "buffer"))
                    size, pointer = c.sizeof(value), c.byref(value)
                    dependencies.append(argument)
                elif isinstance(argument, LocalMemory):
                    if type(argument.nbytes) is not int or argument.nbytes < 1:
                        raise ValueError("positive local argument bytes required")
                    local_bytes += argument.nbytes
                    size, pointer = argument.nbytes, None
                elif type(argument) in (c.c_uint64, c.c_uint32, c.c_int32, c.c_double):
                    size, pointer = c.sizeof(argument), c.byref(argument)
                else:
                    raise TypeError(
                        "explicit buffer, local memory or typed scalar argument required"
                    )
                _check(
                    self.api.clSetKernelArg(fn.handle, index, size, pointer),
                    "clSetKernelArg",
                )
            if local_bytes > shape.local_bytes:
                raise ValueError(
                    "local kernel arguments exceed the declared schedule workspace"
                )
            limit = c.c_size_t()
            _check(
                self.api.clGetKernelWorkGroupInfo(
                    fn.handle,
                    self._device,
                    0x11B0,
                    c.sizeof(limit),
                    c.byref(limit),
                    None,
                ),
                "compiled workgroup limit",
            )
            if shape.workgroup_threads > limit.value:
                raise ValueError(
                    "schedule exceeds the compiled kernel's workgroup limit"
                )
            # Query after setting local arguments: OpenCL includes both static
            # local declarations and dynamic argument storage in this value.
            total_local = c.c_uint64()
            _check(
                self.api.clGetKernelWorkGroupInfo(
                    fn.handle,
                    self._device,
                    0x11B2,
                    c.sizeof(total_local),
                    c.byref(total_local),
                    None,
                ),
                "compiled local memory",
            )
            if total_local.value > self.device["local_memory_bytes"]:
                raise ValueError("compiled kernel exceeds device local memory")
            waits = (c.c_void_p * len(wait_for))(
                *(self._require(e, "event") for e in wait_for)
            )
            # OpenCL requires flushing producer queues before their events can
            # be consumed by another queue; eager vendor submission is optional.
            producers = {
                resource
                for event in wait_for
                for resource in event.dependencies
                if resource.kind == "queue"
                and resource is not queue
                and not event.completed
            }
            for producer in producers:
                _check(self.api.clFlush(self._require(producer, "queue")), "clFlush")
            global_items = shape.padded_items(items)
            if global_items > c.c_size_t(-1).value:
                raise ValueError("global work size exceeds the native address width")
            global_size, local_size, event = (
                c.c_size_t(global_items),
                c.c_size_t(shape.workgroup_threads),
                c.c_void_p(),
            )
            _check(
                self.api.clEnqueueNDRangeKernel(
                    queue.handle,
                    fn.handle,
                    1,
                    None,
                    c.byref(global_size),
                    c.byref(local_size),
                    len(wait_for),
                    waits if wait_for else None,
                    c.byref(event),
                ),
                "clEnqueueNDRangeKernel",
            )
            return self._own("event", event.value, dependencies=dependencies)
        finally:
            self.release(fn)

    def wait(self, event):
        handle = c.c_void_p(self._require(event, "event"))
        code = self.api.clWaitForEvents(1, c.byref(handle))
        # -14 denotes a terminal execution failure, so retained resources can
        # still be released even though the scientific result is invalid.
        if code in (0, -14):
            object.__setattr__(event, "completed", True)
        _check(code, "clWaitForEvents")

    def elapsed_nanoseconds(self, event):
        self.wait(event)
        stamps = []
        for key in (0x1282, 0x1283):
            value = c.c_uint64()
            _check(
                self.api.clGetEventProfilingInfo(
                    event.handle, key, c.sizeof(value), c.byref(value), None
                ),
                "event profiling",
            )
            stamps.append(value.value)
        return stamps[1] - stamps[0]

    def reduce_sum(self, buffer, count, *, workgroup=64):
        """Reduce FP64 values entirely on device without floating-point atomics.

        Each level uses bounded local scratch and writes one result per
        workgroup; later levels consume those device buffers. The returned
        single-double buffer is caller-owned and does not alias the input.
        This is a correctness primitive, not a tuned library-provider claim.
        """
        if type(count) is not int or count < 1:
            raise ValueError("positive FP64 reduction count required")
        self._bounds(buffer, 0, count * 8)
        if type(workgroup) is not int or workgroup < 2 or workgroup & (workgroup - 1):
            raise ValueError("reduction workgroup must be a power of two >= 2")
        shape = ExecutionShape(workgroup, local_bytes=workgroup * 8)
        shape.validate_for(self.capabilities())
        source = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void reduce_sum(__global const double* input, __global double* output,
                         ulong count, __local double* scratch) {
  size_t i = get_global_id(0), lane = get_local_id(0);
  scratch[lane] = i < count ? input[i] : 0.0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (size_t stride = get_local_size(0) / 2; stride; stride /= 2) {
    if (lane < stride) scratch[lane] += scratch[lane + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane == 0) output[get_group_id(0)] = scratch[0];
}
"""
        compiled = self.compile(source)
        executable = None
        event = None
        owned = []
        try:
            executable = self.link((compiled,))
            current = buffer
            while True:
                groups = (count + workgroup - 1) // workgroup
                output = self.allocate(groups * 8)
                owned.append(output)
                event = self.launch(
                    executable,
                    "reduce_sum",
                    (current, output, c.c_uint64(count), LocalMemory(workgroup * 8)),
                    items=count,
                    shape=shape,
                )
                self.wait(event)
                self.release(event)
                event = None
                if current is not buffer:
                    self.release(current)
                    owned.remove(current)
                if groups == 1:
                    owned.remove(output)
                    return output
                current, count = output, groups
        finally:
            if event is not None:
                if not event.completed:
                    # A failed wait with unknown completion cannot safely leave
                    # borrowed inputs reusable. Drain and invalidate this context.
                    self.close()
                else:
                    self.release(event)
            for resource in owned:
                if not self._closed:
                    self.release(resource)
            if executable is not None and not self._closed:
                self.release(executable)
            if not self._closed:
                self.release(compiled)

    def release(self, resource):
        """Reject stale/cross-context handles and release only completed dependencies."""
        if not isinstance(resource, Resource):
            raise TypeError("a live typed OpenCL resource is required")
        self._require(resource, resource.kind)
        if any(
            r.kind == "event" and not r.completed and resource in r.dependencies
            for r in self._resources.values()
        ):
            raise ValueError(
                "resource is retained by an event; wait before releasing it"
            )
        if resource.kind == "event" and not resource.completed:
            self.wait(resource)
        suffix = {
            "buffer": "MemObject",
            "queue": "CommandQueue",
            "program": "Program",
            "kernel": "Kernel",
            "event": "Event",
        }[resource.kind]
        _check(
            getattr(self.api, "clRelease" + suffix)(resource.handle),
            "clRelease" + suffix,
        )
        object.__setattr__(resource, "handle", 0)
        self._resources.pop(id(resource))

    def close(self):
        """Drain and attempt every native release, even after a vendor error.

        Enqueued commands retain their native objects until completion. Dropping
        our references after a failed finish is therefore legal, but the wrapper
        becomes unusable and reports the errors after all cleanup attempts.
        """
        if self._closed:
            return
        errors = []

        def attempt(function, handle, operation):
            try:
                _check(function(handle), operation)
            except OpenCLError as error:
                errors.append(error)

        for resource in self._resources.values():
            if resource.kind == "queue":
                attempt(self.api.clFinish, resource.handle, "clFinish")
        for resource in list(reversed(self._resources.values())):
            suffix = {
                "buffer": "MemObject",
                "queue": "CommandQueue",
                "program": "Program",
                "kernel": "Kernel",
                "event": "Event",
            }[resource.kind]
            operation = "clRelease" + suffix
            attempt(getattr(self.api, operation), resource.handle, operation)
            object.__setattr__(resource, "handle", 0)
        self._resources.clear()
        if self._context:
            attempt(self.api.clReleaseContext, self._context, "clReleaseContext")
            self._context = None
        self._closed = True
        if errors:
            raise OpenCLError("close", errors[0].code, "\n".join(map(str, errors)))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def ctypes_library():
    """Locate the installed ICD lazily; missing SDK/runtime is explicit unsupported."""
    path = c.util.find_library("OpenCL")
    if path is None:
        raise UnsupportedBackendFeature("no OpenCL ICD library is installed")
    return path
