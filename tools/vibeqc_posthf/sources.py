"""Owned native shell-tile sources behind the CG02 raw-block contract."""

from __future__ import annotations

import ctypes as ct
import threading
from dataclasses import asdict
from itertools import product
from math import prod

import numpy as np
from vibeqc import Atom, Calculator, _native
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.integral.blocks import (
    BlockRequest,
    BlockResponse,
    BlockStatus,
    RawBlock,
    ShellTile,
    TensorLayout,
)
from vibeqc_compiler.integral.ir import IntegralIR, OperatorSpec
from vibeqc_compiler.integral.shell_signature import (
    BasisShell,
    CenterBinding,
    ShellSignature,
    checked_index,
)

# Through-g Hermite/Coulomb recurrences have bounded dimensions. This separate
# conservative allowance includes their numeric scratch, not Python/C++ object
# headers or allocator rounding. It is independent of molecular/tile size.
CPU_SOURCE_SCRATCH = 8 << 20
_KIND = {
    "overlap": 0,
    "hcore": 1,
    "four_center_eri": 2,
    "coulomb_metric": 3,
    "three_center_eri": 4,
}
_DOUBLE = ct.POINTER(ct.c_double)
_SIZE = ct.POINTER(ct.c_size_t)


def pointer(array):
    return array.ctypes.data_as(_DOUBLE)


class NativeSource:
    """Copy a normalized system into a values-only CPU source; no AO N**4 cache.

    Native system/context handles used for construction can be destroyed
    immediately. The independent source remains valid until ``close``. All
    public tensor values use the supplied basis convention and no screening.
    """

    backend = "cpu-reference-native-shell-tiles"
    supported_operators = frozenset(_KIND)
    _fixed_fields = frozenset(
        (
            "atoms",
            "shells",
            "auxiliary_shells",
            "charge",
            "multiplicity",
            "electron_count",
            "representation",
            "geometry_hash",
            "basis_hash",
            "auxiliary_hash",
            "identity",
            "shell_sizes",
            "auxiliary_sizes",
            "nbf",
            "naux",
            "numeric_bytes",
        )
    )

    def __setattr__(self, name, value):
        if name in self._fixed_fields and name in self.__dict__:
            raise AttributeError(
                "source scientific state is immutable; construct a new source"
            )
        super().__setattr__(name, value)

    def __init__(
        self,
        atoms,
        basis="sto-3g",
        *,
        auxiliary_basis=None,
        charge=0,
        multiplicity=None,
        representation="cartesian",
    ):
        self.atoms = tuple(Atom.from_value(a) for a in atoms)
        calculator = Calculator(basis=basis, basis_representation=representation)
        self.shells = calculator._shells_for_atoms(self.atoms)
        if auxiliary_basis is not None:
            # This source ABI has one representation for both AO spaces. A
            # loaded auxiliary record must not silently lose its own choice.
            from vibeqc.calculator import _snapshot_basis

            auxiliary_basis = _snapshot_basis(auxiliary_basis, representation)
        self.auxiliary_shells = (
            ()
            if auxiliary_basis is None
            else calculator._shells_for_atoms(self.atoms, auxiliary_basis)
        )
        if any(s.angular_momentum > 4 for s in (*self.shells, *self.auxiliary_shells)):
            raise ValueError("post-HF sources support through g")
        self.representation = (
            "real_spherical" if representation == "spherical" else representation
        )
        self.charge = charge
        self.electron_count = sum(a.atomic_number for a in self.atoms) - charge
        default_multiplicity = 1 if self.electron_count % 2 == 0 else 2
        self.multiplicity = (
            default_multiplicity if multiplicity is None else multiplicity
        )
        if (
            type(self.multiplicity) is not int
            or self.multiplicity < 1
            or self.multiplicity > self.electron_count + 1
            or (self.electron_count + self.multiplicity - 1) % 2
        ):
            raise ValueError("multiplicity is incompatible with the electron count")
        self.geometry_hash = canonical_hash([asdict(a) for a in self.atoms])
        self.basis_hash = canonical_hash(
            {
                "shells": [asdict(s) for s in self.shells],
                "representation": self.representation,
            }
        )
        self.auxiliary_hash = (
            canonical_hash(
                {
                    "shells": [asdict(s) for s in self.auxiliary_shells],
                    "representation": self.representation,
                }
            )
            if self.auxiliary_shells
            else None
        )
        self.identity = canonical_hash(
            {
                "geometry": self.geometry_hash,
                "basis": self.basis_hash,
                "auxiliary": self.auxiliary_hash,
                "charge": charge,
                "multiplicity": self.multiplicity,
                "screening": 0,
                "backend": self.backend,
            }
        )
        self.shell_sizes = tuple(self._size(s) for s in self.shells)
        self.auxiliary_sizes = tuple(self._size(s) for s in self.auxiliary_shells)
        self.nbf = sum(self.shell_sizes)
        self.naux = sum(self.auxiliary_sizes)
        inventories = (*self.shells, *self.auxiliary_shells)

        def cartesian(s):
            return (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2

        # Count normalized/native and original/Python coefficients, all owned
        # geometry copies, AoView angular/normalization data, and a complete
        # Cartesian expansion for each public AO, including numeric indices.
        # The expansion bound deliberately overcounts sparse spherical terms.
        self.numeric_bytes = (
            128 * len(self.atoms)
            + 64 * len(inventories)
            + 32 * sum(len(s.primitives) for s in inventories)
            + 32 * sum(cartesian(s) for s in inventories)
            + 16 * sum(self._size(s) * cartesian(s) for s in inventories)
        )
        self._library = calculator._library
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        lib = self._library
        lib.vibeqc_posthf_source_create_v1.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.POINTER(ct.c_void_p),
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.vibeqc_posthf_source_destroy_v1.argtypes = [ct.c_void_p]
        lib.vibeqc_posthf_source_destroy_v1.restype = None
        lib.vibeqc_posthf_source_read_v1.argtypes = [
            ct.c_void_p,
            ct.c_int,
            _SIZE,
            _SIZE,
            _DOUBLE,
            ct.c_size_t,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.vibeqc_posthf_uhf_density_v1.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_uint,
            ct.c_double,
            ct.c_int,
            ct.c_double,
            _DOUBLE,
            ct.c_size_t,
            _DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.vibeqc_posthf_rhf_density_v1.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_uint,
            ct.c_double,
            ct.c_int,
            ct.c_double,
            _DOUBLE,
            ct.c_size_t,
            _DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        context, orbital, auxiliary = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
        _native.check(
            lib,
            lib.vibeqc_context_create(
                ct.byref(calculator._context_descriptor()), ct.byref(context)
            ),
        )
        try:
            # Multiplicity only validates the system descriptor here; sources
            # are also usable with imported references and synthetic tensors.
            multiplicity = self.multiplicity
            orbital = calculator._create_native_system(
                context, self.atoms, charge, multiplicity
            )
            if auxiliary_basis is not None:
                auxiliary = calculator._create_native_system(
                    context, self.atoms, charge, multiplicity, auxiliary_basis
                )
            self._call(
                "vibeqc_posthf_source_create_v1",
                orbital,
                auxiliary,
                ct.byref(self._handle),
            )
        finally:
            if orbital:
                lib.vibeqc_system_destroy(orbital)
            if auxiliary:
                lib.vibeqc_system_destroy(auxiliary)
            lib.vibeqc_context_destroy(context)

    def _size(self, shell):
        l = shell.angular_momentum
        return (
            2 * l + 1
            if self.representation == "real_spherical"
            else (l + 1) * (l + 2) // 2
        )

    def _call(self, name, *args):
        error = ct.create_string_buffer(2048)
        if getattr(self._library, name)(*args, error, len(error)):
            raise RuntimeError(error.value.decode())

    def _check_open(self):
        if not self._handle:
            raise RuntimeError("integral source is closed")

    def close(self):
        """Release owned native basis state; existing detached values survive."""
        with self._lock:
            if self._handle:
                self._library.vibeqc_posthf_source_destroy_v1(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()

    def _read(self, kind, begin, shape):
        """Private contiguous global-AO adapter; dimensions checked natively."""
        if kind not in _KIND or len(begin) != len(shape) or not 2 <= len(shape) <= 4:
            raise ValueError("invalid raw tile operator/rank")
        if any(type(i) is not int or i < 0 for i in (*begin, *shape)):
            raise ValueError("raw tile offsets/extents must be nonnegative integers")
        if len(shape) != (
            4 if kind == "four_center_eri" else 3 if kind == "three_center_eri" else 2
        ):
            raise ValueError("raw operator/rank mismatch")
        for value in (*begin, *shape):
            checked_index(value, "raw tile index")
        if prod(shape) > (1 << 28):
            raise ValueError("raw tile exceeds bounded source hard limit")
        b = (ct.c_size_t * 4)(*begin, *((0,) * (4 - len(begin))))
        c = (ct.c_size_t * 4)(*shape, *((1,) * (4 - len(shape))))
        out = np.empty(shape, dtype=np.float64)
        with self._lock:
            self._check_open()
            self._call(
                "vibeqc_posthf_source_read_v1",
                self._handle,
                _KIND[kind],
                b,
                c,
                pointer(out),
                out.size,
            )
        return out

    def requests(self, kind, *, axis_tile=2, budget_bytes=1 << 20):
        """Yield bounded CG02 requests, including partial shell-component tiles."""
        if type(axis_tile) is not int or axis_tile < 1:
            raise ValueError("axis_tile must be positive")
        if kind not in ("four_center_eri", "three_center_eri", "coulomb_metric"):
            raise ValueError("unsupported CG02 value operator")
        roles = OperatorSpec(
            kind,
            tuple(
                range(
                    4
                    if kind == "four_center_eri"
                    else 3
                    if kind == "three_center_eri"
                    else 2
                )
            ),
        ).basis_roles
        inventories = [
            self.auxiliary_shells if r == "auxiliary" else self.shells for r in roles
        ]
        for shell_indices in product(*(range(len(s)) for s in inventories)):
            selected = tuple(inventories[k][s] for k, s in enumerate(shell_indices))
            bindings = tuple(
                CenterBinding(k, s.atom_index) for k, s in enumerate(selected)
            )
            signature = ShellSignature(
                tuple(
                    BasisShell(k, k, s.angular_momentum, roles[k], self.representation)
                    for k, s in enumerate(selected)
                ),
                bindings,
            )
            full = signature.component_shape
            for begin in product(*(range(0, n, axis_tile) for n in full)):
                shape = tuple(min(axis_tile, n - b) for n, b in zip(full, begin))
                consumer = RawBlock(
                    TensorLayout(signature.tensor_indices, shape), budget_bytes
                )
                ir = IntegralIR(
                    signature,
                    OperatorSpec(kind, tuple(range(len(full)))),
                    None,
                    (consumer,),
                )
                yield BlockRequest(
                    f"{self.identity}:{shell_indices}:{begin}",
                    ir,
                    ShellTile(begin, shape),
                    shell_indices=shell_indices,
                )

    def global_offsets(self, request):
        """Resolve validated shell-local CG02 indices into public AO positions."""
        if (
            request.shell_indices is None
            or request.integral.derivative is not None
            or not isinstance(request.consumer, RawBlock)
        ):
            raise ValueError(
                "source requires runtime shells and a values-only raw consumer"
            )
        result = []
        for slot, (role, index, local) in enumerate(
            zip(
                request.integral.operator.basis_roles,
                request.shell_indices,
                request.tile.offsets,
            )
        ):
            shells = self.auxiliary_shells if role == "auxiliary" else self.shells
            sizes = self.auxiliary_sizes if role == "auxiliary" else self.shell_sizes
            if index >= len(shells):
                raise ValueError("shell index outside source")
            declared = request.integral.signature.shells[slot]
            if (
                declared.angular != shells[index].angular_momentum
                or declared.convention != self.representation
                or request.center_bindings[slot].atom_index != shells[index].atom_index
            ):
                raise ValueError("CG02 shell metadata does not match the owned source")
            result.append(sum(sizes[:index]) + local)
        return tuple(result)

    def execute(self, request):
        """Return the existing CG02 response with explicit successful metadata."""
        kind = request.integral.operator.family.value
        if (
            request.integral.derivative is not None
            or kind not in self.supported_operators
        ):
            return BlockResponse(
                request,
                BlockStatus.UNSUPPORTED,
                reason=f"{self.backend} does not implement the requested operator/derivative",
            )
        begin = self.global_offsets(request)
        values = self._read(kind, begin, request.tile.shape)
        # Honor the request layout/sign, including padded physical storage.
        from vibeqc_compiler.integral.blocks import assemble_raw_block

        return assemble_raw_block(request, values.ravel())

    def tile(self, request):
        """Read a dense FP64 provider tile through the CG02 response contract."""
        response = self.execute(request)
        if not isinstance(response, BlockResponse) or response.status != BlockStatus.OK:
            raise RuntimeError("raw integral tile is unavailable")
        return np.fromiter(response.values, dtype=np.float64).reshape(
            request.tile.shape
        )

    def one_electron(self):
        """Return only O(N**2) S/h matrices; no derivative/four-index allocation."""
        return tuple(
            self._read(k, (0, 0), (self.nbf, self.nbf)) for k in ("overlap", "hcore")
        )

    def rhf_density(
        self,
        *,
        backend="cpu",
        device_id=0,
        max_iterations=100,
        tolerance=1e-11,
        df=False,
        metric_threshold=1e-10,
    ):
        """Run the existing HF solver and export a detached density.

        The existing CPU HF solver is a small-system dense oracle. Its setup
        memory is outside the subsequent bounded integral-provider budget.
        CUDA execution must run under the caller's GPU allocation.
        """
        if (
            backend not in ("cpu", "cuda")
            or type(max_iterations) is not int
            or not 0 < max_iterations < 1 << 32
            or not 0 < tolerance <= 1e-6
        ):
            raise ValueError("invalid RHF export controls")
        density = np.empty((self.nbf, self.nbf))
        scalars = np.empty(4)
        with self._lock:
            self._check_open()
            self._call(
                "vibeqc_posthf_rhf_density_v1",
                self._handle,
                int(backend == "cuda"),
                device_id,
                max_iterations,
                tolerance,
                int(df),
                metric_threshold,
                pointer(density),
                density.size,
                pointer(scalars),
            )
        return density, {
            "energy": float(scalars[0]),
            "energy_change": float(scalars[1]),
            "density_rms": float(scalars[2]),
            "iterations": int(scalars[3]),
            "backend": backend,
        }

    def uhf_density(
        self,
        *,
        backend="cpu",
        device_id=0,
        max_iterations=100,
        tolerance=1e-11,
        df=False,
        metric_threshold=1e-10,
    ):
        """Run UHF and export detached alpha then beta AO densities.

        This small-system snapshot bridge preserves the native UHF spin order
        for checked host canonicalization. It does not claim device response.
        """
        if (
            backend not in ("cpu", "cuda")
            or type(max_iterations) is not int
            or not 0 < max_iterations < 1 << 32
            or not 0 < tolerance <= 1e-6
        ):
            raise ValueError("invalid UHF export controls")
        density = np.empty((2, self.nbf, self.nbf))
        scalars = np.empty(4)
        with self._lock:
            self._check_open()
            self._call(
                "vibeqc_posthf_uhf_density_v1",
                self._handle,
                int(backend == "cuda"),
                device_id,
                max_iterations,
                tolerance,
                int(df),
                metric_threshold,
                pointer(density),
                density.size,
                pointer(scalars),
            )
        return density, {
            "energy": float(scalars[0]),
            "energy_change": float(scalars[1]),
            "density_rms": float(scalars[2]),
            "iterations": int(scalars[3]),
            "backend": backend,
        }


class CudaDFSource(NativeSource):
    """CG05 generated DF M/A source with explicit metric/raw-tile host staging.

    Source construction preserves the existing native source's setup accounting
    and rejects/releases an over-budget prepared source. It is separate from
    the preflight transformation budget. No conventional four-center GPU source
    or GPU DF whitening is claimed by this adapter.
    """

    backend = "cuda-generated-df-values-host-staged"
    supported_operators = frozenset(("coulomb_metric", "three_center_eri"))

    def __init__(
        self,
        *args,
        device_id=0,
        tile_capacity=512,
        source_budget_bytes=128 << 20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._df_handle = ct.c_void_p()
        if (
            type(tile_capacity) is not int
            or tile_capacity < 1
            or type(source_budget_bytes) is not int
            or source_budget_bytes < 1
        ):
            raise ValueError("positive generated DF source capacity/budget required")
        lib = self._library
        lib.vibeqc_posthf_df_create_v1.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
            _SIZE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.vibeqc_posthf_df_destroy_v1.argtypes = [ct.c_void_p]
        lib.vibeqc_posthf_df_destroy_v1.restype = None
        lib.vibeqc_posthf_df_read_v1.argtypes = [
            ct.c_void_p,
            ct.c_int,
            _SIZE,
            _SIZE,
            _DOUBLE,
            ct.c_size_t,
            ct.c_char_p,
            ct.c_size_t,
        ]
        diagnostics = (ct.c_size_t * 4)()
        self._call(
            "vibeqc_posthf_df_create_v1",
            self._handle,
            device_id,
            tile_capacity,
            source_budget_bytes,
            ct.byref(self._df_handle),
            diagnostics,
        )
        self.source_host_peak_bytes = int(diagnostics[0])
        self.source_device_bytes = int(diagnostics[1])
        object.__setattr__(
            self,
            "numeric_bytes",
            self.numeric_bytes + self.source_host_peak_bytes + self.source_device_bytes,
        )
        self.tile_capacity = tile_capacity
        lib.vibeqc_posthf_df_metrics_v1.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]

    def _read(self, kind, begin, shape):
        if kind in ("overlap", "hcore"):
            return super()._read(kind, begin, shape)
        if kind not in ("coulomb_metric", "three_center_eri"):
            raise ValueError("generated DF source does not provide conventional ERIs")
        if len(begin) != len(shape) or any(
            type(i) is not int or i < 0 for i in (*begin, *shape)
        ):
            raise ValueError("invalid DF tile")
        if len(shape) != (3 if kind == "three_center_eri" else 2):
            raise ValueError("DF operator/rank mismatch")
        for value in (*begin, *shape):
            checked_index(value, "DF tile index")
        if kind == "three_center_eri" and prod(shape) > self.tile_capacity:
            raise MemoryError("generated DF tile exceeds prepared capacity")
        with self._lock:
            self._check_open()
            if not self._df_handle:
                raise RuntimeError("generated DF source is closed")
            out = np.empty(shape)
            b = (ct.c_size_t * 4)(*begin, *((0,) * (4 - len(begin))))
            n = (ct.c_size_t * 4)(*shape, *((1,) * (4 - len(shape))))
            self._call(
                "vibeqc_posthf_df_read_v1",
                self._df_handle,
                _KIND[kind],
                b,
                n,
                pointer(out),
                out.size,
            )
            return out

    def source_metrics(self):
        """Cumulative synchronized GPU generation and explicit D2H timings."""
        with self._lock:
            self._check_open()
            values = np.empty(2)
            self._call("vibeqc_posthf_df_metrics_v1", self._df_handle, pointer(values))
            return {
                "generation_ms": float(values[0]),
                "transfer_ms": float(values[1]),
                "host_setup_peak_bytes": self.source_host_peak_bytes,
                "device_bytes": self.source_device_bytes,
            }

    def close(self):
        with self._lock:
            if getattr(self, "_df_handle", None):
                self._library.vibeqc_posthf_df_destroy_v1(self._df_handle)
                self._df_handle = ct.c_void_p()
            super().close()
