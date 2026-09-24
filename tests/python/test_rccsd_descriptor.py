"""Current RCCSD method descriptors are mandatory at the public C boundary."""

import ctypes as ct

import pytest
from test_rccsd_public import _calculator, _reference_case
from vibeqc import Atom, _native


@pytest.mark.parametrize(
    "last_field", ["density_fitting_mode", "correlation_memory_budget_bytes"]
)
def test_truncated_method_descriptor_is_rejected(last_field: str) -> None:
    calc = _calculator()
    atoms, _ = _reference_case()
    atoms = tuple(Atom(int(z), tuple(xyz)) for z, xyz in atoms)
    library = calc._library
    context, system, calculation = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    context_descriptor = calc._context_descriptor()
    _native.check(
        library,
        library.vibeqc_context_create(ct.byref(context_descriptor), ct.byref(context)),
    )
    try:
        system = calc._create_native_system(context, atoms, 0, 1)
        method = calc._method_descriptor()
        method.struct_size = getattr(_native.MethodDescriptor, last_field).offset
        method.correlation_memory_budget_bytes = 1 << 63
        status = library.vibeqc_calculation_prepare(
            context, system, ct.byref(method), ct.byref(calculation)
        )
        assert status == _native.STATUS_ABI_MISMATCH
        assert not calculation
    finally:
        if calculation:
            library.vibeqc_calculation_destroy(calculation)
        if system:
            library.vibeqc_system_destroy(system)
        library.vibeqc_context_destroy(context)
