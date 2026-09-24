# Developer Guide

This guide describes how VibeQC is implemented and where new functionality belongs.

## Start here

- [Architecture](architecture.md)
- [Scientific compiler architecture](compiler_architecture.md)
- [Electronic-structure boundaries](electronic_structure_boundaries.md)
- [Electronic method IR](electronic_method_ir.md)
- [Program IR](program_ir.md)
- [Integral IR](integral_ir.md)
- [Tensor IR](tensor_ir.md)

## Implementation topics

- SCF and Fock: [module boundaries](scf_module_boundaries.md), [Fock construction](fock_build.md), [strategy selection](fock_strategies.md), and [safe SCF proposals](scf_proposals.md).
- Integrals: [shell generation](shell_codegen.md), [one-electron values](one_electron_codegen.md), [one-electron derivatives](one_electron_derivatives.md), and [second derivatives](second_integral_derivatives.md).
- Density fitting: [overview](density_fitting.md), [derivatives](df_derivatives.md), [occupied exchange](df_occupied_cuda.md), and [tuning](df_tuning.md).
- DFT and XC: [grids](dft_grid.md), [expressions](xc_expressions.md), [integration](xc_integration.md), [SCF domain](xc_scf_domain.md), and [native CUDA](xc_native_cuda.md).
- Post-HF: [reference and providers](posthf.md), [MP2](mp2.md), [RCCSD](rccsd.md), [Lambda](rccsd_lambda.md), [triples](rccsd_t.md), and [CCSD gradients](ccsd_gradient.md).
- Derivatives and response: [stationary problems](stationary_problem.md), [implicit response](implicit_response.md), [orbital response](response.md), and [Hessians](hessian.md).
- Execution: [TensorIR CUDA](tensor_cuda.md), [tensor precision](tensor_precision.md), [state transport](state_transport.md), and [experimental OpenCL](opencl_backend.md).

See [Extending VibeQC](extending/index.md). VibeQC does not yet promise a stable third-party extension API; that section intentionally reserves the framework future public contracts should fill.

```{toctree}
:hidden:
:maxdepth: 1

architecture
compiler_architecture
electronic_structure_boundaries
electronic_method_ir
program_ir
integral_ir
tensor_ir
array_api_frontend
ccsd_gradient
compiler_gfn1_geometry
compiler_xtb_method_ir
cpu_integral_codegen
density_fitting
density_sources
df_batch_properties
df_ccsdt
df_ccsdt_gradient
df_component_trace
df_derivatives
df_device_replay
df_final_eigensystem
df_final_state
df_generated_residency
df_occupied_cuda
df_occupied_exchange
df_response_panel_reuse
df_response_timeline
df_setup_eigensystem
df_shell_derivatives
df_streamed_panels
df_tuning
dft_d3
dft_grid
extensions
first_directional_derivatives
fock_build
fock_strategies
geometry_pair_ir
hessian
implicit_response
incremental_low_rank
ks_diagnostics
libxc_bulk_capabilities
local_spaces
matrix_function
mp2
mp2-a1
mp2-a1-native-boundary
one_electron_codegen
one_electron_derivatives
opencl_backend
posthf
range_separated_integrals
rccsd
rccsd_bc
rccsd_gpu
rccsd_gpu_solver
rccsd_lambda
rccsd_t
rccsd_t_api
response
scf_module_boundaries
scf_proposals
second_integral_derivatives
shell_codegen
source_registry
spatial_tasks
state_transport
stationary_cuda_diagnostic
stationary_native_consumers
stationary_problem
tensor_cuda
tensor_precision
weighted_eri
xc_contractions
xc_expressions
xc_integration
xc_native_cuda
xc_scf_domain
xtb_native_ownership
extending/index
```
