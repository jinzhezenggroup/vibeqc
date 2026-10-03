# Developer Guide

This guide describes how GenerativeQC is implemented and where new functionality
belongs. Start with the architectural boundaries, then use the topic map below
instead of scanning the source tree.

## Start here

1. [Build and CUDA configuration](build.md)
2. [Architecture](architecture.md)
3. [Scientific compiler architecture](compiler_architecture.md)
4. [Electronic-structure boundaries](electronic_structure_boundaries.md)

The primary IR layers are [Electronic Method IR](electronic_method_ir.md),
[Program IR](program_ir.md), [Integral IR](integral_ir.md), and
[Tensor IR](tensor_ir.md).

## Topic map

- **Compiler and IR** — scientific ownership, lowering, generated sources,
  specialization, tensor programs, and compiler-facing method descriptions.
- **Integrals and SCF/HF** — shell/integral generation, Fock construction,
  range-separated exchange, SCF proposals, and HF force finalization.
- **DFT and XC** — grids, XC expressions/contractions, native CUDA integration,
  Libxc capability import, diagnostics, and D3.
- **Density fitting** — DF storage, streaming, occupied exchange, response,
  residency, replay, final state, and tuning.
- **Post-HF** — MP2, RCCSD, triples, Lambda equations, GPU ownership, and
  correlated gradients.
- **Derivatives and response** — stationary problems, implicit/orbital response,
  first/second derivatives, and Hessians.
- **Execution and backends** — TensorIR CUDA, precision, state transport,
  experimental backends, and extension boundaries.

See [Extending GenerativeQC](extending/index.md) for extension points.
GenerativeQC does not yet promise a stable third-party extension API.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Foundations

build
architecture
compiler_architecture
electronic_structure_boundaries
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Compiler and IR

electronic_method_ir
program_ir
integral_ir
tensor_ir
array_api_frontend
source_registry
spatial_tasks
geometry_pair_ir
matrix_function
local_spaces
compiler_gfn1_geometry
compiler_xtb_method_ir
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Integrals and SCF/HF

shell_codegen
cpu_integral_codegen
one_electron_codegen
one_electron_derivatives
second_integral_derivatives
fock_build
fock_strategies
scf_module_boundaries
scf_proposals
incremental_low_rank
range_separated_integrals
weighted_eri
hf_force_finalization
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: DFT and XC

dft_grid
dft_d3
xc_expressions
xc_integration
xc_contractions
xc_native_cuda
xc_scf_domain
ks_diagnostics
libxc_bulk_capabilities
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Density fitting

density_fitting
density_sources
df_batch_properties
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
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Post-HF

posthf
mp2
rccsd
rccsd_bc
rccsd_gpu
rccsd_lambda
rccsd_t
rccsd_t_api
ccsd_gradient
df_ccsdt
df_ccsdt_gradient
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Derivatives and response

first_directional_derivatives
stationary_problem
stationary_native_consumers
stationary_cuda_diagnostic
implicit_response
response
hessian
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Execution and backends

tensor_cuda
cuda_time_estimator
tensor_precision
state_transport
opencl_backend
extensions
xtb_native_ownership
extending/index
```
