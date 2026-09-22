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
