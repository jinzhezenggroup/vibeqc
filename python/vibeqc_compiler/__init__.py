"""Scientific compilers independent of the user-facing VibeQC runtime.

IntegralIR, TensorIR, XC expressions, DFT grid execution and method composition
own separate subpackages. Importing this package never loads a native library or
probes a GPU.
"""
