"""Scientific compilers independent of the user-facing VibeQC runtime.

IntegralIR, TensorIR, XC expressions and DFT grid execution own separate
subpackages. Importing this package never loads a native library or probes a GPU.
"""
