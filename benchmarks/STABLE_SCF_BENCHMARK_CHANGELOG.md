# SCF benchmark stability change

This branch adds an explicit validator for normal-convergence benchmark results.

The immediate goal is to prevent an unstable repeated workload from being summarized as one cross-engine performance ratio. For example, VibeQC `3/3/3/3/3` versus GPU4PySCF `1/7/4/4/1` is reported as inconclusive rather than pooled into one headline median ratio.

This is intentionally separate from the next fixed-work benchmark step. A fixed-work mode should execute the same declared number of SCF/Fock updates on both engines and remain a diagnostic, while the existing converged endpoint stays the user-facing latency/correctness measurement.

Agent: ChatGPT
Model: GPT-5.6 Sol
