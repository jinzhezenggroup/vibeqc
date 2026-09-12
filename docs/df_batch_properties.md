# Energy-only fleet measurements

`PreparedBatch.execute(properties=("energy",))` requests a true energy-only
replay. The default remains energy plus analytic forces. Every result includes
energy; omitted forces are `None`. A later force replay uses the same prepared
scientific model and rebuilds response caches when its requirements change.
Warm-density snapshots follow the existing update/freeze policy.

The native ABI represents this request with a null force pointer and zero
force count for every result descriptor. Mixed descriptors retain the existing
whole-fleet force schedule and copy only requested outputs. Fleet execution
copies the prepared SCF controls before setting output selection; no persistent
option is mutated. Existing direct and DF backends receive that same per-replay
selection, including host numerical recovery and failed-item isolation.

Resource plans keep their conservative energy-plus-force allowance. Omitting
forces does not expand the forward tensor budget or select a different fitting
model. CUDA DF preparation omits derivative output and response bindings for
energy-only execution, so repeated calls do not continually invalidate a
cache that expects no response state.

The batched one-electron exporter also receives a separate nuclear-derivative
flag. Omitting AO derivative matrices alone still runs nuclear response for
fused force consumers; energy-only requests disable both kinds of derivative.
Executed-trace tests detect either response path being launched accidentally.

Run `benchmarks/issue206_df_matrix.py --energy-only` in a finite Slurm GPU job
with a separate output directory. The driver retains the existing 96/192-AO,
batch-1/batch-4 geometry and five interleaved warm repeats. Both engines omit
force evaluation; missing force values and errors are recorded as null rather
than zero. Cold and warm SCF iterations remain attached to raw results, and
unmatched cross-engine branches cannot establish matched speed claims.

The matrix now explicitly requests energy error <=1e-9 Eh and, for force
endpoints, force error <=1e-8 Eh/bohr. Earlier matrix invocations left the
benchmark's optional error thresholds unset; their stored errors need an
independent numerical audit rather than treating exit status as that audit.
The existing batch numerical-selection rule remains visible: matched repeat
pairs when present, otherwise the final unmatched pair, alongside the maximum
over every recorded pair.
