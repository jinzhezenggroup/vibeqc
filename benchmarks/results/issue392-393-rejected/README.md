# Rejected isolated DF optimizations

This records six candidates that failed the production performance gates for
#392/#393. `early-folding/` contains direct shared stores and the first register
variant; `identity-folding/` adds constant tuple extents and unit coefficients.
None demonstrates an endpoint improvement. `serial-grouping/` regresses both
sizes; `block-grouping/` and `warp-grouping/` have lower 384-AO medians but
regress at 768 AOs. See the
[decision note](../../../.agents/notes/rejected/2026-09-16-df-folding-and-c-grouping.md).

The later candidate directories contain source/native-test patches against
`1c3f2ab8d6df6f06bee526b89a807511ef726301`, build and validation identities,
the complete-endpoint summary, and one compact record per AO size. The records
retain every clean sample's timing, energy, forces, errors, iterations and
metric metadata, workload settings and reference hashes, class-level executed
work/resources, and separate disabled-counter Nsight kernel observations.
Detailed signature rows and transient traces are omitted; their original
ledger, trace, database and measurement hashes are retained. No cold or
changed-geometry performance claim is made. The first direct candidate has
native/memcheck evidence but no recorded full Python CUDA suite; the other five
have 120-test CUDA suite records. Per-candidate validation scopes are explicit.

Before publication, the archival reader was replayed using the original local
trace, measurement, generated-header and Nsight SQLite files; its output matched
the retained 768-AO warp class/work/profile records. Those reader inputs are not
included in this archive, so that replay cannot be repeated from the published
files alone. Their hashes identify the omitted inputs but cannot reconstruct
them. The capture instructions below produce new inputs from a reconstructed
candidate; they do not replay the historical records.

The two `early-folding/*-source.patch` files instead apply to
`c5475c49b331c989fbe0aafb3ce19685d7cb254d`. Their `conditions.json` discloses
compilation overlapping part of the intrusive profiles; clean endpoint timing
was isolated. The register native executable was rebuilt with the subsequent
#395 packet assertions, as recorded in `builds.json`; its library was unchanged.

Every patch entry in `builds.json` has a `path` relative to that manifest.
The baseline patch is already retained at
[`../issue395-df-work/reproduction/source.patch`](../issue395-df-work/reproduction/source.patch),
with base `23091a4575c3b2bf9288ae00bb173935a51364c3`. Its historical digest differs
from each candidate patch; the shared `source.patch` key names the artifact
inside each frozen build, not one common file. Binary and generated-header
identities describe historical artifacts that are not included here.

The following commands reconstruct the baseline and selected candidate, build
them separately, and freeze their libraries and source patches. Run from this
publication checkout's root in the configured CUDA 12.9/Python development
environment. Use a fresh `vibeqc_repro_root`; complete all compilation before
starting the Slurm captures. These are instructions for future reproduction,
not additional measurements made for this publication.

```bash
set -euo pipefail
# Choose early-folding, identity-folding, serial-grouping, block-grouping,
# or warp-grouping. early-folding prepares both direct and register variants.
vibeqc_candidate=block-grouping
vibeqc_archive="$PWD/benchmarks/results/issue392-393-rejected"
vibeqc_repro_root="$PWD/.artifacts/rejected-reproduction"
mkdir -p "$PWD/.artifacts"
mkdir "$vibeqc_repro_root"
export VIBEQC_WORK_CUDA=/group/software/cuda-12.9.1
export CUDACXX="$VIBEQC_WORK_CUDA/bin/nvcc"
export VIBEQC_WORK_PYTHON="$(command -v python)"

freeze_variant() {
  local label=$1 base=$2 source_patch=$3
  shift 3
  local checkout="$vibeqc_repro_root/sources/$label"
  local frozen="$vibeqc_repro_root/frozen/$label"
  git worktree add --detach "$checkout" "$base"
  git -C "$checkout" apply "$source_patch"
  for extra_patch in "$@"; do
    git -C "$checkout" apply "$extra_patch"
  done
  cmake --preset cuda-release-sm120 -S "$checkout"
  cmake --build "$checkout/build/cuda-release-sm120" \
    --target vibeqc vibeqc_df_shell_pairs_tests --parallel 2
  mkdir -p "$frozen"
  cp -L "$checkout/build/cuda-release-sm120/libvibeqc.so" "$frozen/libvibeqc.so"
  cp "$checkout/build/cuda-release-sm120/vibeqc_df_shell_pairs_tests" "$frozen/"
  cp "$checkout/build/cuda-release-sm120/generated/generated_df_shell_derivatives.cuh" "$frozen/"
  cp "$source_patch" "$frozen/source.patch"
}

freeze_variant baseline 23091a4575c3b2bf9288ae00bb173935a51364c3 \
  "$vibeqc_archive/../issue395-df-work/reproduction/source.patch"
case "$vibeqc_candidate" in
  early-folding)
    freeze_variant direct c5475c49b331c989fbe0aafb3ce19685d7cb254d \
      "$vibeqc_archive/early-folding/direct-source.patch"
    freeze_variant register c5475c49b331c989fbe0aafb3ce19685d7cb254d \
      "$vibeqc_archive/early-folding/register-source.patch"
    export VIBEQC_WORK_DIRECT="$vibeqc_repro_root/frozen/direct"
    export VIBEQC_WORK_REGISTER="$vibeqc_repro_root/frozen/register"
    capture_stage=clean-early
    ;;
  identity-folding|serial-grouping|block-grouping|warp-grouping)
    extra_patches=("$vibeqc_archive/$vibeqc_candidate/native-test.patch")
    if [[ -f "$vibeqc_archive/$vibeqc_candidate/cuda-test.patch" ]]; then
      extra_patches+=("$vibeqc_archive/$vibeqc_candidate/cuda-test.patch")
    fi
    freeze_variant candidate 1c3f2ab8d6df6f06bee526b89a807511ef726301 \
      "$vibeqc_archive/$vibeqc_candidate/source.patch" "${extra_patches[@]}"
    export VIBEQC_WORK_CANDIDATE="$vibeqc_repro_root/frozen/candidate"
    capture_stage=clean
    ;;
  *) echo "Unknown candidate: $vibeqc_candidate" >&2; exit 2 ;;
esac

export VIBEQC_WORK_BASELINE="$vibeqc_repro_root/frozen/baseline"
export VIBEQC_WORK_OUTPUT="$vibeqc_repro_root/captures"
export VIBEQC_WORK_CHECKPOINTS="$vibeqc_repro_root/checkpoints"
for aos in 384 768; do
  srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
    --time=00:30:00 bash "$vibeqc_archive/reproduction/run-comparison.sh" checkpoint "$aos"
  srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
    --time=00:30:00 bash "$vibeqc_archive/reproduction/run-comparison.sh" "$capture_stage" "$aos"
done
```

The versioned [capture script](reproduction/run-comparison.sh) preserves Slurm's
device visibility and sets one CPU math thread. It selects the matching
independent reference, creates the common post-cold checkpoint on the baseline,
then measures every variant from that checkpoint with the unchanged three-SCF-
update gate and a separate component pass. `clean` executes baseline-2,
candidate-2, candidate-3, baseline-3; `clean-early` executes baseline-2, direct-2,
register-2, register-3, direct-3, baseline-3. Each suffix is the repeat count.
The endpoint runner refuses to overwrite existing measurements or checkpoints.
New checkpoints have their own identities; do not claim exact historical
density reproduction unless their retained hashes match. Rebuilt binaries also
have new identities until checked against the recorded manifests. Keep profiling
and all other compilation outside clean timing.

For intrusive work attribution run a separate `--policies 0 1 --repeats 1
--trace --cuda-profile` capture under Nsight Systems with
`VIBEQC_DF_SHELL_COUNTERS=1`, then reconcile it with the retained
`reproduction/df_shell_work_ledger.py` using the frozen generated header.
Run that file with `PYTHONPATH=python:.` from the reconstructed candidate
checkout, passing `--trace`, `--measurement`, `--generated-header`, `--nsys`
and `--output`. It reads the executed grouping fields and supports both
historical schedule names; the candidate base's original reducer cannot
decode every added diagnostic. Source
counts do not measure atomic contention. Never infer a promotion from them.

Publication changes only evidence and notes. Production selectors and scientific
tolerances remain at the merged #397/#398 baseline. #392/#393 stay open;
the separate [Rys prototype](../issue394-deferred/README.md) is deferred without
an endpoint performance verdict.
