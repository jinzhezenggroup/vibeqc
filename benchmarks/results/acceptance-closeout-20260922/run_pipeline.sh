#!/usr/bin/env bash
# Keep every device consumer serialized through the single scheduled RTX 5090.
set -u
task_root=/home/jzzeng/codes/vibeqc-acceptance-closeout-20260922
task_evidence=/home/jzzeng/codes/vibeqc-acceptance-evidence-20260922
task_python=/home/jzzeng/codes/vibeqc-torch-boundaries/.venv/bin/python
task_gpu_python=/home/jzzeng/codes/qc/build/gpu4pyscf-venv/bin/python
deadline=$((SECONDS + 3600))
while ! rg -q 'Exit status:' "$task_evidence/build.time" 2>/dev/null; do
  if test "$SECONDS" -ge "$deadline"; then exit 124; fi
  sleep 10
done
if ! rg -q 'Exit status: 0' "$task_evidence/build.time"; then
  echo 'Cold build failed; no GPU evidence will be attributed to it.'
  exit 1
fi
"$task_python" "$task_evidence/capture_incremental.py" || exit $?
cd "$task_root" || exit 1
export PYTHONPATH="$task_root/python:$task_root"
export VIBEQC_LIBRARY="$task_root/build-acceptance/libvibeqc.so"
export CUDACXX=/group/software/cuda-12.9.1/bin/nvcc
export VIBEQC_NVCC="$CUDACXX"
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=01:00:00 \
  "$task_python" "$task_evidence/run_runtime.py" > "$task_evidence/runtime-launch.log" 2>&1
echo "runtime exit=$?"
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  "$task_python" "$task_evidence/r2scan3c_qualification.py" > "$task_evidence/r2scan3c-qualification.log" 2>&1
echo "r2scan3c exit=$?"
for profile in historical strict; do
  for aos in 384 768; do
    for process in 1 2; do
      srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --exclusive --time=00:15:00 \
        "$task_gpu_python" "$task_evidence/direct_factorial.py" \
        --aos "$aos" --profile "$profile" --process "$process" \
        > "$task_evidence/direct-$aos-$profile-$process.log" 2>&1
      echo "direct $aos $profile $process exit=$?"
    done
  done
done
for aos in 384 768; do
  srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --exclusive --time=00:45:00 \
    "$task_python" "$task_evidence/run_df_qualification.py" --aos "$aos" \
    > "$task_evidence/df-$aos-launch.log" 2>&1
  echo "df $aos exit=$?"
done
echo 'All scheduled acceptance commands completed.'
