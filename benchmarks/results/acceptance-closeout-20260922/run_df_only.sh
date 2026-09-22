#!/usr/bin/env bash
set -u
task_evidence=/home/jzzeng/codes/vibeqc-acceptance-evidence-20260922
export CUDACXX=/group/software/cuda-12.9.1/bin/nvcc
export VIBEQC_NVCC="$CUDACXX"
for task_aos in 384 768; do
  srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --exclusive --time=00:45:00 \
    /home/jzzeng/codes/vibeqc-torch-boundaries/.venv/bin/python "$task_evidence/run_df_qualification.py" --aos "$task_aos" \
    > "$task_evidence/df-$task_aos-launch.log" 2>&1
  echo "df $task_aos exit=$?"
done
