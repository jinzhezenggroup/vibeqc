#!/usr/bin/env bash
set -u
: "${SLURM_JOB_ID:?GPU execution requires Slurm}"
task_evidence=/home/jzzeng/codes/vibeqc-acceptance-evidence-20260922
cd /home/jzzeng/codes/vibeqc-acceptance-closeout-20260922 || exit 1
export VIBEQC_LIBRARY=$PWD/build-acceptance/libvibeqc.so
export PYTHONPATH=$PWD/python:$PWD
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export VIBEQC_DF_RESPONSE_STORAGE=panel
/home/jzzeng/codes/vibeqc-torch-boundaries/.venv/bin/python "$task_evidence/reuse_df_reference.py" --aos 768 --repeats 7 --cpu-reference --components-after --control VIBEQC_DF_SERIAL_RESPONSE_DOT --policies 0 --output "$task_evidence/df-768-blas-qualified.json" > "$task_evidence/df-768-blas-qualified.log" 2>&1
echo "blas exit=$?"
unset VIBEQC_DF_RESPONSE_STORAGE
/home/jzzeng/codes/vibeqc-torch-boundaries/.venv/bin/python "$task_evidence/reuse_df_reference.py" --aos 768 --repeats 7 --cpu-reference --components-after --control VIBEQC_DF_RESPONSE_SPACE --policies auto --output "$task_evidence/df-768-auto-qualified.json" > "$task_evidence/df-768-auto-qualified.log" 2>&1
echo "auto exit=$?"
