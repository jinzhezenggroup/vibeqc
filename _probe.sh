echo "=== nvidia-smi ==="
nvidia-smi -L 2>&1 || echo NO_GPU
echo "=== cuda dirs ==="
ls /usr/local/ | grep -i cuda || echo NO_CUDA_USRLOCAL
ls /opt 2>/dev/null | grep -i cuda || echo NO_CUDA_OPT
echo "=== shared df ==="
df -h /inspire/qb-ilm/project/chemicalreaction/czxs25220150 | tail -n 1
echo "=== remote repo state ==="
git -C /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc log --oneline -1
git -C /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc branch -a | head -20
echo DONE