cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc
echo "=== HEAD ==="
git log --oneline -1
echo "=== remotes ==="
git remote -v
echo "=== status ==="
git status --short | head -5
echo "=== branches ==="
git branch -a | head -20
echo DONE