"""Stage 1 for #150 B qz validation: fetch PR branch, create worktree, minimal venv.

Runs ON the qz CPU notebook (general-copy) via inspire notebook exec as:
  python3 /inspire/.../_150b/runner_stage1.py
No shell words needed in the local exec string.
"""

import os
import subprocess
import sys

ROOT = "/inspire/qb-ilm/project/chemicalreaction/czxs25220150"
BASE = f"{ROOT}/projects/vibeqc"
WORK = f"{BASE}/worktrees/issue-150-b"
ASSETS = f"{ROOT}/env-assets/vibeqc-cpu-py311-v1"
PY = f"{ROOT}/env-assets/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11"
BRANCH = "claude/issue-150-b-tiles"


def sh(args, **kw):
    print("+", " ".join(args), flush=True)
    r = subprocess.run(args, capture_output=True, text=True, **kw)
    if r.stdout:
        print(r.stdout, flush=True)
    if r.returncode:
        print(r.stderr, flush=True)
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(args)}")
    return r.stdout


def main():
    print("=== fetch branch ===", flush=True)
    remotes = sh(["git", "-C", BASE, "remote"])
    if "fork" not in remotes.split():
        sh(["git", "-C", BASE, "remote", "add", "fork",
            "https://github.com/xshengrui/vibeqc.git"])
    sh(["git", "-C", BASE, "fetch", "fork", BRANCH])

    print("=== worktree ===", flush=True)
    os.makedirs(f"{BASE}/worktrees", exist_ok=True)
    if not os.path.isdir(WORK):
        sh(["git", "-C", BASE, "worktree", "add", "--detach", WORK,
            f"fork/{BRANCH}"])
    head = sh(["git", "-C", WORK, "rev-parse", "HEAD"]).strip()
    print("WORKTREE_HEAD", head, flush=True)

    print("=== venv ===", flush=True)
    vpy = f"{WORK}/.venv/bin/python"
    if not os.path.exists(vpy):
        sh([PY, "-m", "venv", f"{WORK}/.venv"])
    sh([vpy, "-m", "pip", "install", "--no-index", "--find-links",
        f"{ASSETS}/wheels", "numpy"])

    print("=== imports ===", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "python:."
    sh([vpy, "-c",
        "from tools.vibeqc_cc import triples_energy, cpu_triples_tiles; "
        "from tools.vibeqc_cc.triples_tiles import TriplesTileEnumerator; "
        "from tools.vibeqc_cc.triples_cuda import CudaTriplesTiles; "
        "print('imports OK')"],
        cwd=WORK, env=env)

    print("=== endpoint presence ===", flush=True)
    for mol in ("h2", "he", "h2o", "nh3", "ch4"):
        p = f"{WORK}/tests/reference_data/cc/endpoints/{mol}.npz"
        print(mol, os.path.exists(p), flush=True)
    print("STAGE1_DONE", flush=True)


if __name__ == "__main__":
    main()
