"""Full qz GPU validation runner for #150 B (runs ON the H200 notebook).

Stages: verify repo worktree, ensure venv deps, run tools/validate_cc_triples_tiles.py
with --compile-only (sanity) and then full GPU validation at both budgets and
both chunk sizes. Writes manifests to the shared drive and stdout markers.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = "/inspire/qb-ilm/project/chemicalreaction/czxs25220150"
WORK = f"{ROOT}/projects/vibeqc/worktrees/issue-150-b"
OUT = f"{ROOT}/_150b/results"
CACHE = f"{ROOT}/_150b/cache"
PY = f"{WORK}/.venv/bin/python"
NVCC = "/usr/local/cuda/bin/nvcc"
ARCH = "sm_90"  # H200 = Hopper, compute capability 9.0

log_lines = []


def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)


def sh(args, **kw):
    log("+ " + " ".join(args))
    r = subprocess.run(args, capture_output=True, text=True, **kw)
    if r.stdout:
        log(r.stdout.rstrip())
    if r.returncode:
        log("STDERR: " + r.stderr.rstrip())
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(args)}")
    return r.stdout


def main():
    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    pathlib.Path(CACHE).mkdir(parents=True, exist_ok=True)

    # The H200 notebook image has no git binary; the worktree was created on
    # the CPU notebook side.  HEAD is read from the worktree's .git pointer.
    head_file = pathlib.Path(WORK) / ".git"
    if head_file.is_file():
        # worktree .git is a file: "gitdir: <abs>"
        gitdir = pathlib.Path(head_file.read_text().split(None, 1)[1].strip())
        head_ref = (gitdir / "HEAD").read_text().strip()
        if head_ref.startswith("ref:"):
            ref = head_ref.split(None, 1)[1].strip()
            head = (gitdir / ref).read_text().strip()
        else:
            head = head_ref
    else:
        head = (head_file / "HEAD").read_text().strip()
    log("WORKTREE_HEAD " + head)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{WORK}/python:{WORK}"
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    # numpy + scipy for the validator
    sh([PY, "-m", "pip", "install", "--no-index", "--find-links",
        f"{ROOT}/env-assets/vibeqc-cpu-py311-v1/wheels", "numpy"])

    cmd = [
        PY, f"{WORK}/tools/validate_cc_triples_tiles.py",
        "--output", OUT, "--cache", CACHE,
        "--nvcc", NVCC, "--architecture", ARCH,
        "--budget", "256,512",
    ]
    log("+ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=WORK, env=env)
    log("RETURNCODE " + str(r.returncode))
    log("=== STDOUT ===")
    log(r.stdout.rstrip())
    log("=== STDERR (tail) ===")
    log("\n".join(r.stderr.splitlines()[-40:]))
    if r.returncode:
        raise SystemExit("validator failed")

    manifest = pathlib.Path(OUT) / "manifest.json"
    log("manifest exists: " + str(manifest.exists()))
    if manifest.exists():
        data = json.loads(manifest.read_text())
        pathlib.Path(f"{OUT}/manifest-pretty.json").write_text(
            json.dumps(data, indent=2))
        log("runtime_device: " + json.dumps(data.get("runtime_device")))
        log("molecules: " + str([m["name"] for m in data.get("molecules", [])]))
    log("STAGE_RUN_DONE")


if __name__ == "__main__":
    main()
