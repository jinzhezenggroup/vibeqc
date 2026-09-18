"""Stage 2b: robust NVCC discovery on qz."""

import os
import shutil
import subprocess

CANDIDATES = [
    "/usr/local/cuda/bin/nvcc",
    "/usr/local/cuda-12.9/bin/nvcc",
    "/usr/local/cuda-12.8/bin/nvcc",
    "/usr/local/cuda-12.4/bin/nvcc",
    "/usr/local/cuda-12.2/bin/nvcc",
    "/opt/cuda/bin/nvcc",
]


def main():
    print("=== which nvcc ===", flush=True)
    found = shutil.which("nvcc")
    print("which:", found, flush=True)
    for c in CANDIDATES:
        print(c, os.path.exists(c), flush=True)
    print("=== /usr/local ===", flush=True)
    if os.path.isdir("/usr/local"):
        print(sorted(os.listdir("/usr/local")), flush=True)
    print("=== ldconfig cuda ===", flush=True)
    r = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
    libs = [line.strip() for line in r.stdout.splitlines() if "cuda" in line.lower()]
    for line in libs[:10]:
        print(line, flush=True)
    print("STAGE2B_DONE", flush=True)


if __name__ == "__main__":
    main()
