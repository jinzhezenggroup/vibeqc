"""Stage 2: compile tile plans on qz and report whether NVCC is available.

Runs on the qz CPU notebook. The default workspace venv may lack CUDA; NVCC
on qz is typically under /usr/local/cuda/bin/nvcc.
"""

import os
import subprocess

WORK = "/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc/worktrees/issue-150-b"


def sh(args, **kw):
    print("+", " ".join(args), flush=True)
    r = subprocess.run(args, capture_output=True, text=True, **kw)
    if r.stdout:
        print(r.stdout, flush=True)
    if r.returncode:
        print(r.stderr, flush=True)
        raise SystemExit(f"FAILED ({r.returncode})")
    return r.stdout


def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = "python:."
    vpy = f"{WORK}/.venv/bin/python"
    print("=== NVCC check ===", flush=True)
    for path in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12.9/bin/nvcc"):
        if os.path.exists(path):
            print("NVCC_FOUND", path, flush=True)
            sh([path, "--version"], env=env)
            break
    else:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if r.returncode:
            print("NVCC_MISSING", flush=True)
            raise SystemExit(2)
        print(r.stdout, flush=True)

    print("=== compile plans (h2o, chunk=1) ===", flush=True)
    code = (
        "from tools.vibeqc_cc.triples_tiles import TriplesTileEnumerator\n"
        "from vibeqc_compiler.tensor.cuda_plan import plan_cuda, TensorSchedule\n"
        "from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter\n"
        "from vibeqc_compiler.integral.cuda_target import cuda_target_info\n"
        "from vibeqc_compiler.tensor.cuda_resident import compile_resident\n"
        "import numpy as np\n"
        "data = np.load('tests/reference_data/cc/endpoints/h2o.npz')\n"
        "nocc = int(np.sum(data['occ'] > 0)); nvir = len(data['eps']) - nocc\n"
        "tile = next(iter(TriplesTileEnumerator(nocc, nvir, vir_chunk_size=1)))\n"
        "from tools.vibeqc_cc.triples_tiles import build_tile_triples_program\n"
        "prog = build_tile_triples_program(nocc, tile.a_end, vir_chunk=(tile.a_start, tile.a_end))\n"
        "plan = plan_cuda(prog, cuda_target_info('sm_90'), max_bytes=256<<20, schedule=TensorSchedule())\n"
        "print('plan peak_bytes', plan.peak_bytes)\n"
        "import tempfile, pathlib\n"
        "cache = pathlib.Path(tempfile.mkdtemp())\n"
        "compiler = CudaCompilerAdapter(None, cuda_target_info('sm_90'))\n"
        "try:\n"
        "    art = compile_resident(plan, compiler, cache)\n"
        "    print('COMPILED', art.metadata['key'][:24])\n"
        "except Exception as e:\n"
        "    print('COMPILE_FAILED', type(e).__name__, str(e)[:400])\n"
    )
    sh([vpy, "-c", code], cwd=WORK, env=env)
    print("STAGE2_DONE", flush=True)


if __name__ == "__main__":
    main()
