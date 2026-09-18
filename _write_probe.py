import pathlib
pathlib.Path("/inspire/qb-ilm/project/chemicalreaction/czxs25220150/_150b/_probe_nvcc.py").write_text('''import os, shutil, subprocess
print("which nvcc:", shutil.which("nvcc"))
for p in ("/usr/local/cuda/bin/nvcc",):
    print(p, os.path.exists(p))
if os.path.exists("/usr/local"):
    print(sorted(os.listdir("/usr/local")))
r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
print("nvcc rc:", r.returncode)
print(r.stdout[-400:] if r.stdout else r.stderr[-400:])
r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print("nvidia-smi rc:", r.returncode)
print(r.stdout[:600] if r.stdout else r.stderr[:400])
print("PROBE_DONE")
''')
print("WRITTEN")