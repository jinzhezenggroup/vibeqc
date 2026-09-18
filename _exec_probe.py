import subprocess
r = subprocess.run(["/bin/bash", "/inspire/qb-ilm/project/chemicalreaction/czxs25220150/_150b/_run_probe.sh"], capture_output=True, text=True)
print(r.stdout)
print(r.stderr)
print("RC", r.returncode)