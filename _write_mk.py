import pathlib
p = pathlib.Path("/inspire/qb-ilm/project/chemicalreaction/czxs25220150/_150b/mk.py")
p.write_text('''import os, pathlib
scripts = {
  "py3": ("#!/usr/bin/env bash\\nexec /inspire/qb-ilm/project/chemicalreaction/czxs25220150/env-assets/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11 \\"$@\\"\\n"),
}
for name, body in scripts.items():
    f = pathlib.Path(f"/inspire/qb-ilm/project/chemicalreaction/czxs25220150/_150b/{name}")
    f.write_text(body, newline="\\n")
    f.chmod(0o755)
    print(name, oct(f.stat().st_mode), flush=True)
print("MK_DONE")
''', newline="\n")
print("WRITTEN")