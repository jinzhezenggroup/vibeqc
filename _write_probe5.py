import pathlib
p = pathlib.Path("/inspire/qb-ilm/project/chemicalreaction/czxs25220150/_150b/probe2")
p.write_text('#!/bin/sh\nexec /inspire/qb-ilm/project/chemicalreaction/czxs25220150/env-assets/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11 /inspire/qb-ilm/project/chemicalreaction/czxs25220150/_150b/_probe_nvcc.py\n', newline="\n")
p.chmod(0o755)
print("WRITTEN")