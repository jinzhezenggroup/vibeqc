import pathlib
p = pathlib.Path("/inspire/qb-ilm/project/chemicalreaction/czxs25220150/_150b/envprobe")
p.write_text("#!/bin/sh\nenv > /inspire/qb-ilm/project/chemicalreaction/czxs25220150/_150b/env.out\n/usr/local/cuda/bin/nvcc --version >> /inspire/qb-ilm/project/chemicalreaction/czxs25220150/_150b/env.out 2>&1\n", newline="\n")
p.chmod(0o755)
print("WRITTEN")