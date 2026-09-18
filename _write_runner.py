import base64
import pathlib

src_b64 = ""
src = base64.b64decode(src_b64).decode()
pathlib.Path("/inspire/qb-ilm/project/chemicalreaction/czxs25220150/_150b/_runner_gpu.py").write_text(src, newline="\n")
print("WRITTEN")