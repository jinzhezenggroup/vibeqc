set -euo pipefail

destination=${1:?usage: repair-wheel.sh <destination> <wheel>}
wheel=${2:?usage: repair-wheel.sh <destination> <wheel>}
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
provider_python="$project_dir/build/wheel-openblas-provider/bin/python"

# Resolve by distribution metadata only. Importing scipy_openblas32 would load
# OpenBLAS into the process-global namespace, which the private shim avoids.
provider_path=$(
  "$provider_python" \
    "$project_dir/tools/xtb/resolve-openblas-wheel.py" \
    --manifest \
    "$project_dir/tools/xtb/scipy_openblas32_manifest.json" |
    "$provider_python" -c \
      'import json, sys; print(json.load(sys.stdin)["provider_path"])'
)
test -f "$provider_path"
provider_dir=$(dirname -- "$provider_path")
export LD_LIBRARY_PATH="$provider_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# CUDA user-space providers remain separately distributed NVIDIA packages.
# OpenBLAS is intentionally not excluded: auditwheel follows the GFN2 shim and
# vendors/collision-renames the reviewed provider cohort into the VibeQC wheel.
auditwheel repair -w "$destination" "$wheel" \
  --exclude libcublas.so.12 \
  --exclude libcublasLt.so.12 \
  --exclude libcusolver.so.11 \
  --exclude libcusparse.so.12 \
  --exclude libcudart.so.12 \
  --exclude libnvJitLink.so.12 \
  --exclude libcufft.so.11 \
  --exclude libcurand.so.10 \
  --exclude libcuda.so.1

# Surface whether repeated wheel builds actually reuse C++/CUDA compilation.
ccache --show-stats --verbose
