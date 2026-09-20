set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

export CCACHE_DIR="${CCACHE_DIR:-$project_dir/build/ccache}"
bash "$project_dir/python/ci/install-ccache.sh"

# Keep the reviewed LP64 provider alive beyond PEP 517's temporary build
# environment. auditwheel must resolve the shim's DT_NEEDED edge during repair.
provider_env="$project_dir/build/wheel-openblas-provider"
rm -rf "$provider_env"
python -m venv "$provider_env"
"$provider_env/bin/python" -m pip install --disable-pip-version-check --no-deps   "scipy-openblas32==0.3.34.0.0"

"$provider_env/bin/python"   "$project_dir/tools/xtb/resolve-openblas-wheel.py"   --manifest   "$project_dir/tools/xtb/scipy_openblas32_manifest.json"   >/dev/null
