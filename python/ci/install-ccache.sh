#!/usr/bin/env bash
# Install a CUDA-capable ccache into the manylinux build container used by
# cibuildwheel. This is copied from xTBloom's reviewed wheel infrastructure so
# repeated VibeQC PR runs can reuse compiled C++ and CUDA objects.
#
# AlmaLinux-8-based manylinux_2_28 images ship ccache 3.7, which predates
# nvcc/CUDA support (added in ccache 4.1). Use the pinned upstream musl-static
# release instead; it runs inside the older manylinux container without a host
# glibc dependency.
set -euo pipefail

ccache_version=4.14

declare -A ccache_sha256=(
  [x86_64]=888eaf9697919d6eba77be797e9d9c9e40e734d11d67d25a79b075c3f79f6c77
  [aarch64]=40623f11012a97d60c7d90058d7e3c8c70a3802af53a34067b89e6345c89db69
)

arch="$(uname -m)"
case "$arch" in
  x86_64 | aarch64) ;;
  *)
    echo "unsupported build architecture for ccache: $arch" >&2
    exit 1
    ;;
esac

if command -v ccache >/dev/null 2>&1 &&
   [[ "$(ccache --version | sed -n '1s/.* //p')" != 3.* ]]; then
  echo "ccache is already installed: $(ccache --version | head -n1)"
else
  archive="ccache-${ccache_version}-linux-${arch}-musl-static.tar.xz"
  url="https://github.com/ccache/ccache/releases/download/v${ccache_version}/${archive}"
  download_dir="$(mktemp -d)"
  trap 'rm -rf "$download_dir"' EXIT
  curl --fail --location --retry 3 \
    --output "$download_dir/${archive}" "$url"
  echo "${ccache_sha256[$arch]}  $download_dir/${archive}" | sha256sum --check -
  tar -xJf "$download_dir/${archive}" -C /usr/local/bin \
    --strip-components=1 --wildcards "ccache-${ccache_version}*/ccache"
  chmod 0755 /usr/local/bin/ccache
  echo "installed static ccache ${ccache_version} for ${arch} to /usr/local/bin"
fi

mkdir -p "${CCACHE_DIR:-/root/.cache/ccache}"
