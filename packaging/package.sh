#!/bin/sh
set -eu
export LC_ALL=C

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: sh package.sh OUTPUT.deb [XR_FILES_BINARY]" >&2
    exit 2
fi
output=$1
shift
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
chmod 0755 "$stage"
DESTDIR=$stage PREFIX=/usr sh "$root/packaging/install.sh" "$@"
binary=$stage/usr/bin/xr-files
header=$(readelf -hW "$binary")
case "$header" in
    *ELF64*AArch64*) arch=arm64 ;;
    *ELF64*X86-64*) arch=amd64 ;;
    *ELF64*RISC-V*) arch=riscv64 ;;
    *) echo "unsupported ELF architecture" >&2; exit 1 ;;
esac
glibc=$(readelf --version-info "$binary" | sed -n 's/.*Name: GLIBC_\([0-9.]*\) .*/\1/p' | sort -V | tail -1)
test -n "$glibc"
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/Cargo.toml" | head -1)
test -n "$version"
install -d "$stage/DEBIAN"
cat > "$stage/DEBIAN/control" <<EOF
Package: xr-files
Version: $version
Architecture: $arch
Maintainer: openKylin XR SIG <xr@lists.openkylin.top>
Section: utils
Priority: optional
Depends: python3 (>= 3.9), libc6 (>= $glibc), libgcc-s1, libvulkan1, libopenxr-loader1
Recommends: fonts-noto-cjk, xdg-utils, xr-cinema
Description: 3D spatial file manager for openKylin XR
 Browse, search, preview and organize files in an ergonomic 3D spatial
 workbench and volumetric treemap powered by Rust, wgpu (Vulkan) and SIMD.
EOF
dpkg-deb --root-owner-group --build "$stage" "$output"
