#!/bin/sh
set -eu

if [ "$#" -gt 1 ]; then
    echo "usage: sh install.sh [XR_FILES_BINARY]" >&2
    exit 2
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ "$#" -eq 1 ]; then
    binary=$1
else
    target_dir=${CARGO_TARGET_DIR:-"$root/target"}
    cargo build --manifest-path "$root/Cargo.toml" --target-dir "$target_dir" --release --locked
    binary=$target_dir/${CARGO_BUILD_TARGET:+$CARGO_BUILD_TARGET/}release/xr-files
fi
test -f "$binary"

dest=${DESTDIR:-}${PREFIX:-/usr/local}
install -Dm755 "$binary" "$dest/bin/xr-files"
install -Dm755 "$root/packaging/launch.py" "$dest/lib/xr-files/launch.py"
install -Dm644 "$root/packaging/runtime_control.py" "$dest/lib/xr-files/runtime_control.py"
ln -sfn ../lib/xr-files/launch.py "$dest/bin/xr-files-desktop"
install -Dm644 "$root/packaging/org.openkylin.XRFiles.desktop" \
    "$dest/share/applications/org.openkylin.XRFiles.desktop"
install -Dm644 "$root/packaging/xr-files.json" "$dest/share/xr/applications/xr-files.json"
