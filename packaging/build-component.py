#!/usr/bin/env python3
"""Build the locked xr-files component through xr-build-components."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target_dir = Path(os.environ.get("XR_CARGO_TARGET_DIR", args.output / "cargo-target"))
    env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))
    target = env.get("XR_CARGO_TARGET") or env.get("CARGO_BUILD_TARGET")
    command = ["cargo", "build", "--release", "--locked"]
    if target:
        command += ["--target", target]
    subprocess.run(command, cwd=root, env=env, check=True)
    binary = target_dir / (target or "") / "release/xr-files"
    architecture = subprocess.check_output(["readelf", "-hW", str(binary)], text=True)
    arch = next(
        (
            name
            for elf, name in [("AArch64", "arm64"), ("X86-64", "amd64"), ("RISC-V", "riscv64")]
            if elf in architecture
        ),
        None,
    )
    if not arch:
        raise RuntimeError("Unsupported xr-files binary architecture")
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
            cwd=root,
            env=env,
            text=True,
        )
    )
    version = next(p["version"] for p in metadata["packages"] if p["name"] == "xr-files")
    args.packages.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "sh",
            str(root / "packaging/package.sh"),
            str(args.packages / f"xr-files_{version}_{arch}.deb"),
            str(binary),
        ],
        cwd=root,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
