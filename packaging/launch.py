#!/usr/bin/env python3
"""Launch openKylin XR Files coexisting inside HyperSpace Wayland by default,
or as a standalone foreground OpenXR session only when `--xr` is explicitly passed."""

import glob
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from runtime_control import (
    ACTIVE,
    PRIMARY,
    Monado,
    WiVRnControl,
    runtime_details,
    runtime_directory_for,
)

NAME = "openKylin XR Files"
CONTROL_ERRORS = (
    OSError,
    ValueError,
    RuntimeError,
    AttributeError,
    KeyError,
    subprocess.TimeoutExpired,
)


def files_client(clients):
    return next((c for c in clients if c.name == NAME and c.flags & ACTIVE), None)


def restore(controller, previous, current_app):
    if previous is None or current_app is None or previous.id == current_app.id:
        return
    clients = controller.clients()
    current = next((c for c in clients if c.flags & PRIMARY), None)
    if current and (current.id, current.name) not in {
        (previous.id, previous.name),
        (current_app.id, current_app.name),
    }:
        return
    prior = next(
        (
            c
            for c in clients
            if c.id == previous.id and c.name == previous.name and c.flags & ACTIVE
        ),
        None,
    )
    if prior:
        controller.show(prior)


def discover_active_hyperspace_wayland(environment):
    """If invoked from CLI/SSH without WAYLAND_DISPLAY while HyperSpace (`xr-shell-session`)
    is running for this UID, attach to its private `xr-wayland-*` socket and Vulkan ICD."""
    wayland = environment.get("WAYLAND_DISPLAY")
    xdg_rt = environment.get("XDG_RUNTIME_DIR")
    if wayland and xdg_rt and (Path(xdg_rt) / wayland).is_socket():
        return
    uid = os.getuid()
    pattern = f"/run/user/{uid}/xr-desktop-dev.*/runtime/xr-wayland-*"
    candidates = sorted(
        (Path(p) for p in glob.glob(pattern) if Path(p).is_socket()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        sock = candidates[0]
        environment["XDG_RUNTIME_DIR"] = str(sock.parent)
        environment["WAYLAND_DISPLAY"] = sock.name
    kms_icd = Path("/usr/lib/aarch64-linux-gnu/xr-mesa/icd.d/freedreno_icd.aarch64.json")
    if kms_icd.is_file() and not environment.get("VK_ICD_FILENAMES"):
        environment["VK_DRIVER_FILES"] = str(kms_icd)
        environment["VK_ICD_FILENAMES"] = str(kms_icd)
        extra_ld = "/opt/mix-native-dependencies/lib:/usr/lib/aarch64-linux-gnu/xr-mesa"
        cur_ld = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = f"{extra_ld}:{cur_ld}" if cur_ld else extra_ld


def launch(arguments):
    binary = Path(__file__).resolve().parents[2] / "bin/xr-files"
    exe = str(binary) if binary.is_file() else "xr-files"
    command = [exe, *arguments]
    environment = dict(os.environ)

    options = arguments[: arguments.index("--")] if "--" in arguments else arguments
    if options[:1] == ["ctl"] or {"-h", "--help", "-V", "--version", "--headless"}.intersection(
        options
    ):
        os.execvpe(command[0], command, environment)

    explicit_xr = "--xr" in options

    # Default behavior: coexist inside HyperSpace desktop (keeping Qingkong sky, Dock,
    # other windows, and unified Shell drag/resize/curve controls)!
    if not explicit_xr:
        discover_active_hyperspace_wayland(environment)
        if environment.get("WAYLAND_DISPLAY") or environment.get("DISPLAY"):
            os.execvpe(command[0], command, environment)

    # Explicit `--xr` standalone immersive OpenXR mode (or headless console with no display)
    try:
        backend, library = runtime_details(environment)
    except CONTROL_ERRORS:
        backend, library = "other", None
    _, directory = runtime_directory_for(environment, backend)
    if directory and explicit_xr:
        environment["XDG_RUNTIME_DIR"] = os.environ["XDG_RUNTIME_DIR"] = directory

    if explicit_xr and environment.get("XR_SCENE_APPLICATION_SOCKET"):
        status_command = [exe, "ctl", '{"op":"status"}']

        def ready():
            try:
                status = subprocess.run(
                    status_command,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                return status.returncode == 0 and json.loads(status.stdout).get("state", {}).get(
                    "ready", False
                )
            except CONTROL_ERRORS:
                return False

        if ready():
            os.execvpe(command[0], command, environment)
        started = subprocess.run(
            ["xr-shell-session", "--launch-application", "xr-files"], env=environment
        )
        if started.returncode or arguments == ["--xr"]:
            return started.returncode
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if ready():
                os.execvpe(command[0], command, environment)
            time.sleep(0.15)
        raise RuntimeError("XR Files did not become ready after native foreground handoff")

    controller = None
    if backend == "monado":
        controller = Monado(NAME)
    elif backend == "wivrn" and library:
        controller = WiVRnControl(library, NAME)

    if controller:
        try:
            clients = controller.clients()
        except CONTROL_ERRORS:
            controller = None

    if controller is None:
        os.execvpe(command[0], command, environment)

    xr_cmd = [exe, "--xr", *arguments] if not explicit_xr else command
    previous = next((c for c in clients if c.flags & PRIMARY and c.flags & ACTIVE), None)
    existing = files_client(clients)
    child = subprocess.Popen(xr_cmd, env=environment)
    signal.signal(signal.SIGTERM, lambda sig, _: child.send_signal(sig))
    shown = None
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            code = child.poll()
            if code is not None:
                if code != 0 or existing is None:
                    return code
                current = files_client(controller.clients())
                if current and current.id == existing.id:
                    controller.show(current)
                return code
            target = files_client(controller.clients())
            if target and (existing is None or target.id != existing.id):
                controller.show(target)
                shown = target
                break
            time.sleep(0.15)
    except CONTROL_ERRORS as error:
        print(f"{NAME}: {error}", file=sys.stderr)
    try:
        return child.wait()
    finally:
        try:
            restore(controller, previous, shown)
        except CONTROL_ERRORS as error:
            print(f"{NAME}: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(launch(sys.argv[1:]))
