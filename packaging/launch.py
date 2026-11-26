#!/usr/bin/env python3
"""Launch openKylin XR Files in native OpenXR stereo or HyperSpace Wayland mode."""

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


def launch(arguments):
    binary = Path(__file__).resolve().parents[2] / "bin/xr-files"
    exe = str(binary) if binary.is_file() else "xr-files"
    command = [exe, *arguments]
    environment = dict(os.environ)

    try:
        backend, library = runtime_details(environment)
    except CONTROL_ERRORS:
        backend, library = "other", None
    hyperspace, directory = runtime_directory_for(environment, backend)
    if directory:
        environment["XDG_RUNTIME_DIR"] = os.environ["XDG_RUNTIME_DIR"] = directory

    controller = None
    if backend == "monado":
        controller = Monado(NAME)
    elif backend == "wivrn" and library:
        controller = WiVRnControl(library, NAME)

    options = arguments[: arguments.index("--")] if "--" in arguments else arguments
    non_session_flags = {"--headless", "--window", "-h", "--help", "-V", "--version"}
    runs_xr_session = options[:1] != ["ctl"] and not non_session_flags.intersection(options)

    if runs_xr_session and environment.get("XR_SCENE_APPLICATION_SOCKET"):
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
        if started.returncode or not arguments:
            return started.returncode
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if ready():
                os.execvpe(command[0], command, environment)
            time.sleep(0.15)
        raise RuntimeError("XR Files did not become ready after native foreground handoff")

    if controller and runs_xr_session:
        try:
            clients = controller.clients()
        except CONTROL_ERRORS:
            controller = None
    else:
        controller = None

    if controller is None:
        # When inside HyperSpace private Wayland without multi-client XR session handoff,
        # run in interactive wgpu Wayland window mode so it coexists with xr-shell.
        if hyperspace and runs_xr_session and "--xr" not in options:
            window_cmd = [exe, "--window", *arguments]
            os.execvpe(window_cmd[0], window_cmd, environment)
        os.execvpe(command[0], command, environment)

    xr_cmd = [exe, "--xr", *arguments] if "--xr" not in options else command
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
