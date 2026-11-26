"""Client selection and runtime directory discovery for native HyperSpace launchers."""

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PRIMARY = 1 << 0
ACTIVE = 1 << 1
FOCUSED = 1 << 3
OVERLAY = 1 << 4

CLIENT_ROW = re.compile(
    r"^\tid: (\d+)\tact: ([01])\tdisp: ([01])\tfoc: ([01])\tio: [01]"
    r"\tovly: ([01])\tz: -?\d+\tpid: \d+\t([^\n]*)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Client:
    id: int
    name: str
    flags: int


class Monado:
    def __init__(self, label="OpenXR"):
        self.label = label

    @staticmethod
    def _run(*arguments):
        result = subprocess.run(
            ["monado-ctl", *arguments],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            env=dict(os.environ, LC_ALL="C"),
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "monado-ctl failed")
        return result.stdout

    def clients(self):
        clients = []
        for client_id, active, visible, focused, overlay, name in CLIENT_ROW.findall(self._run()):
            flags = ACTIVE if active == "1" else 0
            if focused == "1":
                flags |= FOCUSED
            if overlay == "1":
                flags |= OVERLAY
            elif visible == "1":
                flags |= PRIMARY
            clients.append(Client(int(client_id), name, flags))
        if not clients:
            raise RuntimeError("Could not read the Monado 25 client list")
        return clients

    def show(self, client):
        client_id = str(client.id)
        if client.flags & PRIMARY and client.flags & FOCUSED:
            return
        if not client.flags & (OVERLAY | PRIMARY):
            self._run("-p", client_id)
        self._run("-f", client_id)
        print(f"{self.label}: showing {client.name}", flush=True)


class WiVRnControl:
    def __init__(self, library, label="OpenXR"):
        self.library = library
        self.label = label

    def _request(self, *arguments):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--wivrn-control",
                str(self.library),
                *arguments,
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=6,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "WiVRn session control failed")
        return json.loads(result.stdout) if result.stdout.strip() else None

    def clients(self):
        return [Client(*row) for row in self._request("list")]

    def show(self, client):
        if client.flags & PRIMARY and client.flags & FOCUSED:
            return
        promote = "keep-primary" if client.flags & (PRIMARY | OVERLAY) else "set-primary"
        self._request("show", str(client.id), promote)
        print(f"{self.label}: showing {client.name}", flush=True)


def wivrn_control(library, operation, *arguments):
    c = ctypes
    pointer, number, string = c.POINTER(c.c_void_p), c.POINTER(c.c_uint32), c.POINTER(c.c_char_p)
    api = c.CDLL(library)
    signatures = {
        "create": (pointer,),
        "destroy": (pointer,),
        "update_client_list": (c.c_void_p,),
        "get_number_clients": (c.c_void_p, number),
        "get_client_id_at_index": (c.c_void_p, c.c_uint32, number),
        "get_client_name": (c.c_void_p, c.c_uint32, string),
        "get_client_state": (c.c_void_p, c.c_uint32, number),
        "set_client_primary": (c.c_void_p, c.c_uint32),
        "set_client_focused": (c.c_void_p, c.c_uint32),
    }
    for name, types in signatures.items():
        function = getattr(api, "mnd_root_" + name)
        function.argtypes = types
        function.restype = None if name == "destroy" else c.c_int

    def call(name, *args):
        if getattr(api, "mnd_root_" + name)(*args) < 0:
            raise RuntimeError(f"WiVRn {name} failed")

    root = c.c_void_p()
    call("create", c.byref(root))
    try:
        call("update_client_list", root)
        if operation == "show":
            client_id, promote = arguments
            if promote == "set-primary":
                call("set_client_primary", root, int(client_id))
            call("set_client_focused", root, int(client_id))
            return
        count = c.c_uint32()
        call("get_number_clients", root, c.byref(count))
        clients = []
        for index in range(count.value):
            client_id, flags, name = c.c_uint32(), c.c_uint32(), c.c_char_p()
            call("get_client_id_at_index", root, index, c.byref(client_id))
            call("get_client_state", root, client_id, c.byref(flags))
            call("get_client_name", root, client_id, c.byref(name))
            clients.append(
                (client_id.value, (name.value or b"").decode(errors="replace"), flags.value)
            )
        print(json.dumps(clients), flush=True)
    finally:
        api.mnd_root_destroy(c.byref(root))


def runtime_details(environment):
    selected = environment.get("XR_RUNTIME_JSON")
    user_config = Path(environment.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    manifests = (
        [Path(selected)]
        if selected
        else [
            user_config / "openxr/1/active_runtime.json",
            Path("/etc/xdg/openxr/1/active_runtime.json"),
        ]
    )
    manifest = next((path for path in manifests if path.is_file()), None)
    if manifest is None:
        return "other", None
    manifest = manifest.resolve()
    runtime = json.loads(manifest.read_text(encoding="utf-8"))["runtime"]
    library = (manifest.parent / runtime["library_path"]).resolve()
    backend = next((kind for kind in ("monado", "wivrn") if kind in library.name.lower()), "other")
    control_library = (
        next(iter(sorted(library.parent.glob("libmonado_wivrn.so*"))), None)
        if backend == "wivrn"
        else None
    )
    return backend, control_library


def runtime_directory_for(environment, backend):
    runtime_directory = environment.get("XR_OPENXR_RUNTIME_DIR")
    wayland = environment.get("WAYLAND_DISPLAY")
    hyperspace = bool(
        runtime_directory or (wayland and Path(wayland).name.startswith("xr-wayland-"))
    )
    if hyperspace and backend in {"monado", "wivrn"} and not runtime_directory:
        socket = "wivrn/comp_ipc" if backend == "wivrn" else "monado_comp_ipc"
        candidates = [environment.get("PIPEWIRE_RUNTIME_DIR"), f"/run/user/{os.getuid()}"]
        runtime_directory = next(
            (path for path in candidates if path and (Path(path) / socket).is_socket()), None
        )
    return hyperspace, runtime_directory


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--wivrn-control":
        wivrn_control(*sys.argv[2:])
    else:
        raise SystemExit("expected --wivrn-control LIBRARY OPERATION")
