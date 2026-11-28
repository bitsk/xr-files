// SPDX-License-Identifier: GPL-3.0-or-later
//! Local Smart Control Interface v1 (`$XDG_RUNTIME_DIR/xr-files/control.sock`)
//! Compatible with HyperSpace private runtime directories, CLI forwarding (`xr-files <dir>`),
//! and voice / AI Agent automation (`xr-files ctl '{"op":"status"}'`).

use crate::app::{SpatialFilesApp, StatusSnapshot, ViewMode};
use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    os::unix::{
        fs::PermissionsExt,
        net::{UnixListener, UnixStream},
    },
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, SyncSender},
    thread,
    time::Duration,
};

const MAX_REQUEST_BYTES: usize = 16 * 1024;

fn parse_envelope(raw: &str) -> Result<(u32, Value, ControlCommand)> {
    let v: Value = serde_json::from_str(raw)?;
    let version = v.get("version").and_then(Value::as_u64).unwrap_or(1) as u32;
    let id = v.get("id").cloned().unwrap_or(Value::Null);
    let cmd_val = v.get("command").cloned().unwrap_or(v);
    let cmd: ControlCommand = serde_json::from_value(cmd_val)?;
    Ok((version, id, cmd))
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ControlCommand {
    Status,
    Open { path: String },
    Search { query: String },
    SetView { view: ViewMode },
    Mkdir { name: String },
    Rename { name: String },
    Trash,
    Undo,
    Up,
    Recenter,
    Quit,
}

#[derive(Debug, Serialize)]
struct ControlReply {
    version: u32,
    id: Value,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<StatusSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

struct PendingRequest {
    version: u32,
    id: Value,
    command: ControlCommand,
    reply_tx: SyncSender<ControlReply>,
}

pub struct ControlServer {
    socket_path: PathBuf,
    rx: Receiver<PendingRequest>,
    pub(crate) wake: crate::wake::Wake,
}

impl Drop for ControlServer {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.socket_path);
    }
}

pub fn default_socket_path() -> PathBuf {
    let base = env::var_os("XR_OPENXR_RUNTIME_DIR")
        .or_else(|| env::var_os("XDG_RUNTIME_DIR"))
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .map(|p| p.join("xr-files"))
        .unwrap_or_else(|| {
            // SAFETY: `geteuid` is thread-safe and cannot fail.
            let uid = unsafe { libc::geteuid() };
            env::temp_dir().join(format!("xr-files-{uid}"))
        });
    base.join("control.sock")
}

impl ControlServer {
    pub fn bind_default() -> Result<Self> {
        Self::bind(&default_socket_path())
    }

    pub fn bind(socket_path: &Path) -> Result<Self> {
        if let Some(parent) = socket_path.parent() {
            fs::create_dir_all(parent)?;
            let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o700));
        }
        if socket_path.exists() {
            // Do not clobber a live server
            if send_raw_command(socket_path, r#"{"op":"status"}"#).is_ok() {
                bail!("已有正在运行的 xr-files 会话: {}", socket_path.display());
            }
            let _ = fs::remove_file(socket_path);
        }
        let listener = UnixListener::bind(socket_path)
            .with_context(|| format!("绑定控制 socket 失败: {}", socket_path.display()))?;
        fs::set_permissions(socket_path, fs::Permissions::from_mode(0o600))?;

        let (tx, rx) = mpsc::channel::<PendingRequest>();
        let wake = crate::wake::Wake::default();
        let worker_wake = wake.clone();
        thread::Builder::new()
            .name("xr-files-ctl".into())
            .spawn(move || accept_loop(listener, tx, worker_wake))?;

        Ok(Self {
            socket_path: socket_path.to_path_buf(),
            rx,
            wake,
        })
    }

    pub fn poll_and_apply(&self, app: &mut SpatialFilesApp) -> bool {
        let mut changed = false;
        while let Ok(req) = self.rx.try_recv() {
            let updates_scene = !matches!(req.command, ControlCommand::Status);
            let reply = execute_command(app, req.version, req.id, req.command);
            changed |= updates_scene && reply.ok;
            let _ = req.reply_tx.send(reply);
        }
        changed
    }
}

fn accept_loop(listener: UnixListener, tx: mpsc::Sender<PendingRequest>, wake: crate::wake::Wake) {
    for incoming in listener.incoming() {
        let Ok(mut stream) = incoming else { continue };
        let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));

        let mut line = String::new();
        {
            let reader = BufReader::new(&stream);
            let mut limited = std::io::Read::take(reader, (MAX_REQUEST_BYTES + 1) as u64);
            if limited.read_line(&mut line).is_err() || line.len() > MAX_REQUEST_BYTES {
                let _ = write_json_line(
                    &mut stream,
                    &ControlReply {
                        version: 1,
                        id: Value::Null,
                        ok: false,
                        state: None,
                        error: Some("请求过大或读取失败".into()),
                    },
                );
                continue;
            }
        }
        match parse_envelope(line.trim()) {
            Ok((version, id, cmd)) => {
                if version != 1 {
                    let _ = write_json_line(
                        &mut stream,
                        &ControlReply {
                            version,
                            id,
                            ok: false,
                            state: None,
                            error: Some("仅支持控制协议 version=1".into()),
                        },
                    );
                    continue;
                }
                let (reply_tx, reply_rx) = mpsc::sync_channel(1);
                if tx
                    .send(PendingRequest {
                        version,
                        id: id.clone(),
                        command: cmd,
                        reply_tx,
                    })
                    .is_ok()
                {
                    wake.notify();
                    if let Ok(reply) = reply_rx.recv_timeout(Duration::from_secs(3)) {
                        let _ = write_json_line(&mut stream, &reply);
                    }
                }
            }
            Err(err) => {
                let _ = write_json_line(
                    &mut stream,
                    &ControlReply {
                        version: 1,
                        id: Value::Null,
                        ok: false,
                        state: None,
                        error: Some(format!("无效 JSON 命令: {err}")),
                    },
                );
            }
        }
    }
}

fn execute_command(
    app: &mut SpatialFilesApp,
    version: u32,
    id: Value,
    cmd: ControlCommand,
) -> ControlReply {
    let outcome: Result<()> = (|| {
        match cmd {
            ControlCommand::Status => {}
            ControlCommand::Open { path } => {
                let p = PathBuf::from(&path);
                ensure!(
                    p.is_absolute() || path == "trash://",
                    "open 需要绝对路径或 trash://"
                );
                if path == "trash://" || p.is_dir() {
                    app.navigate_to(path);
                } else if p.exists() {
                    if let Some(parent) = p.parent() {
                        app.navigate_to(parent.to_string_lossy().into_owned());
                    }
                    if let Some((idx, _)) = app
                        .all_entries
                        .iter()
                        .enumerate()
                        .find(|(_, e)| e.path == p)
                    {
                        app.select_single(idx);
                    }
                } else {
                    bail!("目标路径不存在: {path}");
                }
            }
            ControlCommand::Search { query } => app.set_search_query(query),
            ControlCommand::SetView { view } => app.set_view_mode(view),
            ControlCommand::Mkdir { name } => {
                app.create_folder_named(&name)?;
            }
            ControlCommand::Rename { name } => {
                app.rename_selected(&name)?;
            }
            ControlCommand::Trash => {
                app.trash_or_restore_selected()?;
            }
            ControlCommand::Undo => {
                app.undo_last()?;
            }
            ControlCommand::Up => app.navigate_up(),
            ControlCommand::Recenter => app.request_recenter(),
            ControlCommand::Quit => app.request_shutdown(),
        }
        Ok(())
    })();

    match outcome {
        Ok(()) => ControlReply {
            version,
            id,
            ok: true,
            state: Some(app.status_snapshot()),
            error: None,
        },
        Err(err) => ControlReply {
            version,
            id,
            ok: false,
            state: None,
            error: Some(err.to_string()),
        },
    }
}

fn write_json_line(stream: &mut UnixStream, reply: &ControlReply) -> Result<()> {
    let mut text = serde_json::to_string(reply)?;
    text.push('\n');
    stream.write_all(text.as_bytes())?;
    stream.flush()?;
    Ok(())
}

pub fn send_raw_command(socket_path: &Path, raw_json: &str) -> Result<String> {
    let mut stream = UnixStream::connect(socket_path)
        .with_context(|| format!("连接 {} 失败", socket_path.display()))?;
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    stream.set_write_timeout(Some(Duration::from_secs(3)))?;
    let mut line = raw_json.trim().to_string();
    ensure!(!line.is_empty(), "控制指令不能为空");
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    stream.flush()?;
    let mut reply = String::new();
    BufReader::new(stream).read_line(&mut reply)?;
    if reply.trim().is_empty() {
        return Err(anyhow!("未收到控制端响应"));
    }
    Ok(reply.trim().to_string())
}

/// Attempts to forward an initial path to an already-running `xr-files` instance.
pub fn try_forward_open_to_existing(path: &Path) -> bool {
    let sock = default_socket_path();
    if !sock.exists() {
        return false;
    }
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else if let Ok(cwd) = env::current_dir() {
        cwd.join(path)
    } else {
        return false;
    };
    let req = serde_json::json!({
        "version": 1,
        "id": "forward",
        "command": {
            "op": "open",
            "path": abs.to_string_lossy(),
        }
    });
    if let Ok(resp) = send_raw_command(&sock, &req.to_string()) {
        if let Ok(v) = serde_json::from_str::<Value>(&resp) {
            return v["ok"].as_bool() == Some(true);
        }
    }
    false
}
