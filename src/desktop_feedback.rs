// SPDX-License-Identifier: GPL-3.0-or-later
//! Best-effort focused-window feedback through Shell's installed control client.
//! The bounded worker never stalls rendering or connects directly to hardware.
use crate::app::HapticEvent;
use std::{
    os::unix::fs::FileTypeExt,
    path::PathBuf,
    process::{Command, Stdio},
    sync::mpsc::{self, SyncSender},
    thread,
    time::{Duration, Instant},
};

pub struct DesktopFeedback(Option<SyncSender<(Instant, &'static str)>>);
impl DesktopFeedback {
    pub fn connect() -> Self {
        // Shell is outside the component's private Wayland runtime directory.
        let socket = PathBuf::from(format!("/run/user/{}/xr-desktop/shell.sock", unsafe {
            libc::geteuid()
        }));
        if !socket.metadata().is_ok_and(|m| m.file_type().is_socket()) {
            return Self(None);
        }
        let (tx, rx) = mpsc::sync_channel::<(Instant, &'static str)>(1);
        let worker = thread::Builder::new()
            .name("xr-files-feedback".into())
            .spawn(move || {
                for (at, kind) in rx {
                    if at.elapsed() > Duration::from_millis(100) {
                        continue;
                    }
                    let payload = serde_json::json!({"app_id":"org.openkylin.XRFiles","kind":kind});
                    let Ok(mut child) = Command::new("xr-shellctl")
                        .args(["--socket"])
                        .arg(&socket)
                        .args([
                            "--request",
                            "ApplicationFeedback",
                            "--payload",
                            &payload.to_string(),
                        ])
                        .stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()
                    else {
                        continue;
                    };
                    let deadline = Instant::now() + Duration::from_millis(250);
                    loop {
                        match child.try_wait() {
                            Ok(Some(_)) => break,
                            Ok(None) if Instant::now() < deadline => {
                                thread::sleep(Duration::from_millis(4))
                            }
                            _ => {
                                let _ = child.kill();
                                let _ = child.wait();
                                break;
                            }
                        }
                    }
                }
            });
        Self(worker.ok().map(|_| tx))
    }
    pub fn emit(&self, events: [Option<HapticEvent>; 2]) {
        let Some(tx) = &self.0 else { return };
        // Primary/grip already receive feedback from Shell's surface routing.
        let kind = if events.contains(&Some(HapticEvent::ContextMenu)) {
            Some("context")
        } else if events.contains(&Some(HapticEvent::HoverTick)) {
            Some("hover")
        } else {
            None
        };
        if let Some(kind) = kind {
            let _ = tx.try_send((Instant::now(), kind));
        }
    }
}
