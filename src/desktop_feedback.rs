// SPDX-License-Identifier: GPL-3.0-or-later
//! Bounded, asynchronous feedback through the canonical Shell component client.
use crate::app::HapticEvent;
use std::{
    os::unix::fs::FileTypeExt,
    path::PathBuf,
    sync::mpsc::{self, SyncSender},
    thread,
    time::{Duration, Instant},
};
use xr_shell_protocols::Client;

pub struct DesktopFeedback(Option<SyncSender<(Instant, &'static str)>>);
impl DesktopFeedback {
    pub fn connect() -> Self {
        // Shell is outside the component's private Wayland runtime directory.
        // SAFETY: geteuid has no preconditions and does not mutate process state.
        let uid = unsafe { libc::geteuid() };
        let socket = PathBuf::from(format!("/run/user/{uid}/xr-desktop/shell.sock"));
        Self::connect_path(socket)
    }

    fn connect_path(socket: PathBuf) -> Self {
        if !socket.metadata().is_ok_and(|m| m.file_type().is_socket()) {
            return Self(None);
        }
        let (tx, rx) = mpsc::sync_channel::<(Instant, &'static str)>(1);
        let worker = thread::Builder::new()
            .name("xr-files-feedback".into())
            .spawn(move || {
                let mut client = None;
                loop {
                    let (at, kind) = match rx.recv_timeout(Duration::from_millis(20)) {
                        Ok(event) => event,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            // The shared client also negotiates state notifications.
                            // Drain them while idle so this consumer cannot stall Shell.
                            let healthy = client.as_ref().is_none_or(|c: &Client| loop {
                                match c.recv(0) {
                                    Ok(Some(packet))
                                        if packet.message["type"] != "SessionEnding" => {}
                                    Ok(None) => return true,
                                    _ => return false,
                                }
                            });
                            if !healthy {
                                client = None;
                            }
                            continue;
                        }
                    };
                    if at.elapsed() > Duration::from_millis(100) {
                        continue;
                    }
                    if client.is_none() {
                        client =
                            Client::connect(&socket, "system_ui", &["core.v1", "window.v1"]).ok();
                    }
                    let Some(connection) = client.as_mut() else {
                        continue;
                    };
                    // A reconnect/handshake may outlive the event; never replay it.
                    if at.elapsed() > Duration::from_millis(100) {
                        continue;
                    }
                    let payload = serde_json::json!({"app_id":"org.openkylin.XRFiles","kind":kind});
                    let result = (|| -> anyhow::Result<()> {
                        let request = connection.send("ApplicationFeedback", payload, &[])?;
                        let deadline = Instant::now() + Duration::from_millis(100);
                        while Instant::now() < deadline {
                            if let Some(packet) = connection.recv(10)? {
                                anyhow::ensure!(
                                    packet.message["type"] != "SessionEnding",
                                    "desktop ended"
                                );
                                if packet.message["request_id"] == request {
                                    return Ok(());
                                }
                            }
                        }
                        anyhow::bail!("feedback reply timed out")
                    })();
                    if result.is_err() {
                        client = None;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use xr_shell_protocols::SeqPacket;

    #[test]
    fn feedback_reuses_canonical_connection_and_never_blocks_or_replays_stale_input() {
        let dir = std::env::temp_dir().join(format!("xr-files-feedback-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shell.sock");
        let listener = SeqPacket::listen(&path).unwrap();
        let (seen, received) = mpsc::channel();
        let server = thread::spawn(move || {
            let until = Instant::now() + Duration::from_secs(2);
            let connection = loop {
                if let Some(c) = listener.accept().unwrap() {
                    break c;
                }
                assert!(Instant::now() < until);
                thread::sleep(Duration::from_millis(1));
            };
            assert_eq!(
                connection.recv(1000).unwrap().unwrap().message["type"],
                "Hello"
            );
            connection.send(&serde_json::json!({"type":"Welcome","payload":{"session_generation":"test","component_id":"1","component_generation":"1"}}),&[]).unwrap();
            let first = connection.recv(1000).unwrap().unwrap().message;
            assert_eq!(first["type"], "ApplicationFeedback");
            assert_eq!(first["payload"]["kind"], "context");
            connection.send(&serde_json::json!({"type":"RequestResult","request_id":first["request_id"],"payload":{"routed":false}}),&[]).unwrap();
            seen.send(()).unwrap();
            let second = connection.recv(1000).unwrap().unwrap().message;
            assert_eq!(second["payload"]["kind"], "hover");
            seen.send(()).unwrap();
            // Withhold the second ack. The rendering thread must remain free,
            // and a stale event cannot reconnect and vibrate after the pause.
            thread::sleep(Duration::from_millis(180));
            assert!(listener.accept().unwrap().is_none());
        });
        let feedback = DesktopFeedback::connect_path(path);
        feedback.emit([Some(HapticEvent::ContextMenu), None]);
        received.recv_timeout(Duration::from_secs(2)).unwrap();
        feedback.emit([Some(HapticEvent::HoverTick), None]);
        received.recv_timeout(Duration::from_secs(2)).unwrap();
        feedback
            .0
            .as_ref()
            .unwrap()
            .try_send((Instant::now() - Duration::from_secs(1), "hover"))
            .unwrap();
        let start = Instant::now();
        for _ in 0..500 {
            feedback.emit([Some(HapticEvent::HoverTick), None]);
        }
        assert!(start.elapsed() < Duration::from_millis(50));
        server.join().unwrap();
        drop(feedback);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
