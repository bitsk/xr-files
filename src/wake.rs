// SPDX-License-Identifier: GPL-3.0-or-later
use std::{
    io,
    os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
    sync::{Arc, OnceLock},
};

/// The Wayland loop registers this fd before draining its worker queues. Other
/// render loops already drain them on each frame and need no notifications.
#[derive(Clone, Default)]
pub(crate) struct Wake(Arc<OnceLock<OwnedFd>>);

impl Wake {
    pub fn enable(&self) -> io::Result<RawFd> {
        if self.0.get().is_none() {
            let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            let _ = self.0.set(unsafe { OwnedFd::from_raw_fd(fd) });
        }
        Ok(self.0.get().unwrap().as_raw_fd())
    }

    pub fn notify(&self) {
        if let Some(fd) = self.0.get() {
            let bytes = 1u64.to_ne_bytes();
            loop {
                let rc = unsafe { libc::write(fd.as_raw_fd(), bytes.as_ptr().cast(), bytes.len()) };
                if rc >= 0 || io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                    break;
                }
            }
        }
    }

    pub fn drain(&self) {
        if let Some(fd) = self.0.get() {
            let mut bytes = [0u8; 8];
            loop {
                let rc =
                    unsafe { libc::read(fd.as_raw_fd(), bytes.as_mut_ptr().cast(), bytes.len()) };
                if rc >= 0 || io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                    break;
                }
            }
        }
    }
}
