// SPDX-License-Identifier: GPL-3.0-or-later
//! File system operations, storage volume telemetry, XDG Trash 1.0, clipboard,
//! reversible undo stack, and `xr-cinema` / `xdg-open` launch dispatch.

use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    env,
    ffi::CString,
    fs,
    io::Write,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

pub const TRASH_URI: &str = "trash://";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileKind {
    Directory,
    Image,
    Video,
    Audio,
    Model3D,
    Code,
    Document,
    Archive,
    Executable,
    Other,
}

impl FileKind {
    pub fn classify(path: &Path, is_dir: bool, mode: u32) -> Self {
        if is_dir {
            return Self::Directory;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        match ext.as_str() {
            "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif" | "ico" | "svg" | "tga" | "avif" => {
                Self::Image
            }
            "mp4" | "mkv" | "webm" | "mov" | "avi" | "mpeg" | "mpg" | "m4v" | "m2ts" | "mts"
            | "ogv" => Self::Video,
            "mp3" | "flac" | "wav" | "ogg" | "opus" | "m4a" | "aac" => Self::Audio,
            "obj" | "stl" | "glb" | "gltf" | "ply" | "fbx" | "usd" | "usda" | "usdz" => {
                Self::Model3D
            }
            "rs" | "py" | "c" | "cc" | "cpp" | "h" | "hpp" | "js" | "ts" | "tsx" | "jsx"
            | "wgsl" | "glsl" | "vert" | "frag" | "sh" | "bash" | "toml" | "json" | "yaml"
            | "yml" | "xml" | "html" | "css" | "go" | "java" | "sql" | "cmake" => Self::Code,
            "md" | "txt" | "pdf" | "doc" | "docx" | "wps" | "odt" | "ppt" | "pptx" | "xls"
            | "xlsx" | "csv" | "rtf" | "epub" | "log" => Self::Document,
            "zip" | "tar" | "gz" | "tgz" | "xz" | "bz2" | "zst" | "7z" | "rar" | "deb" | "rpm"
            | "iso" | "img" => Self::Archive,
            "appimage" | "run" | "bin" => Self::Executable,
            _ if mode & 0o111 != 0 => Self::Executable,
            _ => Self::Other,
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            Self::Directory => "文件夹",
            Self::Image => "图像",
            Self::Video => "空间视频",
            Self::Audio => "音频",
            Self::Model3D => "3D 模型",
            Self::Code => "源码配置",
            Self::Document => "文档",
            Self::Archive => "压缩包",
            Self::Executable => "可执行程序",
            Self::Other => "文件",
        }
    }

    pub fn badge_code(self) -> &'static str {
        match self {
            Self::Directory => "DIR",
            Self::Image => "IMG",
            Self::Video => "VID",
            Self::Audio => "AUD",
            Self::Model3D => "3D",
            Self::Code => "SRC",
            Self::Document => "DOC",
            Self::Archive => "PKG",
            Self::Executable => "BIN",
            Self::Other => "DAT",
        }
    }

    /// Linear RGBA accent tint used on 3D volumetric cards and 3D treemap blocks.
    pub fn accent_rgba(self) -> [f32; 4] {
        match self {
            Self::Directory => [0.22, 0.58, 0.98, 1.0],
            Self::Image => [0.18, 0.80, 0.64, 1.0],
            Self::Video => [0.94, 0.36, 0.52, 1.0],
            Self::Audio => [0.95, 0.62, 0.20, 1.0],
            Self::Model3D => [0.64, 0.42, 0.98, 1.0],
            Self::Code => [0.24, 0.76, 0.92, 1.0],
            Self::Document => [0.46, 0.66, 0.96, 1.0],
            Self::Archive => [0.88, 0.68, 0.26, 1.0],
            Self::Executable => [0.34, 0.86, 0.48, 1.0],
            Self::Other => [0.54, 0.62, 0.74, 1.0],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub path: PathBuf,
    pub kind: FileKind,
    pub size: u64,
    pub modified_unix: u64,
    pub readonly: bool,
    pub hidden: bool,
    pub is_symlink: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trash_original: Option<PathBuf>,
    #[serde(skip)]
    pub trash_info_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortBy {
    Name,
    Size,
    Modified,
    Kind,
}

impl SortBy {
    pub fn next(self) -> Self {
        match self {
            Self::Name => Self::Size,
            Self::Size => Self::Modified,
            Self::Modified => Self::Kind,
            Self::Kind => Self::Name,
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            Self::Name => "按名称",
            Self::Size => "按大小",
            Self::Modified => "按时间",
            Self::Kind => "按类型",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bookmark {
    pub label: String,
    pub target: String,
    pub icon_tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageVolume {
    pub name: String,
    pub mount_point: PathBuf,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipboardOp {
    Copy,
    Cut,
}

#[derive(Debug, Clone)]
pub struct FileClipboard {
    pub op: ClipboardOp,
    pub paths: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub enum UndoRecord {
    CreatedDir(PathBuf),
    CreatedFile(PathBuf),
    Renamed {
        from: PathBuf,
        to: PathBuf,
    },
    Copied {
        created: Vec<PathBuf>,
    },
    Moved {
        moves: Vec<(PathBuf, PathBuf)>, // (original_path, current_path)
    },
    Trashed {
        items: Vec<TrashedRecord>,
    },
}

#[derive(Debug, Clone)]
pub struct TrashedRecord {
    pub original_path: PathBuf,
    pub trashed_file: PathBuf,
    pub info_file: PathBuf,
}

impl UndoRecord {
    pub fn summary(&self) -> String {
        match self {
            Self::CreatedDir(p) => format!("新建文件夹 {}", file_name_str(p)),
            Self::CreatedFile(p) => format!("新建文件 {}", file_name_str(p)),
            Self::Renamed { to, .. } => format!("重命名 {}", file_name_str(to)),
            Self::Copied { created } => format!("复制 {} 项", created.len()),
            Self::Moved { moves } => format!("移动 {} 项", moves.len()),
            Self::Trashed { items } => format!("移入回收站 {} 项", items.len()),
        }
    }
}

pub fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

pub fn trash_root() -> PathBuf {
    if let Some(xdg) = env::var_os("XDG_DATA_HOME").map(PathBuf::from) {
        xdg.join("Trash")
    } else {
        home_dir().join(".local/share/Trash")
    }
}

/// Discovers standard user bookmarks (both Chinese UKUI and English XDG directories).
pub fn discover_bookmarks() -> Vec<Bookmark> {
    let home = home_dir();
    let mut out = vec![Bookmark {
        label: "主目录".into(),
        target: home.to_string_lossy().into_owned(),
        icon_tag: "HOME".into(),
    }];
    let candidates = [
        ("桌面", &["桌面", "Desktop"][..], "DESK"),
        ("文档", &["文档", "Documents"][..], "DOCS"),
        ("下载", &["下载", "Downloads"][..], "DOWN"),
        ("图片", &["图片", "Pictures"][..], "PICS"),
        ("视频", &["视频", "Videos"][..], "VIDS"),
        ("音乐", &["音乐", "Music"][..], "MUSC"),
    ];
    for (label, names, tag) in candidates {
        if let Some(found) = names.iter().map(|n| home.join(n)).find(|p| p.is_dir()) {
            out.push(Bookmark {
                label: label.into(),
                target: found.to_string_lossy().into_owned(),
                icon_tag: tag.into(),
            });
        }
    }
    out.push(Bookmark {
        label: "回收站".into(),
        target: TRASH_URI.into(),
        icon_tag: "TRSH".into(),
    });
    out
}

/// Queries mounted storage volumes with real `statvfs` capacity telemetry.
pub fn discover_volumes() -> Vec<StorageVolume> {
    let mut mounts = vec![("/", "系统根分区".to_string())];
    let home = home_dir();
    if home != Path::new("/") {
        mounts.push((
            Box::leak(home.to_string_lossy().into_owned().into_boxed_str()),
            "用户空间".to_string(),
        ));
    }
    // Wait: avoid Box::leak! Build owned tuples cleanly.
    drop(mounts);

    let mut candidates: Vec<(PathBuf, String)> = vec![(PathBuf::from("/"), "系统卷 /".into())];
    if let Ok(content) = fs::read_to_string("/proc/mounts") {
        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                continue;
            }
            let mp = unescape_mount_path(parts[1]);
            let fstype = parts[2];
            if matches!(
                fstype,
                "proc"
                    | "sysfs"
                    | "devtmpfs"
                    | "devpts"
                    | "tmpfs"
                    | "cgroup"
                    | "cgroup2"
                    | "pstore"
                    | "securityfs"
                    | "debugfs"
                    | "tracefs"
                    | "configfs"
                    | "fusectl"
                    | "mqueue"
                    | "hugetlbfs"
                    | "autofs"
                    | "squashfs"
                    | "overlay"
            ) {
                continue;
            }
            if mp.starts_with("/media/")
                || mp.starts_with("/run/media/")
                || mp.starts_with("/mnt/")
                || mp == "/home"
            {
                let path = PathBuf::from(&mp);
                let title = path
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| mp.clone());
                if !candidates.iter().any(|(p, _)| p == &path) {
                    candidates.push((path, title));
                }
            }
        }
    }

    let mut volumes = Vec::new();
    let mut seen_devs = Vec::new();
    for (mp, name) in candidates {
        if let Some((dev_id, total, free)) = statvfs_info(&mp) {
            if total == 0 || seen_devs.contains(&dev_id) {
                continue;
            }
            seen_devs.push(dev_id);
            volumes.push(StorageVolume {
                name,
                mount_point: mp,
                total_bytes: total,
                free_bytes: free,
            });
        }
    }
    volumes
}

fn unescape_mount_path(raw: &str) -> String {
    raw.replace("\\040", " ").replace("\\011", "\t")
}

fn statvfs_info(path: &Path) -> Option<(u64, u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let meta = fs::metadata(path).ok()?;
    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated path and `stat` is zero-initialized.
    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut stat) != 0 {
            return None;
        }
        let frsize = if stat.f_frsize > 0 {
            stat.f_frsize as u64
        } else {
            stat.f_bsize as u64
        };
        let total = (stat.f_blocks as u64).saturating_mul(frsize);
        let avail = (stat.f_bavail as u64).saturating_mul(frsize);
        Some((meta.dev(), total, avail))
    }
}

/// Reads entries of a regular filesystem directory or `trash://`.
pub fn read_location_entries(location: &str, show_hidden: bool) -> Result<Vec<FileEntry>> {
    if location == TRASH_URI {
        return read_trash_entries();
    }
    let dir = Path::new(location);
    let rd = fs::read_dir(dir).with_context(|| format!("无法读取目录 {}", dir.display()))?;
    let mut entries = Vec::new();
    for item in rd {
        let Ok(item) = item else { continue };
        let name = item.file_name().to_string_lossy().into_owned();
        let hidden = name.starts_with('.');
        if hidden && !show_hidden {
            continue;
        }
        let path = item.path();
        let symlink_meta = fs::symlink_metadata(&path);
        let is_symlink = symlink_meta
            .as_ref()
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);
        let meta = fs::metadata(&path).or(symlink_meta);
        let Ok(meta) = meta else { continue };
        let is_dir = meta.is_dir();
        #[cfg(unix)]
        let mode = {
            use std::os::unix::fs::PermissionsExt;
            meta.permissions().mode()
        };
        #[cfg(not(unix))]
        let mode = 0u32;

        let size = if is_dir {
            // Fast shallow child count proxy or immediate metadata size until background scan updates it
            shallow_dir_estimate(&path).unwrap_or(4096)
        } else {
            meta.len()
        };
        let modified_unix = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        entries.push(FileEntry {
            name,
            kind: FileKind::classify(&path, is_dir, mode),
            path,
            size,
            modified_unix,
            readonly: meta.permissions().readonly(),
            hidden,
            is_symlink,
            trash_original: None,
            trash_info_file: None,
        });
    }
    Ok(entries)
}

/// Computes a fast shallow byte sum of immediate files in `dir` so 3D Treemap has meaningful
/// directory weights without stalling on deep filesystem trees.
fn shallow_dir_estimate(dir: &Path) -> Option<u64> {
    let rd = fs::read_dir(dir).ok()?;
    let mut sum = 4096u64;
    for entry in rd.take(128).flatten() {
        if let Ok(m) = entry.metadata() {
            if m.is_file() {
                sum = sum.saturating_add(m.len());
            } else if m.is_dir() {
                sum = sum.saturating_add(8192);
            }
        }
    }
    Some(sum)
}

pub fn sort_entries(entries: &mut [FileEntry], sort_by: SortBy, ascending: bool) {
    entries.sort_by(|a, b| {
        // Keep directories grouped ahead of files in primary order
        let dir_cmp = (b.kind == FileKind::Directory).cmp(&(a.kind == FileKind::Directory));
        if dir_cmp != std::cmp::Ordering::Equal {
            return dir_cmp;
        }
        let ord = match sort_by {
            SortBy::Name => a
                .name
                .to_ascii_lowercase()
                .cmp(&b.name.to_ascii_lowercase())
                .then_with(|| a.name.cmp(&b.name)),
            SortBy::Size => a.size.cmp(&b.size).then_with(|| a.name.cmp(&b.name)),
            SortBy::Modified => a
                .modified_unix
                .cmp(&b.modified_unix)
                .then_with(|| a.name.cmp(&b.name)),
            SortBy::Kind => a
                .kind
                .badge_code()
                .cmp(b.kind.badge_code())
                .then_with(|| a.name.to_ascii_lowercase().cmp(&b.name.to_ascii_lowercase())),
        };
        if ascending {
            ord
        } else {
            ord.reverse()
        }
    });
}

// ============================================================================
// XDG FreeDesktop Trash 1.0 Implementation
// ============================================================================

pub fn read_trash_entries() -> Result<Vec<FileEntry>> {
    let root = trash_root();
    let files_dir = root.join("files");
    let info_dir = root.join("info");
    if !files_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in fs::read_dir(&files_dir)? {
        let Ok(item) = item else { continue };
        let path = item.path();
        let name = item.file_name().to_string_lossy().into_owned();
        let Ok(meta) = fs::symlink_metadata(&path) else {
            continue;
        };
        let is_dir = meta.is_dir();
        use std::os::unix::fs::PermissionsExt;
        let mode = meta.permissions().mode();
        let info_file = info_dir.join(format!("{name}.trashinfo"));
        let orig = parse_trashinfo_original_path(&info_file);
        let modified_unix = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        out.push(FileEntry {
            name,
            kind: FileKind::classify(&path, is_dir, mode),
            size: if is_dir {
                shallow_dir_estimate(&path).unwrap_or(4096)
            } else {
                meta.len()
            },
            path,
            modified_unix,
            readonly: meta.permissions().readonly(),
            hidden: false,
            is_symlink: meta.file_type().is_symlink(),
            trash_original: orig,
            trash_info_file: Some(info_file),
        });
    }
    Ok(out)
}

fn parse_trashinfo_original_path(info_file: &Path) -> Option<PathBuf> {
    let text = fs::read_to_string(info_file).ok()?;
    for line in text.lines() {
        if let Some(raw) = line.trim().strip_prefix("Path=") {
            return Some(PathBuf::from(percent_decode(raw)));
        }
    }
    None
}

fn percent_encode_path(path: &Path) -> String {
    let mut out = String::new();
    for &b in path.as_os_str().as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'/' | b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(val) = u8::from_str_radix(hex, 16) {
                    out.push(val);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn format_utc_iso8601(secs: u64) -> String {
    let time_t = secs as libc::time_t;
    // SAFETY: `gmtime_r` writes into stack-allocated `tm` without global state.
    unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        if libc::gmtime_r(&time_t, &mut tm).is_null() {
            return "1970-01-01T00:00:00".into();
        }
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec
        )
    }
}

pub fn trash_paths(paths: &[PathBuf]) -> Result<UndoRecord> {
    return trash_paths_into(paths, &trash_root());
}

pub fn trash_paths_into(paths: &[PathBuf], trash_base: &Path) -> Result<UndoRecord> {
    ensure!(!paths.is_empty(), "未选择要移入回收站的项目");
    let files_dir = trash_base.join("files");
    let info_dir = trash_base.join("info");
    fs::create_dir_all(&files_dir)?;
    fs::create_dir_all(&info_dir)?;

    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let deletion_date = format_utc_iso8601(now_secs);
    let mut records = Vec::with_capacity(paths.len());

    for src in paths {
        let abs = if src.is_absolute() {
            src.clone()
        } else {
            env::current_dir()?.join(src)
        };
        ensure!(
            fs::symlink_metadata(&abs).is_ok(),
            "项目不存在: {}",
            abs.display()
        );
        let base_name = abs
            .file_name()
            .ok_or_else(|| anyhow!("无法将根目录移入回收站"))?
            .to_string_lossy()
            .into_owned();

        let (trashed_file, info_file) = reserve_trash_slot(&files_dir, &info_dir, &base_name, &abs, &deletion_date)?;
        if let Err(err) = move_single_path(&abs, &trashed_file) {
            let _ = fs::remove_file(&info_file);
            return Err(err);
        }
        records.push(TrashedRecord {
            original_path: abs,
            trashed_file,
            info_file,
        });
    }
    Ok(UndoRecord::Trashed { items: records })
}

fn reserve_trash_slot(
    files_dir: &Path,
    info_dir: &Path,
    base_name: &str,
    original_abs: &Path,
    deletion_date: &str,
) -> Result<(PathBuf, PathBuf)> {
    let encoded = percent_encode_path(original_abs);
    let payload = format!("[Trash Info]\nPath={encoded}\nDeletionDate={deletion_date}\n");
    for attempt in 0..10_000u32 {
        let candidate_name = if attempt == 0 {
            base_name.to_string()
        } else {
            let (stem, ext) = split_stem_ext(base_name);
            if ext.is_empty() {
                format!("{stem}.{attempt}")
            } else {
                format!("{stem}.{attempt}.{ext}")
            }
        };
        let target_file = files_dir.join(&candidate_name);
        if target_file.exists() {
            continue;
        }
        let info_file = info_dir.join(format!("{candidate_name}.trashinfo"));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&info_file)
        {
            Ok(mut f) => {
                f.write_all(payload.as_bytes())?;
                return Ok((target_file, info_file));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    bail!("回收站同名槽位已满")
}

pub fn restore_trash_entry(entry: &FileEntry) -> Result<PathBuf> {
    let orig = entry
        .trash_original
        .clone()
        .unwrap_or_else(|| home_dir().join(&entry.name));
    if let Some(parent) = orig.parent() {
        fs::create_dir_all(parent)?;
    }
    let dest = unique_destination_path(&orig);
    move_single_path(&entry.path, &dest)?;
    if let Some(info) = &entry.trash_info_file {
        let _ = fs::remove_file(info);
    }
    Ok(dest)
}

// ============================================================================
// Create / Rename / Copy / Move / Undo Operations
// ============================================================================

pub fn create_directory(parent: &Path, desired_name: &str) -> Result<(PathBuf, UndoRecord)> {
    let clean = sanitize_leaf_name(desired_name)?;
    let dest = unique_destination_path(&parent.join(clean));
    fs::create_dir(&dest).with_context(|| format!("创建文件夹失败: {}", dest.display()))?;
    Ok((dest.clone(), UndoRecord::CreatedDir(dest)))
}

pub fn create_empty_file(parent: &Path, desired_name: &str) -> Result<(PathBuf, UndoRecord)> {
    let clean = sanitize_leaf_name(desired_name)?;
    let dest = unique_destination_path(&parent.join(clean));
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&dest)
        .with_context(|| format!("创建文件失败: {}", dest.display()))?;
    Ok((dest.clone(), UndoRecord::CreatedFile(dest)))
}

pub fn rename_path(src: &Path, new_name: &str) -> Result<(PathBuf, UndoRecord)> {
    let clean = sanitize_leaf_name(new_name)?;
    let parent = src.parent().ok_or_else(|| anyhow!("无效路径"))?;
    let dst = parent.join(clean);
    if dst == src {
        bail!("新名称与原名称相同");
    }
    ensure!(!dst.exists(), "目标名称已存在: {}", dst.display());
    fs::rename(src, &dst).with_context(|| format!("重命名失败: {}", src.display()))?;
    Ok((
        dst.clone(),
        UndoRecord::Renamed {
            from: src.to_path_buf(),
            to: dst,
        },
    ))
}

pub fn copy_paths_to_dir(sources: &[PathBuf], target_dir: &Path) -> Result<UndoRecord> {
    ensure!(!sources.is_empty(), "没有可复制的项目");
    ensure!(target_dir.is_dir(), "目标不是文件夹: {}", target_dir.display());
    let mut created = Vec::with_capacity(sources.len());
    for src in sources {
        let leaf = src
            .file_name()
            .ok_or_else(|| anyhow!("无效源路径"))?
            .to_string_lossy();
        let desired = if src.parent() == Some(target_dir) {
            let (stem, ext) = split_stem_ext(&leaf);
            if ext.is_empty() {
                target_dir.join(format!("{stem} (副本)"))
            } else {
                target_dir.join(format!("{stem} (副本).{ext}"))
            }
        } else {
            target_dir.join(leaf.as_ref())
        };
        let dst = unique_destination_path(&desired);
        if src.is_dir() && dst.starts_with(src) {
            bail!("不能将文件夹复制到其自身子目录中");
        }
        copy_recursive(src, &dst)?;
        created.push(dst);
    }
    Ok(UndoRecord::Copied { created })
}

pub fn move_paths_to_dir(sources: &[PathBuf], target_dir: &Path) -> Result<UndoRecord> {
    ensure!(!sources.is_empty(), "没有可移动的项目");
    ensure!(target_dir.is_dir(), "目标不是文件夹: {}", target_dir.display());
    let mut moves = Vec::with_capacity(sources.len());
    for src in sources {
        if src == target_dir {
            bail!("不能将文件夹移动到自身");
        }
        if src.is_dir() && target_dir.starts_with(src) {
            bail!("不能将文件夹移动到其自身子目录中");
        }
        if src.parent() == Some(target_dir) {
            continue;
        }
        let leaf = src.file_name().ok_or_else(|| anyhow!("无效源路径"))?;
        let dst = unique_destination_path(&target_dir.join(leaf));
        move_single_path(src, &dst)?;
        moves.push((src.clone(), dst));
    }
    ensure!(!moves.is_empty(), "项目已在目标文件夹中");
    Ok(UndoRecord::Moved { moves })
}

pub fn execute_undo(record: UndoRecord) -> Result<String> {
    let desc = record.summary();
    match record {
        UndoRecord::CreatedDir(dir) => {
            fs::remove_dir(&dir).with_context(|| format!("撤销新建文件夹失败（目录非空）: {}", dir.display()))?;
        }
        UndoRecord::CreatedFile(file) => {
            let meta = fs::metadata(&file)?;
            ensure!(meta.len() == 0, "文件已写入内容，拒绝自动删除");
            fs::remove_file(&file)?;
        }
        UndoRecord::Renamed { from, to } => {
            ensure!(!from.exists(), "原路径已被占用: {}", from.display());
            fs::rename(&to, &from)?;
        }
        UndoRecord::Copied { created } => {
            for p in created.into_iter().rev() {
                if p.is_dir() {
                    fs::remove_dir_all(&p)?;
                } else if p.exists() {
                    fs::remove_file(&p)?;
                }
            }
        }
        UndoRecord::Moved { moves } => {
            for (orig, cur) in moves.into_iter().rev() {
                let target = unique_destination_path(&orig);
                move_single_path(&cur, &target)?;
            }
        }
        UndoRecord::Trashed { items } => {
            for item in items.into_iter().rev() {
                if let Some(parent) = item.original_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let target = unique_destination_path(&item.original_path);
                move_single_path(&item.trashed_file, &target)?;
                let _ = fs::remove_file(&item.info_file);
            }
        }
    }
    Ok(format!("已撤销：{desc}"))
}

fn sanitize_leaf_name(name: &str) -> Result<String> {
    let trimmed = name.trim();
    ensure!(!trimmed.is_empty(), "名称不能为空");
    ensure!(
        trimmed != "." && trimmed != ".." && !trimmed.contains('/') && !trimmed.contains('\0'),
        "名称包含非法路径字符"
    );
    Ok(trimmed.to_string())
}

pub fn unique_destination_path(desired: &Path) -> PathBuf {
    if fs::symlink_metadata(desired).is_err() {
        return desired.to_path_buf();
    }
    let parent = desired.parent().unwrap_or_else(|| Path::new("."));
    let leaf = desired
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "item".into());
    let (stem, ext) = split_stem_ext(&leaf);
    for idx in 2..10_000u32 {
        let cand = if ext.is_empty() {
            parent.join(format!("{stem} ({idx})"))
        } else {
            parent.join(format!("{stem} ({idx}).{ext}"))
        };
        if fs::symlink_metadata(&cand).is_err() {
            return cand;
        }
    }
    parent.join(format!("{leaf}.new"))
}

fn split_stem_ext(name: &str) -> (&str, &str) {
    if let Some(stripped) = name.strip_prefix('.') {
        if let Some((s, e)) = stripped.rsplit_once('.') {
            return (&name[..s.len() + 1], e);
        }
        return (name, "");
    }
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() && !ext.is_empty() => (stem, ext),
        _ => (name, ""),
    }
}

fn move_single_path(src: &Path, dst: &Path) -> Result<()> {
    if fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    // Cross-device EXDEV fallback: recursive copy + remove original
    copy_recursive(src, dst)?;
    let meta = fs::symlink_metadata(src)?;
    if meta.is_dir() && !meta.file_type().is_symlink() {
        fs::remove_dir_all(src)?;
    } else {
        fs::remove_file(src)?;
    }
    Ok(())
}

fn copy_recursive(src: &Path, dst: &Path) -> Result<()> {
    let meta = fs::symlink_metadata(src)?;
    if meta.file_type().is_symlink() {
        let target = fs::read_link(src)?;
        std::os::unix::fs::symlink(target, dst)?;
        return Ok(());
    }
    if meta.is_dir() {
        fs::create_dir_all(dst)?;
        for item in fs::read_dir(src)? {
            let item = item?;
            copy_recursive(&item.path(), &dst.join(item.file_name()))?;
        }
    } else {
        fs::copy(src, dst)?;
    }
    Ok(())
}

fn file_name_str(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

// ============================================================================
// External Open & `xr-cinema` Spatial Video Handoff
// ============================================================================

pub fn open_with_system_or_cinema(entry: &FileEntry, force_cinema: bool) -> Result<String> {
    if force_cinema || entry.kind == FileKind::Video {
        for candidate in ["xr-cinema-desktop", "/usr/bin/xr-cinema-desktop", "xr-cinema"] {
            if command_exists(candidate) {
                Command::new(candidate)
                    .arg(&entry.path)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .with_context(|| format!("启动空间影院失败: {candidate}"))?;
                return Ok(format!("已交由空间影院播放：{}", entry.name));
            }
        }
        if force_cinema {
            bail!("未检测到已安装的 xr-cinema（空间影院）");
        }
    }
    Command::new("xdg-open")
        .arg(&entry.path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("调用 xdg-open 打开 {} 失败", entry.name))?;
    Ok(format!("已打开：{}", entry.name))
}

fn command_exists(cmd: &str) -> bool {
    if cmd.contains('/') {
        return Path::new(cmd).is_file();
    }
    env::var_os("PATH")
        .map(|paths| env::split_paths(&paths).any(|dir| dir.join(cmd).is_file()))
        .unwrap_or(false)
}

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut val = bytes as f64;
    let mut unit = 0usize;
    while val >= 1024.0 && unit + 1 < UNITS.len() {
        val /= 1024.0;
        unit += 1;
    }
    if val >= 100.0 {
        format!("{val:.0} {}", UNITS[unit])
    } else {
        format!("{val:.1} {}", UNITS[unit])
    }
}

pub fn format_timestamp_ymd_hm(secs: u64) -> String {
    if secs == 0 {
        return "-".into();
    }
    let time_t = secs as libc::time_t;
    // SAFETY: `localtime_r` writes into local `tm`.
    unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&time_t, &mut tm).is_null() {
            return "-".into();
        }
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_lifecycle_copy_move_rename_trash_and_undo_roundtrip() {
        let base = env::temp_dir().join(format!(
            "xr-files-fs-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let work = base.join("work");
        let trash = base.join("Trash");
        fs::create_dir_all(&work).unwrap();

        // 1. Create dir & file + undo create
        let (sub, undo_dir) = create_directory(&work, "空间项目").unwrap();
        assert!(sub.is_dir());
        let (doc, _undo_file) = create_empty_file(&work, "notes.md").unwrap();
        fs::write(&doc, "# openKylin XR\nhello spatial world\n").unwrap();

        // 2. Rename + undo
        let (renamed, undo_ren) = rename_path(&doc, "readme_xr.md").unwrap();
        assert!(renamed.is_file() && !doc.exists());
        execute_undo(undo_ren).unwrap();
        assert!(doc.is_file() && !renamed.exists());

        // 3. Copy into same dir (auto `(副本)`) + undo
        let undo_cp = copy_paths_to_dir(&[doc.clone()], &work).unwrap();
        assert!(work.join("notes (副本).md").is_file());
        execute_undo(undo_cp).unwrap();
        assert!(!work.join("notes (副本).md").exists());

        // 4. Move into subdir + undo
        let undo_mv = move_paths_to_dir(&[doc.clone()], &sub).unwrap();
        assert!(sub.join("notes.md").is_file() && !doc.exists());
        execute_undo(undo_mv).unwrap();
        assert!(doc.is_file() && !sub.join("notes.md").exists());

        // 5. XDG Trash + Undo restore
        let undo_trash = trash_paths_into(&[doc.clone()], &trash).unwrap();
        assert!(!doc.exists());
        assert!(trash.join("files/notes.md").is_file());
        assert!(trash.join("info/notes.md.trashinfo").is_file());
        execute_undo(undo_trash).unwrap();
        assert!(doc.is_file());

        // Clean empty subdir via undo
        execute_undo(undo_dir).unwrap();
        assert!(!sub.exists());

        let _ = fs::remove_dir_all(&base);
    }
}
