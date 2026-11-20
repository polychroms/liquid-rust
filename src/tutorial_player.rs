use eframe::egui;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const TUTORIAL_WIDTH: usize = 1920;
pub const TUTORIAL_HEIGHT: usize = 1080;
const TUTORIAL_FPS: f64 = 30.0;
pub const TUTORIAL_DURATION_SECS: f64 = 64.0;
pub const TUTORIAL_MP4_URL: &str = "https://fluids.polychroms.dev/media/liquid-rust-training.mp4";
pub const TUTORIAL_WEB_URL: &str = "https://fluids.polychroms.dev/?tutorial=1";

#[derive(Clone, Debug, PartialEq)]
pub enum TutorialStatus {
    Idle,
    Loading(String),
    Playing,
    Paused,
    Ended,
    Error(String),
}

struct SharedState {
    status: TutorialStatus,
    latest_frame: Option<(u64, egui::ColorImage)>,
    pos_secs: f64,
}

pub struct TutorialPlayer {
    shared: Arc<Mutex<SharedState>>,
    stop_flag: Arc<AtomicBool>,
    frame_seq: Arc<AtomicU64>,
    children: Arc<Mutex<Vec<Child>>>,
    worker: Option<JoinHandle<()>>,
    texture: Option<egui::TextureHandle>,
    uploaded_frame_id: u64,
    cache_dir: Option<PathBuf>,
    pub muted: bool,
}

impl TutorialPlayer {
    pub fn new(cache_dir: Option<PathBuf>) -> Self {
        Self {
            shared: Arc::new(Mutex::new(SharedState {
                status: TutorialStatus::Idle,
                latest_frame: None,
                pos_secs: 0.0,
            })),
            stop_flag: Arc::new(AtomicBool::new(false)),
            frame_seq: Arc::new(AtomicU64::new(0)),
            children: Arc::new(Mutex::new(Vec::new())),
            worker: None,
            texture: None,
            uploaded_frame_id: 0,
            cache_dir,
            muted: false,
        }
    }

    pub fn status(&self) -> TutorialStatus {
        self.shared.lock().unwrap().status.clone()
    }

    pub fn position_secs(&self) -> f64 {
        self.shared.lock().unwrap().pos_secs
    }

    pub fn texture(&mut self, ctx: &egui::Context) -> Option<&egui::TextureHandle> {
        let frame_opt = {
            let mut s = self.shared.lock().unwrap();
            match &s.latest_frame {
                Some((id, _)) if *id != self.uploaded_frame_id => s.latest_frame.take(),
                _ => None,
            }
        };
        if let Some((id, img)) = frame_opt {
            self.uploaded_frame_id = id;
            if let Some(tex) = &mut self.texture {
                tex.set(img, egui::TextureOptions::LINEAR);
            } else {
                self.texture = Some(ctx.load_texture(
                    "tutorial_video_frame",
                    img,
                    egui::TextureOptions::LINEAR,
                ));
            }
        }
        self.texture.as_ref()
    }

    pub fn start(&mut self, ctx: egui::Context, start_secs: f64) {
        self.stop_worker();
        let clamped_start = start_secs.clamp(0.0, (TUTORIAL_DURATION_SECS - 0.1).max(0.0));
        {
            let mut s = self.shared.lock().unwrap();
            s.status = TutorialStatus::Loading("Preparing tutorial video...".to_string());
            s.pos_secs = clamped_start;
        }
        self.stop_flag.store(false, Ordering::SeqCst);

        let shared = Arc::clone(&self.shared);
        let stop_flag = Arc::clone(&self.stop_flag);
        let frame_seq = Arc::clone(&self.frame_seq);
        let children = Arc::clone(&self.children);
        let cache_dir = self.cache_dir.clone();
        let muted = self.muted;

        self.worker = Some(std::thread::spawn(move || {
            run_playback_worker(
                ctx,
                shared,
                stop_flag,
                frame_seq,
                children,
                cache_dir,
                clamped_start,
                muted,
            );
        }));
    }

    pub fn pause(&mut self) {
        self.stop_worker();
        let mut s = self.shared.lock().unwrap();
        if matches!(s.status, TutorialStatus::Playing | TutorialStatus::Loading(_)) {
            s.status = TutorialStatus::Paused;
        }
    }

    pub fn stop(&mut self) {
        self.stop_worker();
        let mut s = self.shared.lock().unwrap();
        s.status = TutorialStatus::Idle;
        s.pos_secs = 0.0;
        s.latest_frame = None;
    }

    pub fn toggle_play(&mut self, ctx: egui::Context) {
        match self.status() {
            TutorialStatus::Playing | TutorialStatus::Loading(_) => self.pause(),
            TutorialStatus::Paused => {
                let pos = self.position_secs();
                self.start(ctx, pos);
            }
            TutorialStatus::Idle | TutorialStatus::Ended | TutorialStatus::Error(_) => {
                self.start(ctx, 0.0);
            }
        }
    }

    pub fn seek(&mut self, ctx: egui::Context, target_secs: f64) {
        let clamped = target_secs.clamp(0.0, (TUTORIAL_DURATION_SECS - 0.1).max(0.0));
        match self.status() {
            TutorialStatus::Playing | TutorialStatus::Ended | TutorialStatus::Paused => {
                self.start(ctx, clamped);
            }
            _ => {
                self.shared.lock().unwrap().pos_secs = clamped;
            }
        }
    }

    pub fn toggle_mute(&mut self, ctx: egui::Context) {
        self.muted = !self.muted;
        if self.status() == TutorialStatus::Playing {
            let pos = self.position_secs();
            self.start(ctx, pos);
        }
    }

    fn stop_worker(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        {
            let mut kids = self.children.lock().unwrap();
            for mut child in kids.drain(..) {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for TutorialPlayer {
    fn drop(&mut self) {
        self.stop_worker();
    }
}

fn configure_bg_command(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_playback_worker(
    ctx: egui::Context,
    shared: Arc<Mutex<SharedState>>,
    stop_flag: Arc<AtomicBool>,
    frame_seq: Arc<AtomicU64>,
    children: Arc<Mutex<Vec<Child>>>,
    cache_dir: Option<PathBuf>,
    start_secs: f64,
    muted: bool,
) {
    let mp4_path = match resolve_or_download_mp4(cache_dir.as_deref(), &shared, &stop_flag, &ctx) {
        Ok(p) => p,
        Err(msg) => {
            if !stop_flag.load(Ordering::SeqCst) {
                shared.lock().unwrap().status = TutorialStatus::Error(msg);
                ctx.request_repaint();
            }
            return;
        }
    };

    if stop_flag.load(Ordering::SeqCst) {
        return;
    }

    let Some(mp4_str) = mp4_path.to_str() else {
        shared.lock().unwrap().status = TutorialStatus::Error("Invalid tutorial path".to_string());
        ctx.request_repaint();
        return;
    };

    let mut vcmd = Command::new("ffmpeg");
    vcmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        &format!("{:.3}", start_secs),
        "-i",
        mp4_str,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-r",
        &format!("{}", TUTORIAL_FPS as u32),
        "pipe:1",
    ])
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    configure_bg_command(&mut vcmd);

    let mut vchild = match vcmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            shared.lock().unwrap().status = TutorialStatus::Error(format!(
                "ffmpeg not available ({}); install ffmpeg or open tutorial in browser.",
                e
            ));
            ctx.request_repaint();
            return;
        }
    };

    let Some(mut stdout) = vchild.stdout.take() else {
        let _ = vchild.kill();
        let _ = vchild.wait();
        return;
    };
    children.lock().unwrap().push(vchild);

    let frame_bytes = TUTORIAL_WIDTH * TUTORIAL_HEIGHT * 3;
    let mut buf = vec![0u8; frame_bytes];
    if stdout.read_exact(&mut buf).is_err() {
        cleanup_children(&children);
        if !stop_flag.load(Ordering::SeqCst) {
            shared.lock().unwrap().status = TutorialStatus::Ended;
            ctx.request_repaint();
        }
        return;
    }

    if stop_flag.load(Ordering::SeqCst) {
        cleanup_children(&children);
        return;
    }

    // Start audio companion right when frame 0 decodes so A/V remain aligned
    if !muted {
        spawn_audio_companion(mp4_str, start_secs, &children);
    }

    let t0 = Instant::now();
    let mut frame_idx: u64 = 0;

    loop {
        let target_offset = Duration::from_secs_f64(frame_idx as f64 / TUTORIAL_FPS);
        while !stop_flag.load(Ordering::SeqCst) {
            let elapsed = t0.elapsed();
            if elapsed >= target_offset {
                break;
            }
            std::thread::sleep((target_offset - elapsed).min(Duration::from_millis(16)));
        }

        if stop_flag.load(Ordering::SeqCst) {
            break;
        }

        let cur_pos = (start_secs + (frame_idx as f64 / TUTORIAL_FPS)).min(TUTORIAL_DURATION_SECS);
        let fid = frame_seq.fetch_add(1, Ordering::Relaxed) + 1;
        let img = egui::ColorImage::from_rgb([TUTORIAL_WIDTH, TUTORIAL_HEIGHT], &buf);
        {
            let mut s = shared.lock().unwrap();
            s.status = TutorialStatus::Playing;
            s.latest_frame = Some((fid, img));
            s.pos_secs = cur_pos;
        }
        ctx.request_repaint();
        frame_idx += 1;

        if stdout.read_exact(&mut buf).is_err() {
            if !stop_flag.load(Ordering::SeqCst) {
                let mut s = shared.lock().unwrap();
                s.status = TutorialStatus::Ended;
                s.pos_secs = TUTORIAL_DURATION_SECS;
                ctx.request_repaint();
            }
            break;
        }
    }

    cleanup_children(&children);
}

fn cleanup_children(children: &Arc<Mutex<Vec<Child>>>) {
    let mut kids = children.lock().unwrap();
    for mut child in kids.drain(..) {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn spawn_audio_companion(mp4_str: &str, start_secs: f64, children: &Arc<Mutex<Vec<Child>>>) {
    let mut ffplay = Command::new("ffplay");
    ffplay
        .args([
            "-nodisp",
            "-autoexit",
            "-vn",
            "-loglevel",
            "quiet",
            "-ss",
            &format!("{:.3}", start_secs),
            mp4_str,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    configure_bg_command(&mut ffplay);
    if let Ok(child) = ffplay.spawn() {
        children.lock().unwrap().push(child);
        return;
    }

    #[cfg(target_os = "linux")]
    {
        for player in ["pw-play", "paplay", "aplay"] {
            let mut dec = Command::new("ffmpeg");
            dec.args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                &format!("{:.3}", start_secs),
                "-i",
                mp4_str,
                "-vn",
                "-f",
                "wav",
                "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
            configure_bg_command(&mut dec);
            if let Ok(mut dec_child) = dec.spawn() {
                if let Some(wav_out) = dec_child.stdout.take() {
                    let mut pcmd = Command::new(player);
                    if player == "pw-play" {
                        pcmd.arg("-");
                    }
                    pcmd.stdin(Stdio::from(wav_out))
                        .stdout(Stdio::null())
                        .stderr(Stdio::null());
                    configure_bg_command(&mut pcmd);
                    if let Ok(play_child) = pcmd.spawn() {
                        let mut kids = children.lock().unwrap();
                        kids.push(dec_child);
                        kids.push(play_child);
                        return;
                    }
                }
                let _ = dec_child.kill();
                let _ = dec_child.wait();
            }
        }
    }

    #[cfg(target_os = "macos")]
    if start_secs < 0.05 {
        let mut cmd = Command::new("afplay");
        cmd.arg(mp4_str)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure_bg_command(&mut cmd);
        if let Ok(child) = cmd.spawn() {
            children.lock().unwrap().push(child);
        }
    }
}

pub fn find_existing_tutorial_mp4(cache_dir: Option<&Path>) -> Option<PathBuf> {
    const MIN_BYTES: u64 = 1_000_000;
    let mut candidates = Vec::new();
    if let Some(dir) = cache_dir {
        candidates.push(dir.join("liquid-rust-training.mp4"));
    }
    candidates.push(PathBuf::from("web/media/liquid-rust-training.mp4"));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join("media/liquid-rust-training.mp4"));
            candidates.push(exe_dir.join("liquid-rust-training.mp4"));
        }
    }
    candidates
        .into_iter()
        .find(|p| std::fs::metadata(p).map(|m| m.len() >= MIN_BYTES).unwrap_or(false))
}

fn resolve_or_download_mp4(
    cache_dir: Option<&Path>,
    shared: &Arc<Mutex<SharedState>>,
    stop_flag: &AtomicBool,
    ctx: &egui::Context,
) -> Result<PathBuf, String> {
    if let Some(found) = find_existing_tutorial_mp4(cache_dir) {
        return Ok(found);
    }

    let dir = cache_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("liquid-rust"));
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Failed to prepare tutorial cache directory: {}", e))?;

    let target = dir.join("liquid-rust-training.mp4");
    let part = dir.join("liquid-rust-training.mp4.part");

    {
        shared.lock().unwrap().status =
            TutorialStatus::Loading("Downloading tutorial video (~13 MB)...".to_string());
        ctx.request_repaint();
    }

    let resp = ureq::get(TUTORIAL_MP4_URL)
        .timeout(Duration::from_secs(30))
        .call()
        .map_err(|e| format!("Download failed: {}", e))?;

    let mut reader = resp.into_reader();
    let mut out = std::fs::File::create(&part)
        .map_err(|e| format!("Cannot write tutorial cache file: {}", e))?;
    let mut chunk = [0u8; 65536];
    loop {
        if stop_flag.load(Ordering::SeqCst) {
            let _ = std::fs::remove_file(&part);
            return Err("Cancelled".to_string());
        }
        let n = reader
            .read(&mut chunk)
            .map_err(|e| format!("Download stream interrupted: {}", e))?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut out, &chunk[..n])
            .map_err(|e| format!("Failed writing tutorial video: {}", e))?;
    }
    drop(out);

    std::fs::rename(&part, &target)
        .map_err(|e| format!("Failed to finalize tutorial cache file: {}", e))?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::{find_existing_tutorial_mp4, TutorialPlayer, TutorialStatus};

    #[test]
    fn rejects_undersized_cached_file_and_initializes_idle() {
        let dir = std::env::temp_dir().join(format!("lr_tut_player_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tiny = dir.join("liquid-rust-training.mp4");
        std::fs::write(&tiny, b"partial").unwrap();

        let found = find_existing_tutorial_mp4(Some(&dir));
        assert!(found.as_ref().is_none_or(|p| p != &tiny));

        let mut player = TutorialPlayer::new(Some(dir.clone()));
        assert_eq!(player.status(), TutorialStatus::Idle);
        player.pause();
        assert_eq!(player.status(), TutorialStatus::Idle);
        player.stop();
        assert_eq!(player.position_secs(), 0.0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
