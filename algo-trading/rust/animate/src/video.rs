//! Stream raw RGBA frames into an ffmpeg process.

use std::io::Write;
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};

pub struct Encoder {
    child: Child,
    stdin: Option<ChildStdin>,
}

impl Encoder {
    /// Start ffmpeg writing `path`: H.264 for `.mp4`/`.mov`, a palette GIF for `.gif`.
    pub fn spawn(path: &Path, width: u32, height: u32, fps: u32, bitrate_kbps: u32)
                 -> Result<Encoder, String> {
        let gif = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gif"));
        let size = format!("{width}x{height}");
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", &size,
                  "-r", &fps.to_string(), "-i", "-"]);
        if gif {
            cmd.args(["-vf", "split[a][b];[a]palettegen[p];[b][p]paletteuse", "-loop", "0"]);
        } else {
            cmd.args(["-c:v", "libx264", "-b:v", &format!("{bitrate_kbps}k"), "-pix_fmt",
                      "yuv420p", "-movflags", "+faststart"]);
        }
        cmd.arg(path).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound =>
                "ffmpeg not found on the PATH (e.g. `brew install ffmpeg`)".to_string(),
            _ => format!("could not start ffmpeg: {e}"),
        })?;
        let stdin = child.stdin.take();
        Ok(Encoder { child, stdin })
    }

    pub fn write(&mut self, frame: &[u8]) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("encoder already finished")?;
        stdin.write_all(frame).map_err(|e| format!("ffmpeg stopped accepting frames: {e}"))
    }

    /// Close the input and wait for ffmpeg; surfaces its error output on failure.
    pub fn finish(mut self) -> Result<(), String> {
        drop(self.stdin.take());
        let out = self.child.wait_with_output().map_err(|e| format!("ffmpeg failed: {e}"))?;
        if out.status.success() {
            Ok(())
        } else {
            Err(format!("ffmpeg exited with {}: {}", out.status,
                        String::from_utf8_lossy(&out.stderr).trim()))
        }
    }
}
