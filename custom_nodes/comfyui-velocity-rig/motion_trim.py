"""Motion Trim: cut the dead head and tail of a generated clip so each cut is all action.

Image-to-video models often hold the first frame, ease in, and drift at the end. Per-frame motion energy (mean absolute
difference between neighbouring frames) finds where the action is; frames below `threshold` x the clip's peak at the
head and tail are dropped (with `pad` frames kept), then the kept span can be sped up a little.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch


def energy(frames: torch.Tensor) -> list[float]:
    """frames: T,H,W,C in 0..1. Returns T values (first = second)."""
    if frames.shape[0] < 2:
        return [0.0] * frames.shape[0]
    x = frames[:, ::4, ::4].float().mean(-1)
    d = (x[1:] - x[:-1]).abs().mean((1, 2)) * 255.0
    v = d.tolist()
    return [v[0]] + v


def window(e: list[float], threshold: float = 0.25, pad: int = 3, min_frames: int = 12, head: int = 4) -> tuple[int, int]:
    """Return [start, end) of the frames to keep. The first `head` frames always go (LTX opens on the still plate)."""
    n = len(e)
    if n == 0:
        return 0, 0
    peak = max(e[head:] or e) or 1.0
    act = [i for i, v in enumerate(e) if i >= head and v >= threshold * peak]
    if not act:
        return min(head, n - 1), n
    s, t = max(head, act[0] - pad), min(n, act[-1] + 1 + pad)
    while t - s < min(min_frames, n):
        s, t = max(0, s - 1), min(n, t + 1)
    return s, t


def trim_frames(frames: torch.Tensor, threshold: float = 0.25, pad: int = 3, speed: float = 1.0, min_frames: int = 12):
    """IMAGE batch -> (trimmed batch, start, end). speed > 1 drops frames evenly."""
    s, t = window(energy(frames), threshold, pad, min_frames)
    x = frames[s:t]
    if speed > 1.0 and x.shape[0] > 2:
        idx = torch.arange(0, x.shape[0], speed).long().clamp(max=x.shape[0] - 1)
        x = x[idx]
    return x, s, t


def trim_video(src: Path, dest: Path, fps: float, frames: torch.Tensor, threshold: float = 0.25, pad: int = 3,
               speed: float = 1.2, min_frames: int = 12) -> tuple[int, int]:
    """Cut src (video + audio) to its action window and speed it up; frames = the decoded src frames."""
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    s, t = window(energy(frames), threshold, pad, min_frames)
    a, b = s / fps, t / fps
    vf = f"[0:v]trim=start={a:.4f}:end={b:.4f},setpts=(PTS-STARTPTS)/{speed},fps={int(fps)}[v]"
    af = f"[0:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS,atempo={speed}[a]"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-filter_complex", f"{vf};{af}",
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k", str(dest)])
    if r.returncode != 0:  # no audio stream
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-filter_complex", vf, "-map", "[v]",
                        "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(dest)], check=True)
    return s, t
