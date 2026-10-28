"""Hard-cut reel from shot clips: trim each clip head (LTX opens on the still first frame), keep LTX audio."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def assemble(clips: list[Path], dest: Path, fps: int = 24, trim_frames: int = 6) -> Path:
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    args = [ffmpeg, "-y", "-loglevel", "error"]
    for c in clips:
        args += ["-i", str(c)]
    skip = trim_frames / fps
    parts = [f"[{n}:v]trim=start={skip},setpts=PTS-STARTPTS,fps={fps},format=yuv420p[v{n}];"
             f"[{n}:a]atrim=start={skip},asetpts=PTS-STARTPTS,aresample=48000[a{n}];" for n in range(len(clips))]
    chain = "".join(parts) + "".join(f"[v{n}][a{n}]" for n in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=1[v][a]"
    args += ["-filter_complex", chain, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "17", "-c:a", "aac",
             "-b:a", "192k", "-movflags", "+faststart", str(dest)]
    subprocess.run(args, check=True)
    return dest
