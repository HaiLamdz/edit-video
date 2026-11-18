"""Subprocess wrappers for FFmpeg and FFprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MediaToolError(RuntimeError):
    """Raised when ffmpeg or ffprobe is missing or exits with a non-zero code."""


def require_binary(name: str) -> str:
    """Return full executable path for ffmpeg/ffprobe or raise an actionable error."""
    binary = shutil.which(name)
    if not binary:
        raise MediaToolError(
            f"Không tìm thấy '{name}' trên hệ thống. "
            "Hãy cài FFmpeg trước (VD Ubuntu/Debian: `sudo apt install ffmpeg`, "
            "macOS: `brew install ffmpeg`)."
        )
    return binary


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    """Execute ffmpeg with standardized error reporting."""
    ffmpeg_bin = require_binary("ffmpeg")
    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaToolError(f"FFmpeg lỗi ({proc.returncode}):\n{proc.stderr.strip()}")


def probe_media(path: Path) -> dict[str, Any]:
    """Run ffprobe and return parsed JSON stream + format metadata."""
    ffprobe_bin = require_binary("ffprobe")
    cmd = [
        ffprobe_bin,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaToolError(f"FFprobe lỗi với file {path.name}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def probe_duration(path: Path) -> float:
    """Return exact duration in seconds for an audio or video file."""
    info = probe_media(path)
    raw_duration = info.get("format", {}).get("duration")
    if raw_duration is not None:
        return float(raw_duration)
    for stream in info.get("streams", []):
        if "duration" in stream:
            return float(stream["duration"])
    raise MediaToolError(f"Không đọc được thời lượng của file: {path}")


def _parse_fps(rate_str: str | None) -> float:
    if not rate_str or rate_str == "0/0":
        return 30.0
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 30.0
    return float(rate_str)


def _extract_rotation(video_stream: dict[str, Any]) -> int:
    tags = video_stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return int(float(tags["rotate"]))
        except ValueError:
            pass
    for side_data in video_stream.get("side_data_list") or []:
        if "rotation" in side_data:
            try:
                return int(float(side_data["rotation"]))
            except ValueError:
                pass
    return 0


def parse_video_info(path: Path, info: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized display dimensions, duration, fps, audio flag, and creation_time."""
    streams = info.get("streams", [])
    fmt = info.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise MediaToolError(f"File {path.name} không chứa luồng video.")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    rotation = abs(_extract_rotation(video_stream)) % 360
    if rotation in (90, 270):
        width, height = height, width

    duration_raw = fmt.get("duration") or video_stream.get("duration") or 0.0
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))

    creation_time = (
        (fmt.get("tags") or {}).get("creation_time")
        or (video_stream.get("tags") or {}).get("creation_time")
    )
    if not creation_time:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        creation_time = mtime.isoformat()

    return {
        "width": width,
        "height": height,
        "duration_sec": float(duration_raw),
        "fps": round(fps, 3),
        "has_audio": has_audio,
        "creation_time": str(creation_time),
    }


def extract_frame_jpeg(video_path: Path, timestamp_sec: float, out_path: Path, max_width: int = 512) -> None:
    """Extract a single resized JPEG frame at the given timestamp using FFmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-ss",
            f"{max(0.0, timestamp_sec):.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale='min({max_width},iw)':-2",
            "-q:v",
            "4",
            str(out_path),
        ]
    )
