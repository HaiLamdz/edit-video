"""Stage 1: Ingest raw clips, read timestamps/metadata, sort chronologically, extract frames."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from auto_vlog.config import Settings
from auto_vlog.media_utils import extract_frame_jpeg, parse_video_info, probe_media
from auto_vlog.models import ClipMetadata, IngestManifest, SampledFrame

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


def compute_sample_timestamps(duration_sec: float, max_frames: int) -> list[float]:
    """Choose evenly spaced timestamps across a clip while skipping shaky edges."""
    if duration_sec <= 0 or max_frames <= 0:
        return []
    if duration_sec <= 1.2 or max_frames == 1:
        return [round(duration_sec * 0.5, 2)]

    margin = min(0.35, duration_sec * 0.08)
    start = margin
    end = max(start, duration_sec - margin)
    count = min(max_frames, max(2, int(round(duration_sec / 2.0))))
    if count == 1:
        return [round((start + end) * 0.5, 2)]

    step = (end - start) / (count - 1)
    return [round(start + i * step, 2) for i in range(count)]


def _sortable_timestamp(iso_str: str) -> float:
    normalized = iso_str.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


def run_ingest(settings: Settings) -> IngestManifest:
    """Scan `input_clips/`, sort clips chronologically, and extract preview frames."""
    settings.ensure_dirs()
    frames_dir = settings.work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(
        p for p in settings.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not raw_files:
        raise FileNotFoundError(
            f"Không tìm thấy video nào trong '{settings.input_dir}'. "
            "Hãy copy các clip quay trong ngày (.mp4/.mov) vào thư mục đó trước."
        )

    parsed_items: list[tuple[float, Path, dict]] = []
    for path in raw_files:
        info = probe_media(path)
        meta = parse_video_info(path, info)
        sort_key = _sortable_timestamp(meta["creation_time"])
        parsed_items.append((sort_key, path, meta))

    parsed_items.sort(key=lambda item: (item[0], item[1].name))

    clips: list[ClipMetadata] = []
    for idx, (_, path, meta) in enumerate(parsed_items, start=1):
        clip_id = f"clip_{idx:02d}"
        timestamps = compute_sample_timestamps(meta["duration_sec"], settings.frames_per_clip)
        sampled_frames: list[SampledFrame] = []

        for frame_idx, ts in enumerate(timestamps, start=1):
            frame_path = frames_dir / f"{clip_id}_f{frame_idx:02d}.jpg"
            extract_frame_jpeg(path, ts, frame_path)
            sampled_frames.append(
                SampledFrame(timestamp_sec=ts, image_path=str(frame_path))
            )

        clips.append(
            ClipMetadata(
                clip_id=clip_id,
                filename=path.name,
                path=str(path),
                creation_time=meta["creation_time"],
                duration_sec=round(meta["duration_sec"], 3),
                width=meta["width"],
                height=meta["height"],
                fps=meta["fps"],
                has_audio=meta["has_audio"],
                sampled_frames=sampled_frames,
            )
        )
        logger.info("Ingested %s (%s, %.1fs)", clip_id, path.name, meta["duration_sec"])

    manifest = IngestManifest(clips=clips)
    manifest_path = settings.work_dir / "ingest_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
