"""Stage 4: Render final vertical vlog with xfade transitions, sidechain music ducking, and ASS captions."""

from __future__ import annotations

import logging
from pathlib import Path

from auto_vlog.config import Settings
from auto_vlog.media_utils import run_ffmpeg
from auto_vlog.models import AlignedSegment, RenderTimeline, TimedBeat

logger = logging.getLogger(__name__)

MUSIC_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def find_background_music(music_dir: Path) -> Path | None:
    """Return the first audio track found in `music/`, if any."""
    if not music_dir.exists():
        return None
    tracks = sorted(
        p for p in music_dir.iterdir()
        if p.is_file() and p.suffix.lower() in MUSIC_EXTENSIONS
    )
    return tracks[0] if tracks else None


def _build_atempo_chain(speed_factor: float) -> str:
    """Decompose playback speed factor into valid FFmpeg atempo filters (0.5..2.0)."""
    factors: list[float] = []
    remaining = speed_factor
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    return ",".join(f"atempo={f:.4f}" for f in factors)


def _render_segment_clip(
    seg: AlignedSegment,
    out_path: Path,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Extract and normalize a single source segment to uniform resolution, FPS, and PCM stereo audio."""
    duration = seg.end_sec - seg.start_sec
    rendered_dur = seg.rendered_duration_sec
    vf_parts = [
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
        f"fps={fps}",
        "setsar=1",
        "format=yuv420p",
    ]
    if abs(seg.playback_rate - 1.0) > 1e-3:
        pts_scale = 1.0 / seg.playback_rate
        vf_parts.append(f"setpts={pts_scale:.4f}*PTS")
    vf_parts.extend([f"trim=0:{rendered_dur:.3f}", "setpts=PTS-STARTPTS"])

    args = [
        "-ss",
        f"{seg.start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        seg.source_path,
    ]

    if seg.has_audio and seg.keep_ambient_audio:
        af_parts = ["aformat=sample_rates=44100:channel_layouts=stereo", "volume=0.18"]
        if abs(seg.playback_rate - 1.0) > 1e-3:
            af_parts.append(_build_atempo_chain(seg.playback_rate))
        af_parts.extend(["apad", f"atrim=0:{rendered_dur:.3f}", "asetpts=PTS-STARTPTS"])
        filter_complex = f"[0:v]{','.join(vf_parts)}[v];[0:a]{','.join(af_parts)}[a]"
    else:
        args.extend(["-f", "lavfi", "-t", f"{rendered_dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo"])
        filter_complex = f"[0:v]{','.join(vf_parts)}[v];[1:a]atrim=0:{rendered_dur:.3f},asetpts=PTS-STARTPTS[a]"

    args.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
    )
    run_ffmpeg(args)


def _render_beat_clip(
    beat: TimedBeat,
    render_dir: Path,
    width: int,
    height: int,
    fps: int,
) -> Path:
    """Render all segments of one beat and concatenate them with hard cuts."""
    seg_paths: list[Path] = []
    for s_idx, seg in enumerate(beat.segments, start=1):
        seg_out = render_dir / f"beat_{beat.beat_index:02d}_seg_{s_idx:02d}.mkv"
        _render_segment_clip(seg, seg_out, width, height, fps)
        seg_paths.append(seg_out)

    beat_out = render_dir / f"beat_{beat.beat_index:02d}.mkv"
    if len(seg_paths) == 1:
        seg_paths[0].replace(beat_out)
        return beat_out

    inputs: list[str] = []
    concat_inputs = ""
    for i, path in enumerate(seg_paths):
        inputs.extend(["-i", str(path)])
        concat_inputs += f"[{i}:v][{i}:a]"

    filter_complex = f"{concat_inputs}concat=n={len(seg_paths)}:v=1:a=1[v][a]"
    run_ffmpeg(
        [
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "pcm_s16le",
            str(beat_out),
        ]
    )
    return beat_out


def compute_xfade_steps(
    beats: list[TimedBeat],
    transition_sec: float,
) -> list[dict[str, float | str]]:
    """Compute exact xfade transition name, duration, and offset for chaining beat clips."""
    steps: list[dict[str, float | str]] = []
    hard_cut_sec = min(0.04, transition_sec)

    for idx in range(len(beats) - 1):
        cur = beats[idx]
        t_dur = hard_cut_sec if cur.transition_to_next == "none" else transition_sec
        style = "fade" if cur.transition_to_next == "none" else cur.transition_to_next
        offset = max(0.1, cur.visual_start_sec + cur.visual_duration_sec - t_dur)
        steps.append({"style": style, "duration": round(t_dur, 3), "offset": round(offset, 3)})

    return steps


def _stitch_beats_with_xfade(
    beat_paths: list[Path],
    beats: list[TimedBeat],
    transition_sec: float,
    out_path: Path,
) -> None:
    """Join beat clips together using smooth FFmpeg xfade + acrossfade transitions."""
    if len(beat_paths) == 1:
        beat_paths[0].replace(out_path)
        return

    steps = compute_xfade_steps(beats, transition_sec)
    inputs: list[str] = []
    for path in beat_paths:
        inputs.extend(["-i", str(path)])

    filter_lines: list[str] = []
    v_prev = "[0:v]"
    a_prev = "[0:a]"
    for idx, step in enumerate(steps, start=1):
        v_next = f"[{idx}:v]"
        a_next = f"[{idx}:a]"
        v_out = "[vout]" if idx == len(steps) else f"[vx{idx}]"
        a_out = "[aout]" if idx == len(steps) else f"[ax{idx}]"

        filter_lines.append(
            f"{v_prev}{v_next}xfade=transition={step['style']}:duration={step['duration']:.3f}:offset={step['offset']:.3f}{v_out}"
        )
        filter_lines.append(
            f"{a_prev}{a_next}acrossfade=d={step['duration']:.3f}:c1=tri:c2=tri{a_out}"
        )
        v_prev = v_out
        a_prev = a_out

    run_ffmpeg(
        [
            *inputs,
            "-filter_complex",
            ";".join(filter_lines),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
    )


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape path special characters for FFmpeg filter arguments like `ass=...`."""
    return (
        str(path)
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
    )


def run_render(settings: Settings, output_filename: str = "vlog_final.mp4") -> Path:
    """Execute Stage 4: render all beats, chain transitions, duck music, and burn subtitles."""
    timeline_path = settings.work_dir / "timeline.json"
    if not timeline_path.exists():
        raise FileNotFoundError("Chưa có `work/timeline.json`. Hãy chạy bước `voice` trước.")

    timeline = RenderTimeline.model_validate_json(timeline_path.read_text(encoding="utf-8"))
    render_dir = settings.work_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Rendering %d story beats...", len(timeline.beats))
    beat_paths = [
        _render_beat_clip(
            beat=beat,
            render_dir=render_dir,
            width=settings.target_width,
            height=settings.target_height,
            fps=settings.target_fps,
        )
        for beat in timeline.beats
    ]

    stitched_video = render_dir / "stitched_beats.mkv"
    _stitch_beats_with_xfade(
        beat_paths=beat_paths,
        beats=timeline.beats,
        transition_sec=timeline.transition_sec,
        out_path=stitched_video,
    )

    # Build combined voiceover track aligned to exact `voice_start_sec` offsets.
    voice_track = render_dir / "voice_full.wav"
    voice_inputs: list[str] = []
    voice_filters: list[str] = []
    mix_labels = ""
    for idx, beat in enumerate(timeline.beats):
        voice_inputs.extend(["-i", beat.audio_path])
        delay_ms = max(0, int(round(beat.voice_start_sec * 1000)))
        voice_filters.append(
            f"[{idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,adelay={delay_ms}|{delay_ms}[vo{idx}]"
        )
        mix_labels += f"[vo{idx}]"

    voice_filters.append(
        f"{mix_labels}amix=inputs={len(timeline.beats)}:normalize=0:dropout_transition=0[voice_sum]"
    )
    run_ffmpeg(
        [
            *voice_inputs,
            "-filter_complex",
            ";".join(voice_filters),
            "-map",
            "[voice_sum]",
            "-t",
            f"{timeline.total_duration_sec:.3f}",
            str(voice_track),
        ]
    )

    music_path = find_background_music(settings.music_dir)
    ass_escaped = _escape_ffmpeg_filter_path(Path(timeline.subtitles_ass_path))
    final_output = settings.output_dir / output_filename

    final_inputs = ["-i", str(stitched_video), "-i", str(voice_track)]
    filter_parts = [f"[0:v]ass='{ass_escaped}'[v_sub]"]

    if music_path:
        logger.info("Mixing background music with sidechain ducking: %s", music_path.name)
        fade_out_start = max(0.0, timeline.total_duration_sec - 1.5)
        final_inputs.extend(["-stream_loop", "-1", "-i", str(music_path)])
        filter_parts.extend(
            [
                "[1:a]asplit=2[vo_main][vo_sc]",
                (
                    f"[2:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                    f"atrim=0:{timeline.total_duration_sec:.3f},asetpts=PTS-STARTPTS,"
                    f"volume=0.28,afade=t=out:st={fade_out_start:.3f}:d=1.5[bgm]"
                ),
                "[bgm][vo_sc]sidechaincompress=threshold=0.02:ratio=6:attack=40:release=350[bgm_ducked]",
                "[0:a][vo_main][bgm_ducked]amix=inputs=3:normalize=0:duration=first[a_mix]",
            ]
        )
    else:
        logger.info("No music file found in `music/`; rendering with voiceover + ambient audio.")
        filter_parts.append("[0:a][1:a]amix=inputs=2:normalize=0:duration=first[a_mix]")

    run_ffmpeg(
        [
            *final_inputs,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[v_sub]",
            "-map",
            "[a_mix]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(final_output),
        ]
    )
    return final_output
