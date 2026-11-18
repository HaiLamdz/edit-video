"""Stage 3: Audio-first TTS generation, segment alignment, and ASS subtitle building."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import edge_tts
from openai import OpenAI

from auto_vlog.config import Settings
from auto_vlog.media_utils import probe_duration
from auto_vlog.models import (
    AlignedSegment,
    ClipMetadata,
    IngestManifest,
    RenderTimeline,
    SelectedSegment,
    SubtitleCue,
    TimedBeat,
    VlogPlan,
)

logger = logging.getLogger(__name__)


def _build_openai_client(settings: Settings) -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("Thiếu OPENAI_API_KEY trong file `.env`.")
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def synthesize_tts(settings: Settings, text: str, out_path: Path, client: OpenAI | None) -> None:
    """Generate voiceover audio for a single story beat."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if settings.tts_provider == "edge":
        asyncio.run(edge_tts.Communicate(text=text, voice=settings.edge_tts_voice).save(str(out_path)))
        return

    if client is None:
        client = _build_openai_client(settings)

    kwargs: dict = {
        "model": settings.openai_tts_model,
        "voice": settings.openai_tts_voice,
        "input": text,
        "response_format": "mp3",
    }
    if settings.openai_tts_model == "gpt-4o-mini-tts":
        kwargs["instructions"] = (
            "Đọc tiếng Việt tự nhiên, ấm áp, nhịp vừa phải như đang kể chuyện vlog đời thường."
        )

    with client.audio.speech.with_streaming_response.create(**kwargs) as response:
        response.stream_to_file(out_path)


def align_segments_to_duration(
    segments: list[SelectedSegment],
    clips_by_id: dict[str, ClipMetadata],
    target_duration_sec: float,
    min_seg_sec: float = 0.8,
) -> list[AlignedSegment]:
    """Adjust clip segment bounds (or gently slow down if source clips are too short) to fit target_duration_sec."""
    raw_items: list[dict] = []
    for seg in segments:
        clip = clips_by_id[seg.clip_id]
        start = max(0.0, min(seg.start_sec, max(0.0, clip.duration_sec - 0.2)))
        end = min(clip.duration_sec, max(start + 0.2, seg.end_sec))
        raw_items.append(
            {
                "clip": clip,
                "start": start,
                "end": end,
                "keep_ambient_audio": seg.keep_ambient_audio,
            }
        )

    if not raw_items:
        raise ValueError("Beat không có segment nào.")

    current_total = sum(item["end"] - item["start"] for item in raw_items)

    # Case 1: Selected clips are longer than audio target -> trim proportionally (keeping at least min_seg_sec).
    if current_total > target_duration_sec:
        while len(raw_items) > 1 and len(raw_items) * min_seg_sec > target_duration_sec:
            raw_items.pop()

        current_total = sum(item["end"] - item["start"] for item in raw_items)
        if current_total > target_duration_sec:
            scale = target_duration_sec / current_total
            for item in raw_items:
                new_len = max(0.4, (item["end"] - item["start"]) * scale)
                item["end"] = item["start"] + new_len

            # Correct residual rounding difference on the last segment.
            diff = target_duration_sec - sum(item["end"] - item["start"] for item in raw_items)
            raw_items[-1]["end"] = max(raw_items[-1]["start"] + 0.3, raw_items[-1]["end"] + diff)

    # Case 2: Selected clips are shorter than audio target -> expand within source clip bounds first.
    elif current_total < target_duration_sec:
        needed = target_duration_sec - current_total
        for item in raw_items:
            if needed <= 1e-3:
                break
            clip_dur = item["clip"].duration_sec
            room_after = max(0.0, clip_dur - item["end"])
            add_after = min(needed, room_after)
            item["end"] += add_after
            needed -= add_after

            if needed > 1e-3:
                room_before = max(0.0, item["start"])
                add_before = min(needed, room_before)
                item["start"] -= add_before
                needed -= add_before

    aligned: list[AlignedSegment] = []
    for item in raw_items:
        aligned.append(
            AlignedSegment(
                clip_id=item["clip"].clip_id,
                source_path=item["clip"].path,
                start_sec=round(item["start"], 3),
                end_sec=round(item["end"], 3),
                playback_rate=1.0,
                keep_ambient_audio=item["keep_ambient_audio"],
                has_audio=item["clip"].has_audio,
            )
        )

    # If clips hit physical source bounds and are still shorter, gently slow down playback cleanly.
    actual_total = sum(seg.rendered_duration_sec for seg in aligned)
    if actual_total < target_duration_sec - 0.02 and actual_total > 0:
        rate = max(0.55, actual_total / target_duration_sec)
        for seg in aligned:
            seg.playback_rate = round(rate, 4)

    return aligned


def split_text_to_cues(
    text: str,
    start_sec: float,
    duration_sec: float,
    time_label: str = "",
    max_words_per_cue: int = 6,
) -> list[SubtitleCue]:
    """Split narration text into balanced short caption cues proportional to character length."""
    words = text.strip().split()
    if not words:
        return []

    chunks: list[str] = []
    for i in range(0, len(words), max_words_per_cue):
        chunks.append(" ".join(words[i : i + max_words_per_cue]))

    total_chars = sum(max(1, len(c)) for c in chunks)
    cursor = start_sec
    cues: list[SubtitleCue] = []
    for idx, chunk in enumerate(chunks):
        if idx == len(chunks) - 1:
            end = start_sec + duration_sec
        else:
            share = max(1, len(chunk)) / total_chars
            end = cursor + duration_sec * share
        cues.append(
            SubtitleCue(
                start_sec=round(cursor, 3),
                end_sec=round(end, 3),
                text=chunk,
                time_label=time_label,
            )
        )
        cursor = end
    return cues


def extract_whisper_cues(
    client: OpenAI,
    model: str,
    audio_path: Path,
    offset_sec: float,
    time_label: str = "",
    max_words_per_cue: int = 6,
) -> list[SubtitleCue]:
    """Use OpenAI Whisper word-level timestamps to build tight captions."""
    with audio_path.open("rb") as f:
        transcript = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="vi",
        )

    words = getattr(transcript, "words", None) or []
    if not words:
        return []

    cues: list[SubtitleCue] = []
    for i in range(0, len(words), max_words_per_cue):
        group = words[i : i + max_words_per_cue]
        w_start = group[0]["start"] if isinstance(group[0], dict) else group[0].start
        w_end = group[-1]["end"] if isinstance(group[-1], dict) else group[-1].end
        tokens = [
            (w["word"] if isinstance(w, dict) else w.word).strip()
            for w in group
        ]
        cues.append(
            SubtitleCue(
                start_sec=round(offset_sec + float(w_start), 3),
                end_sec=round(offset_sec + float(w_end), 3),
                text=" ".join(t for t in tokens if t),
                time_label=time_label,
            )
        )
    return cues


def _format_ass_time(seconds: float) -> str:
    cs_total = max(0, int(round(seconds * 100)))
    hours, rem = divmod(cs_total, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def build_ass_subtitles(
    cues: list[SubtitleCue],
    beat_badges: list[tuple[float, float, str]],
    width: int,
    height: int,
) -> str:
    """Render ASS subtitle file with safe-zone caption styling and optional corner time badges."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,DejaVu Sans,54,&H00FFFFFF,&H0000FFFF,&H00121212,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,70,70,290,1
Style: TimeBadge,DejaVu Sans,34,&H00FFFFFF,&H000000FF,&H00181818,&H90000000,-1,0,0,0,100,100,0,0,3,10,0,7,55,55,130,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.rstrip()]

    for start, end, label in beat_badges:
        if not label:
            continue
        lines.append(
            f"Dialogue: 1,{_format_ass_time(start)},{_format_ass_time(end)},TimeBadge,,0,0,0,,{label}"
        )

    for cue in cues:
        clean_text = cue.text.replace("\n", " ").strip()
        if not clean_text:
            continue
        lines.append(
            f"Dialogue: 0,{_format_ass_time(cue.start_sec)},{_format_ass_time(cue.end_sec)},Caption,,0,0,0,,{clean_text}"
        )

    return "\n".join(lines) + "\n"


def compute_beat_schedule(
    audio_durations: list[float],
    transitions: list[str],
    transition_sec: float,
    beat_pause_sec: float,
) -> list[dict[str, float]]:
    """Compute drift-free voice start, visual start, and visual duration for each beat."""
    schedule: list[dict[str, float]] = []
    visual_cursor = 0.0
    hard_cut_sec = min(0.04, transition_sec)

    for idx, audio_dur in enumerate(audio_durations):
        is_last = idx == len(audio_durations) - 1
        if is_last:
            t_out = 0.0
        elif transitions[idx] == "none":
            t_out = hard_cut_sec
        else:
            t_out = transition_sec

        # Lead voice slightly after incoming crossfade begins so transitions feel smooth and unhurried.
        prev_t = schedule[-1]["t_out"] if idx > 0 else 0.0
        voice_start = visual_cursor + (min(0.12, prev_t * 0.4) if idx > 0 else 0.0)
        visual_dur = audio_dur + beat_pause_sec + t_out

        schedule.append(
            {
                "voice_start_sec": round(voice_start, 3),
                "visual_start_sec": round(visual_cursor, 3),
                "visual_duration_sec": round(visual_dur, 3),
                "t_out": round(t_out, 3),
            }
        )
        visual_cursor += audio_dur + beat_pause_sec

    return schedule


def run_voice_and_subtitles(settings: Settings) -> RenderTimeline:
    """Execute Stage 3: synthesize beat voiceovers, align clips, and create subtitles."""
    manifest_path = settings.work_dir / "ingest_manifest.json"
    plan_path = settings.work_dir / "vlog_plan.json"
    if not manifest_path.exists() or not plan_path.exists():
        raise FileNotFoundError(
            "Thiếu `ingest_manifest.json` hoặc `vlog_plan.json`. Hãy chạy `ingest` và `director` trước."
        )

    manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    plan = VlogPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    clips_by_id = {c.clip_id: c for c in manifest.clips}

    needs_openai = settings.tts_provider != "edge" or settings.use_whisper_subs
    client = _build_openai_client(settings) if (needs_openai and settings.openai_api_key) else None

    audio_dir = settings.work_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_paths: list[Path] = []
    audio_durations: list[float] = []
    for idx, beat in enumerate(plan.beats, start=1):
        audio_path = audio_dir / f"beat_{idx:02d}.mp3"
        logger.info("Generating TTS for beat %d/%d...", idx, len(plan.beats))
        synthesize_tts(settings, beat.narration, audio_path, client)
        dur = probe_duration(audio_path)
        audio_paths.append(audio_path)
        audio_durations.append(dur)

    schedule = compute_beat_schedule(
        audio_durations=audio_durations,
        transitions=[b.transition_to_next for b in plan.beats],
        transition_sec=settings.transition_sec,
        beat_pause_sec=settings.beat_pause_sec,
    )

    timed_beats: list[TimedBeat] = []
    all_cues: list[SubtitleCue] = []
    beat_badges: list[tuple[float, float, str]] = []

    for idx, beat in enumerate(plan.beats):
        slot = schedule[idx]
        audio_path = audio_paths[idx]
        audio_dur = audio_durations[idx]

        aligned_segments = align_segments_to_duration(
            segments=beat.segments,
            clips_by_id=clips_by_id,
            target_duration_sec=slot["visual_duration_sec"],
        )

        cues: list[SubtitleCue] = []
        if settings.use_whisper_subs and client is not None:
            try:
                cues = extract_whisper_cues(
                    client=client,
                    model=settings.openai_transcribe_model,
                    audio_path=audio_path,
                    offset_sec=slot["voice_start_sec"],
                    time_label=beat.time_label,
                )
            except Exception as exc:
                logger.warning("Whisper alignment fallback for beat %d: %s", beat.beat_index, exc)

        if not cues:
            cues = split_text_to_cues(
                text=beat.narration,
                start_sec=slot["voice_start_sec"],
                duration_sec=audio_dur,
                time_label=beat.time_label,
            )
        all_cues.extend(cues)

        badge_end = slot["visual_start_sec"] + slot["visual_duration_sec"] - slot["t_out"]
        if beat.time_label:
            beat_badges.append((slot["visual_start_sec"], badge_end, beat.time_label))

        timed_beats.append(
            TimedBeat(
                beat_index=beat.beat_index,
                time_label=beat.time_label,
                narration=beat.narration,
                transition_to_next=beat.transition_to_next,
                audio_path=str(audio_path),
                audio_duration_sec=round(audio_dur, 3),
                voice_start_sec=slot["voice_start_sec"],
                visual_start_sec=slot["visual_start_sec"],
                visual_duration_sec=slot["visual_duration_sec"],
                segments=aligned_segments,
            )
        )

    total_duration = timed_beats[-1].visual_start_sec + timed_beats[-1].visual_duration_sec
    ass_text = build_ass_subtitles(
        cues=all_cues,
        beat_badges=beat_badges,
        width=settings.target_width,
        height=settings.target_height,
    )
    ass_path = settings.work_dir / "subtitles.ass"
    ass_path.write_text(ass_text, encoding="utf-8")

    timeline = RenderTimeline(
        title=plan.title,
        total_duration_sec=round(total_duration, 3),
        transition_sec=settings.transition_sec,
        beats=timed_beats,
        subtitles_ass_path=str(ass_path),
    )
    (settings.work_dir / "timeline.json").write_text(
        timeline.model_dump_json(indent=2), encoding="utf-8"
    )
    return timeline
