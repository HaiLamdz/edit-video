"""Unit tests for core timeline math, alignment, rotation metadata, and ASS subtitles."""

from __future__ import annotations

from pathlib import Path

from auto_vlog.director import sanitize_vlog_plan
from auto_vlog.ingest import compute_sample_timestamps
from auto_vlog.media_utils import parse_video_info
from auto_vlog.models import (
    ClipMetadata,
    IngestManifest,
    SelectedSegment,
    StoryBeat,
    TimedBeat,
    VlogPlan,
)
from auto_vlog.render import compute_xfade_steps
from auto_vlog.voice_sub import (
    align_segments_to_duration,
    build_ass_subtitles,
    compute_beat_schedule,
    split_text_to_cues,
)


def _dummy_clip(clip_id: str = "clip_01", duration: float = 10.0) -> ClipMetadata:
    return ClipMetadata(
        clip_id=clip_id,
        filename=f"{clip_id}.mp4",
        path=f"/tmp/{clip_id}.mp4",
        creation_time="2026-11-18T07:30:00+00:00",
        duration_sec=duration,
        width=1080,
        height=1920,
        fps=30.0,
        has_audio=True,
    )


def test_parse_video_info_handles_vertical_rotation() -> None:
    fake_probe = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "tags": {"rotate": "90"},
            },
            {"codec_type": "audio"},
        ],
        "format": {
            "duration": "12.5",
            "tags": {"creation_time": "2026-11-18T08:00:00Z"},
        },
    }
    meta = parse_video_info(Path("/tmp/test.mp4"), fake_probe)
    assert meta["width"] == 1080
    assert meta["height"] == 1920
    assert meta["has_audio"] is True
    assert meta["duration_sec"] == 12.5


def test_compute_sample_timestamps_skips_edges() -> None:
    stamps = compute_sample_timestamps(10.0, 5)
    assert len(stamps) == 5
    assert stamps[0] > 0.0
    assert stamps[-1] < 10.0
    assert stamps == sorted(stamps)


def test_compute_beat_schedule_and_xfade_zero_drift() -> None:
    audio_durs = [4.0, 5.2, 3.8, 6.0]
    transitions = ["fade", "none", "dissolve", "fade"]
    schedule = compute_beat_schedule(
        audio_durations=audio_durs,
        transitions=transitions,
        transition_sec=0.30,
        beat_pause_sec=0.35,
    )
    assert len(schedule) == 4
    # Voice end of beat i must never overlap voice start of beat i+1
    for i in range(len(schedule) - 1):
        voice_end = schedule[i]["voice_start_sec"] + audio_durs[i]
        assert voice_end < schedule[i + 1]["voice_start_sec"]

    timed = [
        TimedBeat(
            beat_index=i + 1,
            time_label="",
            narration="test",
            transition_to_next=transitions[i],  # type: ignore[arg-type]
            audio_path=f"/tmp/b{i}.mp3",
            audio_duration_sec=audio_durs[i],
            voice_start_sec=schedule[i]["voice_start_sec"],
            visual_start_sec=schedule[i]["visual_start_sec"],
            visual_duration_sec=schedule[i]["visual_duration_sec"],
            segments=[],
        )
        for i in range(len(audio_durs))
    ]
    steps = compute_xfade_steps(timed, transition_sec=0.30)
    assert len(steps) == 3

    # Simulate FFmpeg xfade total duration across chained beats: sum(L_i) - sum(T_i)
    chained_len = sum(b.visual_duration_sec for b in timed) - sum(float(s["duration"]) for s in steps)
    expected_len = timed[-1].visual_start_sec + timed[-1].visual_duration_sec
    assert abs(chained_len - expected_len) < 1e-2


def test_align_segments_expands_and_trims_to_exact_target() -> None:
    clips = {"clip_01": _dummy_clip("clip_01", duration=12.0)}

    # Case 1: expand short request within source clip limits
    segs_short = [SelectedSegment(clip_id="clip_01", start_sec=2.0, end_sec=4.0)]
    aligned = align_segments_to_duration(segs_short, clips, target_duration_sec=5.5)
    total = sum(s.rendered_duration_sec for s in aligned)
    assert abs(total - 5.5) < 0.02

    # Case 2: trim longer request down to target
    segs_long = [
        SelectedSegment(clip_id="clip_01", start_sec=1.0, end_sec=5.0),
        SelectedSegment(clip_id="clip_01", start_sec=6.0, end_sec=10.0),
    ]
    aligned_trim = align_segments_to_duration(segs_long, clips, target_duration_sec=4.0)
    total_trim = sum(s.rendered_duration_sec for s in aligned_trim)
    assert abs(total_trim - 4.0) < 0.02

    # Case 3: source clip physically shorter than target -> smooth slowdown fallback
    tiny_clips = {"clip_01": _dummy_clip("clip_01", duration=3.0)}
    segs_tiny = [SelectedSegment(clip_id="clip_01", start_sec=0.0, end_sec=2.5)]
    aligned_slow = align_segments_to_duration(segs_tiny, tiny_clips, target_duration_sec=4.5)
    assert aligned_slow[0].playback_rate < 1.0
    assert abs(aligned_slow[0].rendered_duration_sec - 4.5) < 0.05


def test_sanitize_vlog_plan_clamps_bounds_and_drops_unknown() -> None:
    manifest = IngestManifest(clips=[_dummy_clip("clip_01", duration=6.0)])
    raw_plan = VlogPlan(
        title="1 ngày đi Tam Đảo",
        mood="chill",
        beats=[
            StoryBeat(
                beat_index=1,
                narration="Sáng sớm tụi mình xuất phát.",
                segments=[
                    SelectedSegment(clip_id="clip_01", start_sec=-1.0, end_sec=99.0),
                    SelectedSegment(clip_id="clip_99", start_sec=0.0, end_sec=2.0),
                ],
            ),
            StoryBeat(
                beat_index=2,
                narration="Lên tới nơi trời mù sương.",
                segments=[SelectedSegment(clip_id="clip_01", start_sec=1.0, end_sec=3.0)],
            ),
            StoryBeat(
                beat_index=3,
                narration="Chiều ghé quán cà phê.",
                segments=[SelectedSegment(clip_id="clip_01", start_sec=2.0, end_sec=5.0)],
            ),
        ],
    )
    clean = sanitize_vlog_plan(raw_plan, manifest)
    assert len(clean.beats[0].segments) == 1
    assert clean.beats[0].segments[0].start_sec == 0.0
    assert clean.beats[0].segments[0].end_sec == 6.0


def test_ass_subtitle_generation() -> None:
    cues = split_text_to_cues(
        "Sáng sớm tụi mình bắt đầu lên đường đi Tam Đảo ngắm mây",
        start_sec=0.0,
        duration_sec=4.5,
        time_label="07:30 • Xuất phát",
    )
    assert len(cues) >= 2
    ass = build_ass_subtitles(cues, [(0.0, 4.5, "07:30 • Xuất phát")], 1080, 1920)
    assert "Style: Caption" in ass
    assert "Style: TimeBadge" in ass
    assert "07:30 • Xuất phát" in ass
