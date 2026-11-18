"""Pydantic data contracts shared across all pipeline stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SampledFrame(BaseModel):
    """A single keyframe extracted from a raw clip for Vision analysis."""

    timestamp_sec: float
    image_path: str


class ClipMetadata(BaseModel):
    """Metadata and sampled preview frames for one raw input video."""

    clip_id: str
    filename: str
    path: str
    creation_time: str
    duration_sec: float
    width: int
    height: int
    fps: float
    has_audio: bool
    sampled_frames: list[SampledFrame] = Field(default_factory=list)


class IngestManifest(BaseModel):
    """Output of Stage 1 (ingest)."""

    clips: list[ClipMetadata]


class BestWindow(BaseModel):
    """Clean, usable time window identified inside a single raw clip."""

    start_sec: float = Field(description="Start second of clean usable footage")
    end_sec: float = Field(description="End second of clean usable footage")
    description: str = Field(description="Brief Vietnamese description of what is visible")
    keep_ambient_audio: bool = Field(
        default=False,
        description="True if ambient sound is pleasant (food sizzling, nature, coffee), False if noisy traffic/wind",
    )


class ClipAnalysis(BaseModel):
    """Vision analysis result for a single raw clip."""

    clip_id: str
    summary: str = Field(description="Concise Vietnamese summary of the clip content")
    mood_tags: list[str] = Field(default_factory=list)
    best_windows: list[BestWindow] = Field(default_factory=list)


class SelectedSegment(BaseModel):
    """One video cut chosen by the AI director inside a story beat."""

    clip_id: str = Field(description="Exact clip_id from the manifest")
    start_sec: float = Field(description="Cut start timestamp in seconds")
    end_sec: float = Field(description="Cut end timestamp in seconds")
    keep_ambient_audio: bool = Field(
        default=False,
        description="Whether to keep soft original clip audio underneath narration",
    )
    visual_note: str = Field(default="", description="Why this visual matches the narration")


TransitionStyle = Literal["fade", "dissolve", "smoothleft", "smoothright", "slideup", "none"]


class StoryBeat(BaseModel):
    """A single narrative beat (1-2 sentences of voiceover + matching visuals)."""

    beat_index: int
    time_label: str = Field(
        default="",
        description="Short corner label like '07:30 • Xuất phát' or '12:15 • Ăn trưa'",
    )
    narration: str = Field(description="Natural Vietnamese storytelling voiceover (1-2 short sentences)")
    transition_to_next: TransitionStyle = Field(
        default="fade",
        description="Transition style from this beat to the next beat",
    )
    segments: list[SelectedSegment] = Field(min_length=1)


class VlogPlan(BaseModel):
    """Output of Stage 2 (director) — editable before Stage 3."""

    title: str = Field(description="Short title for the daily vlog")
    mood: str = Field(description="Overall vibe, e.g. chill, upbeat, nostalgic")
    beats: list[StoryBeat] = Field(min_length=3, max_length=10)


class AlignedSegment(BaseModel):
    """Video segment whose exact bounds and playback rate are locked to voiceover timing."""

    clip_id: str
    source_path: str
    start_sec: float
    end_sec: float
    playback_rate: float = 1.0
    keep_ambient_audio: bool = False
    has_audio: bool = True

    @property
    def rendered_duration_sec(self) -> float:
        raw = max(0.0, self.end_sec - self.start_sec)
        return raw / max(0.1, self.playback_rate)


class SubtitleCue(BaseModel):
    """Single caption line on the final timeline."""

    start_sec: float
    end_sec: float
    text: str
    time_label: str = ""


class TimedBeat(BaseModel):
    """Story beat aligned to exact generated voiceover audio duration."""

    beat_index: int
    time_label: str
    narration: str
    transition_to_next: TransitionStyle
    audio_path: str
    audio_duration_sec: float
    voice_start_sec: float
    visual_start_sec: float
    visual_duration_sec: float
    segments: list[AlignedSegment]


class RenderTimeline(BaseModel):
    """Output of Stage 3 (voice_sub) consumed by Stage 4 (render)."""

    title: str
    total_duration_sec: float
    transition_sec: float
    beats: list[TimedBeat]
    subtitles_ass_path: str
