"""Configuration loading for auto-vlog."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables and `.env`."""

    project_root: Path
    input_dir: Path
    music_dir: Path
    work_dir: Path
    output_dir: Path

    openai_api_key: str | None
    openai_base_url: str | None
    openai_vision_model: str
    openai_director_model: str

    tts_provider: str
    openai_tts_model: str
    openai_tts_voice: str
    edge_tts_voice: str

    use_whisper_subs: bool
    openai_transcribe_model: str

    target_width: int
    target_height: int
    target_fps: int
    target_min_sec: int
    target_max_sec: int
    transition_sec: float
    beat_pause_sec: float
    frames_per_clip: int

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> Settings:
        """Load settings from `.env` in project_root (or CWD)."""
        root = (project_root or Path.cwd()).resolve()
        load_dotenv(root / ".env", override=False)

        transition_sec = max(0.0, _env_float("TRANSITION_SEC", 0.30))
        beat_pause_sec = max(transition_sec, _env_float("BEAT_PAUSE_SEC", 0.35))

        return cls(
            project_root=root,
            input_dir=root / "input_clips",
            music_dir=root / "music",
            work_dir=root / "work",
            output_dir=root / "output",
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
            openai_director_model=os.getenv("OPENAI_DIRECTOR_MODEL", "gpt-4o"),
            tts_provider=os.getenv("TTS_PROVIDER", "openai").strip().lower(),
            openai_tts_model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            openai_tts_voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),
            edge_tts_voice=os.getenv("EDGE_TTS_VOICE", "vi-VN-HoaiMyNeural"),
            use_whisper_subs=_env_bool("USE_WHISPER_SUBS", True),
            openai_transcribe_model=os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1"),
            target_width=_env_int("TARGET_WIDTH", 1080),
            target_height=_env_int("TARGET_HEIGHT", 1920),
            target_fps=_env_int("TARGET_FPS", 30),
            target_min_sec=_env_int("TARGET_MIN_SEC", 40),
            target_max_sec=_env_int("TARGET_MAX_SEC", 60),
            transition_sec=transition_sec,
            beat_pause_sec=beat_pause_sec,
            frames_per_clip=_env_int("FRAMES_PER_CLIP", 5),
        )

    def ensure_dirs(self) -> None:
        """Create required project directories if missing."""
        for directory in (self.input_dir, self.music_dir, self.work_dir, self.output_dir):
            directory.mkdir(parents=True, exist_ok=True)
